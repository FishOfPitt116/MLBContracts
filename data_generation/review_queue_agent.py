"""
Review Queue Agent

An AI-powered agent that automatically processes the review queue to match
Spotrac players with their FanGraphs IDs.

Uses Strands SDK with GPT-4o-mini to:
1. Gather context from Spotrac (contracts, position, bio)
2. Search FanGraphs via pybaseball
3. Reason about the match and commit or flag for human review

The agent uses a retry mechanism:
- Each player is attempted once per run
- After MAX_AGENT_ATTEMPTS (3) failed attempts, status changes to pending_human
- Human review then takes over for truly difficult cases

Usage:
    python -m data_generation.review_queue_agent              # Process all pending
    python -m data_generation.review_queue_agent --dry-run    # Show reasoning only
    python -m data_generation.review_queue_agent --limit 10   # Process first 10
    python -m data_generation.review_queue_agent --verbose    # Detailed output
    python -m data_generation.review_queue_agent --status     # Show queue stats
"""

import os
import sys
from argparse import ArgumentParser
from datetime import datetime
from typing import Dict, List, Optional

from pybaseball import batting_stats, pitching_stats, cache

# Enable pybaseball caching
cache.enable()

from .records import (
    Player,
    ReviewQueueItem,
    QUEUE_STATUS_PENDING_AGENT,
    QUEUE_STATUS_PENDING_HUMAN,
    QUEUE_STATUS_RESOLVED,
    MAX_AGENT_ATTEMPTS,
)
from .save import (
    read_review_queue,
    remove_review_queue_item,
    update_review_queue_item,
    write_players_to_file,
    read_players_from_file,
)
from .player_lookup import normalize_name, add_player_to_cache
from .spotrac import (
    get_player_page_contracts,
    get_player_page_position,
    get_player_page_bio,
    normalize_team,
)


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """You are a player matching agent. Your job is to match Spotrac players to their FanGraphs IDs.

For each queue item, you have:
- player_name: The name from Spotrac
- spotrac_link: URL to their Spotrac page
- contract_year: Year of the contract we're trying to match
- candidates: Previous search results (may be empty or unreliable)
- previous_notes: Notes from previous matching attempts (if any)

WORKFLOW:
1. Use Spotrac tools to understand WHO this player is:
   - What teams did they play for? (get_spotrac_contracts or get_spotrac_bio)
   - What position do they play? (get_spotrac_position)
   - Any other identifying info like draft year?

2. Use search_fangraphs_players() to find matching players. This is a flexible tool with optional filters:
   - first_name, last_name: Search by name (try variations if needed)
   - team: Filter by team abbreviation (NYY, LAD, etc.)
   - year: Search a specific year
   - start_year, end_year: Search a year range
   - player_type: Filter by "batter" or "pitcher"

   Search strategies:
   - Start with full name: first_name="Mike", last_name="Trout"
   - Try name variations: B.J.->BJ, Mike->Michael, Christopher->Chris
   - If too many results, add team filter: last_name="Smith", team="NYY", year=2015
   - If no results, try last name only: last_name="Carpenter"
   - Combine filters to narrow down: last_name="Gonzalez", team="BAL", player_type="pitcher"

3. If you find a confident match:
   - Call commit_match() with the FanGraphs ID and your reasoning

4. If you cannot find a match or are uncertain:
   - Call flag_for_review() explaining what you tried and why uncertain

MATCHING RULES:
- A match must have overlapping career years with the contract
- Team + year is strong evidence (player on same team in same year)
- Position should match (pitcher vs batter)
- Be conservative - flag uncertain cases rather than commit wrong matches
- For common names with multiple candidates, use team history to disambiguate

NAME VARIATIONS TO TRY:
- B.J. / B. J. -> BJ
- Mike -> Michael
- Chris -> Christopher
- Matt -> Matthew
- Jonathon -> Jon / Jonathan
- Nate -> Nathan / Nathaniel
- Phil -> Philip / Phillip
- Vincente -> Vicente

IMPORTANT:
- Always call exactly ONE resolution tool (commit_match OR flag_for_review)
- Never guess - if truly uncertain, flag for human review
- Minor leaguers may not be in FanGraphs - flag these cases
- If previous attempts failed, try different approaches before flagging again"""


# =============================================================================
# TOOL IMPLEMENTATIONS
# =============================================================================

# --- Spotrac Tools ---

def tool_get_spotrac_contracts(spotrac_link: str) -> List[Dict]:
    """Get contract history from a player's Spotrac page."""
    return get_player_page_contracts(spotrac_link)


def tool_get_spotrac_position(spotrac_link: str) -> str:
    """Get player's primary position from Spotrac page."""
    return get_player_page_position(spotrac_link)


def tool_get_spotrac_bio(spotrac_link: str) -> Dict:
    """Get biographical info from a player's Spotrac page."""
    result = get_player_page_bio(spotrac_link)
    if result.get("birth_date"):
        result["birth_date"] = result["birth_date"].strftime("%Y-%m-%d")
    return result


# --- PyBaseball Tools ---

def tool_search_fangraphs_players(
    first_name: str = None,
    last_name: str = None,
    team: str = None,
    year: int = None,
    start_year: int = None,
    end_year: int = None,
    player_type: str = None,
) -> List[Dict]:
    """
    Flexible search for players in FanGraphs database.

    All parameters are optional, but at least one of first_name, last_name, or team must be provided.

    Args:
        first_name: Filter by first name (uses normalized matching for accents, periods, etc.)
        last_name: Filter by last name (uses normalized matching)
        team: Filter by team abbreviation (e.g., "NYY", "LAD", "TEX")
        year: Search only this specific year
        start_year: Start of year range to search (default: 2008)
        end_year: End of year range to search (default: current year)
        player_type: Filter by "batter" or "pitcher" (default: search both)

    Returns:
        List of matching players with fg_id, name, first_year, last_year, type, teams

    Examples:
        - Search by full name: first_name="Mike", last_name="Trout"
        - Search by last name only: last_name="Smith" (returns all Smiths)
        - Search by team and year: team="NYY", year=2015
        - Search pitchers named David: first_name="David", player_type="pitcher"
        - Combine filters: last_name="Carpenter", team="ATL", start_year=2013, end_year=2015
    """
    # Validate inputs
    if not any([first_name, last_name, team]):
        return [{"error": "Must provide at least one of: first_name, last_name, or team"}]

    # Normalize team if provided
    if team:
        team = normalize_team(team)

    # Set year range
    current_year = datetime.now().year
    if year:
        # Single year search
        search_start = year
        search_end = year
    else:
        search_start = start_year if start_year else 2008
        search_end = end_year if end_year else current_year

    # Normalize names for comparison
    first_normalized = normalize_name(first_name) if first_name else None
    last_normalized = normalize_name(last_name) if last_name else None

    results_by_id: Dict[int, Dict] = {}

    def matches_filters(row, stats_type: str) -> bool:
        """Check if a row matches all provided filters."""
        # Check player type filter
        if player_type:
            if player_type == "batter" and stats_type != "batter":
                return False
            if player_type == "pitcher" and stats_type != "pitcher":
                return False

        # Check name filters
        name = str(row['Name'])
        parts = name.split(" ", 1)
        if len(parts) != 2:
            return False

        fg_first, fg_last = parts

        if first_normalized and normalize_name(fg_first) != first_normalized:
            return False
        if last_normalized and normalize_name(fg_last) != last_normalized:
            return False

        # Check team filter
        if team:
            row_team = normalize_team(str(row.get('Team', '')))
            if row_team != team:
                return False

        return True

    def add_result(row, year: int, stats_type: str):
        """Add or update a result entry."""
        fg_id = int(row['IDfg'])
        row_team = str(row.get('Team', ''))

        if fg_id not in results_by_id:
            results_by_id[fg_id] = {
                'fg_id': fg_id,
                'name': row['Name'],
                'first_year': year,
                'last_year': year,
                'type': stats_type,
                'teams': [row_team] if row_team else []
            }
        else:
            results_by_id[fg_id]['first_year'] = min(results_by_id[fg_id]['first_year'], year)
            results_by_id[fg_id]['last_year'] = max(results_by_id[fg_id]['last_year'], year)
            if row_team and row_team not in results_by_id[fg_id]['teams']:
                results_by_id[fg_id]['teams'].append(row_team)

    # Search through years
    for search_year in range(search_start, search_end + 1):
        # Search batting stats (unless filtering for pitchers only)
        if player_type != "pitcher":
            try:
                batting = batting_stats(search_year, qual=0)
                if not batting.empty and 'Name' in batting.columns and 'IDfg' in batting.columns:
                    for _, row in batting.iterrows():
                        if matches_filters(row, "batter"):
                            add_result(row, search_year, "batter")
            except Exception:
                pass

        # Search pitching stats (unless filtering for batters only)
        if player_type != "batter":
            try:
                pitching = pitching_stats(search_year, qual=0)
                if not pitching.empty and 'Name' in pitching.columns and 'IDfg' in pitching.columns:
                    for _, row in pitching.iterrows():
                        if matches_filters(row, "pitcher"):
                            add_result(row, search_year, "pitcher")
            except Exception:
                pass

    return list(results_by_id.values())


def tool_get_player_season_stats(fangraphs_id: int, year: int) -> Dict:
    """Get a player's stats for a specific season to verify they were active."""
    result = {
        'fg_id': fangraphs_id,
        'year': year,
        'found': False,
        'team': None,
        'type': None,
    }

    try:
        batting = batting_stats(year, qual=0)
        if not batting.empty and 'IDfg' in batting.columns:
            player_row = batting[batting['IDfg'] == fangraphs_id]
            if not player_row.empty:
                result['found'] = True
                result['type'] = 'batting'
                result['team'] = str(player_row['Team'].iloc[0])
                result['games'] = int(player_row['G'].iloc[0]) if 'G' in player_row.columns else None
                return result
    except Exception:
        pass

    try:
        pitching = pitching_stats(year, qual=0)
        if not pitching.empty and 'IDfg' in pitching.columns:
            player_row = pitching[pitching['IDfg'] == fangraphs_id]
            if not player_row.empty:
                result['found'] = True
                result['type'] = 'pitching'
                result['team'] = str(player_row['Team'].iloc[0])
                result['games'] = int(player_row['G'].iloc[0]) if 'G' in player_row.columns else None
                return result
    except Exception:
        pass

    return result


# --- Resolution Tools ---

# Global state for tracking current item and results (set by process_queue_item)
_current_item: Optional[ReviewQueueItem] = None
_resolution_result: Optional[Dict] = None
_dry_run: bool = False
_existing_players: Dict[str, Player] = {}


def tool_commit_match(fangraphs_id: int, reasoning: str) -> Dict:
    """
    Commit a match - associate the FanGraphs ID with this player.
    Call this when you're confident about the match.
    """
    global _resolution_result

    if _current_item is None:
        return {"success": False, "error": "No current queue item"}

    # Build player_id from spotrac link
    link_id = _current_item.spotrac_link.split("/id/")[1].split("/")[0]
    player_id = f"{_current_item.last_name}_{link_id}"

    if player_id in _existing_players:
        _resolution_result = {
            "action": "skipped",
            "reason": f"Player {player_id} already exists in dataset",
        }
        return {"success": False, "error": f"Player {player_id} already exists"}

    _resolution_result = {
        "action": "commit",
        "fangraphs_id": fangraphs_id,
        "reasoning": reasoning,
        "player_id": player_id,
    }

    if not _dry_run:
        # Create and save the player
        player = Player(
            player_id=player_id,
            fangraphs_id=fangraphs_id,
            first_name=_current_item.first_name,
            last_name=_current_item.last_name,
            position="",  # Will be filled in on next scrape
            spotrac_link=_current_item.spotrac_link,
        )
        write_players_to_file([player])
        add_player_to_cache(player)
        remove_review_queue_item(_current_item)
        _existing_players[player_id] = player

    return {"success": True, "player_id": player_id, "fangraphs_id": fangraphs_id}


def tool_flag_for_review(reasoning: str) -> Dict:
    """
    Flag this player - you couldn't find a confident match.
    The attempt count will be incremented and reasoning recorded.
    """
    global _resolution_result

    if _current_item is None:
        return {"success": False, "error": "No current queue item"}

    _resolution_result = {
        "action": "flag",
        "reasoning": reasoning,
    }

    # Note: The actual queue update happens in process_queue_item after we return
    return {"success": True, "action": "flagged"}


# =============================================================================
# AGENT SETUP
# =============================================================================

def create_agent():
    """Create and return the Strands agent with all tools."""
    try:
        from strands import Agent, tool
        from strands.models.openai import OpenAIModel
    except ImportError:
        print("Error: strands-agents package not installed.")
        print("Install with: pip install strands-agents openai")
        print("")
        print("Note: strands-agents requires Python >= 3.10")
        print(f"Current Python version: {sys.version}")
        sys.exit(1)

    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set.")
        sys.exit(1)

    # Create model
    model = OpenAIModel(
        model_id="gpt-4o-mini",
        params={"temperature": 0.1}
    )

    # Wrap our tool functions with @tool decorator
    @tool
    def get_spotrac_contracts(spotrac_link: str) -> List[Dict]:
        """Get contract history from a player's Spotrac page. Returns list of contracts with year, team, value. Use this to understand what teams the player played for."""
        return tool_get_spotrac_contracts(spotrac_link)

    @tool
    def get_spotrac_position(spotrac_link: str) -> str:
        """Get player's primary position from Spotrac page. Returns position code like SP, CF, 1B. Use this to verify pitcher vs batter."""
        return tool_get_spotrac_position(spotrac_link)

    @tool
    def get_spotrac_bio(spotrac_link: str) -> Dict:
        """Get biographical info from a player's Spotrac page. Returns birth_date, draft_year, debut_year, and teams list."""
        return tool_get_spotrac_bio(spotrac_link)

    @tool
    def search_fangraphs_players(
        first_name: str = None,
        last_name: str = None,
        team: str = None,
        year: int = None,
        start_year: int = None,
        end_year: int = None,
        player_type: str = None,
    ) -> List[Dict]:
        """
        Flexible search for players in FanGraphs. All parameters optional but must provide at least one of first_name, last_name, or team.

        Args:
            first_name: Filter by first name (normalized matching handles accents, periods)
            last_name: Filter by last name
            team: Filter by team abbreviation (NYY, LAD, TEX, etc.)
            year: Search only this specific year
            start_year: Start of year range (default 2008)
            end_year: End of year range (default current year)
            player_type: Filter by "batter" or "pitcher"

        Examples:
            - Full name: first_name="Mike", last_name="Trout"
            - Last name only: last_name="Smith"
            - Team + year: team="NYY", year=2015
            - Combine: last_name="Carpenter", team="ATL", year=2013
        """
        return tool_search_fangraphs_players(
            first_name=first_name,
            last_name=last_name,
            team=team,
            year=year,
            start_year=start_year,
            end_year=end_year,
            player_type=player_type,
        )

    @tool
    def get_player_season_stats(fangraphs_id: int, year: int) -> Dict:
        """Get a player's stats for a specific season. Use to verify player was active and on which team."""
        return tool_get_player_season_stats(fangraphs_id, year)

    @tool
    def commit_match(fangraphs_id: int, reasoning: str) -> Dict:
        """Commit a match - associate the FanGraphs ID with this player. Call when confident about the match."""
        return tool_commit_match(fangraphs_id, reasoning)

    @tool
    def flag_for_review(reasoning: str) -> Dict:
        """Flag this player - you couldn't find a confident match. Explain what you tried."""
        return tool_flag_for_review(reasoning)

    # Create agent with all tools
    agent = Agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        tools=[
            get_spotrac_contracts,
            get_spotrac_position,
            get_spotrac_bio,
            search_fangraphs_players,
            get_player_season_stats,
            commit_match,
            flag_for_review,
        ]
    )

    return agent


# =============================================================================
# PROCESSING LOGIC
# =============================================================================

def process_queue_item(agent, item: ReviewQueueItem, verbose: bool = False) -> Dict:
    """Process a single queue item through the agent."""
    global _current_item, _resolution_result

    _current_item = item
    _resolution_result = None

    # Build the prompt for this item
    candidates_str = ""
    if item.candidates:
        candidates_str = "\n".join([f"  - FG ID {fg_id}: {desc}" for fg_id, desc in item.candidates.items()])
    else:
        candidates_str = "  (no previous candidates)"

    previous_notes = ""
    if item.agent_notes:
        previous_notes = f"\n\nPrevious Attempt Notes (attempt {item.attempt_count}):\n{item.agent_notes}"

    prompt = f"""Please match this player to their FanGraphs ID:

Player Name: {item.first_name} {item.last_name}
Spotrac Link: {item.spotrac_link}
Contract Year: {item.contract_year}

Previous Candidate Matches:
{candidates_str}{previous_notes}

Use the available tools to gather information and then call either commit_match() or flag_for_review()."""

    if verbose:
        print(f"\n{'='*60}")
        print(f"Processing: {item.first_name} {item.last_name} ({item.contract_year})")
        print(f"Spotrac: {item.spotrac_link}")
        print(f"Attempt: {item.attempt_count + 1}/{MAX_AGENT_ATTEMPTS}")
        print(f"{'='*60}")

    # Run the agent
    try:
        response = agent(prompt)

        if verbose:
            print(f"\nAgent response: {response}")

        if _resolution_result:
            return _resolution_result
        else:
            return {"action": "error", "reason": "Agent did not call resolution tool"}

    except Exception as e:
        return {"action": "error", "reason": str(e)}


def show_queue_status():
    """Display statistics about the review queue."""
    queue = read_review_queue()

    if not queue:
        print("Review queue is empty.")
        return

    pending_agent = [i for i in queue if i.status == QUEUE_STATUS_PENDING_AGENT]
    pending_human = [i for i in queue if i.status == QUEUE_STATUS_PENDING_HUMAN]
    resolved = [i for i in queue if i.status == QUEUE_STATUS_RESOLVED]

    # Count by attempt level
    by_attempts = {}
    for item in pending_agent:
        by_attempts[item.attempt_count] = by_attempts.get(item.attempt_count, 0) + 1

    print(f"\n{'='*60}")
    print(f"Review Queue Status")
    print(f"{'='*60}")
    print(f"Total items:     {len(queue)}")
    print(f"Pending (agent): {len(pending_agent)}")
    print(f"Pending (human): {len(pending_human)}")
    print(f"Resolved:        {len(resolved)}")
    print(f"\nPending agent items by attempt count:")
    for attempts in sorted(by_attempts.keys()):
        print(f"  {attempts} attempts: {by_attempts[attempts]}")
    print(f"{'='*60}\n")


def run_agent(limit: int = None, dry_run: bool = False, verbose: bool = False):
    """Run the review queue agent."""
    global _dry_run, _existing_players

    _dry_run = dry_run
    _existing_players = {p.player_id: p for p in read_players_from_file()}

    # Read queue and filter to items needing agent review
    all_items = read_review_queue()
    queue = [item for item in all_items if item.needs_agent_review()]

    if not queue:
        print("No players pending agent review.")
        show_queue_status()
        return

    if limit:
        queue = queue[:limit]

    print(f"\n{'='*60}")
    print(f"Review Queue Agent")
    print(f"{'='*60}")
    print(f"Items to process: {len(queue)}")
    print(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}")
    print(f"{'='*60}\n")

    # Create agent
    agent = create_agent()

    # Track results
    committed = 0
    flagged = 0
    escalated = 0  # Moved to pending_human
    errors = 0

    for i, item in enumerate(queue, 1):
        print(f"\n[{i}/{len(queue)}] {item.first_name} {item.last_name} ({item.contract_year}) - attempt {item.attempt_count + 1}/{MAX_AGENT_ATTEMPTS}")

        result = process_queue_item(agent, item, verbose)

        if result.get("action") == "commit":
            print(f"  -> MATCHED: FG ID {result.get('fangraphs_id')}")
            if result.get('reasoning'):
                print(f"     Reasoning: {result.get('reasoning', 'N/A')[:80]}...")
            committed += 1

        elif result.get("action") == "flag":
            reasoning = result.get('reasoning', 'No reasoning provided')
            new_attempt_count = item.attempt_count + 1

            # Append reasoning to notes
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            new_note = f"[Attempt {new_attempt_count} - {timestamp}] {reasoning}"
            if item.agent_notes:
                new_notes = f"{item.agent_notes}\n{new_note}"
            else:
                new_notes = new_note

            if new_attempt_count >= MAX_AGENT_ATTEMPTS:
                # Escalate to human review
                print(f"  -> ESCALATED to human review after {MAX_AGENT_ATTEMPTS} attempts")
                print(f"     Last reasoning: {reasoning[:80]}...")
                if not _dry_run:
                    item.attempt_count = new_attempt_count
                    item.status = QUEUE_STATUS_PENDING_HUMAN
                    item.agent_notes = new_notes
                    update_review_queue_item(item)
                escalated += 1
            else:
                # Keep in queue for next run
                print(f"  -> FLAGGED (attempt {new_attempt_count}/{MAX_AGENT_ATTEMPTS})")
                print(f"     Reasoning: {reasoning[:80]}...")
                if not _dry_run:
                    item.attempt_count = new_attempt_count
                    item.agent_notes = new_notes
                    update_review_queue_item(item)
                flagged += 1

        elif result.get("action") == "skipped":
            print(f"  -> SKIPPED: {result.get('reason')}")
            if not _dry_run:
                remove_review_queue_item(item)

        else:
            print(f"  -> ERROR: {result.get('reason', 'Unknown error')}")
            errors += 1

    # Summary
    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    print(f"Processed:  {len(queue)}")
    print(f"Committed:  {committed}")
    print(f"Flagged:    {flagged} (will retry next run)")
    print(f"Escalated:  {escalated} (sent to human review)")
    print(f"Errors:     {errors}")
    if dry_run:
        print(f"\n(DRY RUN - no changes were made)")
    print(f"{'='*60}\n")


def main():
    parser = ArgumentParser(description="AI-powered review queue processor")
    parser.add_argument("--limit", type=int, help="Process only the first N items")
    parser.add_argument("--dry-run", action="store_true", help="Show reasoning without making changes")
    parser.add_argument("--verbose", action="store_true", help="Show detailed agent output")
    parser.add_argument("--status", action="store_true", help="Show queue status and exit")
    args = parser.parse_args()

    if args.status:
        show_queue_status()
    else:
        run_agent(limit=args.limit, dry_run=args.dry_run, verbose=args.verbose)


if __name__ == "__main__":
    main()
