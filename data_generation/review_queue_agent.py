"""
Review Queue Agent

An AI-powered agent that automatically processes the review queue to match
Spotrac players with their FanGraphs IDs.

Uses Strands SDK with GPT-4o-mini to:
1. Gather context from Spotrac (contracts, position, bio)
2. Search FanGraphs via pybaseball
3. Reason about the match and commit or flag for human review

Usage:
    python -m data_generation.review_queue_agent              # Process all
    python -m data_generation.review_queue_agent --dry-run    # Show reasoning only
    python -m data_generation.review_queue_agent --limit 10   # Process first 10
    python -m data_generation.review_queue_agent --verbose    # Detailed output
"""

import csv
import os
import sys
from argparse import ArgumentParser
from datetime import datetime
from typing import Dict, List, Optional

from pybaseball import batting_stats, pitching_stats, cache

# Enable pybaseball caching
cache.enable()

from .records import Player, ReviewQueueItem
from .save import (
    read_review_queue,
    remove_review_queue_item,
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

# File for logging flagged items
HUMAN_REVIEW_LOG = "dataset/human_review_log.csv"


# =============================================================================
# SYSTEM PROMPT
# =============================================================================

SYSTEM_PROMPT = """You are a player matching agent. Your job is to match Spotrac players to their FanGraphs IDs.

For each queue item, you have:
- player_name: The name from Spotrac
- spotrac_link: URL to their Spotrac page
- contract_year: Year of the contract we're trying to match
- candidates: Previous search results (may be empty or unreliable)

WORKFLOW:
1. Use Spotrac tools to understand WHO this player is:
   - What teams did they play for? (get_spotrac_contracts or get_spotrac_bio)
   - What position do they play? (get_spotrac_position)
   - Any other identifying info like draft year?

2. Use PyBaseball tools to find matching players:
   - Search by name (try variations if needed: B.J.→BJ, Mike→Michael, Christopher→Chris)
   - Search by team + year if name search fails or returns too many
   - Verify career timeline matches

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
- B.J. / B. J. → BJ
- Mike → Michael
- Chris → Christopher
- Matt → Matthew
- Jonathon → Jon / Jonathan
- Nate → Nathan / Nathaniel
- Phil → Philip / Phillip

IMPORTANT:
- Always call exactly ONE resolution tool (commit_match OR flag_for_review)
- Never guess - if truly uncertain, flag for human review
- Minor leaguers may not be in FanGraphs - flag these cases"""


# =============================================================================
# TOOL IMPLEMENTATIONS
# =============================================================================

# --- Spotrac Tools ---

def tool_get_spotrac_contracts(spotrac_link: str) -> List[Dict]:
    """
    Get contract history from a player's Spotrac page.

    Returns list of contracts with year, team, value, and contract_type.
    Use this to understand what teams the player played for.

    Args:
        spotrac_link: Full URL to the player's Spotrac page
    """
    return get_player_page_contracts(spotrac_link)


def tool_get_spotrac_position(spotrac_link: str) -> str:
    """
    Get player's primary position from Spotrac page.

    Returns position code like "SP", "CF", "1B", etc.
    Use this to verify pitcher vs batter and filter searches.

    Args:
        spotrac_link: Full URL to the player's Spotrac page
    """
    return get_player_page_position(spotrac_link)


def tool_get_spotrac_bio(spotrac_link: str) -> Dict:
    """
    Get biographical info from a player's Spotrac page.

    Returns birth_date, draft_year, debut_year, and teams list.
    Useful for narrowing down player identity.

    Args:
        spotrac_link: Full URL to the player's Spotrac page
    """
    result = get_player_page_bio(spotrac_link)
    # Convert datetime to string for JSON serialization
    if result.get("birth_date"):
        result["birth_date"] = result["birth_date"].strftime("%Y-%m-%d")
    return result


# --- PyBaseball Tools ---

def tool_search_players_by_name(first_name: str, last_name: str, start_year: int = 2010, end_year: int = None) -> List[Dict]:
    """
    Search FanGraphs for players by name across a range of years.

    Returns list of matching players with fg_id, name, first_year, last_year.
    Handles name normalization (accents, periods, etc.)

    Args:
        first_name: Player's first name
        last_name: Player's last name
        start_year: First year to search (default 2010)
        end_year: Last year to search (default current year)
    """
    if end_year is None:
        end_year = datetime.now().year

    first_normalized = normalize_name(first_name)
    last_normalized = normalize_name(last_name)

    results_by_id: Dict[int, Dict] = {}

    for year in range(start_year, end_year + 1):
        try:
            # Search batting stats
            batting = batting_stats(year, qual=0)
            if not batting.empty and 'Name' in batting.columns and 'IDfg' in batting.columns:
                for _, row in batting.iterrows():
                    name = str(row['Name'])
                    parts = name.split(" ", 1)
                    if len(parts) == 2:
                        fg_first, fg_last = parts
                        if (normalize_name(fg_first) == first_normalized and
                            normalize_name(fg_last) == last_normalized):
                            fg_id = int(row['IDfg'])
                            if fg_id not in results_by_id:
                                results_by_id[fg_id] = {
                                    'fg_id': fg_id,
                                    'name': name,
                                    'first_year': year,
                                    'last_year': year,
                                    'type': 'batter'
                                }
                            else:
                                results_by_id[fg_id]['first_year'] = min(results_by_id[fg_id]['first_year'], year)
                                results_by_id[fg_id]['last_year'] = max(results_by_id[fg_id]['last_year'], year)
        except Exception:
            pass

        try:
            # Search pitching stats
            pitching = pitching_stats(year, qual=0)
            if not pitching.empty and 'Name' in pitching.columns and 'IDfg' in pitching.columns:
                for _, row in pitching.iterrows():
                    name = str(row['Name'])
                    parts = name.split(" ", 1)
                    if len(parts) == 2:
                        fg_first, fg_last = parts
                        if (normalize_name(fg_first) == first_normalized and
                            normalize_name(fg_last) == last_normalized):
                            fg_id = int(row['IDfg'])
                            if fg_id not in results_by_id:
                                results_by_id[fg_id] = {
                                    'fg_id': fg_id,
                                    'name': name,
                                    'first_year': year,
                                    'last_year': year,
                                    'type': 'pitcher'
                                }
                            else:
                                results_by_id[fg_id]['first_year'] = min(results_by_id[fg_id]['first_year'], year)
                                results_by_id[fg_id]['last_year'] = max(results_by_id[fg_id]['last_year'], year)
        except Exception:
            pass

    return list(results_by_id.values())


def tool_search_players_by_team_year(team: str, year: int, position: str = None) -> List[Dict]:
    """
    Find all players on a team in a given year.

    Returns list of players with fg_id, name, position.
    Use this when name search returns multiple candidates and you know the team.

    Args:
        team: Team abbreviation (e.g., "NYY", "LAD", "TEX")
        year: Season year
        position: Optional position filter ("P" for pitchers, None for batters)
    """
    team = normalize_team(team)

    results = []
    seen_ids = set()

    try:
        if position != "P":
            batting = batting_stats(year, qual=0)
            if not batting.empty and 'Team' in batting.columns:
                for _, row in batting.iterrows():
                    if normalize_team(str(row['Team'])) == team:
                        fg_id = int(row['IDfg'])
                        if fg_id not in seen_ids:
                            seen_ids.add(fg_id)
                            results.append({
                                'fg_id': fg_id,
                                'name': row['Name'],
                                'team': team,
                                'year': year,
                            })
    except Exception:
        pass

    try:
        if position is None or position == "P":
            pitching = pitching_stats(year, qual=0)
            if not pitching.empty and 'Team' in pitching.columns:
                for _, row in pitching.iterrows():
                    if normalize_team(str(row['Team'])) == team:
                        fg_id = int(row['IDfg'])
                        if fg_id not in seen_ids:
                            seen_ids.add(fg_id)
                            results.append({
                                'fg_id': fg_id,
                                'name': row['Name'],
                                'team': team,
                                'year': year,
                            })
    except Exception:
        pass

    return results


def tool_get_player_career_range(fangraphs_id: int) -> Dict:
    """
    Get first and last year a player was active in FanGraphs.

    Returns dict with first_year, last_year.
    Use this to verify career timeline matches contract year.

    Args:
        fangraphs_id: FanGraphs player ID
    """
    first_year = None
    last_year = None
    current_year = datetime.now().year

    # Check years from 2000 to current
    for year in range(2000, current_year + 1):
        try:
            batting = batting_stats(year, qual=0)
            if not batting.empty and 'IDfg' in batting.columns:
                if fangraphs_id in batting['IDfg'].values:
                    if first_year is None:
                        first_year = year
                    last_year = year
        except Exception:
            pass

        try:
            pitching = pitching_stats(year, qual=0)
            if not pitching.empty and 'IDfg' in pitching.columns:
                if fangraphs_id in pitching['IDfg'].values:
                    if first_year is None:
                        first_year = year
                    last_year = year
        except Exception:
            pass

    return {
        'fg_id': fangraphs_id,
        'first_year': first_year,
        'last_year': last_year,
    }


def tool_get_player_season_stats(fangraphs_id: int, year: int) -> Dict:
    """
    Get a player's stats for a specific season.

    Returns dict with team, stats type (batting/pitching), and key stats.
    Use this to verify a player was active in a specific year.

    Args:
        fangraphs_id: FanGraphs player ID
        year: Season year
    """
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
    The player will be added to the dataset and removed from the queue.

    Args:
        fangraphs_id: The FanGraphs player ID to associate
        reasoning: Brief explanation of why this is the correct match
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
    Flag this player for human review.

    Call this when you cannot find a confident match.
    The player will be logged for manual review but remain in the queue.

    Args:
        reasoning: Explanation of what you tried and why you're uncertain
    """
    global _resolution_result

    if _current_item is None:
        return {"success": False, "error": "No current queue item"}

    _resolution_result = {
        "action": "flag",
        "reasoning": reasoning,
    }

    if not _dry_run:
        # Log to human review file
        file_exists = os.path.isfile(HUMAN_REVIEW_LOG)
        with open(HUMAN_REVIEW_LOG, mode="a", newline="") as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(["first_name", "last_name", "contract_year", "spotrac_link", "reasoning", "flagged_at"])
            writer.writerow([
                _current_item.first_name,
                _current_item.last_name,
                _current_item.contract_year,
                _current_item.spotrac_link,
                reasoning,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ])
        # Remove from queue since it's now in the human review log
        remove_review_queue_item(_current_item)

    return {"success": True, "action": "flagged_for_review"}


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
    def search_players_by_name(first_name: str, last_name: str, start_year: int = 2010, end_year: int = None) -> List[Dict]:
        """Search FanGraphs for players by name. Returns list of matching players with fg_id, name, first_year, last_year. Try name variations if no results."""
        return tool_search_players_by_name(first_name, last_name, start_year, end_year)

    @tool
    def search_players_by_team_year(team: str, year: int, position: str = None) -> List[Dict]:
        """Find all players on a team in a given year. Returns list with fg_id, name. Use when you know the team from Spotrac."""
        return tool_search_players_by_team_year(team, year, position)

    @tool
    def get_player_career_range(fangraphs_id: int) -> Dict:
        """Get first and last year a player was active. Use to verify career timeline matches contract year."""
        return tool_get_player_career_range(fangraphs_id)

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
        """Flag this player for human review. Call when you cannot find a confident match."""
        return tool_flag_for_review(reasoning)

    # Create agent with all tools
    agent = Agent(
        model=model,
        system_prompt=SYSTEM_PROMPT,
        tools=[
            get_spotrac_contracts,
            get_spotrac_position,
            get_spotrac_bio,
            search_players_by_name,
            search_players_by_team_year,
            get_player_career_range,
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

    prompt = f"""Please match this player to their FanGraphs ID:

Player Name: {item.first_name} {item.last_name}
Spotrac Link: {item.spotrac_link}
Contract Year: {item.contract_year}

Previous Candidate Matches:
{candidates_str}

Use the available tools to gather information and then call either commit_match() or flag_for_review()."""

    if verbose:
        print(f"\n{'='*60}")
        print(f"Processing: {item.first_name} {item.last_name} ({item.contract_year})")
        print(f"Spotrac: {item.spotrac_link}")
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


def run_agent(limit: int = None, dry_run: bool = False, verbose: bool = False):
    """Run the review queue agent."""
    global _dry_run, _existing_players

    _dry_run = dry_run
    _existing_players = {p.player_id: p for p in read_players_from_file()}

    queue = read_review_queue()

    if not queue:
        print("No players in review queue.")
        return

    if limit:
        queue = queue[:limit]

    print(f"\n{'='*60}")
    print(f"Review Queue Agent")
    print(f"{'='*60}")
    print(f"Queue size: {len(queue)} players")
    print(f"Mode: {'DRY RUN' if dry_run else 'LIVE'}")
    print(f"{'='*60}\n")

    # Create agent
    agent = create_agent()

    # Track results
    committed = 0
    flagged = 0
    errors = 0

    for i, item in enumerate(queue, 1):
        print(f"\n[{i}/{len(queue)}] {item.first_name} {item.last_name} ({item.contract_year})")

        result = process_queue_item(agent, item, verbose)

        if result.get("action") == "commit":
            print(f"  -> MATCHED: FG ID {result.get('fangraphs_id')}")
            print(f"     Reasoning: {result.get('reasoning', 'N/A')[:80]}...")
            committed += 1
        elif result.get("action") == "flag":
            print(f"  -> FLAGGED for human review")
            print(f"     Reasoning: {result.get('reasoning', 'N/A')[:80]}...")
            flagged += 1
        elif result.get("action") == "skipped":
            print(f"  -> SKIPPED: {result.get('reason')}")
        else:
            print(f"  -> ERROR: {result.get('reason', 'Unknown error')}")
            errors += 1

    # Summary
    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    print(f"Processed: {len(queue)}")
    print(f"Committed: {committed}")
    print(f"Flagged:   {flagged}")
    print(f"Errors:    {errors}")
    if dry_run:
        print(f"\n(DRY RUN - no changes were made)")
    print(f"{'='*60}\n")


def main():
    parser = ArgumentParser(description="AI-powered review queue processor")
    parser.add_argument("--limit", type=int, help="Process only the first N items")
    parser.add_argument("--dry-run", action="store_true", help="Show reasoning without making changes")
    parser.add_argument("--verbose", action="store_true", help="Show detailed agent output")
    args = parser.parse_args()

    run_agent(limit=args.limit, dry_run=args.dry_run, verbose=args.verbose)


if __name__ == "__main__":
    main()
