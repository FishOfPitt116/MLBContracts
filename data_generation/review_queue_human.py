"""
Review Queue Processor (Human Review)

Interactively processes players that the AI agent couldn't match after multiple attempts.
These items have status=pending_human and include agent_notes explaining what was tried.

For each queued player, allows the user to:
1. Select from candidate matches
2. Enter a different name to search
3. Skip the player entirely (removes from queue)
4. Mark as unresolvable (keeps in queue as resolved with no match)

Once resolved, the player is added to the dataset and removed from the queue.

Usage:
    python -m data_generation.review_queue              # Process pending_human items
    python -m data_generation.review_queue --status     # Show queue statistics
    python -m data_generation.review_queue --all        # Process all items (including pending_agent)
"""

from argparse import ArgumentParser
from datetime import datetime
from typing import Dict, List

from .records import (
    Player,
    ReviewQueueItem,
    QUEUE_STATUS_PENDING_AGENT,
    QUEUE_STATUS_PENDING_HUMAN,
    QUEUE_STATUS_RESOLVED,
)
from .save import (
    read_review_queue,
    remove_review_queue_item,
    update_review_queue_item,
    write_players_to_file,
    read_players_from_file,
)
from .player_lookup import (
    get_players_by_name,
    filter_players_by_year,
    get_career_span_from_stats,
)
from .fangraphs_search import (
    search_fangraphs_by_name_range,
)


def display_queue_status():
    """Show current queue status with breakdown by status"""
    queue = read_review_queue()

    if not queue:
        print("\nReview queue is empty.")
        return

    pending_agent = [i for i in queue if i.status == QUEUE_STATUS_PENDING_AGENT]
    pending_human = [i for i in queue if i.status == QUEUE_STATUS_PENDING_HUMAN]
    resolved = [i for i in queue if i.status == QUEUE_STATUS_RESOLVED]

    print(f"\n{'='*60}")
    print(f"Review Queue Status")
    print(f"{'='*60}")
    print(f"Total items:     {len(queue)}")
    print(f"Pending (agent): {len(pending_agent)}")
    print(f"Pending (human): {len(pending_human)}")
    print(f"Resolved:        {len(resolved)}")
    print(f"{'='*60}")

    if pending_human:
        print(f"\nItems awaiting human review:")
        for i, item in enumerate(pending_human[:10], 1):
            print(f"  {i}. {item.first_name} {item.last_name} ({item.contract_year})")
            print(f"     Attempts: {item.attempt_count}, Added: {item.added_at.strftime('%Y-%m-%d')}")
        if len(pending_human) > 10:
            print(f"  ... and {len(pending_human) - 10} more")
    else:
        print("\nNo items awaiting human review.")
        if pending_agent:
            print(f"Run the agent first: python -m data_generation.review_queue_agent")

    print()


def process_queue(include_all: bool = False):
    """
    Interactively process items in the review queue.

    Args:
        include_all: If True, process all items. If False, only process pending_human items.
    """
    all_items = read_review_queue()

    if include_all:
        queue = all_items
    else:
        queue = [item for item in all_items if item.status == QUEUE_STATUS_PENDING_HUMAN]

    if not queue:
        if include_all:
            print("No players in queue.")
        else:
            print("No players pending human review.")
            print("Run the agent first: python -m data_generation.review_queue_agent")
        return

    existing_players = {p.player_id: p for p in read_players_from_file()}
    processed = 0
    skipped = 0
    marked_unresolvable = 0

    print(f"\nProcessing {len(queue)} queued players...\n")

    for item in queue:
        print(f"\n{'='*60}")
        print(f"Player: {item.first_name} {item.last_name}")
        print(f"Contract Year: {item.contract_year}")
        print(f"Spotrac: {item.spotrac_link}")
        print(f"Status: {item.status}")
        print(f"Agent Attempts: {item.attempt_count}")
        print(f"{'='*60}")

        # Show agent notes if available
        if item.agent_notes:
            print(f"\nAgent Notes:")
            print("-" * 40)
            # Format notes nicely
            for line in item.agent_notes.split('\n'):
                print(f"  {line}")
            print("-" * 40)

        if item.candidates:
            print("\nCandidate matches:")
            candidate_list = list(item.candidates.items())
            for i, (fg_id, desc) in enumerate(candidate_list):
                print(f"  [{i}] {desc} (FanGraphs ID: {fg_id})")
        else:
            print("\nNo candidate matches found.")

        print("\nOptions:")
        print("  [0-N] Select a candidate by number")
        print("  [s]   Search for a different name")
        print("  [x]   Skip this player (remove from queue)")
        print("  [u]   Mark as unresolvable (minor leaguer, etc.)")
        print("  [q]   Quit processing")

        choice = input("\nChoice: ").strip().lower()

        if choice == 'q':
            print("Exiting queue processor.")
            break

        if choice == 'x':
            print(f"Removing {item.first_name} {item.last_name} from queue")
            remove_review_queue_item(item)
            skipped += 1
            continue

        if choice == 'u':
            print(f"Marking {item.first_name} {item.last_name} as unresolvable")
            item.status = QUEUE_STATUS_RESOLVED
            item.agent_notes += f"\n[Human Review] Marked as unresolvable - {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            update_review_queue_item(item)
            marked_unresolvable += 1
            continue

        if choice == 's':
            # Search for a different name
            fangraphs_id = search_player_interactive(item.contract_year)
            if fangraphs_id == -1:
                print("No valid player selected, skipping.")
                skipped += 1
                continue
        else:
            # Try to select a candidate by index
            try:
                idx = int(choice)
                candidate_list = list(item.candidates.items())
                if 0 <= idx < len(candidate_list):
                    fangraphs_id = candidate_list[idx][0]
                else:
                    print("Invalid selection, skipping.")
                    skipped += 1
                    continue
            except ValueError:
                print("Invalid input, skipping.")
                skipped += 1
                continue

        # Create the player record
        player = create_player_from_queue_item(item, fangraphs_id, existing_players)
        if player:
            write_players_to_file([player])
            remove_review_queue_item(item)
            existing_players[player.player_id] = player
            print(f"Added {player.first_name} {player.last_name} to dataset.")
            processed += 1
        else:
            print("Failed to create player record.")
            skipped += 1

    print(f"\n{'='*60}")
    print(f"Queue processing complete:")
    print(f"  Added to dataset:      {processed}")
    print(f"  Marked unresolvable:   {marked_unresolvable}")
    print(f"  Skipped/removed:       {skipped}")
    print(f"{'='*60}\n")


def search_player_interactive(contract_year: int) -> int:
    """
    Interactive search for a player by name.

    Searches both local cache and FanGraphs stats directly,
    combines results and deduplicates by FanGraphs ID.
    """
    first_name = input("Enter first name: ").strip()
    last_name = input("Enter last name: ").strip()

    if not first_name or not last_name:
        return -1

    # Combine results from local cache and FanGraphs search
    candidates: Dict[int, str] = {}

    # Step 1: Search local cache
    cached_players = get_players_by_name(first_name, last_name)
    filtered_cached = filter_players_by_year(cached_players, contract_year)

    for player in filtered_cached:
        career_span = get_career_span_from_stats(player.player_id)
        if career_span:
            candidates[player.fangraphs_id] = f"{player.first_name} {player.last_name} ({career_span[0]}-{career_span[1]}) [cached]"
        else:
            candidates[player.fangraphs_id] = f"{player.first_name} {player.last_name} (career unknown) [cached]"

    # Step 2: Search FanGraphs stats directly
    # Search around the contract year (+/- 2 years)
    search_start = max(2000, contract_year - 2)
    search_end = min(datetime.now().year, contract_year + 2)

    print(f"Searching FanGraphs for {first_name} {last_name} ({search_start}-{search_end})...")
    fangraphs_results = search_fangraphs_by_name_range(first_name, last_name, search_start, search_end)

    for result in fangraphs_results:
        fg_id = result['fg_id']
        if fg_id not in candidates:  # Don't overwrite cached results
            first_year = result.get('first_year', '?')
            last_year = result.get('last_year', '?')
            candidates[fg_id] = f"{result['name']} ({first_year}-{last_year}) [FanGraphs]"

    if not candidates:
        print("No players found.")
        return -1

    print("\nMatches found:")
    candidate_list = list(candidates.items())
    for i, (fg_id, desc) in enumerate(candidate_list):
        print(f"  [{i}] {desc} (FanGraphs ID: {fg_id})")

    try:
        idx = int(input("\nSelect player number (-1 to cancel): "))
        if idx == -1:
            return -1
        if 0 <= idx < len(candidate_list):
            return candidate_list[idx][0]
        print("Invalid selection.")
        return -1
    except (ValueError, IndexError):
        return -1


def create_player_from_queue_item(item: ReviewQueueItem, fangraphs_id: int, existing_players: dict) -> Player:
    """Create a Player object from a queue item and selected FanGraphs ID"""
    # Build player_id from spotrac link - extract numeric ID: .../id/5166/max-scherzer -> 5166
    link_id = item.spotrac_link.split("/id/")[1].split("/")[0]
    player_id = f"{item.last_name}_{link_id}"

    if player_id in existing_players:
        print(f"Player {player_id} already exists in dataset.")
        remove_review_queue_item(item)
        return None

    return Player(
        player_id=player_id,
        fangraphs_id=fangraphs_id,
        first_name=item.first_name,
        last_name=item.last_name,
        position="",  # Will be filled in on next scrape
        spotrac_link=item.spotrac_link,
    )


def main():
    parser = ArgumentParser(description="Process the player review queue (human review)")
    parser.add_argument("--status", action="store_true", help="Show queue status without processing")
    parser.add_argument("--all", action="store_true", help="Process all items, not just pending_human")
    args = parser.parse_args()

    if args.status:
        display_queue_status()
    else:
        display_queue_status()
        if input("\nProcess queue now? [y/N]: ").strip().lower() == 'y':
            process_queue(include_all=args.all)


if __name__ == "__main__":
    main()
