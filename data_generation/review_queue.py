"""
Review Queue Processor

Interactively processes players that were queued during non-interactive scraping runs.
For each queued player, allows the user to:
1. Select from candidate matches
2. Enter a different name to search
3. Skip the player entirely

Once resolved, the player is added to the dataset and removed from the queue.
"""

from argparse import ArgumentParser
from datetime import datetime
from typing import Dict, List

from .records import Player
from .save import (
    read_review_queue,
    remove_review_queue_item,
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
    """Show current queue status"""
    queue = read_review_queue()
    print(f"\n{'='*60}")
    print(f"Review Queue: {len(queue)} players pending")
    print(f"{'='*60}\n")

    if not queue:
        print("No players in queue. Run with --non-interactive to populate.")
        return

    for i, item in enumerate(queue, 1):
        candidate_count = len(item.candidates)
        print(f"  {i}. {item.first_name} {item.last_name} ({item.contract_year})")
        print(f"     Candidates: {candidate_count}, Added: {item.added_at.strftime('%Y-%m-%d')}")


def process_queue():
    """Interactively process all items in the review queue"""
    queue = read_review_queue()

    if not queue:
        print("No players in queue.")
        return

    existing_players = {p.player_id: p for p in read_players_from_file()}
    processed = 0
    skipped = 0

    print(f"\nProcessing {len(queue)} queued players...\n")

    for item in queue:
        print(f"\n{'='*60}")
        print(f"Player: {item.first_name} {item.last_name}")
        print(f"Contract Year: {item.contract_year}")
        print(f"Spotrac: {item.spotrac_link}")
        print(f"{'='*60}")

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
        print("  [x]   Skip this player")
        print("  [q]   Quit processing")

        choice = input("\nChoice: ").strip().lower()

        if choice == 'q':
            print("Exiting queue processor.")
            break

        if choice == 'x':
            print(f"Skipping {item.first_name} {item.last_name}")
            skipped += 1
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
    print(f"Queue processing complete: {processed} added, {skipped} skipped")
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


def create_player_from_queue_item(item, fangraphs_id: int, existing_players: dict) -> Player:
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
    parser = ArgumentParser(description="Process the player review queue")
    parser.add_argument("--status", action="store_true", help="Show queue status without processing")
    args = parser.parse_args()

    if args.status:
        display_queue_status()
    else:
        display_queue_status()
        if input("\nProcess queue now? [y/N]: ").strip().lower() == 'y':
            process_queue()


if __name__ == "__main__":
    main()
