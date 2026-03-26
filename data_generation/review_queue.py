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
from pybaseball import playerid_lookup
import pandas as pd

from .records import Player
from .save import (
    read_review_queue,
    remove_review_queue_item,
    write_players_to_file,
    read_players_from_file,
)
from .spotrac import (
    get_baseball_reference_link,
    get_player_birthday,
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
    """Interactive search for a player by name"""
    first_name = input("Enter first name: ").strip()
    last_name = input("Enter last name: ").strip()

    if not first_name or not last_name:
        return -1

    id_df = playerid_lookup(last_name, first_name, fuzzy=True)

    if id_df.empty:
        print("No players found.")
        return -1

    # Filter by contract year
    id_df["mlb_played_first"] = pd.to_numeric(id_df["mlb_played_first"], errors='coerce')
    id_df["mlb_played_last"] = pd.to_numeric(id_df["mlb_played_last"], errors='coerce')
    id_df = id_df.dropna(subset=["mlb_played_first", "mlb_played_last"])

    filtered_df = id_df[
        (id_df["mlb_played_first"] - 1 <= contract_year) &
        (id_df["mlb_played_last"] + 1 >= contract_year)
    ]

    if filtered_df.empty:
        print("No players found matching contract year.")
        return -1

    print("\nMatches found:")
    for i, (_, row) in enumerate(filtered_df.iterrows()):
        print(f"  [{i}] {row['name_first']} {row['name_last']} ({int(row['mlb_played_first'])}-{int(row['mlb_played_last'])})")

    try:
        idx = int(input("Select player number (-1 to cancel): "))
        if idx == -1:
            return -1
        return int(filtered_df.iloc[idx]["key_fangraphs"])
    except (ValueError, IndexError):
        return -1


def create_player_from_queue_item(item, fangraphs_id: int, existing_players: dict) -> Player:
    """Create a Player object from a queue item and selected FanGraphs ID"""
    # Build player_id from spotrac link
    link_id = item.spotrac_link.split("/")[-1]
    player_id = f"{item.last_name}_{link_id}"

    if player_id in existing_players:
        print(f"Player {player_id} already exists in dataset.")
        remove_review_queue_item(item)
        return None

    print(f"Fetching player details for FanGraphs ID {fangraphs_id}...")

    baseball_reference_link = get_baseball_reference_link(fangraphs_id)
    birthday = get_player_birthday(baseball_reference_link) if baseball_reference_link else None

    return Player(
        player_id=player_id,
        fangraphs_id=fangraphs_id,
        first_name=item.first_name,
        last_name=item.last_name,
        position="",  # Will be filled in on next scrape
        birth_date=birthday,
        spotrac_link=item.spotrac_link,
        baseball_reference_link=baseball_reference_link
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
