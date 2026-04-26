"""
Dataset Join Script

Joins contracts with player stats from the prior season(s) to create
analysis-ready datasets for modeling.

For each contract signed in year Y, attaches the player's stats from year Y-1
(the season before signing). Includes ALL window sizes (1, 3, 5, 10-year) as
separate columns in a single row per contract.

Output format: Wide format with columns like bat_war_1y, bat_war_3y, etc.
"""

from argparse import ArgumentParser
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import csv
import os
import shutil

from .records import Salary, BatterStats, PitcherStats, Player
from .save import (
    read_contracts_from_file,
    read_players_from_file,
    read_batter_stats,
    read_pitcher_stats,
)

CURRENT_YEAR = datetime.now().year

# Output file
CONTRACTS_WITH_STATS_FILE = "dataset/contracts_with_stats.csv"
LEGACY_BACKUP_FILE = "dataset/contracts_with_stats_legacy.csv"

# Window sizes to include
WINDOW_SIZES = [1, 3, 5, 10]

# Stats field names (extracted from BatterStats/PitcherStats dataclasses)
BATTER_STAT_FIELDS = [
    "games", "plate_appearances", "at_bats", "hits", "doubles", "triples",
    "home_runs", "runs", "rbis", "stolen_bases", "caught_stealing", "walks",
    "strikeouts", "batting_avg", "on_base_pct", "slugging_pct", "ops",
    "wrc_plus", "war", "babip", "iso", "bb_pct", "k_pct", "hard_hit_pct", "barrel_pct"
]

PITCHER_STAT_FIELDS = [
    "games", "games_started", "innings_pitched", "wins", "losses", "saves",
    "holds", "strikeouts", "walks", "hits_allowed", "home_runs_allowed",
    "era", "whip", "k_per_9", "bb_per_9", "fip", "xfip", "siera", "war",
    "k_pct", "bb_pct", "k_bb_ratio", "ground_ball_pct", "fly_ball_pct", "hard_hit_pct"
]

# Contract fields (in order)
CONTRACT_FIELDS = [
    "contract_id", "player_id", "age", "service_time", "contract_year",
    "duration", "value", "contract_type", "position", "stats_year"
]


def _generate_stats_headers() -> List[str]:
    """Generate column headers for all stats across all windows."""
    headers = []
    # Batting stats with window suffixes
    for field in BATTER_STAT_FIELDS:
        for window in WINDOW_SIZES:
            headers.append(f"bat_{field}_{window}y")
    # Pitching stats with window suffixes
    for field in PITCHER_STAT_FIELDS:
        for window in WINDOW_SIZES:
            headers.append(f"pit_{field}_{window}y")
    return headers


def _get_all_headers() -> List[str]:
    """Get complete list of CSV headers."""
    return CONTRACT_FIELDS + _generate_stats_headers()


def _build_stats_lookup(
    batter_stats: List[BatterStats],
    pitcher_stats: List[PitcherStats]
) -> Tuple[Dict, Dict]:
    """
    Build lookup dicts for quick stats access.
    Key: (player_id, year, window_years)
    """
    batter_lookup = {stat.get_key(): stat for stat in batter_stats}
    pitcher_lookup = {stat.get_key(): stat for stat in pitcher_stats}
    return batter_lookup, pitcher_lookup


def _build_player_lookup(players: List[Player]) -> Dict[str, Player]:
    """Build lookup dict for players by player_id"""
    return {player.player_id: player for player in players}


def _get_all_window_stats(
    player_id: str,
    contract_year: int,
    batter_lookup: Dict,
    pitcher_lookup: Dict
) -> Tuple[Dict, Optional[int]]:
    """
    Get stats from ALL windows for the year before contract was signed.
    Returns dict with all stats columns and the stats_year.
    """
    stats_year = contract_year - 1
    result = {}

    for window in WINDOW_SIZES:
        key = (player_id, stats_year, window)
        suffix = f"_{window}y"

        # Batting stats
        batter = batter_lookup.get(key)
        for field in BATTER_STAT_FIELDS:
            col_name = f"bat_{field}{suffix}"
            result[col_name] = getattr(batter, field, None) if batter else None

        # Pitching stats
        pitcher = pitcher_lookup.get(key)
        for field in PITCHER_STAT_FIELDS:
            col_name = f"pit_{field}{suffix}"
            result[col_name] = getattr(pitcher, field, None) if pitcher else None

    # Determine if we have any stats at all to set stats_year
    has_any_stats = any(v is not None for v in result.values())
    actual_stats_year = stats_year if has_any_stats else None

    return result, actual_stats_year


def create_contract_row(
    contract: Salary,
    player: Optional[Player],
    batter_lookup: Dict,
    pitcher_lookup: Dict
) -> Dict:
    """Create a single row dict for a contract with all window stats."""
    stats_dict, stats_year = _get_all_window_stats(
        contract.player_id,
        contract.year,
        batter_lookup,
        pitcher_lookup
    )

    # Build the row
    row = {
        "contract_id": contract.contract_id,
        "player_id": contract.player_id,
        "age": contract.age,
        "service_time": contract.service_time,
        "contract_year": contract.year,
        "duration": contract.duration,
        "value": contract.value,
        "contract_type": contract.type,
        "position": player.position if player else "",
        "stats_year": stats_year,
    }
    row.update(stats_dict)

    return row


def _is_legacy_format(filepath: str) -> bool:
    """Check if existing file uses the old format (has stats_window column)."""
    if not os.path.isfile(filepath):
        return False

    with open(filepath, mode="r", newline="") as file:
        reader = csv.reader(file)
        headers = next(reader, [])
        return "stats_window" in headers


def _migrate_legacy_file():
    """Backup legacy format file if it exists."""
    if os.path.isfile(CONTRACTS_WITH_STATS_FILE) and _is_legacy_format(CONTRACTS_WITH_STATS_FILE):
        print(f"Detected legacy format. Backing up to {LEGACY_BACKUP_FILE}")
        shutil.copy2(CONTRACTS_WITH_STATS_FILE, LEGACY_BACKUP_FILE)
        os.remove(CONTRACTS_WITH_STATS_FILE)
        return True
    return False


def read_existing_joined_contracts() -> Dict[str, int]:
    """
    Read existing joined contracts and return dict of contract_id -> contract_year.
    Used to skip already-joined records.
    """
    existing = {}
    if not os.path.isfile(CONTRACTS_WITH_STATS_FILE):
        return existing

    with open(CONTRACTS_WITH_STATS_FILE, mode="r", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            contract_id = row["contract_id"]
            contract_year = int(row["contract_year"])
            existing[contract_id] = contract_year

    return existing


def _format_value(value) -> str:
    """Format a value for CSV output."""
    if value is None:
        return ""
    return str(value)


def write_contracts_with_stats(contracts: List[Dict], overwrite: bool = False):
    """Write combined contracts to CSV."""
    file_exists = os.path.isfile(CONTRACTS_WITH_STATS_FILE)
    headers = _get_all_headers()

    if overwrite and file_exists:
        os.remove(CONTRACTS_WITH_STATS_FILE)
        file_exists = False

    # Read existing to avoid duplicates
    existing = read_existing_joined_contracts() if not overwrite else {}

    # Separate new records from current-year updates
    new_records = []
    update_records = []

    for contract in contracts:
        contract_id = contract["contract_id"]
        if contract_id not in existing:
            new_records.append(contract)
        elif existing[contract_id] == CURRENT_YEAR:
            # Current year record - needs update
            update_records.append(contract)

    # If we have updates, rewrite the file
    if update_records:
        # Read all existing, update matching records, write all
        all_records = {}
        if os.path.isfile(CONTRACTS_WITH_STATS_FILE):
            with open(CONTRACTS_WITH_STATS_FILE, mode="r", newline="") as file:
                reader = csv.DictReader(file)
                for row in reader:
                    all_records[row["contract_id"]] = row

        # Update with new data
        for contract in update_records + new_records:
            formatted_row = {k: _format_value(v) for k, v in contract.items()}
            all_records[contract["contract_id"]] = formatted_row

        # Rewrite
        with open(CONTRACTS_WITH_STATS_FILE, mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=headers)
            writer.writeheader()
            for row in all_records.values():
                writer.writerow(row)
    else:
        # Just append new records
        with open(CONTRACTS_WITH_STATS_FILE, mode="a", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=headers)
            if not file_exists:
                writer.writeheader()
            for contract in new_records:
                formatted_row = {k: _format_value(v) for k, v in contract.items()}
                writer.writerow(formatted_row)


def main(overwrite: bool = False):
    """
    Main join function.

    Args:
        overwrite: Force re-join of all contracts
    """
    print(f"\n{'='*60}")
    print(f"Joining contracts with all window sizes ({', '.join(str(w) for w in WINDOW_SIZES)}-year)")
    print(f"{'='*60}\n")

    # Check for legacy format and migrate if needed
    migrated = _migrate_legacy_file()
    if migrated:
        overwrite = True  # Force full regeneration after migration

    # Load all data
    print("Loading datasets...")
    contracts = read_contracts_from_file()
    players = read_players_from_file()
    batter_stats = read_batter_stats()
    pitcher_stats = read_pitcher_stats()

    print(f"  Contracts: {len(contracts)}")
    print(f"  Players: {len(players)}")
    print(f"  Batter stats: {len(batter_stats)}")
    print(f"  Pitcher stats: {len(pitcher_stats)}")

    # Build lookups
    player_lookup = _build_player_lookup(players)
    batter_lookup, pitcher_lookup = _build_stats_lookup(batter_stats, pitcher_stats)

    # Read existing joined data for skip logic
    existing = read_existing_joined_contracts() if not overwrite else {}

    # Process contracts
    joined_contracts = []
    skipped = 0
    contracts_with_any_stats = 0

    print("\nProcessing contracts...")

    for contract in contracts:
        contract_id = contract.contract_id

        # Skip if already joined and not current year
        if contract_id in existing:
            if contract.year != CURRENT_YEAR:
                skipped += 1
                continue

        player = player_lookup.get(contract.player_id)
        row = create_contract_row(contract, player, batter_lookup, pitcher_lookup)

        # Track if this contract has any stats
        if row["stats_year"] is not None:
            contracts_with_any_stats += 1

        joined_contracts.append(row)

    print(f"\nResults:")
    print(f"  Contracts to write: {len(joined_contracts)}")
    print(f"  Skipped (already joined): {skipped}")
    print(f"  With stats (any window): {contracts_with_any_stats}")
    print(f"  Output columns: {len(_get_all_headers())}")

    # Write results
    if joined_contracts:
        print(f"\nWriting to {CONTRACTS_WITH_STATS_FILE}...")
        write_contracts_with_stats(joined_contracts, overwrite)

    print(f"\n{'='*60}")
    print("Join complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = ArgumentParser(description="Join contracts with player stats (all windows)")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Force re-join of all contracts"
    )
    args = parser.parse_args()

    main(overwrite=args.overwrite)
