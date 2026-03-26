"""
Dataset Join Script

Joins contracts with player stats from the prior season(s) to create
analysis-ready datasets for modeling.

For each contract signed in year Y, attaches the player's stats from year Y-1
(the season before signing). Supports multiple window sizes (1, 3, 5-year).

Follows the same incremental update pattern as other scripts:
- Skip historical years if already joined
- Always refresh current year
- Use --overwrite to force full re-join
"""

from argparse import ArgumentParser
from dataclasses import dataclass, fields
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import csv
import os

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


@dataclass
class ContractWithStats:
    """Contract joined with both batting and pitching stats from prior season"""
    # Contract fields
    contract_id: str
    player_id: str
    age: Optional[int]
    service_time: Optional[float]
    contract_year: int
    duration: int
    value: float
    contract_type: str

    # Player fields
    position: str

    # Stats metadata
    stats_year: Optional[int]
    stats_window: Optional[int]

    # Batting stats (prefixed with bat_)
    bat_games: Optional[int] = None
    bat_plate_appearances: Optional[int] = None
    bat_at_bats: Optional[int] = None
    bat_hits: Optional[int] = None
    bat_doubles: Optional[int] = None
    bat_triples: Optional[int] = None
    bat_home_runs: Optional[int] = None
    bat_runs: Optional[int] = None
    bat_rbis: Optional[int] = None
    bat_stolen_bases: Optional[int] = None
    bat_caught_stealing: Optional[int] = None
    bat_walks: Optional[int] = None
    bat_strikeouts: Optional[int] = None
    bat_batting_avg: Optional[float] = None
    bat_on_base_pct: Optional[float] = None
    bat_slugging_pct: Optional[float] = None
    bat_ops: Optional[float] = None
    bat_wrc_plus: Optional[float] = None
    bat_war: Optional[float] = None
    bat_babip: Optional[float] = None
    bat_iso: Optional[float] = None
    bat_bb_pct: Optional[float] = None
    bat_k_pct: Optional[float] = None
    bat_hard_hit_pct: Optional[float] = None
    bat_barrel_pct: Optional[float] = None

    # Pitching stats (prefixed with pit_)
    pit_games: Optional[int] = None
    pit_games_started: Optional[int] = None
    pit_innings_pitched: Optional[float] = None
    pit_wins: Optional[int] = None
    pit_losses: Optional[int] = None
    pit_saves: Optional[int] = None
    pit_holds: Optional[int] = None
    pit_strikeouts: Optional[int] = None
    pit_walks: Optional[int] = None
    pit_hits_allowed: Optional[int] = None
    pit_home_runs_allowed: Optional[int] = None
    pit_era: Optional[float] = None
    pit_whip: Optional[float] = None
    pit_k_per_9: Optional[float] = None
    pit_bb_per_9: Optional[float] = None
    pit_fip: Optional[float] = None
    pit_xfip: Optional[float] = None
    pit_siera: Optional[float] = None
    pit_war: Optional[float] = None
    pit_k_pct: Optional[float] = None
    pit_bb_pct: Optional[float] = None
    pit_k_bb_ratio: Optional[float] = None
    pit_ground_ball_pct: Optional[float] = None
    pit_fly_ball_pct: Optional[float] = None
    pit_hard_hit_pct: Optional[float] = None


def _get_dataclass_headers(dataclass_type) -> List[str]:
    """Extract field names from a dataclass to use as CSV headers"""
    return [field.name for field in fields(dataclass_type)]


def _get_dataclass_values(instance) -> List:
    """Extract field values from a dataclass instance"""
    return [getattr(instance, field.name) if getattr(instance, field.name) is not None else ""
            for field in fields(instance)]


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


def _get_prior_stats(
    player_id: str,
    contract_year: int,
    window_years: int,
    batter_lookup: Dict,
    pitcher_lookup: Dict
) -> Tuple[Optional[BatterStats], Optional[PitcherStats]]:
    """
    Get stats from the season before the contract was signed.
    For a contract in year Y, get stats from year Y-1.
    """
    stats_year = contract_year - 1
    key = (player_id, stats_year, window_years)

    batter = batter_lookup.get(key)
    pitcher = pitcher_lookup.get(key)

    return batter, pitcher


def create_contract_with_stats(
    contract: Salary,
    player: Player,
    batter_stats: Optional[BatterStats],
    pitcher_stats: Optional[PitcherStats],
    window_years: int
) -> ContractWithStats:
    """Create a ContractWithStats by joining contract with both batting and pitching stats"""
    # Determine stats_year from whichever stats exist
    stats_year = None
    if batter_stats:
        stats_year = batter_stats.year
    elif pitcher_stats:
        stats_year = pitcher_stats.year

    return ContractWithStats(
        contract_id=contract.contract_id,
        player_id=contract.player_id,
        age=contract.age,
        service_time=contract.service_time,
        contract_year=contract.year,
        duration=contract.duration,
        value=contract.value,
        contract_type=contract.type,
        position=player.position if player else "",
        stats_year=stats_year,
        stats_window=window_years,

        # Batting stats
        bat_games=batter_stats.games if batter_stats else None,
        bat_plate_appearances=batter_stats.plate_appearances if batter_stats else None,
        bat_at_bats=batter_stats.at_bats if batter_stats else None,
        bat_hits=batter_stats.hits if batter_stats else None,
        bat_doubles=batter_stats.doubles if batter_stats else None,
        bat_triples=batter_stats.triples if batter_stats else None,
        bat_home_runs=batter_stats.home_runs if batter_stats else None,
        bat_runs=batter_stats.runs if batter_stats else None,
        bat_rbis=batter_stats.rbis if batter_stats else None,
        bat_stolen_bases=batter_stats.stolen_bases if batter_stats else None,
        bat_caught_stealing=batter_stats.caught_stealing if batter_stats else None,
        bat_walks=batter_stats.walks if batter_stats else None,
        bat_strikeouts=batter_stats.strikeouts if batter_stats else None,
        bat_batting_avg=batter_stats.batting_avg if batter_stats else None,
        bat_on_base_pct=batter_stats.on_base_pct if batter_stats else None,
        bat_slugging_pct=batter_stats.slugging_pct if batter_stats else None,
        bat_ops=batter_stats.ops if batter_stats else None,
        bat_wrc_plus=batter_stats.wrc_plus if batter_stats else None,
        bat_war=batter_stats.war if batter_stats else None,
        bat_babip=batter_stats.babip if batter_stats else None,
        bat_iso=batter_stats.iso if batter_stats else None,
        bat_bb_pct=batter_stats.bb_pct if batter_stats else None,
        bat_k_pct=batter_stats.k_pct if batter_stats else None,
        bat_hard_hit_pct=batter_stats.hard_hit_pct if batter_stats else None,
        bat_barrel_pct=batter_stats.barrel_pct if batter_stats else None,

        # Pitching stats
        pit_games=pitcher_stats.games if pitcher_stats else None,
        pit_games_started=pitcher_stats.games_started if pitcher_stats else None,
        pit_innings_pitched=pitcher_stats.innings_pitched if pitcher_stats else None,
        pit_wins=pitcher_stats.wins if pitcher_stats else None,
        pit_losses=pitcher_stats.losses if pitcher_stats else None,
        pit_saves=pitcher_stats.saves if pitcher_stats else None,
        pit_holds=pitcher_stats.holds if pitcher_stats else None,
        pit_strikeouts=pitcher_stats.strikeouts if pitcher_stats else None,
        pit_walks=pitcher_stats.walks if pitcher_stats else None,
        pit_hits_allowed=pitcher_stats.hits_allowed if pitcher_stats else None,
        pit_home_runs_allowed=pitcher_stats.home_runs_allowed if pitcher_stats else None,
        pit_era=pitcher_stats.era if pitcher_stats else None,
        pit_whip=pitcher_stats.whip if pitcher_stats else None,
        pit_k_per_9=pitcher_stats.k_per_9 if pitcher_stats else None,
        pit_bb_per_9=pitcher_stats.bb_per_9 if pitcher_stats else None,
        pit_fip=pitcher_stats.fip if pitcher_stats else None,
        pit_xfip=pitcher_stats.xfip if pitcher_stats else None,
        pit_siera=pitcher_stats.siera if pitcher_stats else None,
        pit_war=pitcher_stats.war if pitcher_stats else None,
        pit_k_pct=pitcher_stats.k_pct if pitcher_stats else None,
        pit_bb_pct=pitcher_stats.bb_pct if pitcher_stats else None,
        pit_k_bb_ratio=pitcher_stats.k_bb_ratio if pitcher_stats else None,
        pit_ground_ball_pct=pitcher_stats.ground_ball_pct if pitcher_stats else None,
        pit_fly_ball_pct=pitcher_stats.fly_ball_pct if pitcher_stats else None,
        pit_hard_hit_pct=pitcher_stats.hard_hit_pct if pitcher_stats else None,
    )


def read_existing_joined_contracts() -> Dict[Tuple[str, int], int]:
    """
    Read existing joined contracts and return dict of (contract_id, stats_window) -> contract_year.
    Used to skip already-joined records.
    """
    existing = {}
    if not os.path.isfile(CONTRACTS_WITH_STATS_FILE):
        return existing

    with open(CONTRACTS_WITH_STATS_FILE, mode="r", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            contract_id = row["contract_id"]
            stats_window = int(row["stats_window"]) if row["stats_window"] else 0
            contract_year = int(row["contract_year"])
            # Key includes window to support multiple window sizes per contract
            existing[(contract_id, stats_window)] = contract_year

    return existing


def write_contracts_with_stats(contracts: List[ContractWithStats], overwrite: bool = False):
    """Write combined contracts to CSV"""
    file_exists = os.path.isfile(CONTRACTS_WITH_STATS_FILE)
    headers = _get_dataclass_headers(ContractWithStats)

    if overwrite and file_exists:
        os.remove(CONTRACTS_WITH_STATS_FILE)
        file_exists = False

    # Read existing to avoid duplicates
    existing = read_existing_joined_contracts() if not overwrite else {}

    # Separate new records from current-year updates
    new_records = []
    update_records = []

    for contract in contracts:
        key = (contract.contract_id, contract.stats_window)
        if key not in existing:
            new_records.append(contract)
        elif existing[key] == CURRENT_YEAR:
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
                    key = (row["contract_id"], int(row["stats_window"]) if row["stats_window"] else 0)
                    all_records[key] = row

        # Update with new data
        for contract in update_records + new_records:
            key = (contract.contract_id, contract.stats_window)
            all_records[key] = dict(zip(headers, _get_dataclass_values(contract)))

        # Rewrite
        with open(CONTRACTS_WITH_STATS_FILE, mode="w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=headers)
            writer.writeheader()
            for row in all_records.values():
                writer.writerow(row)
    else:
        # Just append new records
        with open(CONTRACTS_WITH_STATS_FILE, mode="a", newline="") as file:
            writer = csv.writer(file)
            if not file_exists:
                writer.writerow(headers)
            for contract in new_records:
                writer.writerow(_get_dataclass_values(contract))


def main(window_years: int = 1, overwrite: bool = False):
    """
    Main join function.

    Args:
        window_years: Stats window to use (1=single season, 3=3-year, 5=5-year)
        overwrite: Force re-join of all contracts
    """
    print(f"\n{'='*60}")
    print(f"Joining contracts with {window_years}-year stats window")
    print(f"{'='*60}\n")

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
    no_bat_stats = 0
    no_pit_stats = 0
    two_way_players = 0

    print("\nProcessing contracts...")

    for contract in contracts:
        key = (contract.contract_id, window_years)

        # Skip if already joined and not current year
        if key in existing:
            if contract.year != CURRENT_YEAR:
                skipped += 1
                continue

        player = player_lookup.get(contract.player_id)
        batter, pitcher = _get_prior_stats(
            contract.player_id,
            contract.year,
            window_years,
            batter_lookup,
            pitcher_lookup
        )

        # Track stats availability
        if not batter:
            no_bat_stats += 1
        if not pitcher:
            no_pit_stats += 1
        if batter and pitcher:
            two_way_players += 1

        joined_contracts.append(create_contract_with_stats(
            contract, player, batter, pitcher, window_years
        ))

    print(f"\nResults:")
    print(f"  Contracts to write: {len(joined_contracts)}")
    print(f"  Skipped (already joined): {skipped}")
    print(f"  With batting stats: {len(joined_contracts) - no_bat_stats}")
    print(f"  With pitching stats: {len(joined_contracts) - no_pit_stats}")
    print(f"  Two-way players (both): {two_way_players}")

    # Write results
    if joined_contracts:
        print(f"\nWriting to {CONTRACTS_WITH_STATS_FILE}...")
        write_contracts_with_stats(joined_contracts, overwrite)

    print(f"\n{'='*60}")
    print("Join complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    parser = ArgumentParser(description="Join contracts with player stats")
    parser.add_argument(
        "--window",
        type=int,
        default=1,
        choices=[1, 3, 5, 10],
        help="Stats window size (1=single season, 3/5/10=accumulated)"
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Force re-join of all contracts"
    )
    args = parser.parse_args()

    main(window_years=args.window, overwrite=args.overwrite)
