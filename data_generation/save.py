import csv
import os
from dataclasses import fields
from datetime import datetime
from typing import List, Tuple

from .records import Player, Salary, BatterStats, PitcherStats

PLAYERS_FILE = "dataset/players.csv"
CONTRACTS_FILE = "dataset/contracts_spotrac.csv"
BATTER_STATS_FILE = "dataset/batter_stats.csv"
PITCHER_STATS_FILE = "dataset/pitcher_stats.csv"

def _get_dataclass_headers(dataclass_type) -> List[str]:
    """Extract field names from a dataclass to use as CSV headers"""
    return [field.name for field in fields(dataclass_type)]


def _get_dataclass_values(instance) -> List:
    """Extract field values from a dataclass instance in the same order as headers"""
    return [_optional_value(getattr(instance, field.name)) for field in fields(instance)]


def write_players_to_file(players: List[Player], overwrite: bool = False):
    file_exists = os.path.isfile(PLAYERS_FILE)
    if overwrite and file_exists:
        with open(PLAYERS_FILE, mode="w", newline="") as file:
            # wipe existing file contents
            file.truncate()
            writer = csv.writer(file)
            writer.writerow(["player_id", "fangraphs_id", "first_name", "last_name", "position", "birth_date", "spotrac_link", "baseball_reference_link"])
    existing_players = {player.player_id: player for player in read_players_from_file()}
    with open(PLAYERS_FILE, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["player_id", "fangraphs_id", "first_name", "last_name", "position", "birth_date", "spotrac_link", "baseball_reference_link"])
        for player in players:
            if player.player_id not in existing_players:
                writer.writerow([
                    player.player_id,
                    player.fangraphs_id,
                    player.first_name,
                    player.last_name,
                    player.position,
                    player.birth_date.strftime("%Y-%m-%d") if player.birth_date else "",
                    player.spotrac_link or "",
                    player.baseball_reference_link or ""
                ])

def read_players_from_file() -> List[Player]:
    players = []
    if not os.path.isfile(PLAYERS_FILE):
        return players
    with open(PLAYERS_FILE, mode="r", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            players.append(Player(
                player_id=row["player_id"],
                fangraphs_id=int(row["fangraphs_id"]),
                first_name=row["first_name"],
                last_name=row["last_name"],
                position=row["position"],
                birth_date=datetime.strptime(row["birth_date"], "%Y-%m-%d") if row["birth_date"] != "" else None,
                spotrac_link=row["spotrac_link"] if row["spotrac_link"] != "" else None,
                baseball_reference_link=row["baseball_reference_link"] if row["baseball_reference_link"] != "" else None
            ))
    return players

def write_contracts_to_file(contracts: List[Salary], overwrite: bool = False):
    file_exists = os.path.isfile(CONTRACTS_FILE)
    if overwrite and file_exists:
        with open(CONTRACTS_FILE, mode="w", newline="") as file:
            # wipe existing file contents
            file.truncate()
            writer = csv.writer(file)
            writer.writerow(["contract_id", "player_id", "age", "service_time", "year", "duration", "value", "type"])
    existing_contracts = {contract.contract_id: contract for contract in read_contracts_from_file()}
    with open(CONTRACTS_FILE, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["contract_id", "player_id", "age", "service_time", "year", "duration", "value", "type"])
        for contract in contracts:
            if contract.contract_id not in existing_contracts:
                writer.writerow([
                    contract.contract_id,
                    contract.player_id,
                    contract.age if contract.age != None else -1,
                    contract.service_time if contract.service_time != None else -1,
                    contract.year,
                    contract.duration,
                    contract.value,
                    contract.type
                ])

def read_contracts_from_file() -> List[Salary]:
    contracts = []
    if not os.path.isfile(CONTRACTS_FILE):
        return contracts
    with open(CONTRACTS_FILE, mode="r", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            contracts.append(Salary(
                contract_id=row["contract_id"],
                player_id=row["player_id"],
                age=int(row["age"]) if row["age"] != "-1" else None,
                service_time=float(row["service_time"]) if row["service_time"] != "-1" else None,
                year=int(row["year"]),
                duration=int(row["duration"]),
                value=float(row["value"]),
                type=row["type"]
            ))
    return contracts

# Helper function to convert None to empty string for CSV writing
def _optional_value(val):
    return val if val is not None else ""

# Helper function to parse optional numeric values from CSV
def _parse_optional_int(val):
    return int(val) if val != "" else None

def _parse_optional_float(val):
    return float(val) if val != "" else None

def write_batter_stats(batter_stats: List[BatterStats], overwrite: bool = False):
    """Write batting statistics to CSV file"""
    file_exists = os.path.isfile(BATTER_STATS_FILE)

    # Get headers directly from BatterStats dataclass
    headers = _get_dataclass_headers(BatterStats)

    # Handle overwrite
    if overwrite and file_exists:
        with open(BATTER_STATS_FILE, mode="w", newline="") as file:
            file.truncate()
            writer = csv.writer(file)
            writer.writerow(headers)

    # Read existing stats to avoid duplicates
    existing_stats = {stat.get_key(): stat for stat in read_batter_stats()}

    # Write stats
    with open(BATTER_STATS_FILE, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(headers)

        for stat in batter_stats:
            if stat.get_key() not in existing_stats:
                writer.writerow(_get_dataclass_values(stat))


def write_pitcher_stats(pitcher_stats: List[PitcherStats], overwrite: bool = False):
    """Write pitching statistics to CSV file"""
    file_exists = os.path.isfile(PITCHER_STATS_FILE)

    # Get headers directly from PitcherStats dataclass
    headers = _get_dataclass_headers(PitcherStats)

    # Handle overwrite
    if overwrite and file_exists:
        with open(PITCHER_STATS_FILE, mode="w", newline="") as file:
            file.truncate()
            writer = csv.writer(file)
            writer.writerow(headers)

    # Read existing stats to avoid duplicates
    existing_stats = {stat.get_key(): stat for stat in read_pitcher_stats()}

    # Write stats
    with open(PITCHER_STATS_FILE, mode="a", newline="") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(headers)

        for stat in pitcher_stats:
            if stat.get_key() not in existing_stats:
                writer.writerow(_get_dataclass_values(stat))


def read_batter_stats() -> List[BatterStats]:
    """Read batting statistics from CSV file"""
    batter_stats = []

    if os.path.isfile(BATTER_STATS_FILE):
        with open(BATTER_STATS_FILE, mode="r", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                batter_stats.append(BatterStats(
                    player_id=row["player_id"],
                    year=int(row["year"]),
                    window_years=int(row["window_years"]),
                    games=_parse_optional_int(row["games"]),
                    plate_appearances=_parse_optional_int(row["plate_appearances"]),
                    at_bats=_parse_optional_int(row["at_bats"]),
                    hits=_parse_optional_int(row["hits"]),
                    doubles=_parse_optional_int(row["doubles"]),
                    triples=_parse_optional_int(row["triples"]),
                    home_runs=_parse_optional_int(row["home_runs"]),
                    runs=_parse_optional_int(row["runs"]),
                    rbis=_parse_optional_int(row["rbis"]),
                    stolen_bases=_parse_optional_int(row["stolen_bases"]),
                    caught_stealing=_parse_optional_int(row["caught_stealing"]),
                    walks=_parse_optional_int(row["walks"]),
                    strikeouts=_parse_optional_int(row["strikeouts"]),
                    batting_avg=_parse_optional_float(row["batting_avg"]),
                    on_base_pct=_parse_optional_float(row["on_base_pct"]),
                    slugging_pct=_parse_optional_float(row["slugging_pct"]),
                    ops=_parse_optional_float(row["ops"]),
                    wrc_plus=_parse_optional_float(row["wrc_plus"]),
                    war=_parse_optional_float(row["war"]),
                    babip=_parse_optional_float(row["babip"]),
                    iso=_parse_optional_float(row["iso"]),
                    bb_pct=_parse_optional_float(row["bb_pct"]),
                    k_pct=_parse_optional_float(row["k_pct"]),
                    hard_hit_pct=_parse_optional_float(row["hard_hit_pct"]),
                    barrel_pct=_parse_optional_float(row["barrel_pct"]),
                ))

    return batter_stats


def read_pitcher_stats() -> List[PitcherStats]:
    """Read pitching statistics from CSV file"""
    pitcher_stats = []

    if os.path.isfile(PITCHER_STATS_FILE):
        with open(PITCHER_STATS_FILE, mode="r", newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                pitcher_stats.append(PitcherStats(
                    player_id=row["player_id"],
                    year=int(row["year"]),
                    window_years=int(row["window_years"]),
                    games=_parse_optional_int(row["games"]),
                    games_started=_parse_optional_int(row["games_started"]),
                    innings_pitched=_parse_optional_float(row["innings_pitched"]),
                    wins=_parse_optional_int(row["wins"]),
                    losses=_parse_optional_int(row["losses"]),
                    saves=_parse_optional_int(row["saves"]),
                    holds=_parse_optional_int(row["holds"]),
                    strikeouts=_parse_optional_int(row["strikeouts"]),
                    walks=_parse_optional_int(row["walks"]),
                    hits_allowed=_parse_optional_int(row["hits_allowed"]),
                    home_runs_allowed=_parse_optional_int(row["home_runs_allowed"]),
                    era=_parse_optional_float(row["era"]),
                    whip=_parse_optional_float(row["whip"]),
                    k_per_9=_parse_optional_float(row["k_per_9"]),
                    bb_per_9=_parse_optional_float(row["bb_per_9"]),
                    fip=_parse_optional_float(row["fip"]),
                    xfip=_parse_optional_float(row["xfip"]),
                    siera=_parse_optional_float(row["siera"]),
                    war=_parse_optional_float(row["war"]),
                    k_pct=_parse_optional_float(row["k_pct"]),
                    bb_pct=_parse_optional_float(row["bb_pct"]),
                    k_bb_ratio=_parse_optional_float(row["k_bb_ratio"]),
                    ground_ball_pct=_parse_optional_float(row["ground_ball_pct"]),
                    fly_ball_pct=_parse_optional_float(row["fly_ball_pct"]),
                    hard_hit_pct=_parse_optional_float(row["hard_hit_pct"]),
                ))
    
    return pitcher_stats


def write_stats_to_file(batter_stats: List[BatterStats], pitcher_stats: List[PitcherStats], overwrite: bool = False):
    """Convenience function to write both batting and pitching stats in one call"""
    write_batter_stats(batter_stats, overwrite)
    write_pitcher_stats(pitcher_stats, overwrite)


def read_stats_from_file() -> Tuple[List[BatterStats], List[PitcherStats]]:
    """Convenience function to read both batting and pitching stats in one call"""
    return read_batter_stats(), read_pitcher_stats()
