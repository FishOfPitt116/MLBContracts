import csv
import os
from datetime import datetime
from typing import List

from records import Player, Salary

PLAYERS_FILE = "dataset/players.csv"
CONTRACTS_FILE = "dataset/contracts_spotrac.csv"

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
