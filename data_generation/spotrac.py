from argparse import ArgumentParser
from bs4 import BeautifulSoup
from bs4.element import Tag
import pandas as pd
from pybaseball import playerid_lookup, playerid_reverse_lookup
import requests
from typing import Dict, List, Tuple

from log_stream import LogStream
from records import Player, Salary
from save import write_players_to_file, write_contracts_to_file, read_players_from_file, read_contracts_from_file

PRE_ARB_URL = "https://www.spotrac.com/mlb/pre-arbitration/_/year/{year}/"
ARB_URL = "https://www.spotrac.com/mlb/arbitration/_/year/{year}"
FREE_AGENT_URL = "https://www.spotrac.com/mlb/free-agents/_/year/{year}/level/mlb"

PRE_ARB_EXTENSIONS_URL = "https://www.spotrac.com/mlb/contracts/extensions/_/year/{year}/type/pre-arbitration-extension/"
ARB_EXTENSIONS_URL = "https://www.spotrac.com/mlb/contracts/extensions/_/year/{year}/type/arbitration-extension/"
VETERAN_EXTENSIONS_URL = "https://www.spotrac.com/mlb/contracts/extensions/_/year/{year}/type/extension/"

PLAYER_OBJECT_CACHE = {player.player_id: player for player in read_players_from_file()}
CONTRACT_OBJECT_CACHE = {contract.contract_id: contract for contract in read_contracts_from_file()}

LOG_STREAM = LogStream("SPOTRAC DATA GENERATION")

def fetch_spotrac_data(url: str) -> BeautifulSoup:
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    if response.status_code == 200:
        return BeautifulSoup(response.text, "html.parser")
    return None

def sanitize_string(s: str) -> str:
    return s.replace("\n", "").strip()

def get_player_id(columns: List[Tag], headers: Dict[str, int]) -> str:
    name = sanitize_string(columns[headers['player']].get_text()).split(" ", 1)
    if name[1].endswith("QO"):
        name[1] = name[1][:-2]  # Remove last two characters
    spotrac_link = sanitize_string(columns[headers['player']].find("a")["href"])
    link_id = spotrac_link.split("/")[-1]

    return f"{name[1]}_{link_id}"

def get_fangraphs_id(first_name: str, last_name: str, contract_year: int, fuzzy: bool = False) -> int:
    # Handle names with periods by adding a space after the first period
    if "." in first_name:
        first_name = first_name.replace(".", ". ").strip()

    id_df = playerid_lookup(last_name, first_name, fuzzy)

    # Note that playerid_lookup could return a player with id -1 if they haven't had a qualified season
    if len(id_df) == 1:
        LOG_STREAM.player_mapping(first_name, last_name, id_df, 0)
        return id_df["key_fangraphs"][0]
    
    if len(id_df) == 0:
        if not fuzzy:
            # if no players found, get fuzzy results
            return get_fangraphs_id(first_name, last_name, contract_year, fuzzy=True)
        print(f"Could not find player with name {first_name} {last_name}. Please provide first and last name.\nIf you do not know the player's name, please enter 'exit' to not create this player object.")
        first_name = input("First Name: ")
        if first_name.lower() == "exit":
            return -1
        last_name = input("Last Name: ")
        return get_fangraphs_id(first_name, last_name, contract_year, fuzzy=fuzzy)
    
    # Remove duplicate rows based on key_fangraphs
    id_df = id_df.drop_duplicates(subset=["key_fangraphs"])
    
    # Convert mlb_played_first and mlb_played_last to numeric, setting errors to NaN
    id_df["mlb_played_first"] = pd.to_numeric(id_df["mlb_played_first"], errors='coerce')
    id_df["mlb_played_last"] = pd.to_numeric(id_df["mlb_played_last"], errors='coerce')

    # Drop rows where either value could not be converted
    id_df = id_df.dropna(subset=["mlb_played_first", "mlb_played_last"])

    # Convert remaining valid values to integers
    id_df["mlb_played_first"] = id_df["mlb_played_first"].astype(int)
    id_df["mlb_played_last"] = id_df["mlb_played_last"].astype(int)
    
    # Apply contract year filtering
    filtered_df = id_df[
        (id_df["mlb_played_first"] - 1 <= contract_year) & 
        (id_df["mlb_played_last"] + 1 >= contract_year)
    ]

    # If multiple players remain, prompt user for selection
    if not filtered_df.empty:
        id_df = filtered_df  # Use the filtered list if possible
    else:
        LOG_STREAM.player_mapping_error(first_name, last_name)
        return -1

    if len(id_df) == 1 or 0 in id_df.index:
        LOG_STREAM.player_mapping(first_name, last_name, id_df, 0, iloc=True, low_confidence=(0 in id_df.index))
        return id_df["key_fangraphs"].iloc[0]
    
    print(f"Multiple players found with name {first_name} {last_name}. Please select the correct player from the list below. Type '-1' if none apply and the record will be ignored.")
    print(id_df)
    index = int(input("Enter the index number of the correct player: "))
    if index == -1:
        return -1
    LOG_STREAM.player_mapping(first_name, last_name, id_df, index)
    return id_df["key_fangraphs"][index]

def get_baseball_reference_link(fangraphs_id: int) -> str:
    player_df = playerid_reverse_lookup(fangraphs_id, key_type='fangraphs')
    if player_df.empty:
        return None
    baseball_reference_id = player_df["key_bbref"][0]
    return f"https://www.baseball-reference.com/players/{baseball_reference_id[0]}/{baseball_reference_id}.shtml"


def get_table_headers(table: Tag) -> List[str]:
    LOG_STREAM.write( [ sanitize_string(header.get_text()).replace("$", " ").split(" ")[0].lower() for header in table.find("thead").find_all("th") ] )
    return { sanitize_string(header.get_text()).replace("$", " ").split(" ")[0].lower() : i for i, header in enumerate(table.find("thead").find_all("th")) }

def extract_player_data(row: Tag, headers: Dict[str, int], contract_year: int) -> Player:
    columns = row.find_all('td')
    name = sanitize_string(columns[headers['player']].get_text()).split(" ", 1)
    if name[1].endswith("QO"):
        name[1] = name[1][:-2]  # Remove last two characters
    player_id = get_player_id(columns, headers)

    # if the player doesn't have a contract value, skip them
    value_str = sanitize_string(columns[headers['value']].get_text())
    if value_str == 'N/A':
        return None
    if (float(value_str.replace("$", "").replace(",", "")) / 1_000_000) == 0:
        return None

    if player_id in PLAYER_OBJECT_CACHE:
        return PLAYER_OBJECT_CACHE[player_id]
    
    fangraphs_id = get_fangraphs_id(name[0], name[1], contract_year)
    if fangraphs_id == -1:
        return None

    spotrac_link = sanitize_string(columns[headers['player']].find("a")["href"])
    baseball_reference_link = get_baseball_reference_link(fangraphs_id)

    player_soup = fetch_spotrac_data(spotrac_link)
    # TODO: check spotrac link for birthday
    birthday = None

    return Player(
        player_id=player_id,
        fangraphs_id=fangraphs_id,
        first_name=name[0],
        last_name=name[1],
        position=sanitize_string(columns[headers['pos']].get_text()),
        birth_date=birthday,
        spotrac_link=spotrac_link,
        baseball_reference_link=baseball_reference_link
    )

def extract_salary_data(row: Tag, player_obj: Player, year: int, salary_type: str, headers: Dict[str, int]) -> Salary:
    columns = row.find_all('td')
    value_str = sanitize_string(columns[headers['value']].get_text())
    if value_str == 'N/A':
        return None
    return Salary(
        contract_id=f"{player_obj.player_id}_{year}",
        player_id=player_obj.player_id,
        age=int(sanitize_string(columns[headers['age']].get_text())) if 'age' in headers else None,  # TODO: fix to calculate age when Player has birthdate filled
        service_time=float(sanitize_string(columns[headers['yos']].get_text())) if 'yos' in headers else None, # TODO: maybe calculate service time? Might not matter for free agents
        year=year,
        duration=int(sanitize_string(columns[headers['yrs']].get_text())) if 'yrs' in headers else 1,
        value=float(value_str.replace("$", "").replace(",", "")) / 1_000_000,
        type=salary_type
    )

def get_records(url: str, year: int, salary_type: str) -> Tuple[List[Player], List[Salary]]:
    soup = fetch_spotrac_data(url)
    if not soup:
        return [], []
    
    players, salaries = [], []
    table = soup.find("table")
    if not table:
        return players, salaries
    headers = get_table_headers(table)
    LOG_STREAM.write(headers)
    for row in table.find("tbody").find_all("tr"):
        player = extract_player_data(row, headers, year)
        if not player:
            continue
        if player.player_id not in PLAYER_OBJECT_CACHE:
            PLAYER_OBJECT_CACHE[player.player_id] = player

        salary = extract_salary_data(row, player, year, salary_type, headers)
        if not salary:
            continue
        if salary.contract_id not in CONTRACT_OBJECT_CACHE:
            CONTRACT_OBJECT_CACHE[salary.contract_id] = salary

        players.append(PLAYER_OBJECT_CACHE[player.player_id])
        if salary:
            salaries.append(salary)

    return players, salaries

def get_pre_arb_records(year: int) -> Tuple[List[Player], List[Salary]]:
    settled = get_records(PRE_ARB_URL.format(year=year), year, "pre-arb")
    extensions = get_records(PRE_ARB_EXTENSIONS_URL.format(year=year), year, "pre-arb")
    return (settled[0]+extensions[0], settled[1]+extensions[1])

def get_arb_records(year: int) -> Tuple[List[Player], List[Salary]]:
    settled = get_records(ARB_URL.format(year=year), year, "arb")
    extensions = get_records(ARB_EXTENSIONS_URL.format(year=year), year, "arb")
    return (settled[0]+extensions[0], settled[1]+extensions[1])

def get_free_agent_records(year: int) -> Tuple[List[Player], List[Salary]]:
    contracts = get_records(FREE_AGENT_URL.format(year=year), year, "free-agent")
    extensions = get_records(VETERAN_EXTENSIONS_URL.format(year=year), year, "free-agent")
    return (contracts[0]+extensions[0], contracts[1]+extensions[1])

def main(start_year, end_year=None):
    for year in range(start_year, end_year+1 if end_year else start_year+1):
        pre_arb_players, pre_arb_salaries = get_pre_arb_records(year)
        arb_players, arb_salaries = get_arb_records(year)
        free_agent_players, free_agent_salaries = get_free_agent_records(year)
        print(f"Year {year} Pre-Arb Players: {len(pre_arb_players)}")
        print(f"Year {year} Arb Players: {len(arb_players)}")
        print(f"Year {year} Free Agent Players: {len(free_agent_players)}")
        print(f"Writing {year} records to file...")
        write_players_to_file(pre_arb_players + arb_players + free_agent_players)
        write_contracts_to_file(pre_arb_salaries + arb_salaries + free_agent_salaries)
        print(f"Finished writing {year} records to file.")

if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument("--start-year", type=int, required=True)
    args.add_argument("--end-year", type=int)
    args = args.parse_args()
    main(args.start_year, args.end_year)