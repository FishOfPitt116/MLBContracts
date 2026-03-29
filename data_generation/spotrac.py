from argparse import ArgumentParser
from bs4 import BeautifulSoup
from bs4.element import Tag
from datetime import datetime
import requests
import time
from typing import Dict, List, Optional, Tuple

# Global flag for non-interactive mode (set via CLI)

from .log_stream import LogStream
from .records import Player, Salary, ReviewQueueItem
from .save import write_players_to_file, write_contracts_to_file, read_players_from_file, read_contracts_from_file, write_review_queue_item
from .player_lookup import (
    get_players_by_name,
    get_player_by_fangraphs_id,
    filter_players_by_year,
    add_player_to_cache,
)
from .fangraphs_search import (
    search_fangraphs_by_name,
    search_fangraphs_by_name_range,
    get_player_age_for_year,
)

CURRENT_YEAR = datetime.now().year

PRE_ARB_URL = "https://www.spotrac.com/mlb/pre-arbitration/_/year/{year}/"
ARB_URL = "https://www.spotrac.com/mlb/arbitration/_/year/{year}"
FREE_AGENT_URL = "https://www.spotrac.com/mlb/free-agents/_/year/{year}/level/mlb"

PRE_ARB_EXTENSIONS_URL = "https://www.spotrac.com/mlb/contracts/extensions/_/year/{year}/type/pre-arbitration-extension/"
ARB_EXTENSIONS_URL = "https://www.spotrac.com/mlb/contracts/extensions/_/year/{year}/type/arbitration-extension/"
VETERAN_EXTENSIONS_URL = "https://www.spotrac.com/mlb/contracts/extensions/_/year/{year}/type/extension/"

PLAYER_OBJECT_CACHE = {player.player_id: player for player in read_players_from_file()}
CONTRACT_OBJECT_CACHE = {contract.contract_id: contract for contract in read_contracts_from_file()}

LOG_STREAM = LogStream("SPOTRAC DATA GENERATION")

# Non-interactive mode flag - when True, ambiguous players are queued instead of prompting
NON_INTERACTIVE = False

def _year_exists_in_dataset(year: int) -> bool:
    """
    Check if a year already exists in the contracts dataset.

    Assumption: If a year != CURRENT_YEAR exists in the dataset, all contracts from that year
    have been successfully written (complete scrape with all contract types: pre-arb, arb, free-agent).
    This assumes that the main() function completes all fetches and writes for a year before moving on.

    Note: In rare cases of mid-scrape failures, partial data could exist. Users can use --overwrite
    to force re-fetch in these scenarios.

    Returns True if year exists in dataset, False otherwise.
    """
    for contract in CONTRACT_OBJECT_CACHE.values():
        if contract.year == year:
            return True
    return False

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
    # Extract numeric ID from URL: .../id/5166/max-scherzer -> 5166
    link_id = spotrac_link.split("/id/")[1].split("/")[0]

    return f"{name[1]}_{link_id}"

def get_fangraphs_id(first_name: str, last_name: str, contract_year: int, spotrac_link: str = "") -> int:
    """
    Get FanGraphs ID for a player, using local cache first then FanGraphs search.

    Flow:
    1. Check local player cache by name
    2. Filter candidates by career span (from local stats)
    3. If not found locally, search FanGraphs stats directly
    4. If ambiguous, queue for review or prompt user

    Args:
        first_name: Player's first name
        last_name: Player's last name
        contract_year: Year of the contract (used to filter candidates)
        spotrac_link: Spotrac URL for the player (for queuing)

    Returns:
        FanGraphs player ID, or -1 if not found/skipped
    """
    # Handle names with periods by adding a space after the first period
    if "." in first_name:
        first_name = first_name.replace(".", ". ").strip()

    # Step 1: Check local cache first
    cached_players = get_players_by_name(first_name, last_name)

    if cached_players:
        # Filter by career span around contract year
        filtered_players = filter_players_by_year(cached_players, contract_year)

        if len(filtered_players) == 1:
            player = filtered_players[0]
            LOG_STREAM.write(f"Found player in cache: {first_name} {last_name} -> FG ID {player.fangraphs_id}")
            return player.fangraphs_id

        if len(filtered_players) > 1:
            # Multiple matches in cache - need to resolve
            candidates = _build_candidates_from_players(filtered_players)
            return _resolve_multiple_candidates(first_name, last_name, contract_year, spotrac_link, candidates)

    # Step 2: Search FanGraphs stats directly
    # Search around the contract year (+/- 2 years to catch edge cases)
    search_start = max(2000, contract_year - 2)
    search_end = min(CURRENT_YEAR, contract_year + 2)

    fangraphs_results = search_fangraphs_by_name_range(first_name, last_name, search_start, search_end)

    if len(fangraphs_results) == 1:
        fg_id = fangraphs_results[0]['fg_id']
        LOG_STREAM.write(f"Found player in FanGraphs: {first_name} {last_name} -> FG ID {fg_id}")
        return fg_id

    if len(fangraphs_results) == 0:
        # No matches found
        if NON_INTERACTIVE:
            _queue_for_review(first_name, last_name, contract_year, spotrac_link, {})
            return -1

        print(f"Could not find player with name {first_name} {last_name}. Please provide first and last name.")
        print("If you do not know the player's name, please enter 'exit' to not create this player object.")
        new_first = input("First Name: ")
        if new_first.lower() == "exit":
            return -1
        new_last = input("Last Name: ")
        return get_fangraphs_id(new_first, new_last, contract_year, spotrac_link)

    # Multiple matches from FanGraphs - need to resolve
    candidates = _build_candidates_from_fangraphs(fangraphs_results)
    return _resolve_multiple_candidates(first_name, last_name, contract_year, spotrac_link, candidates)


def _build_candidates_from_players(players: List[Player]) -> Dict[int, str]:
    """Build a dict of fangraphs_id -> display string from Player objects"""
    from .player_lookup import get_career_span_from_stats

    candidates = {}
    for player in players:
        fg_id = player.fangraphs_id
        career_span = get_career_span_from_stats(player.player_id)
        if career_span:
            candidates[fg_id] = f"{player.first_name} {player.last_name} ({career_span[0]}-{career_span[1]})"
        else:
            candidates[fg_id] = f"{player.first_name} {player.last_name} (career unknown)"
    return candidates


def _build_candidates_from_fangraphs(results: List[Dict]) -> Dict[int, str]:
    """Build a dict of fangraphs_id -> display string from FanGraphs search results"""
    candidates = {}
    for result in results:
        fg_id = result['fg_id']
        name = result['name']
        first_year = result.get('first_year', '?')
        last_year = result.get('last_year', '?')
        candidates[fg_id] = f"{name} ({first_year}-{last_year})"
    return candidates


def _resolve_multiple_candidates(first_name: str, last_name: str, contract_year: int,
                                  spotrac_link: str, candidates: Dict[int, str]) -> int:
    """Handle multiple candidate matches - either queue for review or prompt user"""
    if NON_INTERACTIVE:
        _queue_for_review(first_name, last_name, contract_year, spotrac_link, candidates)
        return -1

    print(f"\nMultiple players found with name {first_name} {last_name}.")
    print("Please select the correct player from the list below.")
    print("Type '-1' if none apply and the record will be ignored.\n")

    candidate_list = list(candidates.items())
    for i, (fg_id, desc) in enumerate(candidate_list):
        print(f"  [{i}] {desc} (FanGraphs ID: {fg_id})")

    try:
        index = int(input("\nEnter the number of the correct player: "))
        if index == -1:
            return -1
        if 0 <= index < len(candidate_list):
            return candidate_list[index][0]
        print("Invalid selection.")
        return -1
    except ValueError:
        print("Invalid input.")
        return -1


def _queue_for_review(first_name: str, last_name: str, contract_year: int, spotrac_link: str, candidates: Dict[int, str]):
    """Add a player to the review queue for manual resolution"""
    item = ReviewQueueItem(
        first_name=first_name,
        last_name=last_name,
        contract_year=contract_year,
        spotrac_link=spotrac_link,
        candidates=candidates,
        added_at=datetime.now()
    )
    write_review_queue_item(item)
    print(f"  → Queued for review: {first_name} {last_name} ({contract_year})")

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
    if value_str == 'N/A' or value_str == '':
        return None
    if (float(value_str.replace("$", "").replace(",", "")) / 1_000_000) == 0:
        return None

    if player_id in PLAYER_OBJECT_CACHE:
        LOG_STREAM.write(f"Found player in cache: {player_id}")
        return PLAYER_OBJECT_CACHE[player_id]

    spotrac_link = sanitize_string(columns[headers['player']].find("a")["href"])
    fangraphs_id = get_fangraphs_id(name[0], name[1], contract_year, spotrac_link)
    if fangraphs_id == -1:
        return None

    player = Player(
        player_id=player_id,
        fangraphs_id=fangraphs_id,
        first_name=name[0],
        last_name=name[1],
        position=sanitize_string(columns[headers['pos']].get_text()),
        spotrac_link=spotrac_link,
    )

    # Add to lookup cache so subsequent lookups find this player
    add_player_to_cache(player)

    return player

def extract_salary_data(row: Tag, player_obj: Player, year: int, salary_type: str, headers: Dict[str, int]) -> Salary:
    columns = row.find_all('td')

    value_str = sanitize_string(columns[headers['value']].get_text())
    if value_str == 'N/A':
        return None

    if 'type' in headers:
        type_string = sanitize_string(columns[headers['type']].get_text())
        if type_string == 'Estimate':
            return None

    age = int(sanitize_string(columns[headers['age']].get_text())) if 'age' in headers else None
    # If age is not in Spotrac data, get it from FanGraphs stats
    if age is None:
        age = get_player_age_for_year(player_obj.fangraphs_id, year)

    return Salary(
        contract_id=f"{player_obj.player_id}_{year}",
        player_id=player_obj.player_id,
        age=age,
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

def main(start_year, end_year=None, overwrite=False, non_interactive=False):
    global NON_INTERACTIVE
    NON_INTERACTIVE = non_interactive

    for year in range(start_year, end_year+1 if end_year else start_year+1):
        # Skip fetching if year already exists and is not the current year
        if not overwrite and _year_exists_in_dataset(year) and year != CURRENT_YEAR:
            print(f"\n{'='*60}")
            print(f"Year {year} already exists in dataset (not current year)")
            print(f"Skipping scrape for year {year}")
            print(f"{'='*60}\n")
            continue

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
    args.add_argument("--overwrite", action="store_true", help="Force re-fetch of data even if year exists in dataset")
    args.add_argument("--non-interactive", action="store_true", help="Queue ambiguous players for review instead of prompting")
    args = args.parse_args()
    main(args.start_year, args.end_year, args.overwrite, args.non_interactive)