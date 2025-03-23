from bs4 import BeautifulSoup
from bs4.element import Tag
from pybaseball import playerid_lookup
import requests
from typing import Dict, List, Tuple

from records import Player, Salary

PRE_ARB_URL = "https://www.spotrac.com/mlb/pre-arbitration/_/year/{year}/"
ARB_URL = "https://www.spotrac.com/mlb/arbitration/_/year/{year}"
FREE_AGENT_URL = "https://www.spotrac.com/mlb/free-agents/_/year/{year}/level/mlb"

PRE_ARB_EXTENSIONS_URL = "https://www.spotrac.com/mlb/contracts/extensions/_/year/{year}/type/pre-arbitration-extension/"
ARB_EXTENSIONS_URL = "https://www.spotrac.com/mlb/contracts/extensions/_/year/{year}/type/arbitration-extension/"
VETERAN_EXTENSIONS_URL = "https://www.spotrac.com/mlb/contracts/extensions/_/year/{year}/type/extension/"

PLAYER_OBJECT_CACHE = {}

def fetch_spotrac_data(url: str) -> BeautifulSoup:
    response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    if response.status_code == 200:
        return BeautifulSoup(response.text, "html.parser")
    return None

def sanitize_string(s: str) -> str:
    return s.replace("\n", "").strip()

# BUG: players with names such as "J.D." or "J.A." or "C.J." will not be found in the playerid_lookup function
# BUG: players with accents in their names need to be added manually 
def get_player_id(columns: List[Tag], headers: Dict[str, int]) -> str:
    name = sanitize_string(columns[headers['player']].get_text()).split(" ", 1)
    spotrac_link = sanitize_string(columns[headers['player']].find("a")["href"])
    link_id = spotrac_link.split("/")[-1]

    return f"{name[1]}_{link_id}"

def get_fangraphs_id(first_name, last_name) -> int:
    id_df = playerid_lookup(last_name, first_name)
    if len(id_df) == 1:
        return id_df["key_fangraphs"][0]
    if len(id_df) == 0:
        print(f"Could not find player with name {first_name} {last_name}. Please provide first and last name.\nIf you do not know the player's name, please enter 'exit' to not create this player object.")
        first_name = input("First Name: ")
        if first_name.lower() == "exit":
            return -1
        last_name = input("Last Name: ")
        return get_fangraphs_id(first_name, last_name)
    print(f"Multiple players found with name {first_name} {last_name}. Please select the correct player from the list below.")
    print(id_df)
    index = int(input("Enter the index number of the correct player: "))
    return id_df["key_fangraphs"][index]

def get_table_headers(table: Tag) -> List[str]:
    print( [ sanitize_string(header.get_text()).replace("$", " ").split(" ")[0].lower() for header in table.find("thead").find_all("th") ] )
    return { sanitize_string(header.get_text()).replace("$", " ").split(" ")[0].lower() : i for i, header in enumerate(table.find("thead").find_all("th")) }

def extract_player_data(row: Tag, headers: Dict[str, int]) -> Player:
    columns = row.find_all('td')
    name = sanitize_string(columns[headers['player']].get_text()).split(" ", 1)
    player_id = get_player_id(columns, headers)

    if player_id in PLAYER_OBJECT_CACHE:
        return PLAYER_OBJECT_CACHE[player_id]
    
    fangraphs_id = get_fangraphs_id(name[0], name[1])
    if fangraphs_id == -1:
        return None

    spotrac_link = sanitize_string(columns[headers['player']].find("a")["href"])
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
        baseball_reference_link=None
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
    print(headers)
    for row in table.find("tbody").find_all("tr"):
        player = extract_player_data(row, headers)
        if not player:
            continue
        if player.player_id not in PLAYER_OBJECT_CACHE:
            PLAYER_OBJECT_CACHE[player.player_id] = player
        salary = extract_salary_data(row, player, year, salary_type, headers)

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
        print(pre_arb_players)
        print(pre_arb_salaries)
        print(arb_players)
        print(arb_salaries)
        print(free_agent_players)
        print(free_agent_salaries)
        print(f"Year {year} Pre-Arb Players: {len(pre_arb_players)}")
        print(f"Year {year} Arb Players: {len(arb_players)}")
        print(f"Year {year} Free Agent Players: {len(free_agent_players)}")

if __name__ == "__main__":
    main(2011)