from bs4 import BeautifulSoup
from bs4.element import Tag
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

def get_table_headers(table: Tag) -> List[str]:
    print( [ sanitize_string(header.get_text()).replace("$", " ").split(" ")[0].lower() for header in table.find("thead").find_all("th") ] )
    return { sanitize_string(header.get_text()).replace("$", " ").split(" ")[0].lower() : i for i, header in enumerate(table.find("thead").find_all("th")) }

def sanitize_string(s: str) -> str:
    return s.replace("\n", "").strip()

def extract_player_data(row: Tag, headers: Dict[str, int]) -> Player:
    columns = row.find_all('td')
    name = sanitize_string(columns[headers['player']].get_text()).split(" ", 1)
    spotrac_link = sanitize_string(columns[headers['player']].find("a")["href"])
    link_id = spotrac_link.split("/")[-1]
    return Player(
        player_id=f"{name[1]}_{link_id}",
        first_name=name[0],
        last_name=name[1],
        position=sanitize_string(columns[headers['pos']].get_text()),
        birth_date=None,  # TODO: fill in
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

def main():
    for year in range(2011, 2026):
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
    main()