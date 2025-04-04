from bs4 import BeautifulSoup
from datetime import datetime
from pybaseball import playerid_reverse_lookup
import re
import requests
from save import read_players_from_file, write_players_to_file, read_contracts_from_file, write_contracts_to_file
import time

def clean_players_with_qo():
    players = read_players_from_file()
    cleaned_players = [player for player in players if not re.match(r".*QO_\d+$", player.player_id)]
    removed_player_ids = {player.player_id for player in players if re.match(r".*QO_\d+$", player.player_id)}

    contracts = []
    cleaned_contracts = []
    
    if not len(removed_player_ids) == 0:
        write_players_to_file(cleaned_players, overwrite=True)
    
        contracts = read_contracts_from_file()
        cleaned_contracts = [contract for contract in contracts if contract.player_id not in removed_player_ids]
        
        write_contracts_to_file(cleaned_contracts, overwrite=True)
    
    print(f"Removed {len(players) - len(cleaned_players)} players with 'QO_<number>' in their player_id.")
    print(f"Removed {len(contracts) - len(cleaned_contracts)} contracts associated with those players.")

def update_baseball_reference_links():
    players = read_players_from_file()
    missing_link_players = [player for player in players if player.baseball_reference_link is None]
    
    if len(missing_link_players) == 0:
        print("No players missing baseball reference links.")
        return
    
    fangraphs_ids = [player.fangraphs_id for player in missing_link_players]
    lookup_df = playerid_reverse_lookup(fangraphs_ids, key_type='fangraphs')
    
    lookup_map = {int(row['key_fangraphs']): f"https://www.baseball-reference.com/players/{row['key_bbref'][0]}/{row['key_bbref']}.shtml" 
                  for _, row in lookup_df.iterrows()}
    
    for player in players:
        if player.fangraphs_id in lookup_map:
            player.baseball_reference_link = lookup_map[player.fangraphs_id]
    
    write_players_to_file(players, overwrite=True)
    print(f"Updated {len(lookup_map)} players with missing baseball reference links.")

def update_player_birthdays():
    players = read_players_from_file()
    players_to_update = [player for player in players if not player.birth_date and player.baseball_reference_link]
    
    if not players_to_update:
        print("No players missing birth dates.")
        return
    
    updated_players = 0
    for player in players_to_update:
        print(f"Updating birthday for {player.first_name} {player.last_name}")
        response = requests.get(player.baseball_reference_link)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            birth_element = soup.find('span', id='necro-birth')
            print(f"Birth element found: {birth_element}")
            if birth_element and 'data-birth' in birth_element.attrs:
                player.birth_date = datetime.strptime(birth_element['data-birth'], "%Y-%m-%d")
                updated_players += 1
            print(f"Updated birthday for {player.first_name} {player.last_name}: {player.birth_date}")
        elif response.status_code == 429:
            print(f"Rate limit exceeded for {player.first_name} {player.last_name}. Exiting loop.")
            print(response.text)
            break
        time.sleep(4) # to avoid rate limiting
    
    write_players_to_file(players, overwrite=True)
    print(f"Updated {updated_players} players with missing birth dates.")

if __name__ == "__main__":
    clean_players_with_qo()
    update_baseball_reference_links()
    update_player_birthdays()