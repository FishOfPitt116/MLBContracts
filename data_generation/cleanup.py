import re
from save import read_players_from_file, write_players_to_file, read_contracts_from_file, write_contracts_to_file

def clean_players_with_qo():
    players = read_players_from_file()
    cleaned_players = [player for player in players if not re.match(r".*QO_\d+$", player.player_id)]
    removed_player_ids = {player.player_id for player in players if re.match(r".*QO_\d+$", player.player_id)}
    
    write_players_to_file(cleaned_players, overwrite=True)
    
    contracts = read_contracts_from_file()
    cleaned_contracts = [contract for contract in contracts if contract.player_id not in removed_player_ids]
    
    write_contracts_to_file(cleaned_contracts, overwrite=True)
    
    print(f"Removed {len(players) - len(cleaned_players)} players with 'QO_<number>' in their player_id.")
    print(f"Removed {len(contracts) - len(cleaned_contracts)} contracts associated with those players.")

if __name__ == "__main__":
    clean_players_with_qo()
