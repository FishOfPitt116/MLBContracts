import re
from .save import read_players_from_file, write_players_to_file, read_contracts_from_file, write_contracts_to_file
from .fangraphs_search import get_player_age_for_year


def clean_players_with_qo():
    """Remove players with 'QO' (qualifying offer) suffix in their player_id"""
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


def fill_missing_contract_ages():
    """Fill in missing contract ages by querying FanGraphs stats"""
    players = read_players_from_file()
    player_id_to_fg_id = {player.player_id: player.fangraphs_id for player in players}

    contracts = read_contracts_from_file()

    contracts_updated = 0

    for contract in contracts:
        if contract.age is None:
            fg_id = player_id_to_fg_id.get(contract.player_id)
            if fg_id:
                age = get_player_age_for_year(fg_id, contract.year)
                if age:
                    contract.age = age
                    contracts_updated += 1
                    print(f"  ✓ {contract.player_id} ({contract.year}): age {age}")

    if contracts_updated > 0:
        write_contracts_to_file(contracts, overwrite=True)
    print(f"Updated ages for {contracts_updated} contracts.")


if __name__ == "__main__":
    clean_players_with_qo()
    fill_missing_contract_ages()
