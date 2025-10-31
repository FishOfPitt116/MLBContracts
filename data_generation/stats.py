from pybaseball import playerid_reverse_lookup, batting_stats, pitching_stats
from typing import Dict, List, Tuple

from data_generation.records import BatterStats, PitcherStats
from .save import read_players_from_file, write_stats_to_file

def get_fangraphs_playerid_list() -> Dict[int, str]:
    return {player.fangraphs_id : player.player_id for player in read_players_from_file() if player.fangraphs_id is not None}


def assemble_stat_records(internal_player_id: str, fangraphs_player_id: str, start_year: int, end_year: int) -> Tuple[List[BatterStats], List[PitcherStats]]:
    # CRITICAL: we have to ensure that the pybaseball cache is enabled to avoid excessive API calls
    for year in range(start_year, end_year+1):
        raw_batting_stats = batting_stats(year)
        player_raw_batting_stats = raw_batting_stats[raw_batting_stats['IDfg'] == fangraphs_player_id]
        print(f"Raw batting stats for {internal_player_id} individual year {year}: {player_raw_batting_stats}")
        raw_pitching_stats = pitching_stats(year)
        player_raw_pitching_stats = raw_pitching_stats[raw_pitching_stats['IDfg'] == fangraphs_player_id]
        print(f"Raw pitching stats for {internal_player_id} individual year {year}: {player_raw_pitching_stats}")
        if year != start_year:
            raw_accumulated_batting_stats = batting_stats(start_year, year, ind=0)
            player_raw_accumulated_batting_stats = raw_accumulated_batting_stats[raw_accumulated_batting_stats['IDfg'] == fangraphs_player_id]
            print(f"Raw accumulated batting stats for {internal_player_id} between {(start_year, year)}: {player_raw_accumulated_batting_stats}")
            raw_accumulated_pitching_stats = pitching_stats(start_year, year, ind=0)
            player_raw_accumulated_pitching_stats = raw_accumulated_pitching_stats[raw_accumulated_pitching_stats['IDfg'] == fangraphs_player_id]
            print(f"Raw accumulated pitching stats for {internal_player_id} between {(start_year, year)}: {player_raw_accumulated_pitching_stats}")
    # TODO: append generated stat records to tracking lists and return those tracking lists
    return [], []



def main():
    fg_player_ids = get_fangraphs_playerid_list()
    player_data = playerid_reverse_lookup(fg_player_ids, key_type='fangraphs')
    for fangraphs_id in fg_player_ids.keys():
        player_info = player_data[player_data['key_fangraphs'] == fangraphs_id]
        if not player_info.empty:
            mlb_played_first = player_info['mlb_played_first'].values[0]
            mlb_played_last = player_info['mlb_played_last'].values[0]
            player_batter_stats, player_pitching_stats = assemble_stat_records(fg_player_ids.get(fangraphs_id),
                                                                               fangraphs_id,
                                                                               mlb_played_first,
                                                                               mlb_played_last)
            print(f"Writing player {fg_player_ids.get(fangraphs_id)} records to file...")
            write_stats_to_file(player_batter_stats, player_pitching_stats)
            print(f"Finished writing {fg_player_ids.get(fangraphs_id)} records to file.")


if __name__ == "__main__":
    main()