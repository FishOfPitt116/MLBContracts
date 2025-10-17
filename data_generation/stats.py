from typing import List
from .save import read_players_from_file

def get_fangraphs_playerid_list() -> List[int]:
    return [player.fangraphs_id for player in read_players_from_file() if player.fangraphs_id is not None]

def main():
    """
    PSEUDOCODE:
    - Pull down the existing stats data from CSV files using read_stats_from_file
    - Get list of player data using pybaseball's playerid_reverse_lookup method
    - For each player ID, for each year they played (between mlb_played_first and mlb_played_last), check if there's a record already in the existing stats data.
        - If not, pull down the stats for that player for that year using pybaseball's batting_stats and/or pitching_stats methods.
            - two calls per player, call 1: {start_season: curr_season, qual: 1, ind: 1}, call 2: {start_season: mlb_played_first, end_season: curr_season, qual: 1, ind: 0}
            - first call gives current year stats, second call gives historical stats for all years prior
            - Append this new stats record to the existing stats data
        - For the most recent year, check if the stats data has been updated (i.e., if the games played is different than what's already recorded)
            - If so, update the existing record with the new stats data.
    - Finally, write the updated stats data back to CSV files using write_stats_to_file
    """
    pass

if __name__ == "__main__":
    main()