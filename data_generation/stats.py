from pybaseball import playerid_reverse_lookup, batting_stats, pitching_stats, cache
from typing import Dict, List, Tuple
import pandas as pd

from data_generation.records import BatterStats, PitcherStats
from data_generation.save import read_players_from_file, write_stats_to_file

def get_fangraphs_playerid_list() -> Dict[int, str]:
    return {player.fangraphs_id: player.player_id for player in read_players_from_file() if player.fangraphs_id is not None}


def create_batter_stat_record(player_id: str, year: int, window_years: int, row: pd.Series) -> BatterStats:
    """Convert a pybaseball batting stats row to a BatterStats record"""
    return BatterStats(
        player_id=player_id,
        year=year,
        window_years=window_years,

        # Counting stats
        games=int(row.get('G', 0)) if pd.notna(row.get('G')) else None,
        plate_appearances=int(row.get('PA', 0)) if pd.notna(row.get('PA')) else None,
        at_bats=int(row.get('AB', 0)) if pd.notna(row.get('AB')) else None,
        hits=int(row.get('H', 0)) if pd.notna(row.get('H')) else None,
        doubles=int(row.get('2B', 0)) if pd.notna(row.get('2B')) else None,
        triples=int(row.get('3B', 0)) if pd.notna(row.get('3B')) else None,
        home_runs=int(row.get('HR', 0)) if pd.notna(row.get('HR')) else None,
        runs=int(row.get('R', 0)) if pd.notna(row.get('R')) else None,
        rbis=int(row.get('RBI', 0)) if pd.notna(row.get('RBI')) else None,
        stolen_bases=int(row.get('SB', 0)) if pd.notna(row.get('SB')) else None,
        caught_stealing=int(row.get('CS', 0)) if pd.notna(row.get('CS')) else None,
        walks=int(row.get('BB', 0)) if pd.notna(row.get('BB')) else None,
        strikeouts=int(row.get('SO', 0)) if pd.notna(row.get('SO')) else None,

        # Rate stats
        batting_avg=float(row.get('AVG', 0.0)) if pd.notna(row.get('AVG')) else None,
        on_base_pct=float(row.get('OBP', 0.0)) if pd.notna(row.get('OBP')) else None,
        slugging_pct=float(row.get('SLG', 0.0)) if pd.notna(row.get('SLG')) else None,
        ops=float(row.get('OPS', 0.0)) if pd.notna(row.get('OPS')) else None,

        # Advanced stats
        wrc_plus=float(row.get('wRC+', 0.0)) if pd.notna(row.get('wRC+')) else None,
        war=float(row.get('WAR', 0.0)) if pd.notna(row.get('WAR')) else None,
        babip=float(row.get('BABIP', 0.0)) if pd.notna(row.get('BABIP')) else None,
        iso=float(row.get('ISO', 0.0)) if pd.notna(row.get('ISO')) else None,

        # Discipline
        bb_pct=float(row.get('BB%', 0.0)) if pd.notna(row.get('BB%')) else None,
        k_pct=float(row.get('K%', 0.0)) if pd.notna(row.get('K%')) else None,

        # Batted ball
        hard_hit_pct=float(row.get('HardHit%', 0.0)) if pd.notna(row.get('HardHit%')) else None,
        barrel_pct=float(row.get('Barrel%', 0.0)) if pd.notna(row.get('Barrel%')) else None,
    )


def create_pitcher_stat_record(player_id: str, year: int, window_years: int, row: pd.Series) -> PitcherStats:
    """Convert a pybaseball pitching stats row to a PitcherStats record"""
    return PitcherStats(
        player_id=player_id,
        year=year,
        window_years=window_years,

        # Counting stats
        games=int(row.get('G', 0)) if pd.notna(row.get('G')) else None,
        games_started=int(row.get('GS', 0)) if pd.notna(row.get('GS')) else None,
        innings_pitched=float(row.get('IP', 0.0)) if pd.notna(row.get('IP')) else None,
        wins=int(row.get('W', 0)) if pd.notna(row.get('W')) else None,
        losses=int(row.get('L', 0)) if pd.notna(row.get('L')) else None,
        saves=int(row.get('SV', 0)) if pd.notna(row.get('SV')) else None,
        holds=int(row.get('HLD', 0)) if pd.notna(row.get('HLD')) else None,
        strikeouts=int(row.get('SO', 0)) if pd.notna(row.get('SO')) else None,
        walks=int(row.get('BB', 0)) if pd.notna(row.get('BB')) else None,
        hits_allowed=int(row.get('H', 0)) if pd.notna(row.get('H')) else None,
        home_runs_allowed=int(row.get('HR', 0)) if pd.notna(row.get('HR')) else None,

        # Rate stats
        era=float(row.get('ERA', 0.0)) if pd.notna(row.get('ERA')) else None,
        whip=float(row.get('WHIP', 0.0)) if pd.notna(row.get('WHIP')) else None,
        k_per_9=float(row.get('K/9', 0.0)) if pd.notna(row.get('K/9')) else None,
        bb_per_9=float(row.get('BB/9', 0.0)) if pd.notna(row.get('BB/9')) else None,

        # Advanced stats
        fip=float(row.get('FIP', 0.0)) if pd.notna(row.get('FIP')) else None,
        xfip=float(row.get('xFIP', 0.0)) if pd.notna(row.get('xFIP')) else None,
        siera=float(row.get('SIERA', 0.0)) if pd.notna(row.get('SIERA')) else None,
        war=float(row.get('WAR', 0.0)) if pd.notna(row.get('WAR')) else None,

        # Discipline
        k_pct=float(row.get('K%', 0.0)) if pd.notna(row.get('K%')) else None,
        bb_pct=float(row.get('BB%', 0.0)) if pd.notna(row.get('BB%')) else None,
        k_bb_ratio=float(row.get('K/BB', 0.0)) if pd.notna(row.get('K/BB')) else None,

        # Batted ball
        ground_ball_pct=float(row.get('GB%', 0.0)) if pd.notna(row.get('GB%')) else None,
        fly_ball_pct=float(row.get('FB%', 0.0)) if pd.notna(row.get('FB%')) else None,
        hard_hit_pct=float(row.get('HardHit%', 0.0)) if pd.notna(row.get('HardHit%')) else None,
    )


def assemble_stat_records(internal_player_id: str, fangraphs_player_id: str, start_year: int, end_year: int) -> Tuple[List[BatterStats], List[PitcherStats]]:
    """
    Collects both individual year stats and accumulated window stats for a player.

    Windows collected:
    - Individual years: window_years=1
    - 3-year windows: Recent performance
    - 5-year windows: Medium-term track record
    - 10-year windows: Long-term success (max to avoid FanGraphs 500 error)

    Args:
        internal_player_id: Your internal player ID
        fangraphs_player_id: FanGraphs player ID for API queries
        start_year: First year player played in MLB
        end_year: Last year player played in MLB

    Returns:
        Tuple of (batter_stats_list, pitcher_stats_list)
    """
    # Note that we cannot for greater than 10 year windows due to a fangraphs limitation: https://github.com/jldbc/pybaseball/issues/492#issuecomment-3741471714
    WINDOW_SIZES = [3, 5, 10]  # Different lookback periods for model

    batter_stats_list = []
    pitcher_stats_list = []

    print(f"\n{'='*60}")
    print(f"Processing {internal_player_id} (FanGraphs ID: {fangraphs_player_id})")
    print(f"Career span: {start_year}-{end_year}")
    print(f"{'='*60}\n")

    # Step 1: Get individual year stats (window_years = 1)
    print("Collecting individual year stats...")
    for year in range(start_year, end_year + 1):
        try:
            # Batting stats for this year
            # qual=0: Include all players regardless of minimum appearance threshold
            # Filtering for statistical relevance happens at model-building time
            raw_batting_stats = batting_stats(year, qual=0)
            player_batting = raw_batting_stats[raw_batting_stats['IDfg'] == fangraphs_player_id]

            if not player_batting.empty:
                print(f"  ✓ Found batting stats for {year}")
                batter_record = create_batter_stat_record(
                    internal_player_id,
                    year,
                    window_years=1,
                    row=player_batting.iloc[0]
                )
                batter_stats_list.append(batter_record)
            else:
                print(f"  - No batting stats for {year}")

            # Pitching stats for this year
            raw_pitching_stats = pitching_stats(year, qual=0)
            player_pitching = raw_pitching_stats[raw_pitching_stats['IDfg'] == fangraphs_player_id]

            if not player_pitching.empty:
                print(f"  ✓ Found pitching stats for {year}")
                pitcher_record = create_pitcher_stat_record(
                    internal_player_id,
                    year,
                    window_years=1,
                    row=player_pitching.iloc[0]
                )
                pitcher_stats_list.append(pitcher_record)
            else:
                print(f"  - No pitching stats for {year}")

        except Exception as e:
            print(f"  ✗ Error fetching year {year}: {e}")
            continue

    # Step 2: Get accumulated stats in rolling windows
    print(f"\nCollecting accumulated stats (windows: {WINDOW_SIZES})...")

    for year in range(start_year, end_year + 1):
        for window in WINDOW_SIZES:
            if year - window + 1 < start_year:
                print(f"  - Skipping {window}yr window ending {year} (insufficient years played)")
                continue  # Not enough years played yet for this window
            window_start = max(start_year, year - window + 1)

            # Only query if we have a valid range (at least 2 years)
            if window_start >= year:
                continue

            try:
                # Accumulated batting stats
                raw_accumulated_batting = batting_stats(window_start, year, ind=0, qual=0)
                player_accumulated_batting = raw_accumulated_batting[
                    raw_accumulated_batting['IDfg'] == fangraphs_player_id
                ]

                if not player_accumulated_batting.empty:
                    print(f"  ✓ Found {window}yr batting window ending {year}")
                    batter_record = create_batter_stat_record(
                        internal_player_id,
                        year,
                        window_years=window,
                        row=player_accumulated_batting.iloc[0]
                    )
                    batter_stats_list.append(batter_record)

                # Accumulated pitching stats
                raw_accumulated_pitching = pitching_stats(window_start, year, ind=0, qual=0)
                player_accumulated_pitching = raw_accumulated_pitching[
                    raw_accumulated_pitching['IDfg'] == fangraphs_player_id
                ]

                if not player_accumulated_pitching.empty:
                    print(f"  ✓ Found {window}yr pitching window ending {year}")
                    pitcher_record = create_pitcher_stat_record(
                        internal_player_id,
                        year,
                        window_years=window,
                        row=player_accumulated_pitching.iloc[0]
                    )
                    pitcher_stats_list.append(pitcher_record)

            except Exception as e:
                print(f"  ✗ Error fetching {window}yr window ending {year}: {e}")
                continue

    print(f"\nCompleted {internal_player_id}:")
    print(f"  - {len(batter_stats_list)} batting records")
    print(f"  - {len(pitcher_stats_list)} pitching records")

    return batter_stats_list, pitcher_stats_list


def main():
    """Main execution: process all players with FanGraphs IDs"""
    fg_player_ids = get_fangraphs_playerid_list()
    player_data = playerid_reverse_lookup(list(fg_player_ids.keys()), key_type='fangraphs')

    total_players = len(fg_player_ids)
    print(f"\n{'='*60}")
    print(f"Starting stats collection for {total_players} players")
    print(f"{'='*60}\n")

    for idx, fangraphs_id in enumerate(fg_player_ids.keys(), 1):
        player_info = player_data[player_data['key_fangraphs'] == fangraphs_id]

        if not player_info.empty:
            mlb_played_first = int(player_info['mlb_played_first'].values[0])
            mlb_played_last = int(player_info['mlb_played_last'].values[0])

            print(f"\n[{idx}/{total_players}] Processing player {fg_player_ids.get(fangraphs_id)}")
            print(f"Career: {mlb_played_first}-{mlb_played_last}")

            player_batter_stats, player_pitching_stats = assemble_stat_records(
                fg_player_ids.get(fangraphs_id),
                fangraphs_id,
                mlb_played_first,
                mlb_played_last
            )

            print(f"Writing records to file...")
            write_stats_to_file(player_batter_stats, player_pitching_stats)
            print(f"✓ Completed {fg_player_ids.get(fangraphs_id)}")
        else:
            print(f"⚠ No career info found for FanGraphs ID {fangraphs_id}")

    print(f"\n{'='*60}")
    print(f"Stats collection complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    print("Enabling pybaseball cache...")
    cache.enable()
    main()