from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple

@dataclass
class Player:
    player_id: str # PK
    fangraphs_id: int
    first_name: str
    last_name: str
    position: str
    birth_date: datetime
    spotrac_link: str
    baseball_reference_link: str

@dataclass
class Salary:
    contract_id: str # PK
    player_id: str # FK
    age: int
    service_time: float
    year: int
    duration: int
    value: float
    type: str # [pre-arb, arb, free-agent]

@dataclass
class Stats:
    """Base class for all player statistics"""
    player_id: str
    year: int
    window_years: int  # 1 for single year, 3/5/10 for accumulated windows
    
    def get_key(self) -> Tuple[str, int, int]:
        """Returns composite key for deduplication and lookups"""
        return (self.player_id, self.year, self.window_years)
    
    def is_single_year(self) -> bool:
        """True if this represents a single season"""
        return self.window_years == 1
    
    def is_accumulated(self) -> bool:
        """True if this represents accumulated stats over multiple years"""
        return self.window_years > 1

@dataclass
class BatterStats(Stats):
    # Counting stats
    games: Optional[int] = None
    plate_appearances: Optional[int] = None
    at_bats: Optional[int] = None
    hits: Optional[int] = None
    doubles: Optional[int] = None
    triples: Optional[int] = None
    home_runs: Optional[int] = None
    runs: Optional[int] = None
    rbis: Optional[int] = None
    stolen_bases: Optional[int] = None
    caught_stealing: Optional[int] = None
    walks: Optional[int] = None
    strikeouts: Optional[int] = None
    
    # Rate stats
    batting_avg: Optional[float] = None
    on_base_pct: Optional[float] = None
    slugging_pct: Optional[float] = None
    ops: Optional[float] = None
    
    # Advanced stats (FanGraphs)
    wrc_plus: Optional[float] = None  # Weighted Runs Created Plus (park/league adjusted)
    war: Optional[float] = None  # Wins Above Replacement
    babip: Optional[float] = None  # Batting Average on Balls In Play
    iso: Optional[float] = None  # Isolated Power
    
    # Discipline
    bb_pct: Optional[float] = None  # Walk percentage
    k_pct: Optional[float] = None  # Strikeout percentage
    
    # Batted ball
    hard_hit_pct: Optional[float] = None
    barrel_pct: Optional[float] = None

@dataclass
class PitcherStats(Stats):    
    # Counting stats
    games: Optional[int] = None
    games_started: Optional[int] = None
    innings_pitched: Optional[float] = None
    wins: Optional[int] = None
    losses: Optional[int] = None
    saves: Optional[int] = None
    holds: Optional[int] = None
    strikeouts: Optional[int] = None
    walks: Optional[int] = None
    hits_allowed: Optional[int] = None
    home_runs_allowed: Optional[int] = None
    
    # Rate stats
    era: Optional[float] = None  # Earned Run Average
    whip: Optional[float] = None  # Walks + Hits per Inning Pitched
    k_per_9: Optional[float] = None  # Strikeouts per 9 innings
    bb_per_9: Optional[float] = None  # Walks per 9 innings
    
    # Advanced stats (FanGraphs)
    fip: Optional[float] = None  # Fielding Independent Pitching
    xfip: Optional[float] = None  # Expected FIP
    siera: Optional[float] = None  # Skill-Interactive ERA
    war: Optional[float] = None  # Wins Above Replacement
    
    # Discipline
    k_pct: Optional[float] = None  # Strikeout percentage
    bb_pct: Optional[float] = None  # Walk percentage
    k_bb_ratio: Optional[float] = None
    
    # Batted ball
    ground_ball_pct: Optional[float] = None
    fly_ball_pct: Optional[float] = None
    hard_hit_pct: Optional[float] = None
