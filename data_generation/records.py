from dataclasses import dataclass
from datetime import datetime

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
class BatterStats:
    # TODO: fill records
    pass

@dataclass
class PitcherStats:
    # TODO: fill records
    pass