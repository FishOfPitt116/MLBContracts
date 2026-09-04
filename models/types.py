from dataclasses import dataclass, field

from agent.phase import PhaseResolution


@dataclass
class YearPrediction:
    player_id: str
    year: int
    phase: str
    method: str
    aav_millions: float
    within_current_cba: bool
    phase_resolution: PhaseResolution
    notes: list = field(default_factory=list)
