from dataclasses import dataclass

from agent.phase import PhaseResolution


@dataclass
class PhaseMismatchError(Exception):
    player_id: str
    year: int
    expected_phase: str
    actual_phase: str
    resolution: PhaseResolution

    def __str__(self):
        return (
            f"{self.player_id} in {self.year}: expected phase={self.expected_phase!r}, "
            f"resolved phase={self.actual_phase!r} (method={self.resolution.method})"
        )
