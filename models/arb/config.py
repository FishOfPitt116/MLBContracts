from agent.config import REPO_ROOT

CONTRACT_TYPE = "arb"
MAX_DURATION = 1
EXCLUDED_PLAYER_IDS = ["Ohtani_24661"]  # two-way player, no real comparable
SP_SPLIT_THRESHOLD = 0.5
STATS_WINDOW_YEARS = 3
MIN_TRAINING_ROWS = 20

CONTRACTS_WITH_STATS_CSV = REPO_ROOT / "dataset" / "contracts_with_stats.csv"

SUPER_TWO = "Super Two"
ARB_1 = "Arb-1"
ARB_2 = "Arb-2"
ARB_3 = "Arb-3"
ARB_BAND_THRESHOLDS = [(3.0, SUPER_TWO), (4.0, ARB_1), (5.0, ARB_2)]  # else ARB_3
SUPPORTED_ARB_BANDS = [SUPER_TWO, ARB_1]

ROLE_FEATURE_SPECS = {
    "batter": ["war_3y", "volume_3y", "service_time", "war_x_service"],
    "SP": ["war_3y", "service_time", "war_x_service"],
    "RP": ["war_3y", "volume_3y", "service_time", "war_x_service"],
}
