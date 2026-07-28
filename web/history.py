"""Reads predictions/history.csv for the sidebar's prediction list.

history.csv (agent/trace.py:append_history) already has exactly what a list
view needs -- one row per prediction with player_id, target_year, phase, the
predicted figures, and confidence -- written at the same time as each full
trace JSON, so it's always in sync and far cheaper to read than opening every
trace file (each of which also carries full message history/token usage,
useful for a future detail view, not a list).

Plain csv (not pandas) deliberately: every value comes back as a string,
which sidesteps NaN/numpy-dtype JSON-serialization edge cases entirely for a
handful of small local files -- the frontend parses numbers itself.

Note: this file also picks up `make backtest-agent` runs, not just
interactive predictions (agent/predict/predict.py:run_prediction is shared by
both) -- surfaced as-is for now (see docs), not filtered.
"""

import csv

from agent.config import HISTORY_CSV, PLAYERS_CSV


def _player_names():
    if not PLAYERS_CSV.exists():
        return {}
    with open(PLAYERS_CSV, newline="") as file:
        return {row["player_id"]: f"{row['first_name']} {row['last_name']}" for row in csv.DictReader(file)}


def load_history():
    """Every prediction row, newest first, with a resolved player_name added."""
    if not HISTORY_CSV.exists():
        return []
    names = _player_names()
    with open(HISTORY_CSV, newline="") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        row["player_name"] = names.get(row["player_id"], row["player_id"])
    rows.reverse()  # file is append-only oldest-first
    return rows
