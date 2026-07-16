"""Predict a single player's contract.

Usage:
    python -m agent.predict --player-id Scherzer_5166 --year 2026
    python -m agent.predict --name "Max Scherzer" --year 2026
    python -m agent.predict --player-id X --year Y --model gpt-5
"""

from argparse import ArgumentParser
from datetime import datetime

import pandas as pd

from agent.config import CONTRACTS_CSV, DEFAULT_MODEL_ID, PLAYERS_CSV
from agent.phase import resolve_phase
from agent.predictor import predict_contract
from agent.prompts import PROMPT_VERSION, SYSTEM_PROMPT, build_prediction_prompt
from agent.trace import append_history, new_run_id, write_trace


def lookup_player(player_id=None, name=None):
    """Find a player row in players.csv by id or 'First Last' name."""
    players = pd.read_csv(PLAYERS_CSV)
    if player_id:
        matches = players[players["player_id"] == player_id]
    else:
        full_names = (players["first_name"] + " " + players["last_name"]).str.lower()
        matches = players[full_names == name.strip().lower()]
    if matches.empty:
        raise SystemExit(f"Player not found in {PLAYERS_CSV}: {player_id or name}")
    if len(matches) > 1:
        print(f"Multiple players match '{name}'; using the first. Candidates:")
        for _, row in matches.iterrows():
            print(f"  {row['player_id']} ({row['position']})")
    return matches.iloc[0]


def actual_aav_for(player_id, year):
    """Actual AAV if an observed contract row exists for that year, else None."""
    contracts = pd.read_csv(CONTRACTS_CSV).drop_duplicates(subset="contract_id")
    row = contracts[(contracts["player_id"] == player_id) & (contracts["year"] == year)]
    if row.empty:
        return None
    row = row.iloc[0]
    if row["duration"] < 1 or row["value"] <= 0:
        return None
    return float(row["value"]) / int(row["duration"])


def run_prediction(player_row, year, model_id, quiet=False):
    """Predict one player-year; persist trace + history. Returns the prediction."""
    player_id = player_row["player_id"]
    player_name = f"{player_row['first_name']} {player_row['last_name']}"

    resolution = resolve_phase(player_id, year)
    user_prompt = build_prediction_prompt(
        player_name=player_name,
        position=player_row["position"],
        player_id=player_id,
        target_year=year,
        phase_resolution=resolution,
    )

    prediction, messages, usage, latency = predict_contract(user_prompt, model_id)

    run_id = new_run_id(player_id)
    trace_path = write_trace(
        run_id=run_id,
        model_id=model_id,
        prompt_version=PROMPT_VERSION,
        phase_resolution=resolution,
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        messages=messages,
        prediction=prediction,
        usage=usage,
        latency_seconds=latency,
    )
    actual = actual_aav_for(player_id, year)
    append_history(
        run_id=run_id,
        player_id=player_id,
        target_year=year,
        phase=resolution.phase,
        prediction=prediction,
        model_id=model_id,
        prompt_version=PROMPT_VERSION,
        trace_path=trace_path,
        actual_aav=actual,
    )

    if not quiet:
        print(f"\n{player_name} ({player_id}) — {year} [{resolution.phase}, {resolution.method}]")
        for note in resolution.notes:
            print(f"  note: {note}")
        print(
            f"\nPrediction: {prediction.duration_years}yr x ${prediction.aav_millions}M AAV "
            f"= ${prediction.total_value_millions}M total "
            f"(range ${prediction.aav_low_millions}M-${prediction.aav_high_millions}M, "
            f"confidence {prediction.confidence})"
        )
        if actual is not None:
            print(f"Actual AAV: ${actual:.3f}M")
        note = prediction.arithmetic_note()
        if note:
            print(f"Arithmetic check: {note}")
        print(f"\nReasoning: {prediction.reasoning}")
        print("\nCitations:")
        for citation in prediction.citations:
            print(f"  [{citation.source_type}] {citation.claim}")
            print(f"      basis: {citation.basis}")
        print(f"\nTrace: {trace_path}")
    return prediction


def main():
    parser = ArgumentParser(description="Predict an MLB player's contract for a season.")
    who = parser.add_mutually_exclusive_group(required=True)
    who.add_argument("--player-id", help="Player id, e.g. Scherzer_5166")
    who.add_argument("--name", help='Player name, e.g. "Max Scherzer"')
    parser.add_argument("--year", type=int, default=datetime.now().year, help="Target season")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, help="OpenAI model id")
    args = parser.parse_args()

    player_row = lookup_player(args.player_id, args.name)
    run_prediction(player_row, args.year, args.model)


if __name__ == "__main__":
    main()
