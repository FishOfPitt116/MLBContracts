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
from agent.phase import PhaseResolution, resolve_phase
from agent.predict.predictor import predict_contract
from agent.predict.prompts import PROMPT_VERSION, SYSTEM_PROMPT, build_prediction_prompt
from agent.predict.schema import Citation, ContractPrediction
from agent.trace import append_history, new_run_id, write_trace


def _known_prediction(resolution):
    """Build a ContractPrediction straight from an on-record contract, no LLM call.

    Used when resolution.known_value is set (agent/phase.py): the target year's
    figure is already public record (an exact contract row, or a year covered
    by an earlier multi-year deal), so "predicting" it would just be re-deriving
    a known fact at LLM cost/latency.
    """
    kv = resolution.known_value
    span = (
        str(kv["start_year"])
        if kv["start_year"] == kv["end_year"]
        else f"{kv['start_year']}-{kv['end_year']}"
    )
    note = f"{kv['contract_id']}: {kv['duration_years']}yr / ${kv['total_value_millions']}M ({span})"
    return ContractPrediction(
        aav_millions=kv["aav_millions"],
        duration_years=kv["duration_years"],
        total_value_millions=kv["total_value_millions"],
        aav_low_millions=kv["aav_millions"],
        aav_high_millions=kv["aav_millions"],
        reasoning=f"Already on record, not a prediction: {note}.",
        citations=[
            Citation(
                source_type="tool",
                claim=(
                    f"{kv['contract_id']}: {kv['duration_years']}yr / "
                    f"${kv['total_value_millions']}M (${kv['aav_millions']}M AAV)"
                ),
                basis="dataset/contracts_spotrac.csv (Spotrac-sourced contract record)",
                tool_name="resolve_phase",
                tool_call_ref=kv["contract_id"],
            )
        ],
        confidence="high",
    )


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


def run_prediction(player_row, year, model_id, quiet=False, resolution=None, force_predict=False):
    """Predict one player-year; persist trace + history. Returns (prediction, trace_path).

    resolution: an optional pre-resolved PhaseResolution (e.g. a synthetic
    hypothetical-free-agent one from predict_for). Defaults to resolve_phase().

    force_predict: if True, always calls the LLM even when resolution.known_value
    is set. Used by the backtest harness, which deliberately measures LLM
    accuracy against known historical outcomes — everywhere else (predict_for,
    the direct CLI), a known year skips the LLM call entirely.
    """
    player_id = player_row["player_id"]
    player_name = f"{player_row['first_name']} {player_row['last_name']}"

    resolution = resolution if resolution is not None else resolve_phase(player_id, year)

    if resolution.known_value is not None and not force_predict:
        prediction = _known_prediction(resolution)
        user_prompt = "(skipped: target year is already on record — see phase_resolution.known_value)"
        messages, usage, latency = [], None, 0.0
    else:
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
    return prediction, trace_path


def predict_for(player_id, year, mode="predict", model_id=None):
    """Resolve a player_id + year (+ mode) into a prediction summary dict.

    mode="predict" resolves phase deterministically as usual — if the target
    year turns out to already be on record (resolve_phase's known_value), no
    LLM call happens at all; see run_prediction. mode="hypothetical_free_agent"
    ignores the player's actual contract status and forces a free-agent
    valuation, for requests like "what would this player get in free agency
    right now" (never on-record by construction, always a real prediction).

    Used by agent/predict/tools.py's predict_tool, so the orchestrator agent
    can invoke a prediction without going through the direct-flags CLI.
    """
    model_id = model_id or DEFAULT_MODEL_ID
    player_row = lookup_player(player_id=player_id)

    if mode == "hypothetical_free_agent":
        age = resolve_phase(player_id, year).age_estimate
        resolution = PhaseResolution(
            player_id=player_id,
            year=year,
            phase="free-agent",
            method="hypothetical",
            service_time_estimate=None,
            age_estimate=age,
            notes=[
                "user requested a hypothetical free-agent valuation; "
                "current contract status intentionally ignored"
            ],
        )
    else:
        resolution = resolve_phase(player_id, year)

    prediction, trace_path = run_prediction(
        player_row, year, model_id, quiet=True, resolution=resolution
    )
    return {
        "aav_millions": prediction.aav_millions,
        "duration_years": prediction.duration_years,
        "total_value_millions": prediction.total_value_millions,
        "aav_low_millions": prediction.aav_low_millions,
        "aav_high_millions": prediction.aav_high_millions,
        "confidence": prediction.confidence,
        "reasoning": prediction.reasoning,
        "trace_path": str(trace_path),
    }


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
