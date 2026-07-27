"""Backtest the prediction agent against historical contracts.

Samples N contracts per phase (deterministically, via --seed), predicts each
with the Phase 0 agent, and scores predicted vs actual AAV using the shared
metrics in agent/metrics.py.

CAVEAT — training-data contamination: in Phase 0 the LLM has likely memorized
many historical contract outcomes, so backtests on past contracts measure
recall as much as prediction skill. Treat these numbers as harness
scaffolding and a citation-quality check; predictions for future seasons
accumulating in predictions/history.csv are the honest test.

Usage:
    python -m agent.backtest --n-per-phase 5 --seed 42
    python -m agent.backtest --n-per-phase 3 --phase free-agent
"""

import json
from argparse import ArgumentParser
from datetime import datetime, timezone

import pandas as pd

from agent.config import BACKTESTS_DIR, CONTRACTS_CSV, DEFAULT_MODEL_ID, PLAYERS_CSV
from agent.metrics import calculate_all_metrics, format_metrics_report
from agent.predict import run_prediction

PHASES = ["pre-arb", "arb", "free-agent"]

# Tolerance (in $M) for the pct-within-tolerance metric, scaled to the salary
# range typical of each phase.
PHASE_TOLERANCES = {"pre-arb": 0.25, "arb": 1.0, "free-agent": 5.0}

CAVEAT = (
    "CAVEAT: Phase 0 backtests are contaminated — the LLM has likely memorized\n"
    "historical contract outcomes, so these metrics measure recall as much as\n"
    "prediction. Use them as scaffolding + citation-quality checks, not skill."
)


def sample_contracts(n_per_phase, seed, phases):
    contracts = (
        pd.read_csv(CONTRACTS_CSV)
        .drop_duplicates(subset="contract_id")
        .query("duration >= 1 and value > 0")
        .sort_values("contract_id")
        .reset_index(drop=True)
    )
    samples = []
    for phase in phases:
        pool = contracts[contracts["type"] == phase]
        n = min(n_per_phase, len(pool))
        samples.append(pool.sample(n=n, random_state=seed))
    return pd.concat(samples).reset_index(drop=True)


def main():
    parser = ArgumentParser(description="Backtest the contract prediction agent.")
    parser.add_argument("--n-per-phase", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--phase", choices=PHASES, action="append", dest="phases",
                        help="Restrict to one or more phases (default: all)")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, help="OpenAI model id")
    args = parser.parse_args()
    phases = args.phases or PHASES

    print(CAVEAT)
    print()

    players = pd.read_csv(PLAYERS_CSV).set_index("player_id")
    sampled = sample_contracts(args.n_per_phase, args.seed, phases)
    print(f"Backtesting {len(sampled)} contracts ({args.n_per_phase}/phase, seed {args.seed}, "
          f"model {args.model})\n")

    results = []
    for _, contract in sampled.iterrows():
        player_id = contract["player_id"]
        year = int(contract["year"])
        actual_aav = float(contract["value"]) / int(contract["duration"])
        if player_id not in players.index:
            print(f"skip {contract['contract_id']}: no players.csv row")
            continue
        player_row = players.loc[player_id]
        if isinstance(player_row, pd.DataFrame):
            player_row = player_row.iloc[0]
        player_row = player_row.copy()
        player_row["player_id"] = player_id

        print(f"predicting {contract['contract_id']} "
              f"({player_row['first_name']} {player_row['last_name']}, {contract['type']})...")
        try:
            # force_predict: the backtest deliberately measures LLM accuracy against
            # known historical outcomes, so it must not take the known-contract shortcut.
            prediction, _ = run_prediction(player_row, year, args.model, quiet=True, force_predict=True)
        except Exception as error:
            print(f"  FAILED: {error}")
            results.append({
                "contract_id": contract["contract_id"],
                "phase": contract["type"],
                "error": str(error),
            })
            continue
        results.append({
            "contract_id": contract["contract_id"],
            "phase": contract["type"],
            "year": year,
            "actual_aav": actual_aav,
            "predicted_aav": prediction.aav_millions,
            "actual_duration": int(contract["duration"]),
            "predicted_duration": prediction.duration_years,
            "confidence": prediction.confidence,
            "n_citations": len(prediction.citations),
        })

    scored = [r for r in results if "error" not in r]
    summary = {"metrics_by_phase": {}}
    for phase in phases:
        phase_results = [r for r in scored if r["phase"] == phase]
        if not phase_results:
            continue
        y_true = [r["actual_aav"] for r in phase_results]
        y_pred = [r["predicted_aav"] for r in phase_results]
        metrics = calculate_all_metrics(y_true, y_pred, tolerance=PHASE_TOLERANCES[phase])
        summary["metrics_by_phase"][phase] = metrics
        print(f"\n--- {phase} ---")
        print(format_metrics_report(metrics))

    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    summary.update(
        {
            "run_date": datetime.now(timezone.utc).isoformat(),
            "model_id": args.model,
            "seed": args.seed,
            "n_per_phase": args.n_per_phase,
            "caveat": CAVEAT,
            "results": results,
        }
    )
    BACKTESTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = BACKTESTS_DIR / f"backtest_{run_stamp}.json"
    with open(summary_path, mode="w") as file:
        json.dump(summary, file, indent=2, default=str)
    print(f"\nSummary written to {summary_path}")
    failures = len(results) - len(scored)
    if failures:
        print(f"{failures} prediction(s) failed — see summary JSON.")


if __name__ == "__main__":
    main()
