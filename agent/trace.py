"""Reproducibility artifacts: per-run trace JSON + append-only history CSV.

Every prediction run writes:
- predictions/traces/{run_id}.json — everything needed to audit/reproduce the
  run: model id, prompt version, phase resolution, full messages, structured
  output (citations included), token usage.
- a row in predictions/history.csv — so repeated runs for the same player
  over time show how the projection fluctuates as inputs change.
"""

import csv
import json
import uuid
from dataclasses import asdict
from datetime import datetime, timezone

from agent.config import CONVERSATIONS_DIR, HISTORY_CSV, TRACES_DIR

HISTORY_HEADERS = [
    "run_date",
    "run_id",
    "player_id",
    "target_year",
    "phase",
    "predicted_aav",
    "predicted_duration",
    "predicted_total",
    "aav_low",
    "aav_high",
    "actual_aav",
    "confidence",
    "model_id",
    "prompt_version",
    "trace_path",
]


def new_run_id(player_id):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{player_id}_{uuid.uuid4().hex[:8]}"


def write_trace(
    run_id,
    model_id,
    prompt_version,
    phase_resolution,
    system_prompt,
    user_prompt,
    messages,
    prediction,
    usage=None,
    latency_seconds=None,
    traces_dir=None,
):
    """Persist the full run trace as JSON. Returns the trace file path."""
    traces_dir = traces_dir or TRACES_DIR
    traces_dir.mkdir(parents=True, exist_ok=True)
    trace = {
        "run_id": run_id,
        "run_date": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "prompt_version": prompt_version,
        "phase_resolution": asdict(phase_resolution),
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "messages": messages,
        "structured_output": prediction.model_dump(),
        "arithmetic_note": prediction.arithmetic_note(),
        "usage": usage,
        "latency_seconds": latency_seconds,
    }
    path = traces_dir / f"{run_id}.json"
    with open(path, mode="w") as file:
        json.dump(trace, file, indent=2, default=str)
    return path


def append_history(
    run_id,
    player_id,
    target_year,
    phase,
    prediction,
    model_id,
    prompt_version,
    trace_path,
    actual_aav=None,
    history_csv=None,
):
    """Append one prediction row to the history CSV (header written if new)."""
    history_csv = history_csv or HISTORY_CSV
    history_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not history_csv.exists()
    with open(history_csv, mode="a", newline="") as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(HISTORY_HEADERS)
        writer.writerow(
            [
                datetime.now(timezone.utc).isoformat(),
                run_id,
                player_id,
                target_year,
                phase,
                prediction.aav_millions,
                prediction.duration_years,
                prediction.total_value_millions,
                prediction.aav_low_millions,
                prediction.aav_high_millions,
                "" if actual_aav is None else actual_aav,
                prediction.confidence,
                model_id,
                prompt_version,
                str(trace_path),
            ]
        )


def write_conversation_trace(
    run_id,
    model_id,
    initial_request,
    turns,
    prediction_trace_path=None,
    conversations_dir=None,
):
    """Persist the orchestrator's turn-by-turn transcript for one conversation.

    Kept separate from write_trace()'s per-prediction trace: the prediction
    itself stays exactly as comparable across runs as it is today (fixed
    prompt shape); this file records the free-form, variable-length
    conversation that led to invoking it, linking back via
    prediction_trace_path.
    """
    conversations_dir = conversations_dir or CONVERSATIONS_DIR
    conversations_dir.mkdir(parents=True, exist_ok=True)
    trace = {
        "run_id": run_id,
        "run_date": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "initial_request": initial_request,
        "turns": turns,
        "prediction_trace_path": str(prediction_trace_path) if prediction_trace_path else None,
    }
    path = conversations_dir / f"{run_id}.json"
    with open(path, mode="w") as file:
        json.dump(trace, file, indent=2, default=str)
    return path
