"""Tests for trace persistence and the history CSV."""

import csv
import json

from agent.phase import PhaseResolution
from agent.tests.test_schema import make_prediction
from agent.trace import HISTORY_HEADERS, append_history, new_run_id, write_trace


def make_resolution():
    return PhaseResolution(
        player_id="Doe_1",
        year=2026,
        phase="arb",
        method="projected",
        service_time_estimate=3.5,
        age_estimate=26,
        notes=["projected from 2025 service time"],
    )


def test_write_trace(tmp_path):
    pred = make_prediction()
    run_id = new_run_id("Doe_1")
    path = write_trace(
        run_id=run_id,
        model_id="gpt-5-mini",
        prompt_version="p0.1",
        phase_resolution=make_resolution(),
        system_prompt="system",
        user_prompt="user",
        messages=[{"role": "user", "content": "user"}],
        prediction=pred,
        usage={"totalTokens": 123},
        latency_seconds=1.5,
        traces_dir=tmp_path,
    )
    assert path.exists()
    trace = json.loads(path.read_text())
    assert trace["run_id"] == run_id
    assert trace["model_id"] == "gpt-5-mini"
    assert trace["prompt_version"] == "p0.1"
    assert trace["phase_resolution"]["phase"] == "arb"
    assert trace["structured_output"]["aav_millions"] == 10.0
    assert trace["structured_output"]["citations"][0]["source_type"] == "model_knowledge"
    assert trace["arithmetic_note"] is None


def test_append_history_creates_header_once(tmp_path):
    history = tmp_path / "history.csv"
    pred = make_prediction()
    for i in range(2):
        append_history(
            run_id=f"run_{i}",
            player_id="Doe_1",
            target_year=2026,
            phase="arb",
            prediction=pred,
            model_id="gpt-5-mini",
            prompt_version="p0.1",
            trace_path=tmp_path / f"run_{i}.json",
            actual_aav=9.5 if i == 0 else None,
            history_csv=history,
        )
    with open(history, newline="") as file:
        rows = list(csv.reader(file))
    assert rows[0] == HISTORY_HEADERS
    assert len(rows) == 3  # header + 2 data rows
    header_index = {name: i for i, name in enumerate(rows[0])}
    assert rows[1][header_index["actual_aav"]] == "9.5"
    assert rows[2][header_index["actual_aav"]] == ""
    assert rows[1][header_index["predicted_aav"]] == "10.0"
