"""Tests for the known-contract shortcut (agent/predict/predict.py).

A target year whose dollar figure is already on record (resolve_phase's
known_value) should skip the LLM entirely, unless force_predict=True (the
backtest harness, which deliberately measures LLM accuracy against known
historical outcomes).
"""

import pandas as pd
import pytest

from agent.phase import PhaseResolution
from agent.predict import predict as predict_module
from agent.predict.predict import _known_prediction, run_prediction


def _known_resolution():
    return PhaseResolution(
        player_id="Scherzer_5166",
        year=2016,
        phase="free-agent",
        method="known",
        notes=["under contract through 2021 (Scherzer_5166_2015: 7yr/$210.0M free-agent)"],
        known_value={
            "contract_id": "Scherzer_5166_2015",
            "aav_millions": 30.0,
            "duration_years": 7,
            "total_value_millions": 210.0,
        },
    )


def _projected_resolution():
    return PhaseResolution(
        player_id="Doe_1",
        year=2026,
        phase="arb",
        method="projected",
        service_time_estimate=3.5,
        age_estimate=26,
        notes=["projected from 2025 service time"],
    )


def _player_row():
    return pd.Series(
        {
            "player_id": "Scherzer_5166",
            "first_name": "Max",
            "last_name": "Scherzer",
            "position": "SP",
        }
    )


class TestKnownPrediction:
    def test_uses_the_on_record_figure(self):
        prediction = _known_prediction(_known_resolution())
        assert prediction.aav_millions == 30.0
        assert prediction.duration_years == 7
        assert prediction.total_value_millions == 210.0
        assert prediction.aav_low_millions == prediction.aav_high_millions == 30.0
        assert prediction.confidence == "high"

    def test_citation_is_tool_sourced(self):
        prediction = _known_prediction(_known_resolution())
        assert len(prediction.citations) == 1
        citation = prediction.citations[0]
        assert citation.source_type == "tool"
        assert citation.tool_name == "resolve_phase"
        assert citation.tool_call_ref == "Scherzer_5166_2015"


class TestRunPredictionShortcut:
    def _no_llm_allowed(self, *args, **kwargs):
        raise AssertionError("predict_contract should not be called for a known year")

    def test_skips_the_llm_for_a_known_year(self, tmp_path, monkeypatch):
        monkeypatch.setattr(predict_module, "predict_contract", self._no_llm_allowed)
        monkeypatch.setattr("agent.trace.TRACES_DIR", tmp_path)
        monkeypatch.setattr("agent.trace.HISTORY_CSV", tmp_path / "history.csv")

        prediction, trace_path = run_prediction(
            _player_row(), 2016, "gpt-5-mini", quiet=True, resolution=_known_resolution()
        )
        assert prediction.aav_millions == 30.0
        assert trace_path.exists()

    def test_force_predict_calls_the_llm_anyway(self, tmp_path, monkeypatch):
        calls = []

        def fake_predict_contract(user_prompt, model_id):
            calls.append(user_prompt)
            from agent.predict.tests.test_schema import make_prediction

            return make_prediction(), [], {"totalTokens": 1}, 0.01

        monkeypatch.setattr(predict_module, "predict_contract", fake_predict_contract)
        monkeypatch.setattr("agent.trace.TRACES_DIR", tmp_path)
        monkeypatch.setattr("agent.trace.HISTORY_CSV", tmp_path / "history.csv")

        prediction, _ = run_prediction(
            _player_row(),
            2016,
            "gpt-5-mini",
            quiet=True,
            resolution=_known_resolution(),
            force_predict=True,
        )
        assert len(calls) == 1
        assert prediction.aav_millions == 10.0  # from make_prediction(), not the known $30M

    def test_calls_the_llm_when_nothing_is_known(self, tmp_path, monkeypatch):
        calls = []

        def fake_predict_contract(user_prompt, model_id):
            calls.append(user_prompt)
            from agent.predict.tests.test_schema import make_prediction

            return make_prediction(), [], {"totalTokens": 1}, 0.01

        monkeypatch.setattr(predict_module, "predict_contract", fake_predict_contract)
        monkeypatch.setattr("agent.trace.TRACES_DIR", tmp_path)
        monkeypatch.setattr("agent.trace.HISTORY_CSV", tmp_path / "history.csv")

        run_prediction(
            _player_row(), 2026, "gpt-5-mini", quiet=True, resolution=_projected_resolution()
        )
        assert len(calls) == 1
