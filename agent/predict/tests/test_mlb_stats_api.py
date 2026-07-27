"""Tests for agent/predict/mlb_stats_api.py (no real network calls).

requests.get is monkeypatched at the module boundary so retry/backoff logic
can be exercised deterministically; time.sleep is monkeypatched to a no-op so
these run instantly despite testing exponential backoff timing.
"""

import requests
import pytest

from agent.predict import mlb_stats_api as api


def _http_error(status_code):
    response = requests.Response()
    response.status_code = status_code
    error = requests.HTTPError(response=response)
    return error


class TestTransientClassification:
    def test_timeout_and_connection_error_are_transient(self):
        assert api._is_transient(requests.Timeout())
        assert api._is_transient(requests.ConnectionError())

    def test_5xx_and_429_are_transient(self):
        assert api._is_transient(_http_error(500))
        assert api._is_transient(_http_error(503))
        assert api._is_transient(_http_error(429))

    def test_4xx_other_than_429_is_not_transient(self):
        assert not api._is_transient(_http_error(404))
        assert not api._is_transient(_http_error(400))


class TestGetJsonRetry:
    def test_succeeds_without_retry_when_first_call_works(self, monkeypatch):
        calls = []

        class FakeResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {"ok": True}

        def fake_get(*args, **kwargs):
            calls.append(1)
            return FakeResponse()

        monkeypatch.setattr(api.requests, "get", fake_get)
        monkeypatch.setattr(api.time, "sleep", lambda _: None)

        result = api._get_json("http://x", params={})
        assert result == {"ok": True}
        assert len(calls) == 1

    def test_retries_transient_failures_then_succeeds(self, monkeypatch):
        calls = []
        sleeps = []

        class FakeResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {"ok": True}

        def fake_get(*args, **kwargs):
            calls.append(1)
            if len(calls) < 3:
                raise requests.Timeout("slow")
            return FakeResponse()

        monkeypatch.setattr(api.requests, "get", fake_get)
        monkeypatch.setattr(api.time, "sleep", lambda s: sleeps.append(s))

        result = api._get_json("http://x", params={})
        assert result == {"ok": True}
        assert len(calls) == 3
        # exponential backoff: 0.5, 1.0 (2 sleeps between 3 attempts)
        assert sleeps == [api.BACKOFF_BASE_SECONDS, api.BACKOFF_BASE_SECONDS * 2]

    def test_raises_after_exhausting_retries_on_persistent_transient_failure(self, monkeypatch):
        calls = []

        def fake_get(*args, **kwargs):
            calls.append(1)
            raise requests.ConnectionError("down")

        monkeypatch.setattr(api.requests, "get", fake_get)
        monkeypatch.setattr(api.time, "sleep", lambda _: None)

        with pytest.raises(RuntimeError, match="MLB Stats API request failed"):
            api._get_json("http://x", params={})
        assert len(calls) == api.MAX_HTTP_ATTEMPTS

    def test_does_not_retry_non_transient_error(self, monkeypatch):
        calls = []

        class FakeResponse:
            def raise_for_status(self):
                raise _http_error(404)

        def fake_get(*args, **kwargs):
            calls.append(1)
            return FakeResponse()

        monkeypatch.setattr(api.requests, "get", fake_get)
        monkeypatch.setattr(api.time, "sleep", lambda _: (_ for _ in ()).throw(AssertionError("should not sleep")))

        with pytest.raises(RuntimeError, match="MLB Stats API request failed"):
            api._get_json("http://x", params={})
        assert len(calls) == 1  # no retry wasted on a definitive client error


class TestCoerce:
    def test_numeric_strings_become_floats(self):
        assert api._coerce(".313") == pytest.approx(0.313)
        assert api._coerce("24.0") == pytest.approx(24.0)

    def test_non_numeric_strings_pass_through(self):
        assert api._coerce("New York Yankees") == "New York Yankees"

    def test_non_strings_pass_through(self):
        assert api._coerce(7) == 7
        assert api._coerce(None) is None


class TestFetchYearByYear:
    def _splits(self, years):
        return {
            "stats": [
                {
                    "splits": [
                        {
                            "season": str(year),
                            "team": {"name": f"Team{year}"},
                            "stat": {"gamesPlayed": year, "avg": ".300"},
                        }
                        for year in years
                    ]
                }
            ]
        }

    def test_before_year_excludes_target_and_later_seasons(self, monkeypatch):
        monkeypatch.setattr(api, "_get_json", lambda url, params: self._splits([2024, 2025, 2026]))
        rows = api._fetch_year_by_year(1, "hitting", api.BATTING_FIELDS, before_year=2026, min_year=None, max_year=None)
        years = [r["year"] for r in rows]
        assert years == [2024, 2025]

    def test_min_max_year_bounds(self, monkeypatch):
        monkeypatch.setattr(api, "_get_json", lambda url, params: self._splits([2020, 2022, 2024]))
        rows = api._fetch_year_by_year(1, "hitting", api.BATTING_FIELDS, before_year=None, min_year=2021, max_year=2023)
        assert [r["year"] for r in rows] == [2022]

    def test_only_curated_fields_extracted(self, monkeypatch):
        monkeypatch.setattr(api, "_get_json", lambda url, params: self._splits([2024]))
        rows = api._fetch_year_by_year(1, "hitting", ["avg"], before_year=None, min_year=None, max_year=None)
        assert rows[0]["avg"] == pytest.approx(0.3)
        assert "gamesPlayed" not in rows[0]

    def test_empty_stats_block_returns_empty_list(self, monkeypatch):
        monkeypatch.setattr(api, "_get_json", lambda url, params: {"stats": []})
        rows = api._fetch_year_by_year(1, "hitting", api.BATTING_FIELDS, before_year=None, min_year=None, max_year=None)
        assert rows == []


class TestQueryStats:
    def test_unmapped_players_reported_separately(self, monkeypatch):
        monkeypatch.setattr(api, "resolve_mlbam_id", lambda pid: None)
        result = api._query_stats(["Ghost_1"], "hitting", api.BATTING_FIELDS, None, None, None)
        assert result["stats"] == {}
        assert result["unmapped_player_ids"] == ["Ghost_1"]

    def test_mapped_players_get_fetched(self, monkeypatch):
        monkeypatch.setattr(api, "resolve_mlbam_id", lambda pid: 123)
        monkeypatch.setattr(api, "_fetch_year_by_year", lambda *a, **k: [{"year": 2024}])
        result = api._query_stats(["Skubal_26337"], "pitching", api.PITCHING_FIELDS, None, None, None)
        assert result["stats"] == {"Skubal_26337": [{"year": 2024}]}
        assert "unmapped_player_ids" not in result


class TestToolFactories:
    def test_before_year_not_exposed_to_the_model(self):
        for factory in (api.make_batting_stats_tool, api.make_pitching_stats_tool):
            tool_obj = factory(2026)
            exposed = set(tool_obj.tool_spec["inputSchema"]["json"]["properties"].keys())
            assert "before_year" not in exposed
            assert exposed == {"player_ids", "min_year", "max_year"}

    def test_batting_and_pitching_tools_have_distinct_names_and_real_descriptions(self):
        bt = api.make_batting_stats_tool(2026)
        pt = api.make_pitching_stats_tool(2026)
        assert bt.tool_name == "query_batting_stats"
        assert pt.tool_name == "query_pitching_stats"
        assert len(bt.tool_spec["description"]) > 50
        assert len(pt.tool_spec["description"]) > 50
        assert bt.tool_spec["description"] != pt.tool_spec["description"]

    def test_factory_produces_a_working_tool(self, monkeypatch):
        monkeypatch.setattr(api, "resolve_mlbam_id", lambda pid: 669373)
        monkeypatch.setattr(
            api, "_fetch_year_by_year", lambda *a, **k: [{"year": 2025, "era": 3.5}]
        )
        tool_obj = api.make_pitching_stats_tool(2026)
        result = tool_obj(player_ids=["Skubal_26337"])
        assert result["stats"]["Skubal_26337"] == [{"year": 2025, "era": 3.5}]
