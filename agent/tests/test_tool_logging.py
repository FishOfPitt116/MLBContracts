"""Tests for ToolCallLogger (no LLM calls) -- constructs Strands hook events directly.

Strands' default callback handler only prints "Tool #N: <name>" when a call
starts; this hook adds the actual input and result/error for each tool call,
so `make ask` shows what each sub-agent (orchestrator/intake/predict) is
actually doing, not just that something was called.
"""

from strands.hooks import AfterToolCallEvent, BeforeToolCallEvent

from agent.tool_logging import ToolCallLogger, _result_text, _summarize


def _tool_use(name="find_player", input_=None):
    return {"name": name, "input": input_ or {"last_name": "Scherzer"}, "toolUseId": "call_1"}


def test_summarize_renders_dict_compactly():
    assert _summarize({"last_name": "Scherzer"}) == '{"last_name": "Scherzer"}'


def test_summarize_truncates_long_values():
    long_text = "x" * 1000
    summary = _summarize(long_text)
    assert len(summary) < 1000
    assert summary.endswith("... (truncated)")


def test_result_text_prefers_text_blocks():
    result = {"status": "success", "toolUseId": "call_1", "content": [{"text": "found 1 match"}]}
    assert _result_text(result) == "found 1 match"


def test_result_text_falls_back_to_json_blocks():
    result = {
        "status": "success",
        "toolUseId": "call_1",
        "content": [{"json": {"matches": ["Scherzer_5166"]}}],
    }
    assert "Scherzer_5166" in _result_text(result)


def test_before_hook_prints_name_and_input(capsys):
    logger = ToolCallLogger()
    event = BeforeToolCallEvent(
        agent=None,
        selected_tool=None,
        tool_use=_tool_use(),
        invocation_state={},
    )
    logger._on_before(event)
    out = capsys.readouterr().out
    assert "find_player" in out
    assert "Scherzer" in out


def test_after_hook_prints_status_and_result(capsys):
    logger = ToolCallLogger()
    event = AfterToolCallEvent(
        agent=None,
        selected_tool=None,
        tool_use=_tool_use(),
        invocation_state={},
        result={"status": "success", "toolUseId": "call_1", "content": [{"text": "1 match found"}]},
    )
    logger._on_after(event)
    out = capsys.readouterr().out
    assert "find_player" in out
    assert "success" in out
    assert "1 match found" in out


def test_after_hook_prints_failure_on_exception(capsys):
    logger = ToolCallLogger()
    event = AfterToolCallEvent(
        agent=None,
        selected_tool=None,
        tool_use=_tool_use(),
        invocation_state={},
        result={"status": "error", "toolUseId": "call_1", "content": []},
        exception=ValueError("boom"),
    )
    logger._on_after(event)
    out = capsys.readouterr().out
    assert "FAILED" in out
    assert "boom" in out
