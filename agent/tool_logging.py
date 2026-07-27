"""Verbose tool-call logging: prints each tool's request and result to stdout.

Strands' default callback handler only prints "Tool #N: <name>" as a call
starts -- no arguments, no result. Shared across all three sub-agents
(orchestrator, intake, predict) via Agent(hooks=[ToolCallLogger()]), since
"one tool call" here includes structured output itself (Design Decision #7:
a structured_output_model is just one more callable tool internally), so this
also surfaces the final IntakeResult/ContractPrediction/OrchestratorTurn
payload, not just find_player/predict_tool/intake_tool.
"""

import json

from strands.hooks import AfterToolCallEvent, BeforeToolCallEvent, HookProvider, HookRegistry

# Tool inputs/outputs can be arbitrarily large (a full ContractPrediction with
# citations, a long reasoning string); truncate so one call can't flood the terminal.
_MAX_LOG_CHARS = 500


def _summarize(value):
    """Render a tool input/output compactly as a single truncated line."""
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    text = " ".join(text.split())  # collapse newlines/indentation to one line
    if len(text) > _MAX_LOG_CHARS:
        return text[:_MAX_LOG_CHARS] + "... (truncated)"
    return text


def _result_text(result):
    """Pull the human-readable text/json out of a Strands ToolResult's content blocks."""
    parts = []
    for block in result.get("content", []):
        if "text" in block:
            parts.append(block["text"])
        elif "json" in block:
            parts.append(json.dumps(block["json"], default=str))
    return _summarize(" ".join(parts)) if parts else _summarize(result)


class ToolCallLogger(HookProvider):
    """Prints each tool call's name, input, and result/error to stdout."""

    def register_hooks(self, registry: HookRegistry) -> None:
        registry.add_callback(BeforeToolCallEvent, self._on_before)
        registry.add_callback(AfterToolCallEvent, self._on_after)

    def _on_before(self, event: BeforeToolCallEvent) -> None:
        name = event.tool_use.get("name", "?")
        tool_input = event.tool_use.get("input", {})
        print(f"  → {name}({_summarize(tool_input)})")

    def _on_after(self, event: AfterToolCallEvent) -> None:
        name = event.tool_use.get("name", "?")
        if event.exception is not None:
            print(f"  ← {name} FAILED: {event.exception}")
            return
        status = event.result.get("status", "?")
        print(f"  ← {name} [{status}] {_result_text(event.result)}")
