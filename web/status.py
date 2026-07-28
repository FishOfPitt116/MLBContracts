"""Per-session status tracking for the chat UI's "what's happening" indicator.

A HookProvider that watches which top-level tool the orchestrator agent is
calling -- it only has two: intake_tool (resolves player/year/mode) and
predict_tool (runs the actual prediction) -- and records a user-facing phase
string per session. The frontend polls GET /api/status while a POST
/api/chat call is in flight to show something better than a static
"Thinking..." the whole time.

Deliberately lives in web/, not agent/: this exists purely to narrate
progress to the UI, it isn't part of the orchestrator's own behavior.
"""

from strands.hooks import BeforeToolCallEvent, HookProvider, HookRegistry

DEFAULT_STATUS = "Thinking..."

_PHASE_LABELS = {
    "intake_tool": "Looking up player...",
    "predict_tool": "Making prediction...",
}

# session_id -> current status string
_status = {}


class StatusHook(HookProvider):
    """Records the friendly phase label for whichever tool call just started."""

    def __init__(self, session_id):
        self._session_id = session_id

    def register_hooks(self, registry: HookRegistry) -> None:
        registry.add_callback(BeforeToolCallEvent, self._on_before)

    def _on_before(self, event: BeforeToolCallEvent) -> None:
        label = _PHASE_LABELS.get(event.tool_use.get("name", ""))
        if label:
            _status[self._session_id] = label


def get_status(session_id):
    return _status.get(session_id, DEFAULT_STATUS)


def reset_status(session_id):
    _status[session_id] = DEFAULT_STATUS
