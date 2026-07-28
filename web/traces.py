"""Loads a single prediction trace for the sidebar's detail popup.

Returns a trimmed subset of the full trace JSON (agent/trace.py:write_trace)
-- reasoning, citations, predicted figures, phase resolution -- not the full
thing (system_prompt/messages/usage), which is large and meant for deep
debugging, not an at-a-glance popup.

run_id is client-supplied (it comes from the sidebar list), so it's validated
against the same charset new_run_id() actually produces before touching the
filesystem -- this is the one place user input reaches a file path, and
"pass whatever string you like" would let a request read arbitrary files.
"""

import json
import re

from agent.config import TRACES_DIR

_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_]+$")


def load_trace_summary(run_id):
    """Returns a trimmed trace dict, or None if run_id is invalid or not found."""
    if not _RUN_ID_RE.match(run_id):
        return None
    path = TRACES_DIR / f"{run_id}.json"
    if not path.exists():
        return None
    with open(path) as file:
        trace = json.load(file)
    return {
        "run_id": trace.get("run_id"),
        "run_date": trace.get("run_date"),
        "model_id": trace.get("model_id"),
        "prompt_version": trace.get("prompt_version"),
        "phase_resolution": trace.get("phase_resolution"),
        "structured_output": trace.get("structured_output"),
        "arithmetic_note": trace.get("arithmetic_note"),
    }
