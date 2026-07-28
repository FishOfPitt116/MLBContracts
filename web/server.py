"""Very basic local chat server over the orchestrator agent.

Deliberately lives outside agent/ -- this is a consumer of the agent package
(same relationship agent/ask.py's CLI has), not part of its business logic.

One-turn-per-HTTP-request adapter around agent/orchestrator/agent.py: the
orchestrator is the one persistent (non-fresh) agent in the system (Design
Decision #7 in docs/agent/DESIGN.md), so each browser session gets its own
long-lived Agent instance, kept in an in-memory dict keyed by a session_id
the frontend generates once and resends with every message.

Deliberately minimal for a first local pass:
- In-memory sessions only -- lost on server restart, no multi-worker support.
- No auth, no rate limiting, no HTTPS -- localhost use only.
- No per-session turn/clarification-streak caps (see MAX_TURNS /
  MAX_CONVERSATION_TURNS in orchestrator/agent.py's CLI loop) -- a long
  confused conversation just keeps calling the LLM. Fine for local testing;
  revisit before this is anything but a local dev tool.
- No incremental conversation-trace persistence -- agent/ask.py writes a
  trace at the end of a CLI conversation; this server doesn't write one at
  all yet (a session that's abandoned mid-conversation just disappears).

Run with: make web  (see Makefile), then open http://localhost:8000/
"""

import os
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent.config import DEFAULT_MODEL_ID
from agent.orchestrator.agent import get_turn
from web.history import load_history
from web.status import get_status, reset_status, StatusHook
from web.traces import load_trace_summary

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="MLB Contract Prediction Chat")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# session_id -> live orchestrator Agent instance (see module docstring)
_sessions = {}


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    message: str
    done: bool


def _get_or_create_agent(session_id):
    if session_id and session_id in _sessions:
        return session_id, _sessions[session_id]

    if not os.environ.get("OPENAI_API_KEY"):
        raise HTTPException(
            status_code=500,
            detail="OPENAI_API_KEY not set (checked .env at repo root).",
        )
    from agent.orchestrator.agent import create_orchestrator_agent

    new_id = session_id or str(uuid.uuid4())
    _sessions[new_id] = create_orchestrator_agent(
        DEFAULT_MODEL_ID, extra_hooks=[StatusHook(new_id)]
    )
    return new_id, _sessions[new_id]


@app.get("/")
def index():
    # No-store: this is actively edited during local dev and a stale cached
    # copy in the browser silently masks frontend changes (the dev server's
    # --reload only restarts the backend, it can't refresh an open tab).
    return FileResponse(STATIC_DIR / "index.html", headers={"Cache-Control": "no-store"})


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    session_id, agent = _get_or_create_agent(req.session_id)
    reset_status(session_id)
    turn = get_turn(agent, req.message)
    return ChatResponse(session_id=session_id, message=turn.message, done=turn.done)


@app.get("/api/status")
def status(session_id: str):
    return {"status": get_status(session_id)}


@app.get("/api/history")
def history():
    return {"predictions": load_history()}


@app.get("/api/trace/{run_id}")
def trace_detail(run_id: str):
    summary = load_trace_summary(run_id)
    if summary is None:
        raise HTTPException(status_code=404, detail="Trace not found")
    return summary


@app.post("/api/reset")
def reset(session_id: str):
    _sessions.pop(session_id, None)
    return {"ok": True}
