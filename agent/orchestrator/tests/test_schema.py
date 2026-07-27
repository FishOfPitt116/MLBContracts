"""OrchestratorTurn schema validation (no LLM calls)."""

import pytest
from pydantic import ValidationError

from agent.orchestrator.schema import OrchestratorTurn


def test_valid_turn():
    turn = OrchestratorTurn(message="What year would you like this for?", done=False)
    assert turn.done is False


def test_missing_message_rejected():
    with pytest.raises(ValidationError):
        OrchestratorTurn(done=True)


def test_missing_done_rejected():
    with pytest.raises(ValidationError):
        OrchestratorTurn(message="hello")
