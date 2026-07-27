"""run_conversation's loop logic, driven by a fake agent (no real LLM calls)."""

from agent.orchestrator.agent import MAX_TURNS, run_conversation
from agent.orchestrator.schema import OrchestratorTurn


class FakeResult:
    def __init__(self, turn):
        self.structured_output = turn


class FakeAgent:
    """Returns a scripted sequence of OrchestratorTurns, one per call."""

    def __init__(self, scripted_turns):
        self.scripted_turns = list(scripted_turns)
        self.calls = []

    def __call__(self, text, structured_output_model):
        assert structured_output_model is OrchestratorTurn
        self.calls.append(text)
        return FakeResult(self.scripted_turns.pop(0))


def test_resolves_immediately_without_clarification():
    agent = FakeAgent([OrchestratorTurn(message="Scherzer will make $5M in 2026.", done=True)])
    turns = run_conversation("what will Scherzer make in 2026?", agent=agent, ask_fn=lambda _: "")
    assert len(turns) == 1
    assert turns[0]["done"] is True
    assert len(agent.calls) == 1


def test_asks_clarifying_question_then_resolves():
    agent = FakeAgent(
        [
            OrchestratorTurn(message="Which year did you mean?", done=False),
            OrchestratorTurn(message="Scherzer will make $5M in 2026.", done=True),
        ]
    )
    # "2026" answers the clarifying question; "" declines a follow-up and ends it.
    replies = iter(["2026", ""])
    turns = run_conversation(
        "what will Scherzer make?", agent=agent, ask_fn=lambda _: next(replies)
    )
    assert len(turns) == 2
    assert turns[0]["done"] is False
    assert turns[1]["done"] is True
    # the clarifying answer was fed back in as the next call's input
    assert agent.calls == ["what will Scherzer make?", "2026"]


def test_follow_up_question_continues_on_same_agent():
    agent = FakeAgent(
        [
            OrchestratorTurn(message="Scherzer will make $5M in 2026.", done=True),
            OrchestratorTurn(message="Because of his age and injury history.", done=True),
        ]
    )
    replies = iter(["why so low?", ""])
    turns = run_conversation(
        "what will Scherzer make in 2026?", agent=agent, ask_fn=lambda _: next(replies)
    )
    assert len(turns) == 2
    assert turns[0]["done"] is True
    assert turns[1]["done"] is True
    # the follow-up was fed back in as the next call on the same persistent agent
    assert agent.calls == ["what will Scherzer make in 2026?", "why so low?"]


def test_exit_word_other_than_blank_also_ends_conversation():
    agent = FakeAgent([OrchestratorTurn(message="Scherzer will make $5M in 2026.", done=True)])
    turns = run_conversation(
        "what will Scherzer make in 2026?", agent=agent, ask_fn=lambda _: "no thanks"
    )
    assert len(turns) == 1
    assert len(agent.calls) == 1


def test_gives_up_after_max_turns():
    never_done = [OrchestratorTurn(message="Still need more info.", done=False)] * MAX_TURNS
    agent = FakeAgent(never_done)
    turns = run_conversation("vague request", agent=agent, ask_fn=lambda _: "still vague")
    assert len(turns) == MAX_TURNS + 1
    assert turns[-1]["done"] is True
    assert "couldn't resolve" in turns[-1]["message"]
