"""Natural-language front door for the contract prediction agent.

Usage:
    python -m agent.ask "give me the projected contract for Max Scherzer in 2026"
    python -m agent.ask    # prompts interactively if no request is given
"""

import sys
from argparse import ArgumentParser

from agent.config import DEFAULT_MODEL_ID
from agent.orchestrator.agent import run_conversation
from agent.trace import new_run_id, write_conversation_trace


def _extract_prediction_trace_path(messages):
    """Best-effort pull of predict_tool's trace_path out of the raw message log."""
    for message in reversed(messages):
        for block in message.get("content", []):
            tool_result = block.get("toolResult")
            if not tool_result:
                continue
            for item in tool_result.get("content", []):
                data = item.get("json")
                if isinstance(data, dict) and "trace_path" in data:
                    return data["trace_path"]
    return None


def ask(initial_request, model_id=None):
    """Run one full conversation and persist its transcript. Returns the turns."""
    model_id = model_id or DEFAULT_MODEL_ID
    from agent.orchestrator.agent import create_orchestrator_agent

    orchestrator = create_orchestrator_agent(model_id)
    turns = run_conversation(initial_request, model_id=model_id, agent=orchestrator)

    run_id = new_run_id("conversation")
    trace_path = write_conversation_trace(
        run_id=run_id,
        model_id=model_id,
        initial_request=initial_request,
        turns=turns,
        prediction_trace_path=_extract_prediction_trace_path(orchestrator.messages),
    )
    print(f"\nConversation trace: {trace_path}")
    return turns


def main():
    parser = ArgumentParser(description="Ask the contract prediction agent in plain English.")
    parser.add_argument("request", nargs="?", help="Your request, e.g. "
                         '"what will Max Scherzer make in 2026?"')
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, help="OpenAI model id")
    args = parser.parse_args()

    request = args.request or input("What would you like to know? ")
    if not request.strip():
        sys.exit("A request is required.")

    ask(request, model_id=args.model)


if __name__ == "__main__":
    main()
