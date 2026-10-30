from __future__ import annotations

import json

import pytest

from loopx.presentation.codex_activity import CodexActivitySteps


def _command(command: str) -> dict:
    return {"type": "commandExecution", "id": "exec-1", "command": command,
            "commandActions": [{"type": "unknown", "command": command}]}


@pytest.mark.parametrize(
    "token",
    [
        "np" + "m_" + "aB3dE5gH7jK9mN1pQ3sT5vX7zA9cE1gI3kM5",
        "py" + "pi-" + "AgEIcHlwaS5vcmcCJDY0Zjg5YjA1LTY3",
        "AI" + "za" + "SyA1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q",
        "AK" + "IA" + "Q3B7X9Z2N4C6V8M1",
        "AS" + "IA" + "Q7D2W5Y8K3N6R9T4",
        "xo" + "xs-" + "123456789012-abcdefghij",
        "xo" + "xr-" + "123456789012-abcdefghij",
        "ey" + "JhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0."
        "dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U",
    ],
)
def test_provider_tokens_in_commands_are_masked_before_replay(token: str) -> None:
    steps = CodexActivitySteps()
    text = f"deploy --config {token}"
    command = steps.started(_command(text))
    thought = steps.completed({"type": "reasoning", "id": "r", "summary": [text]})

    for step in (command, thought):
        assert token not in json.dumps(step)
        assert "***" in step["title"]
