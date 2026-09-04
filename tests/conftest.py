"""Shared test fixtures — chiefly a fake LLM provider.

Injecting a provider is how the tests avoid the network: every module that
talks to a model takes a `provider` argument, so nothing needs to patch the
Anthropic or OpenAI SDKs.
"""

from __future__ import annotations

from typing import Any

import pytest

from doc_suggester_ch.llm import ToolSpec


class FakeProvider:
    """An LLMProvider stand-in that records calls and replays scripted replies.

    `completions` is a list of replies for complete(); the last one repeats once
    exhausted. `tool_script` is a list of turns for run_tool_loop(): each entry
    is either a string (final answer) or a list of (tool_name, args) to call.
    """

    def __init__(
        self,
        completions: list[str] | None = None,
        tool_script: list[Any] | None = None,
        name: str = "fake",
        complete_error: Exception | None = None,
    ) -> None:
        self.name = name
        self.main_model = "fake-main"
        self.bulk_model = "fake-bulk"
        self._completions = list(completions or ["fake completion"])
        self._tool_script = list(tool_script or ["final answer"])
        self._complete_error = complete_error
        self.complete_calls: list[tuple[str, int]] = []
        self.tool_loop_calls: list[dict[str, Any]] = []
        self.dispatched: list[tuple[str, dict[str, Any]]] = []

    async def complete(self, prompt: str, max_tokens: int = 1024) -> str:
        self.complete_calls.append((prompt, max_tokens))
        if self._complete_error is not None:
            raise self._complete_error
        if len(self._completions) > 1:
            return self._completions.pop(0)
        return self._completions[0]

    async def run_tool_loop(
        self,
        system: str,
        user_content: str,
        tools: list[ToolSpec],
        dispatch: Any,
        on_tool: Any = None,
        max_turns: int = 20,
    ) -> str:
        self.tool_loop_calls.append({
            "system": system,
            "user_content": user_content,
            "tools": tools,
            "max_turns": max_turns,
        })
        for turn in self._tool_script:
            if isinstance(turn, str):
                return turn
            for tool_name, args in turn:
                if on_tool:
                    on_tool(tool_name, args)
                result = await dispatch(tool_name, args)
                self.dispatched.append((tool_name, args))
                self.tool_loop_calls[-1].setdefault("results", []).append(result)
        return "final answer"


@pytest.fixture
def fake_provider() -> FakeProvider:
    return FakeProvider()
