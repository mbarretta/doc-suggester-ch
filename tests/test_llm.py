"""Tests for provider resolution and both providers' wire protocols.

The wire-shape assertions here matter more than usual: they are the only thing
standing between a protocol mistake and a runtime failure against a real API.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from doc_suggester_ch import llm
from doc_suggester_ch.llm import (
    ANTHROPIC,
    OPENAI,
    AnthropicProvider,
    OpenAIProvider,
    ProviderError,
    ToolSpec,
    available_providers,
    extract_json,
    resolve_provider,
)

TOOLS = [
    ToolSpec(
        name="search_docs",
        description="Search the docs",
        schema={"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
    ),
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Neutralize ambient credentials and overrides for every test."""
    for var in (
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
        "DOC_SUGGESTER_CH_PROVIDER",
        "DOC_SUGGESTER_CH_MODEL",
        "DOC_SUGGESTER_CH_BULK_MODEL",
        "DOC_SUGGESTER_CH_OPENAI_REASONING_EFFORT",
    ):
        monkeypatch.delenv(var, raising=False)


# ─── provider resolution ────────────────────────────────────────────────────

def test_available_providers_reflects_keys(monkeypatch):
    assert available_providers() == []
    monkeypatch.setenv("OPENAI_API_KEY", "sk-x")
    assert available_providers() == [OPENAI]
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-y")
    # Anthropic is preferred when both are present
    assert available_providers() == [ANTHROPIC, OPENAI]


def test_resolve_prefers_anthropic_when_both_set(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-a")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-o")
    with patch.object(llm, "AnthropicProvider") as anthropic_cls:
        anthropic_cls.return_value = SimpleNamespace(name=ANTHROPIC)
        assert resolve_provider().name == ANTHROPIC


def test_resolve_falls_back_to_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-o")
    with patch.object(llm, "OpenAIProvider") as openai_cls:
        openai_cls.return_value = SimpleNamespace(name=OPENAI)
        assert resolve_provider().name == OPENAI


def test_resolve_honors_explicit_preference(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-a")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-o")
    with patch.object(llm, "OpenAIProvider") as openai_cls:
        openai_cls.return_value = SimpleNamespace(name=OPENAI)
        assert resolve_provider("openai").name == OPENAI


def test_resolve_honors_env_preference(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-a")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-o")
    monkeypatch.setenv("DOC_SUGGESTER_CH_PROVIDER", "openai")
    with patch.object(llm, "OpenAIProvider") as openai_cls:
        openai_cls.return_value = SimpleNamespace(name=OPENAI)
        assert resolve_provider().name == OPENAI


def test_resolve_errors_with_no_keys():
    with pytest.raises(ProviderError) as exc:
        resolve_provider()
    message = str(exc.value)
    assert "ANTHROPIC_API_KEY" in message
    assert "OPENAI_API_KEY" in message


def test_resolve_errors_when_requested_provider_has_no_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-a")
    with pytest.raises(ProviderError, match="OPENAI_API_KEY is not set"):
        resolve_provider("openai")


def test_resolve_errors_on_unknown_provider(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-a")
    with pytest.raises(ProviderError, match="Unknown provider"):
        resolve_provider("cohere")


def test_blank_key_does_not_count(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "   ")
    assert available_providers() == []


def test_model_overrides(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-o")
    monkeypatch.setenv("DOC_SUGGESTER_CH_MODEL", "my-main")
    monkeypatch.setenv("DOC_SUGGESTER_CH_BULK_MODEL", "my-bulk")
    with patch("openai.AsyncOpenAI"):
        provider = resolve_provider("openai")
    assert provider.main_model == "my-main"
    assert provider.bulk_model == "my-bulk"


def test_default_models_differ_per_provider(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-a")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-o")
    with patch("anthropic.AsyncAnthropic"), patch("openai.AsyncOpenAI"):
        a = resolve_provider("anthropic")
        o = resolve_provider("openai")
    assert a.main_model.startswith("claude-")
    assert a.bulk_model.startswith("claude-")
    assert o.main_model.startswith("gpt-")
    assert o.bulk_model.startswith("gpt-")
    # Bulk should be a cheaper model than main, not the same one
    assert a.main_model != a.bulk_model
    assert o.main_model != o.bulk_model


# ─── extract_json ───────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "text,expected",
    [
        ('{"a": 1}', {"a": 1}),
        ('prose before {"a": 1} prose after', {"a": 1}),
        ('```json\n{"a": 1}\n```', {"a": 1}),
        ('```\n{"a": 1}\n```', {"a": 1}),
        ('Here you go:\n```json\n{"a": 1}\n```\nHope that helps!', {"a": 1}),
    ],
)
def test_extract_json_variants(text, expected):
    assert extract_json(text) == expected


def test_extract_json_raises_without_object():
    with pytest.raises(ValueError, match="no JSON object"):
        extract_json("no json here")


def test_extract_json_raises_on_malformed():
    with pytest.raises(json.JSONDecodeError):
        extract_json('{"a": }')


# ─── Anthropic wire protocol ────────────────────────────────────────────────

def _a_block(type_, **kwargs):
    block = MagicMock()
    block.type = type_
    for key, value in kwargs.items():
        setattr(block, key, value)
    return block


def _anthropic_provider(responses):
    client = AsyncMock()
    client.messages.create = AsyncMock(side_effect=responses)
    with patch("anthropic.AsyncAnthropic", return_value=client):
        provider = AnthropicProvider("claude-main", "claude-bulk")
    return provider, client


async def test_anthropic_complete_uses_bulk_model():
    response = MagicMock(content=[_a_block("text", text="synopsis text")])
    provider, client = _anthropic_provider([response])

    assert await provider.complete("prompt", max_tokens=200) == "synopsis text"
    kwargs = client.messages.create.await_args.kwargs
    assert kwargs["model"] == "claude-bulk"
    assert kwargs["max_tokens"] == 200
    assert kwargs["messages"] == [{"role": "user", "content": "prompt"}]


async def test_anthropic_tool_loop_wire_shape():
    final = MagicMock(stop_reason="end_turn", content=[_a_block("text", text="done")])
    provider, client = _anthropic_provider([final])

    result = await provider.run_tool_loop("SYS", "USER", TOOLS, AsyncMock())

    assert result == "done"
    kwargs = client.messages.create.await_args.kwargs
    assert kwargs["model"] == "claude-main"
    assert kwargs["system"] == "SYS"
    assert kwargs["thinking"] == {"type": "adaptive"}
    # Anthropic takes input_schema, not parameters
    assert kwargs["tools"] == [{
        "name": "search_docs",
        "description": "Search the docs",
        "input_schema": TOOLS[0].schema,
    }]


async def test_anthropic_tool_loop_returns_results_in_one_user_message():
    tool_turn = MagicMock(
        stop_reason="tool_use",
        content=[
            _a_block("thinking", thinking="reasoning"),
            _a_block("tool_use", id="t1", name="search_docs", input={"query": "a"}),
            _a_block("tool_use", id="t2", name="search_docs", input={"query": "b"}),
        ],
    )
    final = MagicMock(stop_reason="end_turn", content=[_a_block("text", text="done")])
    provider, client = _anthropic_provider([tool_turn, final])

    dispatch = AsyncMock(side_effect=["result a", "result b"])
    await provider.run_tool_loop("SYS", "USER", TOOLS, dispatch)

    messages = client.messages.create.await_args.kwargs["messages"]
    # Both results ride in a single user message, in call order
    assert messages[2]["role"] == "user"
    assert [b["tool_use_id"] for b in messages[2]["content"]] == ["t1", "t2"]
    assert [b["content"] for b in messages[2]["content"]] == ["result a", "result b"]


async def test_anthropic_echoes_thinking_blocks_unchanged():
    thinking = _a_block("thinking", thinking="reasoning")
    tool_turn = MagicMock(
        stop_reason="tool_use",
        content=[thinking, _a_block("tool_use", id="t1", name="search_docs", input={})],
    )
    final = MagicMock(stop_reason="end_turn", content=[_a_block("text", text="done")])
    provider, client = _anthropic_provider([tool_turn, final])

    await provider.run_tool_loop("SYS", "USER", TOOLS, AsyncMock(return_value="r"))

    messages = client.messages.create.await_args.kwargs["messages"]
    # The whole content list is echoed back, thinking block included
    assert messages[1]["content"] is tool_turn.content
    assert thinking in messages[1]["content"]


async def test_anthropic_reports_dispatch_failure_as_tool_error():
    tool_turn = MagicMock(
        stop_reason="tool_use",
        content=[_a_block("tool_use", id="t1", name="search_docs", input={})],
    )
    final = MagicMock(stop_reason="end_turn", content=[_a_block("text", text="recovered")])
    provider, client = _anthropic_provider([tool_turn, final])

    dispatch = AsyncMock(side_effect=RuntimeError("boom"))
    assert await provider.run_tool_loop("SYS", "USER", TOOLS, dispatch) == "recovered"

    block = client.messages.create.await_args.kwargs["messages"][2]["content"][0]
    assert block["is_error"] is True
    assert "boom" in block["content"]


async def test_anthropic_handles_refusal():
    refusal = MagicMock(
        stop_reason="refusal",
        content=[],
        stop_details=MagicMock(explanation="policy decline"),
    )
    provider, _ = _anthropic_provider([refusal])

    result = await provider.run_tool_loop("SYS", "USER", TOOLS, AsyncMock())
    assert "declined" in result
    assert "policy decline" in result


async def test_anthropic_calls_on_tool_callback():
    tool_turn = MagicMock(
        stop_reason="tool_use",
        content=[_a_block("tool_use", id="t1", name="search_docs", input={"query": "q"})],
    )
    final = MagicMock(stop_reason="end_turn", content=[_a_block("text", text="done")])
    provider, _ = _anthropic_provider([tool_turn, final])

    seen = []
    await provider.run_tool_loop(
        "SYS", "USER", TOOLS, AsyncMock(return_value="r"),
        on_tool=lambda n, a: seen.append((n, a)),
    )
    assert seen == [("search_docs", {"query": "q"})]


# ─── OpenAI wire protocol ───────────────────────────────────────────────────

def _o_call(call_id, name, arguments):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _o_response(content=None, tool_calls=None, finish_reason="stop"):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish_reason)])


def _openai_provider(responses):
    client = AsyncMock()
    client.chat.completions.create = AsyncMock(side_effect=responses)
    with patch("openai.AsyncOpenAI", return_value=client):
        provider = OpenAIProvider("gpt-main", "gpt-bulk")
    return provider, client


async def test_openai_complete_uses_bulk_model_and_completion_tokens():
    provider, client = _openai_provider([_o_response(content="synopsis text")])

    assert await provider.complete("prompt", max_tokens=200) == "synopsis text"
    kwargs = client.chat.completions.create.await_args.kwargs
    assert kwargs["model"] == "gpt-bulk"
    # Current models reject max_tokens; max_completion_tokens is required
    assert kwargs["max_completion_tokens"] == 200
    assert "max_tokens" not in kwargs
    # Reasoning models reject temperature
    assert "temperature" not in kwargs


async def test_openai_complete_handles_null_content():
    provider, _ = _openai_provider([_o_response(content=None)])
    assert await provider.complete("prompt") == ""


async def test_openai_tool_loop_wire_shape():
    provider, client = _openai_provider([_o_response(content="done")])

    result = await provider.run_tool_loop("SYS", "USER", TOOLS, AsyncMock())

    assert result == "done"
    kwargs = client.chat.completions.create.await_args.kwargs
    assert kwargs["model"] == "gpt-main"
    assert "max_tokens" not in kwargs
    assert "temperature" not in kwargs
    # OpenAI nests the schema under function.parameters
    assert kwargs["tools"] == [{
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": "Search the docs",
            "parameters": TOOLS[0].schema,
        },
    }]
    # System prompt is a message, not a top-level field
    assert kwargs["messages"][0] == {"role": "system", "content": "SYS"}
    assert kwargs["messages"][1] == {"role": "user", "content": "USER"}


async def test_openai_parses_json_string_arguments():
    tool_turn = _o_response(
        tool_calls=[_o_call("c1", "search_docs", '{"query": "primary key"}')],
        finish_reason="tool_calls",
    )
    provider, _ = _openai_provider([tool_turn, _o_response(content="done")])

    dispatch = AsyncMock(return_value="results")
    await provider.run_tool_loop("SYS", "USER", TOOLS, dispatch)

    # Arguments arrive as a JSON string and must reach dispatch as a dict
    dispatch.assert_awaited_once_with("search_docs", {"query": "primary key"})


async def test_openai_echoes_assistant_tool_calls_then_tool_messages():
    tool_turn = _o_response(
        tool_calls=[
            _o_call("c1", "search_docs", '{"query": "a"}'),
            _o_call("c2", "search_docs", '{"query": "b"}'),
        ],
        finish_reason="tool_calls",
    )
    provider, client = _openai_provider([tool_turn, _o_response(content="done")])

    dispatch = AsyncMock(side_effect=["result a", "result b"])
    await provider.run_tool_loop("SYS", "USER", TOOLS, dispatch)

    messages = client.chat.completions.create.await_args.kwargs["messages"]
    assistant = messages[2]
    assert assistant["role"] == "assistant"
    assert [c["id"] for c in assistant["tool_calls"]] == ["c1", "c2"]
    assert assistant["tool_calls"][0]["type"] == "function"
    assert assistant["tool_calls"][0]["function"]["name"] == "search_docs"
    # Unlike Anthropic, each result is its own message keyed by tool_call_id
    assert messages[3] == {"role": "tool", "tool_call_id": "c1", "content": "result a"}
    assert messages[4] == {"role": "tool", "tool_call_id": "c2", "content": "result b"}


async def test_openai_assistant_echo_omits_tool_calls_when_absent():
    provider, client = _openai_provider([_o_response(content="done")])
    await provider.run_tool_loop("SYS", "USER", TOOLS, AsyncMock())
    # Loop ended without tools; nothing to echo, and no empty tool_calls key
    assert client.chat.completions.create.await_count == 1


async def test_openai_handles_malformed_tool_arguments():
    tool_turn = _o_response(
        tool_calls=[_o_call("c1", "search_docs", "{not json")],
        finish_reason="tool_calls",
    )
    provider, client = _openai_provider([tool_turn, _o_response(content="recovered")])

    dispatch = AsyncMock()
    result = await provider.run_tool_loop("SYS", "USER", TOOLS, dispatch)

    assert result == "recovered"
    dispatch.assert_not_awaited()
    tool_message = client.chat.completions.create.await_args.kwargs["messages"][3]
    assert tool_message["tool_call_id"] == "c1"
    assert "could not parse arguments" in tool_message["content"]


async def test_openai_handles_non_object_tool_arguments():
    tool_turn = _o_response(
        tool_calls=[_o_call("c1", "search_docs", '"just a string"')],
        finish_reason="tool_calls",
    )
    provider, client = _openai_provider([tool_turn, _o_response(content="recovered")])

    await provider.run_tool_loop("SYS", "USER", TOOLS, AsyncMock())
    tool_message = client.chat.completions.create.await_args.kwargs["messages"][3]
    assert "could not parse arguments" in tool_message["content"]


async def test_openai_defaults_empty_arguments_to_empty_dict():
    tool_turn = _o_response(
        tool_calls=[_o_call("c1", "search_docs", "")],
        finish_reason="tool_calls",
    )
    provider, _ = _openai_provider([tool_turn, _o_response(content="done")])

    dispatch = AsyncMock(return_value="r")
    await provider.run_tool_loop("SYS", "USER", TOOLS, dispatch)
    dispatch.assert_awaited_once_with("search_docs", {})


async def test_openai_reports_dispatch_failure_to_model():
    tool_turn = _o_response(
        tool_calls=[_o_call("c1", "search_docs", "{}")],
        finish_reason="tool_calls",
    )
    provider, client = _openai_provider([tool_turn, _o_response(content="recovered")])

    dispatch = AsyncMock(side_effect=RuntimeError("boom"))
    assert await provider.run_tool_loop("SYS", "USER", TOOLS, dispatch) == "recovered"

    tool_message = client.chat.completions.create.await_args.kwargs["messages"][3]
    assert "Tool failed: boom" in tool_message["content"]


async def test_openai_handles_content_filter():
    provider, _ = _openai_provider([_o_response(content=None, finish_reason="content_filter")])
    result = await provider.run_tool_loop("SYS", "USER", TOOLS, AsyncMock())
    assert "declined" in result


async def test_openai_reasoning_effort_omitted_by_default():
    provider, client = _openai_provider([_o_response(content="done")])
    await provider.run_tool_loop("SYS", "USER", TOOLS, AsyncMock())
    # Sending reasoning_effort to a model that rejects it is a 400
    assert "reasoning_effort" not in client.chat.completions.create.await_args.kwargs


async def test_openai_reasoning_effort_sent_when_configured(monkeypatch):
    monkeypatch.setenv("DOC_SUGGESTER_CH_OPENAI_REASONING_EFFORT", "high")
    provider, client = _openai_provider([_o_response(content="done")])
    await provider.run_tool_loop("SYS", "USER", TOOLS, AsyncMock())
    assert client.chat.completions.create.await_args.kwargs["reasoning_effort"] == "high"


async def test_openai_calls_on_tool_callback():
    tool_turn = _o_response(
        tool_calls=[_o_call("c1", "search_docs", '{"query": "q"}')],
        finish_reason="tool_calls",
    )
    provider, _ = _openai_provider([tool_turn, _o_response(content="done")])

    seen = []
    await provider.run_tool_loop(
        "SYS", "USER", TOOLS, AsyncMock(return_value="r"),
        on_tool=lambda n, a: seen.append((n, a)),
    )
    assert seen == [("search_docs", {"query": "q"})]


async def test_openai_stops_at_max_turns():
    """A model that keeps calling tools must not loop forever."""
    endless = [
        _o_response(
            tool_calls=[_o_call(f"c{i}", "search_docs", "{}")],
            finish_reason="tool_calls",
        )
        for i in range(10)
    ]
    provider, client = _openai_provider(endless)

    await provider.run_tool_loop(
        "SYS", "USER", TOOLS, AsyncMock(return_value="r"), max_turns=3
    )
    assert client.chat.completions.create.await_count == 3


async def test_anthropic_stops_at_max_turns():
    endless = [
        MagicMock(
            stop_reason="tool_use",
            content=[_a_block("tool_use", id=f"t{i}", name="search_docs", input={})],
        )
        for i in range(10)
    ]
    provider, client = _anthropic_provider(endless)

    await provider.run_tool_loop(
        "SYS", "USER", TOOLS, AsyncMock(return_value="r"), max_turns=3
    )
    assert client.messages.create.await_count == 3


async def test_both_providers_satisfy_the_protocol():
    """Structural check that both classes expose the interface callers use."""
    with patch("anthropic.AsyncAnthropic"), patch("openai.AsyncOpenAI"):
        for provider in (AnthropicProvider("m", "b"), OpenAIProvider("m", "b")):
            assert isinstance(provider.name, str)
            assert provider.main_model == "m"
            assert provider.bulk_model == "b"
            assert callable(provider.complete)
            assert callable(provider.run_tool_loop)
