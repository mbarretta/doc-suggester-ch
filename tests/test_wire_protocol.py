"""End-to-end wire-protocol tests against local servers.

The mocked tests in test_llm.py assert what we *pass to* each SDK. These run
the real SDKs against a local HTTP server that speaks each provider's protocol,
so they also cover request serialization and response parsing — everything
except the model itself.

That matters because a protocol mistake here fails at runtime against a real
API, and neither provider's credentials are guaranteed to be available in CI.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from doc_suggester_ch.llm import AnthropicProvider, OpenAIProvider, ToolSpec

TOOLS = [
    ToolSpec(
        name="search_docs",
        description="Search the ClickHouse docs",
        schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    )
]

TOOL_RESULT = "Title: ClickStack overview\nLink: https://clickhouse.com/docs/clickstack/overview"
FINAL_TEXT = "### 1. [Docs] ClickStack overview"


class _ProtocolServer:
    """Serves a scripted two-turn conversation and records what it received."""

    def __init__(self, responder):
        self.received: list[dict] = []
        self.turn = 0
        responder_ref = responder
        server_ref = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # keep pytest output clean
                pass

            def do_POST(self):
                length = int(self.headers.get("content-length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                server_ref.received.append(body)
                server_ref.turn += 1
                payload = responder_ref(server_ref.turn, body)
                out = json.dumps(payload).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(out)))
                self.end_headers()
                self.wfile.write(out)

        self._httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.port = self._httpd.server_address[1]
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def close(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5)


def _openai_responder(turn: int, body: dict) -> dict:
    if turn == 1:
        message = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_abc",
                "type": "function",
                "function": {
                    "name": "search_docs",
                    "arguments": json.dumps({"query": "ClickStack"}),
                },
            }],
        }
        finish = "tool_calls"
    else:
        message = {"role": "assistant", "content": FINAL_TEXT}
        finish = "stop"
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": body.get("model", "test"),
        "choices": [{"index": 0, "message": message, "finish_reason": finish}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _anthropic_responder(turn: int, body: dict) -> dict:
    if turn == 1:
        content = [
            {"type": "text", "text": "Let me search."},
            {
                "type": "tool_use",
                "id": "toolu_abc",
                "name": "search_docs",
                "input": {"query": "ClickStack"},
            },
        ]
        stop_reason = "tool_use"
    else:
        content = [{"type": "text", "text": FINAL_TEXT}]
        stop_reason = "end_turn"
    return {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": body.get("model", "test"),
        "content": content,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


@pytest.fixture
def openai_server(monkeypatch):
    server = _ProtocolServer(_openai_responder)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-local")
    monkeypatch.setenv("OPENAI_BASE_URL", f"{server.base_url}/v1")
    monkeypatch.delenv("DOC_SUGGESTER_CH_OPENAI_REASONING_EFFORT", raising=False)
    yield server
    server.close()


@pytest.fixture
def anthropic_server(monkeypatch):
    server = _ProtocolServer(_anthropic_responder)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-local")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", server.base_url)
    yield server
    server.close()


async def _run(provider) -> tuple[str, list[tuple[str, dict]]]:
    seen: list[tuple[str, dict]] = []

    async def dispatch(name: str, args: dict) -> str:
        seen.append((name, args))
        return TOOL_RESULT

    result = await provider.run_tool_loop("SYSTEM PROMPT", "USER NOTES", TOOLS, dispatch)
    return result, seen


# ─── OpenAI ─────────────────────────────────────────────────────────────────

async def test_openai_full_loop_through_real_sdk(openai_server):
    result, seen = await _run(OpenAIProvider("gpt-test", "gpt-test-bulk"))

    assert result == FINAL_TEXT
    assert seen == [("search_docs", {"query": "ClickStack"})]
    assert len(openai_server.received) == 2


async def test_openai_first_request_serializes_correctly(openai_server):
    await _run(OpenAIProvider("gpt-test", "gpt-test-bulk"))
    request = openai_server.received[0]

    assert request["model"] == "gpt-test"
    assert request["max_completion_tokens"] > 0
    assert "max_tokens" not in request
    assert "temperature" not in request
    assert "reasoning_effort" not in request
    assert request["tools"][0]["type"] == "function"
    assert request["tools"][0]["function"]["name"] == "search_docs"
    assert request["tools"][0]["function"]["parameters"] == TOOLS[0].schema
    assert [m["role"] for m in request["messages"]] == ["system", "user"]
    assert request["messages"][0]["content"] == "SYSTEM PROMPT"


async def test_openai_second_request_carries_echo_and_tool_result(openai_server):
    await _run(OpenAIProvider("gpt-test", "gpt-test-bulk"))
    messages = openai_server.received[1]["messages"]

    assert [m["role"] for m in messages] == ["system", "user", "assistant", "tool"]
    assert [c["id"] for c in messages[2]["tool_calls"]] == ["call_abc"]
    assert messages[3]["tool_call_id"] == "call_abc"
    assert messages[3]["content"] == TOOL_RESULT


async def test_openai_complete_through_real_sdk(openai_server):
    provider = OpenAIProvider("gpt-test", "gpt-test-bulk")
    # turn 1 returns a tool call, so advance past it for a plain text reply
    openai_server.turn = 1
    text = await provider.complete("summarize this", max_tokens=64)

    assert text == FINAL_TEXT
    request = openai_server.received[-1]
    assert request["model"] == "gpt-test-bulk"
    assert request["max_completion_tokens"] == 64
    assert "tools" not in request


async def test_openai_sends_reasoning_effort_when_configured(openai_server, monkeypatch):
    monkeypatch.setenv("DOC_SUGGESTER_CH_OPENAI_REASONING_EFFORT", "low")
    await _run(OpenAIProvider("gpt-test", "gpt-test-bulk"))
    assert openai_server.received[0]["reasoning_effort"] == "low"


# ─── Anthropic ──────────────────────────────────────────────────────────────

async def test_anthropic_full_loop_through_real_sdk(anthropic_server):
    result, seen = await _run(AnthropicProvider("claude-test", "claude-test-bulk"))

    assert result == FINAL_TEXT
    assert seen == [("search_docs", {"query": "ClickStack"})]
    assert len(anthropic_server.received) == 2


async def test_anthropic_first_request_serializes_correctly(anthropic_server):
    await _run(AnthropicProvider("claude-test", "claude-test-bulk"))
    request = anthropic_server.received[0]

    assert request["model"] == "claude-test"
    assert request["max_tokens"] > 0
    assert request["system"] == "SYSTEM PROMPT"
    assert request["thinking"] == {"type": "adaptive"}
    assert request["tools"][0]["name"] == "search_docs"
    assert request["tools"][0]["input_schema"] == TOOLS[0].schema
    assert [m["role"] for m in request["messages"]] == ["user"]


async def test_anthropic_second_request_echoes_blocks_and_results(anthropic_server):
    await _run(AnthropicProvider("claude-test", "claude-test-bulk"))
    messages = anthropic_server.received[1]["messages"]

    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    # The assistant turn is echoed back with its blocks intact
    echoed = messages[1]["content"]
    assert [b["type"] for b in echoed] == ["text", "tool_use"]
    # Results ride in one user message as tool_result blocks
    results = messages[2]["content"]
    assert len(results) == 1
    assert results[0]["type"] == "tool_result"
    assert results[0]["tool_use_id"] == "toolu_abc"
    assert results[0]["content"] == TOOL_RESULT


async def test_anthropic_complete_through_real_sdk(anthropic_server):
    provider = AnthropicProvider("claude-test", "claude-test-bulk")
    anthropic_server.turn = 1  # skip the tool-call turn
    text = await provider.complete("summarize this", max_tokens=64)

    assert text == FINAL_TEXT
    request = anthropic_server.received[-1]
    assert request["model"] == "claude-test-bulk"
    assert request["max_tokens"] == 64
    assert "tools" not in request


async def test_both_providers_reach_the_same_answer(openai_server, anthropic_server):
    """The point of the abstraction: same inputs, same shape of result."""
    openai_result, openai_seen = await _run(OpenAIProvider("gpt-test", "gpt-bulk"))
    anthropic_result, anthropic_seen = await _run(AnthropicProvider("claude-test", "claude-bulk"))

    assert openai_result == anthropic_result == FINAL_TEXT
    assert openai_seen == anthropic_seen
