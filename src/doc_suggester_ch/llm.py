"""Provider abstraction over Claude and OpenAI.

The tool-use protocols differ structurally — Anthropic returns `tool_use`
content blocks and takes results back as `tool_result` blocks in a user
message, while OpenAI returns `tool_calls` on the assistant message and takes
results back as separate `tool` role messages. Rather than adapt one into the
other, each provider owns its whole loop and they share a neutral interface:

    complete()        one prompt in, text out — for bulk work (synopses, enrichment)
    run_tool_loop()   the multi-turn recommendation loop

Callers describe tools as `ToolSpec` and supply an async `dispatch` callback,
so tool *implementations* stay provider-independent.

Provider selection is by available credentials: ANTHROPIC_API_KEY wins, then
OPENAI_API_KEY. Override with --provider, or DOC_SUGGESTER_CH_PROVIDER.
Model choices can be overridden per provider (see _MODELS) because model names
churn faster than this code will.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Protocol

logger = logging.getLogger(__name__)

ANTHROPIC = "anthropic"
OPENAI = "openai"

_ENV_KEYS = {ANTHROPIC: "ANTHROPIC_API_KEY", OPENAI: "OPENAI_API_KEY"}

# (main model, bulk model). Main drives the tool loop; bulk does the
# high-volume single-shot work where a cheaper model is plenty.
_MODELS = {
    ANTHROPIC: ("claude-opus-5", "claude-haiku-4-5"),
    OPENAI: ("gpt-5.5", "gpt-5.4-mini"),
}

_MAIN_MODEL_ENV = "DOC_SUGGESTER_CH_MODEL"
_BULK_MODEL_ENV = "DOC_SUGGESTER_CH_BULK_MODEL"
_PROVIDER_ENV = "DOC_SUGGESTER_CH_PROVIDER"
# OpenAI reasoning effort is opt-in: sending it to a model that doesn't accept
# it is a 400, and the default (unset) is fine for this workload.
_OPENAI_EFFORT_ENV = "DOC_SUGGESTER_CH_OPENAI_REASONING_EFFORT"

MAX_TOKENS = 16000
MAX_TURNS = 20


@dataclass(frozen=True)
class ToolSpec:
    """A tool, described independently of either provider's wire format."""

    name: str
    description: str
    schema: dict[str, Any]


# (tool_name, tool_input) -> result text
Dispatch = Callable[[str, dict[str, Any]], Awaitable[str]]
# Called before each tool runs, for progress output
OnTool = Callable[[str, dict[str, Any]], None]


class ProviderError(RuntimeError):
    """Raised when no usable provider can be resolved."""


class LLMProvider(Protocol):
    name: str
    main_model: str
    bulk_model: str

    async def complete(self, prompt: str, max_tokens: int = 1024) -> str:
        """Run one prompt through the bulk model and return its text."""
        ...

    async def run_tool_loop(
        self,
        system: str,
        user_content: str,
        tools: list[ToolSpec],
        dispatch: Dispatch,
        on_tool: OnTool | None = None,
        max_turns: int = MAX_TURNS,
    ) -> str:
        """Run the multi-turn tool loop and return the final assistant text."""
        ...


def _models_for(provider: str) -> tuple[str, str]:
    main, bulk = _MODELS[provider]
    return os.environ.get(_MAIN_MODEL_ENV) or main, os.environ.get(_BULK_MODEL_ENV) or bulk


def _has_key(provider: str) -> bool:
    return bool(os.environ.get(_ENV_KEYS[provider], "").strip())


def available_providers() -> list[str]:
    """Providers whose credentials are present, in preference order."""
    return [p for p in (ANTHROPIC, OPENAI) if _has_key(p)]


def resolve_provider(preference: str | None = None) -> LLMProvider:
    """Pick a provider from an explicit preference, the environment, or keys.

    Raises ProviderError with actionable text when nothing is usable.
    """
    preference = (preference or os.environ.get(_PROVIDER_ENV) or "").strip().lower() or None

    if preference:
        if preference not in _ENV_KEYS:
            raise ProviderError(
                f"Unknown provider {preference!r}. Choose one of: {', '.join(_ENV_KEYS)}."
            )
        if not _has_key(preference):
            raise ProviderError(
                f"Provider {preference!r} was requested but {_ENV_KEYS[preference]} is not set."
            )
        return _build(preference)

    for provider in available_providers():
        return _build(provider)

    raise ProviderError(
        "No API key found. Set one of: "
        + ", ".join(f"{key} (for {name})" for name, key in _ENV_KEYS.items())
        + "."
    )


def _build(provider: str) -> LLMProvider:
    main, bulk = _models_for(provider)
    if provider == ANTHROPIC:
        return AnthropicProvider(main, bulk)
    return OpenAIProvider(main, bulk)


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model response.

    Tolerates ```json fences and surrounding prose, which both providers emit
    from time to time despite instructions to return bare JSON.
    """
    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if fenced:
        return json.loads(fenced.group(1))
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("no JSON object in response")
    return json.loads(match.group(0))


class AnthropicProvider:
    """Claude via the Messages API, with adaptive thinking on the tool loop."""

    name = ANTHROPIC

    def __init__(self, main_model: str, bulk_model: str) -> None:
        import anthropic

        self._anthropic = anthropic
        self._client = anthropic.AsyncAnthropic()
        self.main_model = main_model
        self.bulk_model = bulk_model

    async def complete(self, prompt: str, max_tokens: int = 1024) -> str:
        response = await self._client.messages.create(
            model=self.bulk_model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return next((b.text for b in response.content if b.type == "text"), "")

    def _tools(self, tools: list[ToolSpec]) -> list[dict[str, Any]]:
        return [
            {"name": t.name, "description": t.description, "input_schema": t.schema}
            for t in tools
        ]

    async def run_tool_loop(
        self,
        system: str,
        user_content: str,
        tools: list[ToolSpec],
        dispatch: Dispatch,
        on_tool: OnTool | None = None,
        max_turns: int = MAX_TURNS,
    ) -> str:
        import asyncio

        messages: list[dict[str, Any]] = [{"role": "user", "content": user_content}]
        wire_tools = self._tools(tools)
        response = None

        for _ in range(max_turns):
            response = await self._client.messages.create(
                model=self.main_model,
                max_tokens=MAX_TOKENS,
                thinking={"type": "adaptive"},
                system=system,
                tools=wire_tools,
                messages=messages,
            )

            # Append the full content list — thinking blocks must be echoed back
            # unchanged for the model to keep reasoning across tool turns.
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "refusal":
                detail = getattr(getattr(response, "stop_details", None), "explanation", "") or ""
                return f"The request was declined by the model. {detail}".strip()

            if response.stop_reason != "tool_use":
                break

            calls = [b for b in response.content if b.type == "tool_use"]
            for block in calls:
                if on_tool:
                    on_tool(block.name, block.input)

            async def run(block: Any) -> dict[str, Any]:
                try:
                    return {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": await dispatch(block.name, block.input),
                    }
                except Exception as exc:  # noqa: BLE001 — report, don't abort
                    return {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": f"Tool failed: {exc}",
                        "is_error": True,
                    }

            # All results for one assistant turn go back in a single user message.
            results = list(await asyncio.gather(*map(run, calls)))
            messages.append({"role": "user", "content": results})

        if response is None:
            return ""
        return next(
            (b.text for b in response.content if b.type == "text" and b.text.strip()), ""
        )


class OpenAIProvider:
    """OpenAI via Chat Completions.

    Notes on the current models this targets:
      - `max_completion_tokens`, not `max_tokens` (the latter is rejected)
      - no `temperature` — reasoning models reject it, and the default is right
      - tool call arguments arrive as a JSON *string* and must be parsed
    """

    name = OPENAI

    def __init__(self, main_model: str, bulk_model: str) -> None:
        import openai

        self._openai = openai
        self._client = openai.AsyncOpenAI()
        self.main_model = main_model
        self.bulk_model = bulk_model
        self._effort = (os.environ.get(_OPENAI_EFFORT_ENV) or "").strip() or None

    def _extra(self) -> dict[str, Any]:
        return {"reasoning_effort": self._effort} if self._effort else {}

    async def complete(self, prompt: str, max_tokens: int = 1024) -> str:
        response = await self._client.chat.completions.create(
            model=self.bulk_model,
            max_completion_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            **self._extra(),
        )
        return response.choices[0].message.content or ""

    def _tools(self, tools: list[ToolSpec]) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.schema,
                },
            }
            for t in tools
        ]

    async def run_tool_loop(
        self,
        system: str,
        user_content: str,
        tools: list[ToolSpec],
        dispatch: Dispatch,
        on_tool: OnTool | None = None,
        max_turns: int = MAX_TURNS,
    ) -> str:
        import asyncio

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ]
        wire_tools = self._tools(tools)
        message = None

        for _ in range(max_turns):
            response = await self._client.chat.completions.create(
                model=self.main_model,
                max_completion_tokens=MAX_TOKENS,
                messages=messages,
                tools=wire_tools,
                **self._extra(),
            )
            choice = response.choices[0]
            message = choice.message
            calls = list(message.tool_calls or [])

            if choice.finish_reason == "content_filter":
                return "The request was declined by the model (content filter)."

            # Echo the assistant turn back verbatim, including tool_calls, or the
            # follow-up tool messages have nothing to attach to.
            messages.append(
                {
                    "role": "assistant",
                    "content": message.content or "",
                    **(
                        {
                            "tool_calls": [
                                {
                                    "id": c.id,
                                    "type": "function",
                                    "function": {
                                        "name": c.function.name,
                                        "arguments": c.function.arguments,
                                    },
                                }
                                for c in calls
                            ]
                        }
                        if calls
                        else {}
                    ),
                }
            )

            if not calls:
                break

            parsed: list[tuple[Any, dict[str, Any]]] = []
            for call in calls:
                try:
                    args = json.loads(call.function.arguments or "{}")
                    if not isinstance(args, dict):
                        raise ValueError("tool arguments were not a JSON object")
                except (json.JSONDecodeError, ValueError) as exc:
                    logger.debug("bad tool arguments for %s: %s", call.function.name, exc)
                    args = {"__error__": str(exc)}
                parsed.append((call, args))
                if on_tool and "__error__" not in args:
                    on_tool(call.function.name, args)

            async def run(call: Any, args: dict[str, Any]) -> dict[str, Any]:
                if "__error__" in args:
                    content = f"Tool failed: could not parse arguments ({args['__error__']})"
                else:
                    try:
                        content = await dispatch(call.function.name, args)
                    except Exception as exc:  # noqa: BLE001 — report, don't abort
                        content = f"Tool failed: {exc}"
                return {"role": "tool", "tool_call_id": call.id, "content": content}

            results = list(await asyncio.gather(*(run(c, a) for c, a in parsed)))
            # Unlike Anthropic, each result is its own message.
            messages.extend(results)

        return (message.content or "").strip() if message else ""
