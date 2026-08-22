"""Fake vendor clients so the LLM tests need no SDKs, keys or network.

Each fake mimics the exact response shape the real SDK returns, so the
analyzers' extraction code (``.choices[0].message.content`` for OpenAI-wire
providers, content blocks for Anthropic) is genuinely exercised.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from src.llm_analyzers import DIMENSIONS


def verdict_payload(
    score: float = 70.0,
    decision: str = "buy",
    confidence: float = 0.7,
    dimensions: Optional[Dict[str, float]] = None,
    rug_flags: Optional[List[str]] = None,
    positives: Optional[List[str]] = None,
    risks: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """A schema-valid verdict, tweakable per test."""
    return {
        "overall_score": score,
        "decision": decision,
        "confidence": confidence,
        "dimension_scores": dimensions or {dim: round(score / 10, 1) for dim in DIMENSIONS},
        "lore_summary": f"A Base meme token that scored {score}.",
        "key_positives": positives if positives is not None else ["LP is burned", "Ownership renounced"],
        "key_risks": risks if risks is not None else ["Meme coins are reflexive"],
        "rug_flags": rug_flags if rug_flags is not None else [],
        "rationale": f"Scored {score} on the evidence provided.",
    }


def verdict_json(**kwargs: Any) -> str:
    return json.dumps(verdict_payload(**kwargs))


class FakeOpenAIClient:
    """Mimics ``openai.OpenAI`` for chat.completions.

    ``fail_modes`` makes the first N attempts raise, which is how the tests
    drive the json_schema -> json_object -> plain fallback chain.
    """

    def __init__(self, content: str = "", fail_modes: int = 0, error: Optional[Exception] = None) -> None:
        self.content = content
        self.fail_modes = fail_modes
        self.error = error
        self.calls: List[Dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        if len(self.calls) <= self.fail_modes:
            raise RuntimeError("response_format not supported by this model")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))]
        )


class FakeAnthropicClient:
    """Mimics ``anthropic.Anthropic`` messages.create with content blocks."""

    def __init__(
        self,
        content: str = "",
        fail_modes: int = 0,
        error: Optional[Exception] = None,
        as_tool_use: bool = False,
    ) -> None:
        self.content = content
        self.fail_modes = fail_modes
        self.error = error
        self.as_tool_use = as_tool_use
        self.calls: List[Dict[str, Any]] = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        if len(self.calls) <= self.fail_modes:
            raise RuntimeError("output_config is not supported on this model")
        if self.as_tool_use and "tools" in kwargs:
            return SimpleNamespace(
                content=[SimpleNamespace(type="tool_use", name="submit_verdict",
                                         input=json.loads(self.content))]
            )
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=self.content)])


class SlowClient(FakeOpenAIClient):
    """OpenAI-shaped client that blocks, to exercise timeouts and parallelism."""

    def __init__(self, delay: float, content: str = "") -> None:
        super().__init__(content=content)
        self.delay = delay

    def _create(self, **kwargs: Any) -> Any:
        import time

        time.sleep(self.delay)
        return super()._create(**kwargs)


class SlowAnthropicClient(FakeAnthropicClient):
    """Anthropic-shaped client that blocks (the Claude slot needs its own shape)."""

    def __init__(self, delay: float, content: str = "") -> None:
        super().__init__(content=content)
        self.delay = delay

    def _create(self, **kwargs: Any) -> Any:
        import time

        time.sleep(self.delay)
        return super()._create(**kwargs)
