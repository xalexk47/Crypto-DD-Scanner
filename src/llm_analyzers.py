"""Multi-LLM ensemble analysis: Grok (xAI) + Claude (Anthropic) + GPT (OpenAI).

The same structured token payload and the same prompt go to every configured
model **in parallel**; each returns strict JSON against one shared schema; the
answers are then normalized, compared and combined into a consensus verdict.

Why an ensemble rather than one model
-------------------------------------
A single model's meme-coin call is a coin flip dressed up as analysis. Three
independent reads give you something a single call cannot: *agreement*. When
all three land on the same decision the signal is worth acting on; when they
scatter, the spread itself is the finding, and this module surfaces it as
``dissent`` and a reduced confidence rather than hiding it behind an average.
A rug flag raised independently by two models is tracked separately from one
model's hunch for the same reason.

Strict JSON, per vendor
-----------------------
Each provider has a different mechanism for constraining output, so each gets
its native one, with fallbacks when a model or SDK version does not support it:

* **Anthropic** - ``output_config.format`` json_schema -> strict tool call -> plain prompt.
* **OpenAI / xAI** - ``response_format`` json_schema (strict) -> json_object -> plain prompt.

Whatever comes back is run through :func:`coerce_verdict`, which repairs the
common model errors (0-100 dimension scores instead of 0-10, confidence as a
percentage, a missing or invented decision label) instead of discarding the
response. Nothing here raises: a provider that fails returns a verdict with
``ok=False`` and the reason.

Adding a fourth provider is one subclass plus one entry in :data:`ANALYZERS`.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import re
import statistics
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import config
from .models import (
    ConsensusVerdict,
    EnsembleResult,
    LLMVerdict,
    MindshareReport,
    ScoreCard,
    SecurityReport,
    TokenProfile,
    TokenSnapshot,
)
from .utils import clamp, safe_float

logger = logging.getLogger(__name__)

DIMENSIONS: Tuple[str, ...] = config.LLM_DIMENSIONS
DECISIONS: Tuple[str, ...] = config.LLM_DECISIONS
# Ordering matters: ties and conservative fallbacks resolve toward "pass".
DECISION_RANK: Dict[str, int] = {"pass": 0, "watch": 1, "buy": 2, "strong_buy": 3}
RANK_DECISION: Dict[int, str] = {rank: name for name, rank in DECISION_RANK.items()}


# ==========================================================================
# The shared JSON contract
# ==========================================================================
# Deliberately limited to type/enum/description/required/additionalProperties:
# that is the intersection every vendor's strict mode accepts. Numeric bounds
# are stated in descriptions and enforced by coerce_verdict(), because strict
# structured-output modes ignore or reject `minimum`/`maximum`.
def _string_array(description: str) -> Dict[str, Any]:
    return {"type": "array", "items": {"type": "string"}, "description": description}


VERDICT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "overall_score": {
            "type": "number",
            "description": "Overall quality of this opportunity, 0-100. 0 is a certain rug, 100 is exceptional.",
        },
        "decision": {
            "type": "string",
            "enum": list(DECISIONS),
            "description": "Your call. Use pass unless the risk-adjusted case is genuinely good.",
        },
        "confidence": {
            "type": "number",
            "description": "How confident you are in this verdict, 0.0-1.0. Lower it when data is missing.",
        },
        "dimension_scores": {
            "type": "object",
            "properties": {
                "security": {"type": "number", "description": "Contract safety, 0-10."},
                "liquidity": {"type": "number", "description": "Depth and exitability, 0-10."},
                "holders": {"type": "number", "description": "Distribution health, 0-10."},
                "mindshare": {"type": "number", "description": "Attention and momentum, 0-10."},
                "lore": {"type": "number", "description": "Narrative strength and meme-ability, 0-10."},
                "catalyst": {"type": "number", "description": "Upside catalysts and listing potential, 0-10."},
            },
            "required": list(DIMENSIONS),
            "additionalProperties": False,
        },
        "lore_summary": {
            "type": "string",
            "description": "2-4 sentence narrative assessment of what this token is and why anyone cares.",
        },
        "key_positives": _string_array("2-5 concrete reasons this could work."),
        "key_risks": _string_array("2-5 concrete reasons this fails."),
        "rug_flags": _string_array(
            "Specific rug/scam indicators found in the data. Empty array if none."
        ),
        "rationale": {
            "type": "string",
            "description": "One concise paragraph tying the evidence to your decision.",
        },
    },
    "required": [
        "overall_score",
        "decision",
        "confidence",
        "dimension_scores",
        "lore_summary",
        "key_positives",
        "key_risks",
        "rug_flags",
        "rationale",
    ],
    "additionalProperties": False,
}


ANALYST_SYSTEM_PROMPT = """\
You are a professional meme-coin due-diligence analyst. You are deeply cynical \
about hype and precise about risk. Most meme coins are worthless or outright \
fraudulent; your default answer is "pass" and a good token has to earn better.

You will be given a structured JSON payload of on-chain and market data for one \
token. Judge only what is in the payload.

Rules:
- Never invent facts. If a field is missing or null, treat that as unknown and \
say so - unknown is a risk, never a positive.
- Missing contract-security data must LOWER your security score and your \
confidence. Do not assume a token is safe because nothing is flagged.
- A confirmed honeypot, an unsellable balance, a hidden owner or a live mint \
authority is disqualifying: score security 0-1, decision "pass", and list it \
in rug_flags.
- Weigh exitability heavily. A position you cannot sell at size is worthless \
regardless of the chart.
- rug_flags is for concrete indicators present in this payload, not generic \
warnings about meme coins. Return an empty array when there are none.
- A deterministic score computed by the platform's own rules engine may be \
included for reference. Form your own judgement: say so in your rationale if \
you disagree with it, and explain why.
- You are producing research, not financial advice.

Return only the JSON object defined by the schema. No prose outside it, no \
markdown fences."""


# ==========================================================================
# Payload construction
# ==========================================================================
def build_analysis_payload(
    snapshot: TokenSnapshot,
    security: Optional[SecurityReport] = None,
    scorecard: Optional[ScoreCard] = None,
    profile: Optional[TokenProfile] = None,
    mindshare: Optional[MindshareReport] = None,
) -> Dict[str, Any]:
    """Assemble the full structured token payload sent to every model.

    One payload, one prompt, every provider -- that is what makes the verdicts
    comparable. ``None`` is preserved rather than defaulted so the models can
    see what is genuinely unknown.
    """
    payload: Dict[str, Any] = {
        "token": {
            "name": snapshot.name or None,
            "symbol": snapshot.symbol or None,
            "chain": snapshot.chain,
            "address": snapshot.address,
            "age": snapshot.age_label,
            "age_hours": round(snapshot.age_hours, 1) if snapshot.age_hours is not None else None,
            "description": snapshot.description or None,
        },
        "market": {
            "price_usd": snapshot.price_usd or None,
            "market_cap_usd": snapshot.market_cap or None,
            "fdv_usd": snapshot.fdv or None,
            "liquidity_usd": snapshot.liquidity_usd or None,
            "volume_24h_usd": snapshot.volume_24h,
            "volume_6h_usd": snapshot.volume_6h,
            "volume_1h_usd": snapshot.volume_1h,
            "turnover_24h_x_mcap": round(snapshot.turnover_24h, 4) if snapshot.market_cap else None,
            "liquidity_to_mcap_pct": round(snapshot.liquidity_ratio * 100, 2) if snapshot.market_cap else None,
            "volume_acceleration_6h_vs_24h": (
                round(snapshot.volume_acceleration, 2) if snapshot.volume_acceleration is not None else None
            ),
            "price_change_pct": {
                "5m": snapshot.price_change_5m,
                "1h": snapshot.price_change_1h,
                "6h": snapshot.price_change_6h,
                "24h": snapshot.price_change_24h,
            },
            "txns_24h": {
                "buys": snapshot.txns_24h_buys,
                "sells": snapshot.txns_24h_sells,
                "buy_share": round(snapshot.buy_sell_ratio, 3) if snapshot.buy_sell_ratio is not None else None,
            },
            "venues": {
                "pairs": snapshot.pair_count,
                "dexes": snapshot.dex_count,
                "primary_dex": snapshot.dex_id or None,
                "quote_token": snapshot.quote_symbol or None,
            },
            "dexscreener_boosts": snapshot.boosts,
        },
        "socials": [{"kind": link.kind, "url": link.url} for link in snapshot.socials] or None,
        "security": None,
        "platform_deterministic_score": None,
        "dexscreener_profile": None,
        "x_mindshare": None,
    }

    if security is not None and security.available:
        payload["security"] = {
            "source": security.source,
            "is_honeypot": security.is_honeypot,
            "cannot_sell_all": security.cannot_sell_all,
            "buy_tax_pct": security.buy_tax_pct,
            "sell_tax_pct": security.sell_tax_pct,
            "is_open_source": security.is_open_source,
            "is_proxy": security.is_proxy,
            "is_mintable": security.is_mintable,
            "owner_renounced": security.owner_renounced,
            "can_take_back_ownership": security.can_take_back_ownership,
            "hidden_owner": security.hidden_owner,
            "selfdestruct": security.selfdestruct,
            "transfer_pausable": security.transfer_pausable,
            "is_blacklisted": security.is_blacklisted,
            "slippage_modifiable": security.slippage_modifiable,
            "is_freezable": security.is_freezable,
            "lp_burned_pct": security.lp_burned_pct,
            "lp_locked_pct": security.lp_locked_pct,
            "lp_secured_pct": security.lp_secured_pct,
            "holder_count": security.holder_count,
            "top10_pct_including_lp": security.top10_pct,
            "top10_pct_excluding_lp_and_burn": security.top10_pct_adjusted,
            "deployer_holdings_pct": security.creator_percent,
            "provider_warnings": security.warnings or None,
        }
    else:
        payload["security_unavailable_reason"] = (
            security.error if security is not None and security.error
            else "No contract-security data was available for this token."
        )

    if scorecard is not None:
        payload["platform_deterministic_score"] = {
            "composite_0_100": scorecard.composite,
            "decision": scorecard.decision,
            "confidence": scorecard.confidence,
            "vetoed": scorecard.vetoed,
            "veto_reason": scorecard.veto_reason or None,
            "pillars": {c.key: round(c.score, 1) for c in scorecard.components},
        }

    if mindshare is not None and mindshare.available:
        # Sharing Grok's X findings with every model means Claude and GPT can
        # weigh the social signal too, instead of Grok alone having seen it.
        payload["x_mindshare"] = {
            "data_is_live": mindshare.is_live,
            "source": mindshare.source,
            "sentiment": mindshare.sentiment,
            "sentiment_score": mindshare.sentiment_score,
            "attention_score_0_100": mindshare.mindshare_score,
            "post_volume": mindshare.post_volume,
            "trend": mindshare.trend,
            "looks_organic": mindshare.is_organic,
            "summary": mindshare.summary or None,
            "themes": mindshare.themes or None,
            "notable_accounts": mindshare.notable_accounts or None,
            "red_flags": mindshare.red_flags or None,
            "sample_posts": [
                {"handle": p.handle, "text": p.text, "engagement": p.engagement}
                for p in mindshare.sample_posts
            ] or None,
        }
        if not mindshare.is_live:
            payload["x_mindshare"]["caveat"] = (
                "This came from model knowledge, not a live X search. Treat it as "
                "background, not current sentiment."
            )

    if profile is not None and profile.has_content:
        payload["dexscreener_profile"] = {
            "description": profile.description or None,
            "links": [{"kind": link.kind, "url": link.url} for link in profile.links] or None,
            "has_icon": bool(profile.icon_url),
            "has_header_art": bool(profile.header_url),
        }

    return payload


def build_user_prompt(payload: Dict[str, Any]) -> str:
    """Wrap the payload in the instruction every model receives verbatim."""
    return (
        "Analyze this token and return the JSON verdict.\n\n"
        "TOKEN DATA:\n"
        f"{json.dumps(payload, indent=2, default=str, sort_keys=True)}"
    )


# ==========================================================================
# Response normalization / repair
# ==========================================================================
_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")


def extract_json(raw: str) -> Dict[str, Any]:
    """Pull a JSON object out of a model response.

    Strict modes return clean JSON, but the plain-prompt fallback may wrap it
    in fences or prose, so we strip fences and, failing that, take the outermost
    balanced ``{...}`` span.
    """
    text = (raw or "").strip()
    if not text:
        raise ValueError("empty response")

    if text.startswith("```"):
        text = _FENCE_RE.sub("", text).strip()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise ValueError("no JSON object found in response")
        parsed = json.loads(text[start : end + 1])

    if not isinstance(parsed, dict):
        raise ValueError(f"expected a JSON object, got {type(parsed).__name__}")
    return parsed


def _string_list(value: Any, limit: int = 8) -> List[str]:
    """Coerce a model's array field into a clean list of non-empty strings."""
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)):
        return []
    items: List[str] = []
    for item in value:
        if isinstance(item, dict):  # some models wrap entries as {"text": ...}
            item = item.get("text") or item.get("description") or item.get("value") or ""
        text = str(item).strip()
        if text and text.lower() not in ("none", "n/a", "null"):
            items.append(text)
    return items[:limit]


def _normalize_decision(value: Any) -> Optional[str]:
    """Map free-form decision text onto the four allowed values."""
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if text in DECISION_RANK:
        return text
    aliases = {
        "strongbuy": "strong_buy", "strong": "strong_buy", "accumulate": "buy",
        "hold": "watch", "monitor": "watch", "neutral": "watch", "wait": "watch",
        "avoid": "pass", "sell": "pass", "skip": "pass", "no": "pass", "reject": "pass",
    }
    return aliases.get(text)


def decision_from_score(score: float) -> str:
    """Bucket a 0-100 score using the same thresholds as the rules engine."""
    if score >= config.DECISION_THRESHOLDS["Strong Buy"]:
        return "strong_buy"
    if score >= config.DECISION_THRESHOLDS["Buy"]:
        return "buy"
    if score >= config.DECISION_THRESHOLDS["Watch"]:
        return "watch"
    return "pass"


def coerce_verdict(data: Dict[str, Any], provider: str, model: str, raw: str = "") -> LLMVerdict:
    """Normalize and repair one model's JSON into an :class:`LLMVerdict`.

    Repairs (each recorded in ``warnings``, and flagged by ``repaired``):

    * dimension scores returned on a 0-100 scale instead of 0-10
    * confidence returned as a percentage instead of a 0-1 fraction
    * a missing/unrecognised decision, re-derived from the score
    * missing dimensions, filled from the overall score
    """
    warnings: List[str] = []
    repaired = False

    score = clamp(safe_float(data.get("overall_score"), 0.0), 0.0, 100.0)

    decision = _normalize_decision(data.get("decision"))
    if decision is None:
        decision = decision_from_score(score)
        warnings.append(f"Unrecognised decision {data.get('decision')!r}; derived {decision!r} from the score.")
        repaired = True

    confidence = safe_float(data.get("confidence"), 0.0)
    if confidence > 1.0:
        # 85 almost certainly means 85%, not "eighty-five times certain".
        confidence = confidence / 100.0 if confidence <= 100.0 else 1.0
        warnings.append("Confidence looked like a percentage; converted to a 0-1 fraction.")
        repaired = True
    confidence = clamp(confidence, 0.0, 1.0)

    raw_dims = data.get("dimension_scores") or {}
    if not isinstance(raw_dims, dict):
        raw_dims = {}
    # None means "the model did not answer", which is different from a value we
    # simply dislike -- an out-of-range number is clamped, never treated as
    # missing and back-filled from the (possibly high) overall score.
    parsed_dims: Dict[str, Optional[float]] = {
        key: (None if raw_dims.get(key) is None else safe_float(raw_dims.get(key), 0.0))
        for key in DIMENSIONS
    }
    present = [value for value in parsed_dims.values() if value is not None]

    # A model that answers 0-100 per dimension is wrong but recoverable.
    if present and max(present) > 10.0:
        parsed_dims = {
            key: (None if value is None else value / 10.0) for key, value in parsed_dims.items()
        }
        warnings.append("Dimension scores came back on a 0-100 scale; rescaled to 0-10.")
        repaired = True

    dimensions: Dict[str, float] = {}
    missing: List[str] = []
    for key in DIMENSIONS:
        value = parsed_dims[key]
        if value is None:
            missing.append(key)
            value = score / 10.0          # neutral stand-in, not a free pass
        dimensions[key] = round(clamp(value, 0.0, 10.0), 2)
    if missing:
        warnings.append(f"Missing dimension score(s) {', '.join(missing)}; filled from the overall score.")
        repaired = True

    verdict = LLMVerdict(
        provider=provider,
        model=model,
        ok=True,
        overall_score=round(score, 1),
        decision=decision,
        confidence=round(confidence, 3),
        dimension_scores=dimensions,
        lore_summary=str(data.get("lore_summary") or "").strip(),
        key_positives=_string_list(data.get("key_positives")),
        key_risks=_string_list(data.get("key_risks")),
        rug_flags=_string_list(data.get("rug_flags")),
        rationale=str(data.get("rationale") or "").strip(),
        repaired=repaired,
        warnings=warnings,
        raw_excerpt=(raw or "")[:600],
    )
    if not verdict.lore_summary and not verdict.rationale:
        verdict.warnings.append("Model returned no narrative text.")
    return verdict


# ==========================================================================
# Analyzers
# ==========================================================================
@dataclass
class ProviderStatus:
    """Whether a provider can run, and why not when it cannot."""

    provider: str
    ready: bool
    reason: str = ""


class BaseAnalyzer:
    """Common behaviour: timing, error capture, JSON coercion.

    Subclasses implement :meth:`_request_json`, which must return the model's
    raw response text (expected to be a JSON object).
    """

    provider = "base"
    label = "Base"

    def __init__(
        self,
        api_key: str = "",
        model: str = "",
        timeout: Optional[float] = None,
        max_tokens: Optional[int] = None,
        client: Any = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.timeout = timeout if timeout is not None else config.LLM_TIMEOUT_SECONDS
        self.max_tokens = max_tokens if max_tokens is not None else config.LLM_MAX_TOKENS
        # An injected client makes this testable without the vendor SDK.
        self._injected_client = client

    # -- availability ---------------------------------------------------
    def _sdk_module(self) -> str:
        raise NotImplementedError

    def status(self) -> ProviderStatus:
        if self._injected_client is not None:
            return ProviderStatus(self.provider, True)
        if not self.api_key:
            return ProviderStatus(self.provider, False, f"No API key set for {self.label}.")
        module = self._sdk_module()
        try:
            __import__(module)
        except ImportError:
            return ProviderStatus(self.provider, False, f"`pip install {module}` to enable {self.label}.")
        return ProviderStatus(self.provider, True)

    def available(self) -> bool:
        return self.status().ready

    # -- transport ------------------------------------------------------
    def _client(self) -> Any:
        raise NotImplementedError

    def _request_json(self, system: str, user: str, schema: Optional[Dict[str, Any]] = None) -> str:
        raise NotImplementedError

    def request_json(self, system: str, user: str, schema: Optional[Dict[str, Any]] = None) -> str:
        """Public transport hook: prompt in, raw JSON text out.

        Exposed so other modules (the narrative layer in :mod:`src.llm`) can
        reuse this provider's strict-JSON handling with a different schema
        instead of re-implementing each vendor's SDK call.
        """
        return self._request_json(system, user, schema)

    # -- public API -----------------------------------------------------
    def analyze(self, payload: Dict[str, Any]) -> LLMVerdict:
        """Run one model. Never raises -- failures come back as ``ok=False``."""
        started = time.monotonic()
        raw = ""
        try:
            raw = self._request_json(ANALYST_SYSTEM_PROMPT, build_user_prompt(payload))
            verdict = coerce_verdict(extract_json(raw), self.provider, self.model, raw)
        except Exception as exc:  # noqa: BLE001 - one provider must not sink the run
            logger.warning("%s analysis failed: %s", self.provider, exc, exc_info=logger.isEnabledFor(logging.DEBUG))
            return LLMVerdict(
                provider=self.provider,
                model=self.model,
                ok=False,
                error=f"{type(exc).__name__}: {exc}"[:300],
                latency_ms=int((time.monotonic() - started) * 1000),
                raw_excerpt=(raw or "")[:600],
            )
        verdict.latency_ms = int((time.monotonic() - started) * 1000)
        return verdict


class AnthropicAnalyzer(BaseAnalyzer):
    """Claude via the official Anthropic SDK.

    Strict JSON comes from ``output_config.format`` (json_schema), with a
    strict tool call and then a plain prompt as fallbacks for older SDKs or
    models that lack structured outputs.

    Note: current Claude models reject sampling parameters, so no temperature
    is sent here -- determinism is handled by the schema and the prompt.
    """

    provider = "anthropic"
    label = "Claude"

    def __init__(self, api_key: str = "", model: str = "", **kwargs: Any) -> None:
        super().__init__(
            api_key=api_key or config.ANTHROPIC_API_KEY,
            model=model or config.ANTHROPIC_MODEL,
            **kwargs,
        )

    def _sdk_module(self) -> str:
        return "anthropic"

    def _client(self) -> Any:
        if self._injected_client is not None:
            return self._injected_client
        import anthropic

        return anthropic.Anthropic(api_key=self.api_key, timeout=self.timeout)

    @staticmethod
    def _text_blocks(message: Any) -> str:
        return "".join(
            getattr(block, "text", "") for block in getattr(message, "content", [])
            if getattr(block, "type", "") == "text"
        )

    @staticmethod
    def _tool_input(message: Any) -> Optional[Dict[str, Any]]:
        for block in getattr(message, "content", []):
            if getattr(block, "type", "") == "tool_use":
                return getattr(block, "input", None)
        return None

    def _request_json(self, system: str, user: str, schema: Optional[Dict[str, Any]] = None) -> str:
        schema = schema or VERDICT_SCHEMA
        client = self._client()
        messages = [{"role": "user", "content": user}]
        base = {"model": self.model, "max_tokens": self.max_tokens, "system": system, "messages": messages}

        # 1) Structured outputs: the response is guaranteed to parse.
        try:
            message = client.messages.create(
                **base,
                output_config={"format": {"type": "json_schema", "schema": schema}, "effort": "medium"},
            )
            text = self._text_blocks(message)
            if text.strip():
                return text
        except Exception as exc:  # noqa: BLE001
            logger.info("Claude structured output unavailable (%s); trying strict tool use.", exc)

        # 2) A forced strict tool call also guarantees schema-valid JSON.
        try:
            message = client.messages.create(
                **base,
                tools=[{
                    "name": "submit_verdict",
                    "description": "Submit the token due-diligence verdict.",
                    "strict": True,
                    "input_schema": schema,
                }],
                tool_choice={"type": "tool", "name": "submit_verdict"},
            )
            tool_input = self._tool_input(message)
            if tool_input:
                return json.dumps(tool_input)
        except Exception as exc:  # noqa: BLE001
            logger.info("Claude strict tool use unavailable (%s); falling back to prompt-only JSON.", exc)

        # 3) Last resort: ask for JSON in the prompt and repair what comes back.
        message = client.messages.create(**base)
        return self._text_blocks(message)


class OpenAICompatibleAnalyzer(BaseAnalyzer):
    """Shared implementation for every OpenAI-wire-protocol endpoint.

    Tries strict ``json_schema`` first, then ``json_object``, then a plain
    call -- endpoint support for structured outputs varies by model and by
    vendor, and a hard failure here would cost us a whole ensemble member.
    """

    provider = "openai"
    label = "GPT"
    base_url = ""

    def _sdk_module(self) -> str:
        return "openai"

    def _client(self) -> Any:
        if self._injected_client is not None:
            return self._injected_client
        from openai import OpenAI

        kwargs: Dict[str, Any] = {"api_key": self.api_key, "timeout": self.timeout}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return OpenAI(**kwargs)

    @staticmethod
    def _content(response: Any) -> str:
        return (response.choices[0].message.content or "") if getattr(response, "choices", None) else ""

    def _request_json(self, system: str, user: str, schema: Optional[Dict[str, Any]] = None) -> str:
        schema = schema or VERDICT_SCHEMA
        client = self._client()
        base: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "temperature": config.LLM_TEMPERATURE,
        }

        attempts = (
            {"response_format": {
                "type": "json_schema",
                "json_schema": {"name": "token_verdict", "strict": True, "schema": schema},
            }},
            {"response_format": {"type": "json_object"}},
            {},
        )
        last_error: Optional[Exception] = None
        for index, extra in enumerate(attempts):
            try:
                text = self._content(client.chat.completions.create(**base, **extra))
                if text.strip():
                    return text
                last_error = ValueError("empty response")
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if index < len(attempts) - 1:
                    logger.info("%s attempt %d failed (%s); trying a looser JSON mode.", self.label, index + 1, exc)
        raise last_error or ValueError("no response")


class OpenAIAnalyzer(OpenAICompatibleAnalyzer):
    """GPT via api.openai.com."""

    provider = "openai"
    label = "GPT"

    def __init__(self, api_key: str = "", model: str = "", **kwargs: Any) -> None:
        super().__init__(api_key=api_key or config.OPENAI_API_KEY, model=model or config.OPENAI_MODEL, **kwargs)
        self.base_url = config.OPENAI_BASE_URL


class XAIAnalyzer(OpenAICompatibleAnalyzer):
    """Grok via xAI's OpenAI-compatible endpoint at api.x.ai/v1.

    Grok is the most interesting ensemble member for meme coins specifically:
    it has live access to X, which is where meme-coin mindshare actually forms
    and the one signal on-chain data cannot see.
    """

    provider = "xai"
    label = "Grok"

    def __init__(self, api_key: str = "", model: str = "", **kwargs: Any) -> None:
        super().__init__(api_key=api_key or config.XAI_API_KEY, model=model or config.XAI_MODEL, **kwargs)
        self.base_url = config.XAI_BASE_URL


ANALYZERS: Dict[str, type] = {
    "anthropic": AnthropicAnalyzer,
    "claude": AnthropicAnalyzer,
    "openai": OpenAIAnalyzer,
    "gpt": OpenAIAnalyzer,
    "xai": XAIAnalyzer,
    "grok": XAIAnalyzer,
}


def build_analyzers(
    providers: Optional[Sequence[str]] = None,
    clients: Optional[Dict[str, Any]] = None,
) -> Tuple[List[BaseAnalyzer], Dict[str, str]]:
    """Instantiate the requested analyzers.

    Returns ``(ready, skipped)`` where ``skipped`` maps a provider name to a
    plain-English reason (no key, SDK missing, unknown name). ``clients`` injects
    pre-built clients by provider name, which is how the tests avoid the SDKs.
    """
    names = list(providers or config.ENSEMBLE_PROVIDERS)
    ready: List[BaseAnalyzer] = []
    skipped: Dict[str, str] = {}
    seen: set = set()

    for name in names:
        key = (name or "").strip().lower()
        analyzer_cls = ANALYZERS.get(key)
        if analyzer_cls is None:
            skipped[key or "?"] = f"Unknown provider {name!r}."
            continue
        analyzer = analyzer_cls(client=(clients or {}).get(key))
        if analyzer.provider in seen:      # e.g. both "claude" and "anthropic"
            continue
        seen.add(analyzer.provider)
        status = analyzer.status()
        if status.ready:
            ready.append(analyzer)
        else:
            skipped[analyzer.provider] = status.reason
    return ready, skipped


def provider_statuses(clients: Optional[Dict[str, Any]] = None) -> List[ProviderStatus]:
    """Readiness of every known provider - used by the sidebar."""
    statuses: List[ProviderStatus] = []
    for key in ("xai", "anthropic", "openai"):
        analyzer = ANALYZERS[key](client=(clients or {}).get(key))
        statuses.append(analyzer.status())
    return statuses


# ==========================================================================
# Consensus
# ==========================================================================
def _dedupe_key(text: str) -> str:
    """Normalize a sentence for cross-model duplicate detection.

    Catches exact and near-exact repeats (case, punctuation, whitespace). Two
    models phrasing the same idea differently will still produce two entries --
    real semantic clustering would need embeddings, which is deliberately out
    of scope here.
    """
    return re.sub(r"[^a-z0-9 ]", "", (text or "").lower()).strip()


def _merge_statements(verdicts: Sequence[LLMVerdict], attribute: str, limit: int = 8) -> List[str]:
    """Merge a list field across models, most-corroborated first."""
    counts: Dict[str, int] = {}
    originals: Dict[str, str] = {}
    for verdict in verdicts:
        for item in getattr(verdict, attribute, []) or []:
            key = _dedupe_key(item)
            if not key:
                continue
            counts[key] = counts.get(key, 0) + 1
            originals.setdefault(key, item)
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], originals[kv[0]]))
    return [originals[key] for key, _ in ordered[:limit]]


def _weight(verdict: LLMVerdict) -> float:
    """Confidence as an aggregation weight, floored so nothing is silenced."""
    return max(0.15, verdict.confidence)


def build_consensus(verdicts: Sequence[LLMVerdict]) -> Optional[ConsensusVerdict]:
    """Combine model verdicts, preserving disagreement as a first-class signal.

    * Scores are confidence-weighted means.
    * The decision is a confidence-weighted vote, then held to the **more
      conservative** of that vote and the score's own bucket -- when the models
      say "buy" but their average score says "watch", we take "watch".
    * Confidence is scaled by how much the models actually agreed, so three
      models at 90/50/20 produce a low-confidence consensus, not a confident 53.
    """
    usable = [v for v in verdicts if v.ok]
    if not usable:
        return None

    weights = [_weight(v) for v in usable]
    total_weight = sum(weights) or 1.0
    scores = [v.overall_score for v in usable]

    overall = sum(v.overall_score * w for v, w in zip(usable, weights)) / total_weight
    dimensions = {
        dim: round(
            sum(v.dimension_scores.get(dim, 0.0) * w for v, w in zip(usable, weights)) / total_weight, 2
        )
        for dim in DIMENSIONS
    }

    # Weighted decision vote; ties break conservative (lowest rank wins).
    votes: Dict[str, float] = {}
    split: Dict[str, int] = {}
    for verdict, weight in zip(usable, weights):
        votes[verdict.decision] = votes.get(verdict.decision, 0.0) + weight
        split[verdict.decision] = split.get(verdict.decision, 0) + 1
    top = max(votes.values())
    winners = [d for d, w in votes.items() if abs(w - top) < 1e-9]
    voted = RANK_DECISION[min(DECISION_RANK[d] for d in winners)]

    from_score = decision_from_score(overall)
    decision = RANK_DECISION[min(DECISION_RANK[voted], DECISION_RANK[from_score])]

    # Agreement: how tight are the scores, and do the calls actually match?
    spread = max(scores) - min(scores) if len(scores) > 1 else 0.0
    stdev = statistics.pstdev(scores) if len(scores) > 1 else 0.0
    score_agreement = clamp(1.0 - (stdev / 25.0), 0.0, 1.0)
    ranks = [DECISION_RANK[v.decision] for v in usable]
    rank_spread = max(ranks) - min(ranks)
    decision_agreement = {0: 1.0, 1: 0.6, 2: 0.25}.get(rank_spread, 0.0)
    agreement = clamp(0.6 * score_agreement + 0.4 * decision_agreement, 0.0, 1.0)

    mean_confidence = sum(v.confidence for v in usable) / len(usable)
    # A single model has nothing to agree with, so it cannot claim ensemble
    # confidence: cap it rather than pretending one opinion is a consensus.
    confidence = mean_confidence * (agreement if len(usable) > 1 else 0.7)

    # Rug flags: which ones did more than one model find independently?
    flag_counts: Dict[str, int] = {}
    flag_text: Dict[str, str] = {}
    for verdict in usable:
        for flag in {_dedupe_key(f): f for f in verdict.rug_flags}.items():
            key, original = flag
            if not key:
                continue
            flag_counts[key] = flag_counts.get(key, 0) + 1
            flag_text.setdefault(key, original)
    ordered_flags = sorted(flag_counts.items(), key=lambda kv: (-kv[1], flag_text[kv[0]]))
    rug_flags = [flag_text[key] for key, _ in ordered_flags]
    corroborated = [flag_text[key] for key, count in ordered_flags if count >= 2]

    # Dissent: the parts a single averaged number would hide.
    dissent: List[str] = []
    if len(usable) > 1:
        if spread >= 25:
            high = max(usable, key=lambda v: v.overall_score)
            low = min(usable, key=lambda v: v.overall_score)
            dissent.append(
                f"{spread:.0f}-point spread: {high.label} scored {high.overall_score:.0f}, "
                f"{low.label} scored {low.overall_score:.0f}."
            )
        if len(split) > 1:
            calls = ", ".join(
                f"{count}x {config.DECISION_LABELS.get(name, name)}" for name, count in sorted(split.items())
            )
            dissent.append(f"Models split on the call: {calls}.")
        if voted != decision:
            dissent.append(
                f"Vote favoured {config.DECISION_LABELS.get(voted, voted)} but the averaged score implies "
                f"{config.DECISION_LABELS.get(from_score, from_score)}; taking the more conservative call."
            )
        solo_flags = len(rug_flags) - len(corroborated)
        if solo_flags > 0:
            dissent.append(f"{solo_flags} rug flag(s) raised by only one model - verify before acting on them.")

    # Narrative: quote the most confident model rather than blending prose.
    lead = max(usable, key=lambda v: (v.confidence, len(v.lore_summary)))
    agreement_line = (
        f"{len(usable)} model(s) analysed this token; agreement {agreement * 100:.0f}%. "
        if len(usable) > 1 else "Single-model verdict (no cross-check). "
    )

    return ConsensusVerdict(
        overall_score=round(clamp(overall, 0.0, 100.0), 1),
        decision=decision,
        confidence=round(clamp(confidence, 0.0, 1.0), 3),
        dimension_scores=dimensions,
        lore_summary=lead.lore_summary,
        key_positives=_merge_statements(usable, "key_positives"),
        key_risks=_merge_statements(usable, "key_risks"),
        rug_flags=rug_flags,
        rationale=agreement_line + (lead.rationale or ""),
        model_count=len(usable),
        agreement=round(agreement, 3),
        score_spread=round(spread, 1),
        decision_split=split,
        corroborated_rug_flags=corroborated,
        dissent=dissent,
    )


def blend_with_deterministic(
    scorecard: Optional[ScoreCard],
    consensus: Optional[ConsensusVerdict],
    weight: float = config.ENSEMBLE_BLEND_WEIGHT,
) -> Tuple[Optional[float], str, List[str]]:
    """Blend the rules-engine score with the LLM consensus.

    Returns ``(blended_score, decision, notes)``. Two rules that are not up for
    negotiation:

    * The deterministic **security veto wins outright**. Models can be talked
      out of a honeypot by a good story; the rules engine cannot.
    * The blended decision is the more conservative of the blended score's
      bucket and the deterministic decision, so the LLMs can talk a score
      *down* freely but never rescue one the rules engine failed.
    """
    notes: List[str] = []
    if consensus is None or scorecard is None:
        return None, "", notes

    weight = clamp(weight, 0.0, 1.0)
    blended = scorecard.composite * (1 - weight) + consensus.overall_score * weight
    notes.append(
        f"Blended {(1 - weight) * 100:.0f}% rules engine ({scorecard.composite:.0f}) with "
        f"{weight * 100:.0f}% LLM consensus ({consensus.overall_score:.0f})."
    )

    if scorecard.vetoed:
        notes.append(f"Security veto overrides the models: {scorecard.veto_reason}")
        return round(min(blended, 20.0), 1), "pass", notes

    deterministic = _decision_key(scorecard.decision)
    from_blend = decision_from_score(blended)
    decision = RANK_DECISION[min(DECISION_RANK[from_blend], DECISION_RANK[deterministic])]
    if decision != from_blend:
        notes.append(
            f"Held to the rules engine's more conservative {config.DECISION_LABELS[deterministic]}."
        )
    return round(blended, 1), decision, notes


def _decision_key(label: str) -> str:
    """"Strong Buy" -> "strong_buy" (rules engine -> model vocabulary)."""
    key = (label or "").strip().lower().replace(" ", "_")
    return key if key in DECISION_RANK else "pass"


# ==========================================================================
# Parallel runner
# ==========================================================================
def run_ensemble(
    snapshot: TokenSnapshot,
    security: Optional[SecurityReport] = None,
    scorecard: Optional[ScoreCard] = None,
    profile: Optional[TokenProfile] = None,
    mindshare: Optional[MindshareReport] = None,
    providers: Optional[Sequence[str]] = None,
    clients: Optional[Dict[str, Any]] = None,
    blend_weight: float = config.ENSEMBLE_BLEND_WEIGHT,
    timeout: Optional[float] = None,
) -> EnsembleResult:
    """Send one payload to every configured model in parallel and combine.

    All providers run concurrently, so the wall-clock cost is the slowest model
    rather than their sum. A provider that times out is recorded as a failed
    verdict and the others still count -- an ensemble that needs all three to
    answer is just three single points of failure.
    """
    started = time.monotonic()
    analyzers, skipped = build_analyzers(providers, clients)
    result = EnsembleResult(
        requested=[a.provider for a in analyzers],
        skipped=skipped,
        blend_weight=clamp(blend_weight, 0.0, 1.0),
    )

    if not analyzers:
        result.notes.append(
            "No LLM providers are configured. Add an API key to .env "
            "(ANTHROPIC_API_KEY / OPENAI_API_KEY / XAI_API_KEY) to enable the ensemble."
        )
        result.elapsed_ms = int((time.monotonic() - started) * 1000)
        return result

    payload = build_analysis_payload(snapshot, security, scorecard, profile, mindshare)
    wall_clock = timeout if timeout is not None else config.LLM_TIMEOUT_SECONDS + 15

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(analyzers))
    try:
        futures = {executor.submit(analyzer.analyze, payload): analyzer for analyzer in analyzers}
        done, pending = concurrent.futures.wait(futures, timeout=wall_clock)

        for future in futures:
            analyzer = futures[future]
            if future in done:
                try:
                    result.verdicts.append(future.result())
                except Exception as exc:  # noqa: BLE001 - defensive; analyze() catches its own
                    result.verdicts.append(
                        LLMVerdict(provider=analyzer.provider, model=analyzer.model, ok=False,
                                   error=f"{type(exc).__name__}: {exc}"[:300])
                    )
            else:
                future.cancel()
                result.verdicts.append(
                    LLMVerdict(provider=analyzer.provider, model=analyzer.model, ok=False,
                               error=f"Timed out after {wall_clock:.0f}s.")
                )
    finally:
        # Don't block on a hung provider thread: the verdict is already recorded.
        executor.shutdown(wait=False, cancel_futures=True)

    # Preserve the caller's provider order so the UI is stable run to run.
    order = {analyzer.provider: index for index, analyzer in enumerate(analyzers)}
    result.verdicts.sort(key=lambda v: order.get(v.provider, 99))

    result.consensus = build_consensus(result.verdicts)
    if result.consensus is not None:
        blended, decision, notes = blend_with_deterministic(scorecard, result.consensus, result.blend_weight)
        result.blended_score = blended
        result.blended_decision = decision
        result.notes.extend(notes)

    for verdict in result.failed:
        result.notes.append(f"{verdict.label} failed: {verdict.error}")
    for provider, reason in skipped.items():
        result.notes.append(f"{provider} skipped: {reason}")

    result.elapsed_ms = int((time.monotonic() - started) * 1000)
    return result


def ensemble_status_line(providers: Optional[Sequence[str]] = None) -> str:
    """One-line summary of which models would run - for the sidebar."""
    ready, skipped = build_analyzers(providers)
    if not ready:
        return "No LLM providers configured (add keys to .env)"
    names = ", ".join(f"{a.label} ({a.model})" for a in ready)
    suffix = f" · {len(skipped)} unavailable" if skipped else ""
    return f"{len(ready)} model(s) ready: {names}{suffix}"
