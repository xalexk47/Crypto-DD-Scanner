"""X / Twitter mindshare via Grok's server-side search tools.

Why this exists
---------------
Every other signal in this app is on-chain or market data. Meme-coin mindshare
is neither: it forms on X, hours before it shows up in volume. Grok is the only
major model with first-party access to X, which is the entire reason it earns a
place in the stack beyond being a third ensemble opinion.

How it talks to xAI
-------------------
xAI **retired** the original Live Search API (``search_parameters``) on
2026-01-12 -- it now returns 410 Gone. The current mechanism is the server-side
Agent Tools API: you put ``{"type": "x_search"}`` in the ``tools`` array and
xAI runs the search loop itself, returning a finished answer. That travels over
the ordinary OpenAI-compatible ``/v1/chat/completions`` endpoint, so the same
``openai`` SDK the ensemble uses drives it.

Because the exact model id and tool parameter casing move faster than this app,
every one of them is configurable (see ``src.config``), and the call degrades
through a chain rather than failing outright:

1. ``x_search`` tool + strict JSON schema  -> live X data, structured
2. ``x_search`` tool, no schema            -> live X data, JSON repaired on parse
3. no tools at all                         -> model knowledge, ``is_live=False``
4. everything failed                       -> ``available=False`` plus the reason

Step 3 is deliberately labelled, never silently blended with live data: a
model's recollection of a ticker is not the same fact as what X is saying now,
and treating them alike would be the most dangerous thing this module could do.

Run ``python scripts/check_grok.py`` to see what your key actually supports.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from . import config
from .models import MindshareReport, TokenSnapshot, XPost
from .utils import TTLCache, clamp, normalize_address, safe_float, safe_int

logger = logging.getLogger(__name__)

_cache = TTLCache(ttl_seconds=config.CACHE_TTL_MINDSHARE)


MINDSHARE_SYSTEM_PROMPT = """\
You are a crypto social-intelligence analyst. You measure attention on X, and \
you are hard to impress: most meme-coin "buzz" is a handful of bots and paid \
callers talking to each other.

Search X for current discussion of the token you are given, then report what \
you actually found.

Rules:
- Report only what the search returned. If you found little or nothing, say so \
and set post_volume to "none" or "low". Never pad a thin result.
- Distinguish organic discussion from coordinated shilling. Repetitive phrasing, \
bursts of near-identical posts, reply-spam under unrelated accounts and \
engagement-farming are NOT mindshare. Set is_organic false and explain in \
red_flags.
- Prefer accounts with real reach and history over new or anonymous ones.
- sentiment_score runs -1.0 (uniformly bearish) to +1.0 (uniformly bullish). \
Use values near 0 for genuinely mixed or absent discussion.
- mindshare_score (0-100) rates how much attention this token commands RIGHT \
NOW relative to other meme coins, not how good the token is.
- Quote real posts in sample_posts, with the author's handle. Do not invent \
posts, handles, engagement numbers or URLs. Omit a field you did not observe.
- Impersonation of a well-known project or person, giveaway scams and \
"presale" bait belong in red_flags.

Return only the JSON object defined by the schema. No prose outside it."""


MINDSHARE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "sentiment": {
            "type": "string",
            "enum": ["bullish", "mixed", "bearish", "quiet"],
            "description": "Overall tone of discussion. Use 'quiet' when there is barely any.",
        },
        "sentiment_score": {"type": "number", "description": "-1.0 bearish to +1.0 bullish."},
        "mindshare_score": {"type": "number", "description": "0-100 attention right now vs other meme coins."},
        "post_volume": {
            "type": "string",
            "enum": ["none", "low", "moderate", "high", "viral"],
            "description": "How much is actually being posted.",
        },
        "trend": {
            "type": "string",
            "enum": ["accelerating", "steady", "fading", "unknown"],
            "description": "Direction of attention over the search window.",
        },
        "is_organic": {
            "type": "boolean",
            "description": "False if the discussion reads as coordinated shilling or bots.",
        },
        "summary": {"type": "string", "description": "2-4 sentences on what X is saying and who is saying it."},
        "themes": {"type": "array", "items": {"type": "string"}, "description": "Up to 5 recurring narrative tags."},
        "notable_accounts": {
            "type": "array", "items": {"type": "string"},
            "description": "Handles with real reach discussing it, without the @.",
        },
        "sample_posts": {
            "type": "array",
            "description": "Real posts observed. Empty array if none were found.",
            "items": {
                "type": "object",
                "properties": {
                    "handle": {"type": "string", "description": "Author handle without the @."},
                    "text": {"type": "string", "description": "The post text, trimmed."},
                    "url": {"type": "string", "description": "Link to the post, or empty string."},
                    "engagement": {"type": "integer", "description": "Likes plus reposts, or 0 if unknown."},
                },
                "required": ["handle", "text", "url", "engagement"],
                "additionalProperties": False,
            },
        },
        "red_flags": {
            "type": "array", "items": {"type": "string"},
            "description": "Scam, impersonation or coordinated-shilling signals. Empty array if none.",
        },
    },
    "required": [
        "sentiment", "sentiment_score", "mindshare_score", "post_volume", "trend",
        "is_organic", "summary", "themes", "notable_accounts", "sample_posts", "red_flags",
    ],
    "additionalProperties": False,
}

_VOLUME_SCORES = {"none": 0.0, "low": 25.0, "moderate": 55.0, "high": 80.0, "viral": 100.0}


def build_query(snapshot: TokenSnapshot) -> str:
    """Search terms for a token: ticker, name and contract address.

    The contract address matters most -- ticker collisions are rampant (every
    chain has three PEPEs), and a post quoting the CA is unambiguous.
    """
    parts: List[str] = []
    if snapshot.symbol:
        parts.append(f"${snapshot.symbol.upper()}")
    if snapshot.name and snapshot.name.lower() != (snapshot.symbol or "").lower():
        parts.append(snapshot.name)
    if snapshot.address:
        parts.append(snapshot.address)
    return " OR ".join(parts) if parts else snapshot.address


def build_prompt(snapshot: TokenSnapshot, window_hours: int, max_posts: int) -> str:
    """The user turn: what to search for and how to disambiguate it."""
    chain_label = config.get_chain(snapshot.chain).label
    return (
        f"Search X for current discussion of this token and report what you find.\n\n"
        f"Token: {snapshot.name or 'unknown'} (${snapshot.symbol or '?'})\n"
        f"Chain: {chain_label}\n"
        f"Contract address: {snapshot.address}\n"
        f"Search window: the last {window_hours} hours.\n\n"
        f"Ticker collisions are common — several unrelated tokens may share "
        f"${snapshot.symbol or '?'}. Posts quoting the contract address above are "
        f"definitely this token; for ticker-only posts, judge from context whether "
        f"they refer to the {chain_label} token, and ignore them if they clearly do not.\n\n"
        f"Include up to {max_posts} representative posts in sample_posts."
    )


# --------------------------------------------------------------------------
# Response parsing
# --------------------------------------------------------------------------
def _posts(raw: Any, limit: int) -> List[XPost]:
    posts: List[XPost] = []
    if not isinstance(raw, (list, tuple)):
        return posts
    for item in raw[:limit]:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        handle = str(item.get("handle") or item.get("author") or "").strip().lstrip("@")
        if not text and not handle:
            continue
        posts.append(
            XPost(
                handle=handle,
                text=text[:400],
                url=str(item.get("url") or "").strip(),
                engagement=safe_int(item.get("engagement")) or None,
                posted_at=str(item.get("posted_at") or "").strip(),
            )
        )
    return posts


def _strings(raw: Any, limit: int = 6) -> List[str]:
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        return []
    out = []
    for item in raw:
        text = str(item).strip().lstrip("@")
        if text and text.lower() not in ("none", "n/a", "null"):
            out.append(text)
    return out[:limit]


def parse_mindshare(
    data: Dict[str, Any],
    query: str,
    model: str,
    is_live: bool,
    max_posts: int = config.X_SEARCH_MAX_POSTS,
) -> MindshareReport:
    """Normalize Grok's JSON into a MindshareReport, clamping every range."""
    sentiment = str(data.get("sentiment") or "unknown").strip().lower()
    if sentiment not in ("bullish", "mixed", "bearish", "quiet"):
        sentiment = "unknown"

    volume = str(data.get("post_volume") or "unknown").strip().lower()
    if volume not in _VOLUME_SCORES:
        volume = "unknown"

    trend = str(data.get("trend") or "unknown").strip().lower()
    if trend not in ("accelerating", "steady", "fading"):
        trend = "unknown"

    mindshare_score = clamp(safe_float(data.get("mindshare_score"), 0.0), 0.0, 100.0)
    # A model that answers 0-10 or 0-1 for a 0-100 field is a common slip; only
    # rescale when the volume band clearly disagrees with a near-zero score.
    if mindshare_score <= 10 and volume in ("high", "viral"):
        mindshare_score = min(100.0, mindshare_score * 10)

    organic = data.get("is_organic")
    report = MindshareReport(
        query=query,
        available=True,
        is_live=is_live,
        source="x_search" if is_live else "model_knowledge",
        model=model,
        sentiment=sentiment,
        sentiment_score=clamp(safe_float(data.get("sentiment_score"), 0.0), -1.0, 1.0),
        mindshare_score=round(mindshare_score, 1),
        post_volume=volume,
        trend=trend,
        is_organic=bool(organic) if isinstance(organic, bool) else None,
        summary=str(data.get("summary") or "").strip(),
        themes=_strings(data.get("themes"), 5),
        notable_accounts=_strings(data.get("notable_accounts"), 8),
        sample_posts=_posts(data.get("sample_posts"), max_posts),
        red_flags=_strings(data.get("red_flags"), 6),
    )
    if not is_live:
        report.warnings.append(
            "Live X search was unavailable; this reflects the model's training data, "
            "not current activity. Do not read it as today's sentiment."
        )
    return report


def score_from_report(report: Optional[MindshareReport]) -> Optional[float]:
    """Collapse a report into a 0-100 social score for the momentum pillar.

    Blends the model's own attention rating with the observed post volume, then
    applies the judgement calls: coordinated shilling is not mindshare, and a
    bearish crowd is not the same as a quiet one.
    """
    if report is None or not report.available:
        return None

    volume_score = _VOLUME_SCORES.get(report.post_volume)
    parts = [p for p in (report.mindshare_score, volume_score) if p is not None]
    if not parts:
        return None
    score = sum(parts) / len(parts)

    # Sentiment tilts it, but attention still counts: a hated token people are
    # talking about has more mindshare than one nobody mentions.
    score *= 1.0 + (report.sentiment_score * 0.25)

    if report.is_organic is False:
        score *= 0.5
    if report.red_flags:
        score *= max(0.4, 1.0 - 0.15 * len(report.red_flags))
    if report.trend == "accelerating":
        score += 8
    elif report.trend == "fading":
        score -= 8
    if not report.is_live:
        # Stale knowledge should not move the score much in either direction.
        score = 50.0 + (score - 50.0) * 0.4

    return round(clamp(score, 0.0, 100.0), 1)


# --------------------------------------------------------------------------
# The Grok client
# --------------------------------------------------------------------------
class GrokMindshareClient:
    """Queries X through Grok's server-side search tool.

    ``client`` may be injected (the tests do this) so no SDK, key or network is
    needed to exercise the logic.
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = "",
        base_url: str = "",
        timeout: Optional[float] = None,
        client: Any = None,
    ) -> None:
        self.api_key = api_key or config.XAI_API_KEY
        self.model = model or config.X_SEARCH_MODEL
        self.base_url = base_url or config.XAI_BASE_URL
        self.timeout = timeout if timeout is not None else config.LLM_TIMEOUT_SECONDS
        self._injected_client = client

    # -- availability ---------------------------------------------------
    def unavailable_reason(self) -> str:
        if self._injected_client is not None:
            return ""
        if not config.X_SEARCH_ENABLED:
            return "X search is disabled (MEMEDD_X_SEARCH=0)."
        if not self.api_key:
            return "No XAI_API_KEY set — X mindshare needs a Grok key."
        try:
            import openai  # noqa: F401
        except ImportError:
            return "`pip install openai` to enable Grok X search."
        return ""

    def available(self) -> bool:
        return not self.unavailable_reason()

    def _client(self) -> Any:
        if self._injected_client is not None:
            return self._injected_client
        from openai import OpenAI

        return OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)

    # -- the tool definition --------------------------------------------
    def search_tool_variants(self, window_hours: int) -> List[Dict[str, Any]]:
        """Candidate x_search tool payloads for the Agent Tools API.

        The documented minimal form is ``{"type": "x_search"}``; the richer form
        adds the date window and a result cap. Richest first, minimal last, so a
        schema change costs us a rejected request rather than the feature.
        """
        now = datetime.now(timezone.utc)
        from_date = (now - timedelta(hours=max(1, window_hours))).strftime("%Y-%m-%d")
        to_date = now.strftime("%Y-%m-%d")
        tool_type = config.X_SEARCH_TOOL_TYPE
        return [
            {
                "type": tool_type,
                "from_date": from_date,
                "to_date": to_date,
                "max_search_results": config.X_SEARCH_MAX_RESULTS,
            },
            {"type": tool_type, "from_date": from_date, "to_date": to_date},
            {"type": tool_type},
        ]

    def _completion(self, messages: List[Dict[str, str]], **extra: Any) -> Any:
        """Ordinary chat completion - used only for the non-live fallback."""
        return self._client().chat.completions.create(
            model=self.model, messages=messages, **extra
        )

    def _responses_call(self, messages: List[Dict[str, str]], **extra: Any) -> Any:
        """Agent Tools API call (/v1/responses), where x_search is available."""
        client = self._client()
        responses = getattr(client, "responses", None)
        if responses is None:      # SDK too old to expose the endpoint
            raise AttributeError("this openai SDK has no .responses endpoint")
        return responses.create(model=self.model, input=messages, **extra)

    @staticmethod
    def _responses_text(response: Any) -> str:
        """Pull the assistant text out of a Responses API result."""
        text = getattr(response, "output_text", None)
        if isinstance(text, str) and text.strip():
            return text
        parts: List[str] = []
        for item in getattr(response, "output", None) or []:
            for block in getattr(item, "content", None) or []:
                value = getattr(block, "text", None)
                if isinstance(value, str):
                    parts.append(value)
        return "".join(parts)

    @staticmethod
    def _content(response: Any) -> str:
        choices = getattr(response, "choices", None) or []
        if not choices:
            return ""
        return getattr(choices[0].message, "content", "") or ""

    @staticmethod
    def _citations(response: Any) -> List[str]:
        """Pull citation URLs out of whichever field the API used."""
        found: List[str] = []
        for holder in (response, getattr(response, "choices", [None])[0] if getattr(response, "choices", None) else None):
            if holder is None:
                continue
            for attr in ("citations", "search_results", "sources", "annotations"):
                value = getattr(holder, attr, None)
                if isinstance(value, (list, tuple)):
                    for item in value:
                        url = item if isinstance(item, str) else (
                            item.get("url") if isinstance(item, dict) else getattr(item, "url", None)
                        )
                        if url:
                            found.append(str(url))
        return list(dict.fromkeys(found))[:20]

    # -- public API -----------------------------------------------------
    def fetch(self, snapshot: TokenSnapshot, window_hours: Optional[int] = None) -> MindshareReport:
        """Fetch X mindshare, degrading through the fallback chain. Never raises."""
        from .llm_analyzers import extract_json      # local import avoids a cycle

        reason = self.unavailable_reason()
        query = build_query(snapshot)
        if reason:
            return MindshareReport(query=query, available=False, error=reason)

        window = window_hours if window_hours is not None else config.X_SEARCH_WINDOW_HOURS
        max_posts = config.X_SEARCH_MAX_POSTS
        messages = [
            {"role": "system", "content": MINDSHARE_SYSTEM_PROMPT},
            {"role": "user", "content": build_prompt(snapshot, window, max_posts)},
        ]
        # Structured-output syntax differs between the two endpoints.
        chat_schema = {
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "x_mindshare", "strict": True, "schema": MINDSHARE_SCHEMA},
            }
        }
        responses_schema = {
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "x_mindshare",
                    "strict": True,
                    "schema": MINDSHARE_SCHEMA,
                }
            }
        }

        # Live X search runs on the Agent Tools API; the non-live fallback runs
        # on chat completions. Tool-shape rejections fail during request
        # validation, before any model runs, so trying a few costs no tokens.
        attempts: List[Any] = []
        variants = self.search_tool_variants(window)
        for variant in variants:
            attempts.append(("x_search", True, True, {"tools": [variant], **responses_schema}))
        for variant in variants:
            attempts.append(("x_search", True, True, {"tools": [variant]}))
        attempts.append(("model_knowledge", False, False, chat_schema))
        attempts.append(("model_knowledge", False, False, {}))

        started = time.monotonic()
        errors: List[str] = []
        for label, is_live, via_responses, kwargs in attempts:
            try:
                if via_responses:
                    response = self._responses_call(messages, **kwargs)
                    text = self._responses_text(response)
                else:
                    response = self._completion(messages, **kwargs)
                    text = self._content(response)
                if not text.strip():
                    raise ValueError("empty response")
                report = parse_mindshare(extract_json(text), query, self.model, is_live, max_posts)
                report.latency_ms = int((time.monotonic() - started) * 1000)
                report.citations = self._citations(response)
                if errors:
                    report.warnings.append(f"Fell back after: {errors[0]}")
                return report
            except Exception as exc:  # noqa: BLE001 - try the next, cheaper mode
                message = f"{type(exc).__name__}: {exc}"[:200]
                errors.append(message)
                logger.info("Grok mindshare attempt (%s) failed: %s", label, message)

        return MindshareReport(
            query=query,
            available=False,
            error=(
                f"Grok X search failed. First error: {errors[0]} "
                "Run `python scripts/check_grok.py` to see what your key supports."
            ),
            model=self.model,
            latency_ms=int((time.monotonic() - started) * 1000),
        )


def fetch_mindshare(
    snapshot: TokenSnapshot,
    client: Any = None,
    use_cache: bool = True,
    window_hours: Optional[int] = None,
) -> MindshareReport:
    """Cached entry point. Live searches are billed, so repeats are cheap."""
    key = ("mindshare", normalize_address(snapshot.address), snapshot.chain, window_hours)
    if use_cache and client is None:
        cached = _cache.get(key)
        if cached is not None:
            return cached

    report = GrokMindshareClient(client=client).fetch(snapshot, window_hours=window_hours)
    if report.available and client is None:
        _cache.set(key, report)
    return report


def clear_cache() -> None:
    _cache.clear()
