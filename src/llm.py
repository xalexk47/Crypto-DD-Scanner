"""Narrative / lore layer -- and the seam where LLMs plug in.

Today this module ships one working provider, :class:`HeuristicProvider`, which
builds a narrative summary from on-chain and DexScreener metadata alone.  It
needs no API key, so the app is fully functional out of the box.

To add real LLM analysis (Claude / GPT / Grok), implement
:class:`NarrativeProvider` and register it in :data:`PROVIDERS`.  The rest of
the app only ever sees a :class:`~src.models.NarrativeReport`, so nothing
downstream changes.  Scaffolding for each vendor is stubbed below with the
exact call each SDK needs -- uncomment, add the key to ``.env``, done.
"""

from __future__ import annotations

import json
import logging
from typing import Dict, List, Optional, Protocol

from . import config
from .models import NarrativeReport, SecurityReport, TokenSnapshot

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Prompt construction (shared by every LLM provider)
# --------------------------------------------------------------------------
NARRATIVE_SYSTEM_PROMPT = """\
You are a crypto meme-coin analyst. You are cynical about hype and precise
about risk. Given structured on-chain data for a token, produce a short
narrative assessment.

Return STRICT JSON with these keys:
  summary        : 2-4 sentences on what this token is and why anyone cares
  themes         : array of 1-5 short narrative tags (e.g. "base ecosystem", "dog meta")
  bull_case      : array of 2-4 concrete reasons it could run
  bear_case      : array of 2-4 concrete reasons it fails
  mindshare_notes: array of 1-3 observations about attention/community

Never invent facts that are not in the data. If information is missing, say so.
Do not give financial advice. Output JSON only, no markdown fences.
"""


def build_narrative_prompt(snapshot: TokenSnapshot, security: Optional[SecurityReport] = None) -> str:
    """Serialize the token context into a compact prompt payload."""
    payload: Dict[str, object] = {
        "name": snapshot.name,
        "symbol": snapshot.symbol,
        "chain": snapshot.chain,
        "address": snapshot.address,
        "description": snapshot.description or None,
        "age": snapshot.age_label,
        "price_usd": snapshot.price_usd,
        "market_cap_usd": snapshot.market_cap,
        "fdv_usd": snapshot.fdv,
        "liquidity_usd": snapshot.liquidity_usd,
        "volume_24h_usd": snapshot.volume_24h,
        "turnover_24h": round(snapshot.turnover_24h, 3),
        "price_change_pct": {
            "1h": snapshot.price_change_1h,
            "6h": snapshot.price_change_6h,
            "24h": snapshot.price_change_24h,
        },
        "txns_24h": snapshot.txns_24h,
        "buy_share_of_trades": snapshot.buy_sell_ratio,
        "dex_count": snapshot.dex_count,
        "boosts": snapshot.boosts,
        "socials": [{"kind": s.kind, "url": s.url} for s in snapshot.socials],
    }
    if security and security.available:
        payload["security"] = {
            "honeypot": security.is_honeypot,
            "buy_tax_pct": security.buy_tax_pct,
            "sell_tax_pct": security.sell_tax_pct,
            "mintable": security.is_mintable,
            "owner_renounced": security.owner_renounced,
            "lp_secured_pct": security.lp_secured_pct,
            "top10_non_lp_pct": security.top10_pct_adjusted,
            "holder_count": security.holder_count,
            "warnings": security.warnings,
        }
    return json.dumps(payload, indent=2, default=str)


def parse_llm_json(raw: str, model: str, source: str) -> NarrativeReport:
    """Parse a model's JSON reply into a NarrativeReport, tolerating fences."""
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.lstrip().lower().startswith("json"):
            text = text.lstrip()[4:]
        text = text.rsplit("```", 1)[0]
    data = json.loads(text)

    def as_list(key: str) -> List[str]:
        value = data.get(key) or []
        if isinstance(value, str):
            return [value]
        return [str(item) for item in value]

    return NarrativeReport(
        summary=str(data.get("summary", "")).strip(),
        themes=as_list("themes"),
        bull_case=as_list("bull_case"),
        bear_case=as_list("bear_case"),
        mindshare_notes=as_list("mindshare_notes"),
        source=source,
        model=model,
    )


# --------------------------------------------------------------------------
# Provider interface
# --------------------------------------------------------------------------
class NarrativeProvider(Protocol):
    """Anything that can turn token data into a NarrativeReport."""

    name: str

    def available(self) -> bool: ...

    def analyze(self, snapshot: TokenSnapshot, security: Optional[SecurityReport] = None) -> NarrativeReport: ...


# --------------------------------------------------------------------------
# Default provider: deterministic heuristics, zero API keys
# --------------------------------------------------------------------------
class HeuristicProvider:
    """Builds a readable narrative from metadata only.

    This is the placeholder the spec calls for: real prose, real signal, no
    model.  It also doubles as the fallback whenever an LLM call fails.
    """

    name = "heuristic"

    def available(self) -> bool:
        return True

    def analyze(self, snapshot: TokenSnapshot, security: Optional[SecurityReport] = None) -> NarrativeReport:
        symbol = snapshot.symbol or "This token"
        themes: List[str] = []
        bull: List[str] = []
        bear: List[str] = []
        mindshare: List[str] = []

        # --- themes -----------------------------------------------------
        chain_label = config.get_chain(snapshot.chain).label
        themes.append(f"{chain_label} ecosystem")
        text = f"{snapshot.name} {snapshot.symbol} {snapshot.description}".lower()
        for keyword, theme in (
            ("dog", "dog meta"), ("inu", "dog meta"), ("doge", "dog meta"),
            ("cat", "cat meta"), ("pepe", "pepe/frog meta"), ("frog", "pepe/frog meta"),
            ("ai", "AI narrative"), ("agent", "AI agent narrative"),
            ("trump", "politics meta"), ("elon", "Elon meta"),
            ("base", "Base-native meme"), ("game", "gaming narrative"),
            ("moon", "classic moon meme"),
        ):
            if keyword in text and theme not in themes:
                themes.append(theme)

        # --- size / stage ----------------------------------------------
        mcap = snapshot.market_cap
        if mcap:
            if mcap < 1_000_000:
                stage = f"a sub-$1M micro cap (${mcap / 1e3:,.0f}k)"
                bull.append("Micro cap - a single wave of attention can re-rate it multiples higher.")
                bear.append("At this size a handful of wallets can move (or destroy) the price.")
            elif mcap <= 5_000_000:
                stage = f"an early-stage ${mcap / 1e6:.2f}M cap"
                bull.append("Sits in the $1-5M band where meme coins have the most historical room to run.")
            elif mcap <= 25_000_000:
                stage = f"a mid-tier ${mcap / 1e6:.1f}M cap"
                bull.append("Past the survival phase with a real holder base.")
                bear.append("The easy multiple is gone - needs fresh inflows or a listing to re-rate.")
            else:
                stage = f"a large ${mcap / 1e6:.0f}M cap"
                bear.append("Large cap for a meme - upside now requires major new capital.")
        else:
            stage = "a token with no reliable market-cap reading"
            bear.append("Market cap could not be read - size and dilution are unclear.")

        # --- attention --------------------------------------------------
        turnover = snapshot.turnover_24h
        if turnover >= 1.0:
            mindshare.append(f"Traded {turnover:.1f}x its market cap in 24h - currently a focal point.")
            bull.append("Extreme turnover means live attention, not a dormant chart.")
        elif turnover >= 0.25:
            mindshare.append(f"Healthy 24h turnover of {turnover:.2f}x market cap.")
        elif turnover > 0:
            mindshare.append(f"Quiet: only {turnover * 100:.1f}% of market cap traded in 24h.")
            bear.append("Low turnover - attention has moved elsewhere.")
        if snapshot.txns_24h:
            mindshare.append(f"{snapshot.txns_24h:,} trades in 24h across {snapshot.pair_count} pool(s).")
        if snapshot.boosts:
            mindshare.append(f"{snapshot.boosts} paid DexScreener boosts active - promotion is being bought.")

        kinds = {s.kind.lower() for s in snapshot.socials}
        if kinds:
            bull.append(f"Community channels present ({', '.join(sorted(kinds))}) to carry a narrative.")
        else:
            bear.append("No socials listed - no place for a community to form or a narrative to spread.")

        # --- security colour -------------------------------------------
        if security and security.available:
            if security.is_critical:
                bear.insert(0, "Critical contract risk detected - the narrative is irrelevant, the exit is blocked.")
            else:
                if security.owner_renounced:
                    bull.append("Ownership renounced - no owner switch left to flip.")
                if (security.lp_secured_pct or 0) >= 95:
                    bull.append("LP burned/locked - the classic rug vector is closed.")
                elif security.lp_secured_pct is not None and security.lp_secured_pct < 50:
                    bear.append("LP is not meaningfully locked - liquidity can be pulled.")
                if security.top10_pct_adjusted is not None and security.top10_pct_adjusted > 40:
                    bear.append(
                        f"Top 10 non-LP wallets control {security.top10_pct_adjusted:.0f}% - concentrated supply overhang."
                    )
        else:
            bear.append("No contract-security data - treat unverified capabilities as present until proven otherwise.")

        # --- summary prose ---------------------------------------------
        social_bit = f" It links {', '.join(sorted(kinds))}." if kinds else " It lists no social channels."
        move_bit = (
            f"Price is {snapshot.price_change_24h:+.1f}% over 24h on "
            f"${snapshot.volume_24h:,.0f} of volume against ${snapshot.liquidity_usd:,.0f} of liquidity."
        )
        age_bit = (
            f", {snapshot.age_label} old" if snapshot.age_hours is not None else " (pair age unknown)"
        )
        summary = (
            f"{symbol} ({snapshot.name or 'unnamed'}) is {stage} on {chain_label}"
            f"{age_bit}.{social_bit} {move_bit} "
            f"Narrative read here is heuristic only - metadata, not community sentiment. "
            f"Enable an LLM provider for a genuine lore and mindshare assessment."
        )
        if snapshot.description:
            summary += f" Project self-description: \"{snapshot.description[:280].strip()}\""

        return NarrativeReport(
            summary=summary,
            themes=themes[:5],
            bull_case=bull[:4],
            bear_case=bear[:4],
            mindshare_notes=mindshare[:3],
            source="heuristic",
            model="rules-v1",
        )


# --------------------------------------------------------------------------
# LLM providers -- scaffolded, key-gated
# --------------------------------------------------------------------------
class AnthropicProvider:
    """Claude-backed narrative analysis.

    Enable with::

        pip install anthropic
        # .env
        ANTHROPIC_API_KEY=sk-ant-...
        MEMEDD_LLM_PROVIDER=anthropic
        MEMEDD_LLM_MODEL=claude-sonnet-4-5      # or any current model id
    """

    name = "anthropic"
    default_model = "claude-sonnet-4-5"

    def __init__(self, api_key: str = "", model: str = "") -> None:
        self.api_key = api_key or config.ANTHROPIC_API_KEY
        self.model = model or config.LLM_MODEL or self.default_model

    def available(self) -> bool:
        if not self.api_key:
            return False
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return False
        return True

    def analyze(self, snapshot: TokenSnapshot, security: Optional[SecurityReport] = None) -> NarrativeReport:
        import anthropic

        client = anthropic.Anthropic(api_key=self.api_key)
        message = client.messages.create(
            model=self.model,
            max_tokens=1200,
            system=NARRATIVE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_narrative_prompt(snapshot, security)}],
        )
        text = "".join(block.text for block in message.content if getattr(block, "type", "") == "text")
        return parse_llm_json(text, self.model, "anthropic")


class OpenAIProvider:
    """GPT-backed narrative analysis.

    Enable with ``pip install openai``, ``OPENAI_API_KEY=...`` and
    ``MEMEDD_LLM_PROVIDER=openai``.
    """

    name = "openai"
    default_model = "gpt-4o-mini"

    def __init__(self, api_key: str = "", model: str = "") -> None:
        self.api_key = api_key or config.OPENAI_API_KEY
        self.model = model or config.LLM_MODEL or self.default_model

    def available(self) -> bool:
        if not self.api_key:
            return False
        try:
            import openai  # noqa: F401
        except ImportError:
            return False
        return True

    def analyze(self, snapshot: TokenSnapshot, security: Optional[SecurityReport] = None) -> NarrativeReport:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": NARRATIVE_SYSTEM_PROMPT},
                {"role": "user", "content": build_narrative_prompt(snapshot, security)},
            ],
            response_format={"type": "json_object"},
        )
        return parse_llm_json(response.choices[0].message.content or "", self.model, "openai")


class XAIProvider:
    """Grok-backed narrative analysis.

    xAI exposes an OpenAI-compatible endpoint, so this reuses the OpenAI SDK
    with a different ``base_url``.  Grok is the most interesting option here
    because of its live X/Twitter access - exactly the mindshare signal the
    heuristic layer cannot see.
    """

    name = "xai"
    default_model = "grok-4"
    base_url = "https://api.x.ai/v1"

    def __init__(self, api_key: str = "", model: str = "") -> None:
        self.api_key = api_key or config.XAI_API_KEY
        self.model = model or config.LLM_MODEL or self.default_model

    def available(self) -> bool:
        if not self.api_key:
            return False
        try:
            import openai  # noqa: F401
        except ImportError:
            return False
        return True

    def analyze(self, snapshot: TokenSnapshot, security: Optional[SecurityReport] = None) -> NarrativeReport:
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": NARRATIVE_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        build_narrative_prompt(snapshot, security)
                        + "\n\nIf you have live access to X/Twitter, factor in current chatter "
                          "about this ticker and note it in mindshare_notes."
                    ),
                },
            ],
        )
        return parse_llm_json(response.choices[0].message.content or "", self.model, "xai")


PROVIDERS: Dict[str, type] = {
    "none": HeuristicProvider,
    "heuristic": HeuristicProvider,
    "anthropic": AnthropicProvider,
    "claude": AnthropicProvider,
    "openai": OpenAIProvider,
    "gpt": OpenAIProvider,
    "xai": XAIProvider,
    "grok": XAIProvider,
}


def get_provider(name: Optional[str] = None) -> NarrativeProvider:
    """Resolve a provider by name, falling back to the heuristic one."""
    key = (name or config.LLM_PROVIDER or "none").lower()
    provider_cls = PROVIDERS.get(key, HeuristicProvider)
    provider = provider_cls()
    if not provider.available():
        if key not in ("none", "heuristic"):
            logger.info("LLM provider '%s' unavailable (missing key or SDK); using heuristics.", key)
        return HeuristicProvider()
    return provider


def analyze_narrative(
    snapshot: TokenSnapshot,
    security: Optional[SecurityReport] = None,
    provider_name: Optional[str] = None,
) -> NarrativeReport:
    """Produce a narrative report, never raising.

    Any provider failure degrades to the heuristic narrative with a note, so a
    flaky LLM can never take the dashboard down.
    """
    provider = get_provider(provider_name)
    try:
        return provider.analyze(snapshot, security)
    except Exception as exc:  # noqa: BLE001 - a bad LLM call must not break the app
        logger.warning("Narrative provider %s failed: %s", getattr(provider, "name", "?"), exc)
        fallback = HeuristicProvider().analyze(snapshot, security)
        fallback.mindshare_notes.append(f"LLM narrative unavailable ({type(exc).__name__}); showing heuristics.")
        return fallback


def llm_status() -> str:
    """One-line description of the active narrative provider, for the sidebar."""
    provider = get_provider()
    if getattr(provider, "name", "") == "heuristic":
        configured = (config.LLM_PROVIDER or "none").lower()
        if configured in ("none", "heuristic"):
            return "Heuristic narrative (no LLM configured)"
        return f"Heuristic fallback ('{configured}' configured but key/SDK missing)"
    return f"LLM narrative via {provider.name} ({getattr(provider, 'model', '?')})"
