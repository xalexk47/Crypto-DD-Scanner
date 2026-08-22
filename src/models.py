"""Typed data structures that flow through the pipeline.

    raw API JSON -> TokenSnapshot / SecurityReport -> ScoreCard -> AnalysisResult

Keeping these as plain dataclasses (with ``to_dict``) makes JSON export,
SQLite persistence and unit testing trivial, and gives the future LLM layer a
stable contract to consume.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from .utils import age_hours, fmt_age


# --------------------------------------------------------------------------
# Market data
# --------------------------------------------------------------------------
@dataclass
class SocialLink:
    kind: str          # "twitter", "telegram", "website", "discord", ...
    url: str
    label: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TokenSnapshot:
    """Normalized market view of a token, built from its best DexScreener pair."""

    address: str
    chain: str                       # DexScreener chainId, e.g. "base"
    name: str = ""
    symbol: str = ""
    price_usd: float = 0.0
    market_cap: float = 0.0
    fdv: float = 0.0
    liquidity_usd: float = 0.0
    volume_24h: float = 0.0
    volume_6h: float = 0.0
    volume_1h: float = 0.0
    volume_5m: float = 0.0
    price_change_5m: float = 0.0
    price_change_1h: float = 0.0
    price_change_6h: float = 0.0
    price_change_24h: float = 0.0
    txns_24h_buys: int = 0
    txns_24h_sells: int = 0
    txns_1h_buys: int = 0
    txns_1h_sells: int = 0
    pair_address: str = ""
    pair_created_at: Optional[int] = None    # epoch ms
    dex_id: str = ""
    dex_count: int = 1                        # distinct DEXes this token trades on
    pair_count: int = 1
    quote_symbol: str = ""
    url: str = ""
    image_url: str = ""
    description: str = ""
    socials: List[SocialLink] = field(default_factory=list)
    boosts: int = 0
    holders: Optional[int] = None             # filled in from the security feed
    total_supply: Optional[float] = None
    fetched_at: str = ""
    source: str = "dexscreener"

    # -- derived -----------------------------------------------------------
    @property
    def age_hours(self) -> Optional[float]:
        return age_hours(self.pair_created_at)

    @property
    def age_label(self) -> str:
        return fmt_age(self.pair_created_at)

    @property
    def txns_24h(self) -> int:
        return self.txns_24h_buys + self.txns_24h_sells

    @property
    def buy_sell_ratio(self) -> Optional[float]:
        """Buys as a share of 24h trades (0.5 == balanced). None if no trades."""
        total = self.txns_24h
        if total <= 0:
            return None
        return self.txns_24h_buys / total

    @property
    def turnover_24h(self) -> float:
        """24h volume / market cap - the cleanest single momentum proxy."""
        return self.volume_24h / self.market_cap if self.market_cap else 0.0

    @property
    def liquidity_ratio(self) -> float:
        """Liquidity / market cap.  Healthy meme coins sit roughly 5-30%."""
        return self.liquidity_usd / self.market_cap if self.market_cap else 0.0

    @property
    def volume_acceleration(self) -> Optional[float]:
        """Last 6h annualised to 24h vs actual 24h volume. >1 == heating up."""
        if self.volume_24h <= 0 or self.volume_6h <= 0:
            return None
        return (self.volume_6h * 4.0) / self.volume_24h

    @property
    def has_socials(self) -> bool:
        return bool(self.socials)

    def social_url(self, kind: str) -> Optional[str]:
        for link in self.socials:
            if link.kind.lower() == kind.lower():
                return link.url
        return None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["socials"] = [s.to_dict() for s in self.socials]
        data.update(
            age_hours=self.age_hours,
            age_label=self.age_label,
            txns_24h=self.txns_24h,
            buy_sell_ratio=self.buy_sell_ratio,
            turnover_24h=self.turnover_24h,
            liquidity_ratio=self.liquidity_ratio,
        )
        return data


# --------------------------------------------------------------------------
# Security data
# --------------------------------------------------------------------------
@dataclass
class HolderEntry:
    address: str
    percent: float           # 0-100
    tag: str = ""
    is_contract: bool = False
    is_locked: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SecurityReport:
    """Normalized token-security view (GoPlus today, pluggable tomorrow).

    Tri-state booleans matter here: ``None`` means "the provider didn't tell
    us", which we score differently from a confirmed ``False``.
    """

    address: str
    chain: str
    available: bool = False
    source: str = "goplus"
    error: str = ""

    is_honeypot: Optional[bool] = None
    cannot_sell_all: Optional[bool] = None
    buy_tax_pct: Optional[float] = None
    sell_tax_pct: Optional[float] = None
    transfer_tax_pct: Optional[float] = None
    is_open_source: Optional[bool] = None
    is_proxy: Optional[bool] = None
    is_mintable: Optional[bool] = None
    owner_renounced: Optional[bool] = None
    can_take_back_ownership: Optional[bool] = None
    hidden_owner: Optional[bool] = None
    selfdestruct: Optional[bool] = None
    external_call: Optional[bool] = None
    transfer_pausable: Optional[bool] = None
    is_blacklisted: Optional[bool] = None
    is_whitelisted: Optional[bool] = None
    trading_cooldown: Optional[bool] = None
    anti_whale_modifiable: Optional[bool] = None
    slippage_modifiable: Optional[bool] = None
    is_freezable: Optional[bool] = None       # Solana
    is_in_dex: Optional[bool] = None

    owner_address: str = ""
    creator_address: str = ""
    creator_percent: Optional[float] = None   # 0-100
    owner_percent: Optional[float] = None     # 0-100

    lp_locked_pct: Optional[float] = None     # 0-100, burned counts as locked
    lp_burned_pct: Optional[float] = None     # 0-100
    lp_holder_count: Optional[int] = None

    holder_count: Optional[int] = None
    top_holders: List[HolderEntry] = field(default_factory=list)
    top10_pct: Optional[float] = None                 # raw, includes LP/burn
    top10_pct_adjusted: Optional[float] = None        # excludes LP/burn/locked

    total_supply: Optional[float] = None
    token_name: str = ""
    token_symbol: str = ""

    warnings: List[str] = field(default_factory=list)   # hard/serious problems
    notes: List[str] = field(default_factory=list)      # informational
    positives: List[str] = field(default_factory=list)

    @property
    def lp_secured_pct(self) -> Optional[float]:
        """Share of LP that is burned or locked (the number people care about)."""
        parts = [p for p in (self.lp_locked_pct, self.lp_burned_pct) if p is not None]
        if not parts:
            return None
        return min(100.0, max(parts))

    @property
    def is_critical(self) -> bool:
        """True when we should veto the trade regardless of everything else."""
        return bool(self.is_honeypot or self.cannot_sell_all or self.selfdestruct or self.hidden_owner)

    @property
    def max_tax_pct(self) -> Optional[float]:
        taxes = [t for t in (self.buy_tax_pct, self.sell_tax_pct) if t is not None]
        return max(taxes) if taxes else None

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["top_holders"] = [h.to_dict() for h in self.top_holders]
        data["lp_secured_pct"] = self.lp_secured_pct
        data["is_critical"] = self.is_critical
        return data


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
@dataclass
class ComponentScore:
    """One weighted pillar of the composite score."""

    key: str
    label: str
    score: float             # 0-100
    weight: float            # 0-1
    reasons: List[str] = field(default_factory=list)
    confidence: float = 1.0  # 0-1; lowered when the inputs were missing

    @property
    def weighted(self) -> float:
        return self.score * self.weight

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["weighted"] = self.weighted
        return data


@dataclass
class ScoreCard:
    composite: float
    decision: str
    components: List[ComponentScore] = field(default_factory=list)
    positives: List[str] = field(default_factory=list)
    risks: List[str] = field(default_factory=list)
    vetoed: bool = False
    veto_reason: str = ""
    confidence: float = 1.0

    def component(self, key: str) -> Optional[ComponentScore]:
        return next((c for c in self.components if c.key == key), None)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "composite": self.composite,
            "decision": self.decision,
            "vetoed": self.vetoed,
            "veto_reason": self.veto_reason,
            "confidence": self.confidence,
            "components": [c.to_dict() for c in self.components],
            "positives": self.positives,
            "risks": self.risks,
        }


# --------------------------------------------------------------------------
# Risk management
# --------------------------------------------------------------------------
@dataclass
class RiskPlan:
    """Position-sizing output for one idea."""

    portfolio_usd: float
    risk_profile: str
    position_pct: float          # % of portfolio
    position_usd: float
    stop_loss_pct: float         # distance below entry, %
    stop_price: Optional[float]
    max_loss_usd: float
    max_loss_pct_of_portfolio: float
    take_profit_targets: List[Dict[str, Any]] = field(default_factory=list)
    liquidity_capped: bool = False
    liquidity_share_pct: float = 0.0     # position as % of pool liquidity
    est_slippage_pct: float = 0.0
    warnings: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------
# Narrative (LLM-ready)
# --------------------------------------------------------------------------
@dataclass
class NarrativeReport:
    """Lore / narrative summary.

    v1 fills this with deterministic heuristics; the LLM layer in
    ``src/llm.py`` swaps in a richer version behind the same shape.
    """

    summary: str = ""
    themes: List[str] = field(default_factory=list)
    bull_case: List[str] = field(default_factory=list)
    bear_case: List[str] = field(default_factory=list)
    mindshare_notes: List[str] = field(default_factory=list)
    source: str = "heuristic"     # "heuristic" | "anthropic" | "openai" | "xai"
    model: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------
# Top-level result
# --------------------------------------------------------------------------
@dataclass
class AnalysisResult:
    """Everything the UI needs to render one token report."""

    address: str
    chain: str
    ok: bool = True
    error: str = ""
    snapshot: Optional[TokenSnapshot] = None
    security: Optional[SecurityReport] = None
    scorecard: Optional[ScoreCard] = None
    risk_plan: Optional[RiskPlan] = None
    narrative: Optional[NarrativeReport] = None
    data_warnings: List[str] = field(default_factory=list)
    analyzed_at: str = ""

    @property
    def display_name(self) -> str:
        if self.snapshot and (self.snapshot.symbol or self.snapshot.name):
            return f"{self.snapshot.symbol or '?'} - {self.snapshot.name or 'Unknown'}"
        return self.address

    @property
    def composite(self) -> float:
        return self.scorecard.composite if self.scorecard else 0.0

    @property
    def decision(self) -> str:
        return self.scorecard.decision if self.scorecard else "Pass"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "address": self.address,
            "chain": self.chain,
            "ok": self.ok,
            "error": self.error,
            "analyzed_at": self.analyzed_at,
            "data_warnings": self.data_warnings,
            "snapshot": self.snapshot.to_dict() if self.snapshot else None,
            "security": self.security.to_dict() if self.security else None,
            "scorecard": self.scorecard.to_dict() if self.scorecard else None,
            "risk_plan": self.risk_plan.to_dict() if self.risk_plan else None,
            "narrative": self.narrative.to_dict() if self.narrative else None,
        }


@dataclass
class ScanCandidate:
    """One row of Scanner mode - a light-weight, pre-analysis view."""

    snapshot: TokenSnapshot
    quick_score: float
    reasons: List[str] = field(default_factory=list)

    def to_row(self) -> Dict[str, Any]:
        s = self.snapshot
        return {
            "Score": round(self.quick_score, 1),
            "Symbol": s.symbol or "?",
            "Name": s.name or "Unknown",
            "Price": s.price_usd,
            "MCap": s.market_cap,
            "Liq": s.liquidity_usd,
            "Vol 24h": s.volume_24h,
            "Vol/MC": round(s.turnover_24h, 2),
            "Liq/MC": round(s.liquidity_ratio, 3),
            "1h %": s.price_change_1h,
            "24h %": s.price_change_24h,
            "Txns 24h": s.txns_24h,
            "Age": s.age_label,
            "Socials": "yes" if s.has_socials else "no",
            "Chain": s.chain,
            "Address": s.address,
            "DEX": s.dex_id,
            "Notes": "; ".join(self.reasons[:3]),
        }
