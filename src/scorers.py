"""Scoring engine and risk calculator.

The composite score is a weighted blend of six pillars:

===================  ======  =====================================================
Pillar               Weight  What it measures
===================  ======  =====================================================
Security               30%   Honeypot / mint / owner powers / LP lock / taxes
Liquidity Quality      20%   Absolute depth, liquidity-to-mcap ratio, turnover
Holder Distribution    15%   Non-LP top-10 concentration, deployer bag, holders
Mindshare / Momentum   15%   Volume turnover, price action, trade count, flow
Narrative Potential    10%   Socials, branding, meme-ability heuristics
Catalyst / Listings    10%   Venue spread, boosts, volume acceleration, age
===================  ======  =====================================================

Every pillar returns 0-100 plus a ``confidence`` factor.  Missing inputs pull a
pillar toward a neutral 50 and lower confidence rather than silently scoring
0 or 100 -- an unknown is not a virtue and not a crime.

The functions here are pure: same inputs, same outputs, no network.  That makes
them cheap to unit test and safe to reuse from a backtester.
"""

from __future__ import annotations

import math
import re
from typing import List, Optional, Tuple

from . import config
from .models import (
    ComponentScore,
    MindshareReport,
    NarrativeReport,
    RiskPlan,
    ScoreCard,
    SecurityReport,
    TokenSnapshot,
    WalletFlowReport,
)
from .utils import clamp, log_scale, safe_ratio, scale

# Words that reliably show up in meme tokens that catch on.  Crude on purpose:
# this is a v1 heuristic that the LLM layer is meant to replace.
_MEME_KEYWORDS = {
    "pepe", "wojak", "doge", "shib", "inu", "cat", "dog", "frog", "moon", "elon",
    "chad", "based", "brett", "andy", "bonk", "wif", "hat", "toshi", "degen",
    "mog", "turbo", "banana", "monkey", "ape", "bull", "bear", "milady", "retard",
    "gm", "wagmi", "ai", "agent", "meme", "coin", "baby", "trump", "boden", "pump",
}


# ==========================================================================
# Pillar 1 - Security (30%)
# ==========================================================================
def score_security(security: Optional[SecurityReport]) -> ComponentScore:
    """Deduction-based security score.

    Starts at 100 and subtracts for each dangerous capability found.  A
    confirmed honeypot (or equivalent) drops straight to zero and triggers the
    composite veto.
    """
    reasons: List[str] = []
    weight = config.DEFAULT_WEIGHTS.security

    if security is None or not security.available:
        reason = (security.error if security and security.error else "No security provider data available.")
        return ComponentScore(
            key="security",
            label=config.COMPONENT_LABELS["security"],
            score=40.0,           # unknown is penalised, not neutral
            weight=weight,
            reasons=[reason, "Unverified contracts are scored conservatively."],
            confidence=0.25,
        )

    score = 100.0
    known_fields = 0

    def deduct(amount: float, reason: str) -> None:
        nonlocal score
        score -= amount
        reasons.append(f"-{amount:.0f} {reason}")

    # --- fatal ---------------------------------------------------------
    if security.is_honeypot:
        return ComponentScore(
            key="security", label=config.COMPONENT_LABELS["security"], score=0.0, weight=weight,
            reasons=["Honeypot confirmed by security provider - unsellable."], confidence=1.0,
        )
    if security.cannot_sell_all:
        deduct(45, "cannot sell entire balance")
    if security.selfdestruct:
        deduct(35, "selfdestruct present")
    if security.hidden_owner:
        deduct(35, "hidden owner")

    # --- owner powers --------------------------------------------------
    for flag, penalty, label in (
        (security.is_mintable, 25, "supply is mintable"),
        (security.can_take_back_ownership, 20, "ownership reclaimable"),
        (security.transfer_pausable, 20, "transfers pausable"),
        (security.slippage_modifiable, 15, "tax modifiable post-launch"),
        (security.is_blacklisted, 12, "blacklist function"),
        (security.is_freezable, 20, "freeze authority active"),
        (security.anti_whale_modifiable, 5, "anti-whale limits modifiable"),
        (security.trading_cooldown, 5, "trading cooldown"),
        (security.is_whitelisted, 4, "whitelist function"),
        (security.external_call, 5, "external calls in contract"),
        (security.is_proxy, 8, "upgradeable proxy"),
    ):
        if flag is not None:
            known_fields += 1
            if flag:
                deduct(penalty, label)

    if security.is_open_source is not None:
        known_fields += 1
        if not security.is_open_source:
            deduct(25, "source not verified")
    if security.owner_renounced is not None:
        known_fields += 1
        if not security.owner_renounced and not security.hidden_owner:
            deduct(10, "ownership not renounced")
        elif security.owner_renounced:
            reasons.append("Ownership renounced.")

    # --- taxes ---------------------------------------------------------
    max_tax = security.max_tax_pct
    if max_tax is not None:
        known_fields += 1
        if max_tax >= 25:
            deduct(40, f"punitive tax {max_tax:.1f}%")
        elif max_tax >= 10:
            deduct(22, f"high tax {max_tax:.1f}%")
        elif max_tax > 5:
            deduct(10, f"elevated tax {max_tax:.1f}%")
        elif max_tax <= 1:
            reasons.append("Near-zero trading tax.")

    # --- LP security ---------------------------------------------------
    secured = security.lp_secured_pct
    if secured is None:
        reasons.append("LP lock status unknown - assume it can be pulled.")
        deduct(12, "LP lock status unknown")
    else:
        known_fields += 1
        if secured >= 95:
            reasons.append(f"LP {secured:.0f}% burned/locked.")
        elif secured >= 80:
            deduct(6, f"LP only {secured:.0f}% secured")
        elif secured >= 50:
            deduct(18, f"LP only {secured:.0f}% secured")
        else:
            deduct(30, f"LP largely unsecured ({secured:.0f}%)")

    # Confidence scales with how much the provider actually told us.
    confidence = clamp(0.35 + 0.05 * known_fields, 0.35, 1.0)
    return ComponentScore(
        key="security",
        label=config.COMPONENT_LABELS["security"],
        score=clamp(score),
        weight=weight,
        reasons=reasons or ["No dangerous contract capabilities detected."],
        confidence=confidence,
    )


# ==========================================================================
# Pillar 2 - Liquidity quality (20%)
# ==========================================================================
def score_liquidity(snapshot: TokenSnapshot, security: Optional[SecurityReport] = None) -> ComponentScore:
    """Blend absolute depth, depth-to-mcap ratio and turnover sanity."""
    reasons: List[str] = []
    weight = config.DEFAULT_WEIGHTS.liquidity

    if snapshot.liquidity_usd <= 0:
        return ComponentScore(
            key="liquidity", label=config.COMPONENT_LABELS["liquidity"], score=0.0, weight=weight,
            reasons=["No liquidity reported - the pool is empty or the data is broken."], confidence=0.5,
        )

    # (a) Absolute depth: $15k is thin, $750k+ is deep. Log axis.
    depth_score = log_scale(snapshot.liquidity_usd, 15_000, 750_000)
    reasons.append(f"Pool depth ${snapshot.liquidity_usd:,.0f} -> {depth_score:.0f}/100.")

    # (b) Liquidity / market cap. Below ~3% exits get brutal; above ~35% the
    #     market cap is mostly the pool itself (little real float demand).
    ratio = snapshot.liquidity_ratio
    if ratio <= 0:
        ratio_score = 45.0
        reasons.append("Market cap unknown - liquidity ratio could not be judged.")
    elif ratio < 0.03:
        ratio_score = scale(ratio, 0.0, 0.03, 5, 45)
        reasons.append(f"Liquidity is only {ratio * 100:.1f}% of market cap - exits will slip badly.")
    elif ratio <= 0.12:
        ratio_score = scale(ratio, 0.03, 0.12, 45, 100)
        reasons.append(f"Liquidity is {ratio * 100:.1f}% of market cap - workable.")
    elif ratio <= 0.35:
        ratio_score = 100.0
        reasons.append(f"Liquidity is {ratio * 100:.1f}% of market cap - deep relative to size.")
    else:
        ratio_score = scale(ratio, 0.35, 1.2, 100, 60)
        reasons.append(f"Liquidity is {ratio * 100:.1f}% of market cap - most of the 'cap' is the pool.")

    # (c) Turnover sanity: some churn is healthy, 20x pool/day smells washed.
    turnover = safe_ratio(snapshot.volume_24h, snapshot.liquidity_usd)
    if turnover <= 0:
        turnover_score = 25.0
        reasons.append("No 24h volume against the pool - effectively dead.")
    elif turnover < 0.3:
        turnover_score = scale(turnover, 0.0, 0.3, 25, 70)
        reasons.append(f"Low churn ({turnover:.2f}x pool/day).")
    elif turnover <= 6:
        turnover_score = 100.0
        reasons.append(f"Healthy churn ({turnover:.2f}x pool/day).")
    elif turnover <= 20:
        turnover_score = scale(turnover, 6, 20, 100, 55)
        reasons.append(f"Very high churn ({turnover:.1f}x pool/day) - hot, but crowded.")
    else:
        turnover_score = 35.0
        reasons.append(f"Extreme churn ({turnover:.0f}x pool/day) - possible wash trading.")

    score = depth_score * 0.45 + ratio_score * 0.35 + turnover_score * 0.20

    # LP security is a liquidity-quality question too, not just a contract one.
    secured = security.lp_secured_pct if security else None
    if secured is not None:
        if secured >= 95:
            score = min(100.0, score + 5)
            reasons.append("LP burned/locked - depth cannot simply vanish.")
        elif secured < 50:
            score *= 0.75
            reasons.append("Unsecured LP - this depth can be removed at any moment.")

    if snapshot.pair_count > 1:
        reasons.append(f"Depth spread over {snapshot.pair_count} pools on {snapshot.dex_count} DEX(es).")

    return ComponentScore(
        key="liquidity", label=config.COMPONENT_LABELS["liquidity"], score=clamp(score),
        weight=weight, reasons=reasons, confidence=0.9 if snapshot.market_cap else 0.6,
    )


# ==========================================================================
# Pillar 3 - Holder distribution (15%)
# ==========================================================================
def _apply_wallet_flow(
    score: float,
    confidence: float,
    reasons: List[str],
    wallet_flow: Optional[WalletFlowReport],
) -> Tuple[float, float]:
    """Adjust the holder score by which way wallets are actually moving.

    Kept separate from the concentration logic because the two come from
    different providers: a chain can have wallet flow without GoPlus holder
    data, and dropping the flow in that case would throw away the only holder
    signal available.
    """
    if wallet_flow is None or not wallet_flow.available:
        return score, confidence

    if wallet_flow.quiet_accumulation:
        score += 12
        reasons.append(
            "Wallets accumulating while price consolidates - positioning ahead of a move."
        )
    elif wallet_flow.accumulation_verdict == "accumulating":
        score += 6
        reasons.append(
            f"{wallet_flow.accumulating_wallets} wallets accumulating vs "
            f"{wallet_flow.distributing_wallets} distributing."
        )
    elif wallet_flow.accumulation_verdict == "distributing":
        score -= 10
        reasons.append(
            f"Net distribution: {wallet_flow.distributing_wallets} wallets selling vs "
            f"{wallet_flow.accumulating_wallets} buying."
        )

    hold_rate = wallet_flow.early_hold_rate
    if hold_rate is not None and wallet_flow.early_buyers >= 5:
        if hold_rate >= 0.6:
            score += 5
            reasons.append(f"{hold_rate * 100:.0f}% of early buyers still holding.")
        elif hold_rate <= 0.3:
            score -= 8
            reasons.append(f"Only {hold_rate * 100:.0f}% of early buyers still hold.")

    # A wall of one-and-done wallets is farming, not demand.
    if (wallet_flow.fresh_wallet_ratio or 0) >= 0.8:
        score -= 10
        reasons.append(
            f"{wallet_flow.fresh_wallet_ratio * 100:.0f}% of active wallets bought once and "
            "never traded again - looks like farming or bots."
        )

    # A watchlist hit is the user's own judgement, so it outweighs any
    # heuristic here.
    accumulating_hits = [w for w in wallet_flow.watchlist_hits if w.net_tokens > 0]
    leaving_hits = [w for w in wallet_flow.watchlist_hits if w.net_tokens <= 0]
    if accumulating_hits:
        score += min(15, 7 * len(accumulating_hits))
        reasons.append(
            f"{len(accumulating_hits)} wallet(s) from your smart-money list are accumulating."
        )
    if leaving_hits:
        score -= min(15, 7 * len(leaving_hits))
        reasons.append(
            f"{len(leaving_hits)} wallet(s) from your smart-money list are distributing."
        )

    return score, min(0.95, confidence + 0.05)


def score_holders(
    security: Optional[SecurityReport],
    snapshot: Optional[TokenSnapshot] = None,
    wallet_flow: Optional[WalletFlowReport] = None,
) -> ComponentScore:
    """Concentration risk, plus which direction the holders are moving.

    Concentration is a snapshot; flow is the derivative. A tightly held token
    whose wallets are accumulating is a very different proposition from the
    same token being quietly distributed into.
    """
    reasons: List[str] = []
    weight = config.DEFAULT_WEIGHTS.holders

    if security is None or not security.available or security.top10_pct_adjusted is None:
        # No concentration data. Wallet flow is an independent source, so it
        # still applies -- on a chain with no security provider at all it may
        # be the only holder signal available.
        reasons.append("Holder data unavailable - concentration risk is unknown.")
        score, confidence = _apply_wallet_flow(45.0, 0.25, reasons, wallet_flow)
        return ComponentScore(
            key="holders", label=config.COMPONENT_LABELS["holders"], score=clamp(score),
            weight=weight, reasons=reasons, confidence=confidence,
        )

    top10 = security.top10_pct_adjusted
    # <=12% is excellent for a meme coin, >=55% is a single-wallet time bomb.
    if top10 <= 12:
        conc_score = 100.0
    elif top10 <= 25:
        conc_score = scale(top10, 12, 25, 100, 75)
    elif top10 <= 40:
        conc_score = scale(top10, 25, 40, 75, 45)
    elif top10 <= 55:
        conc_score = scale(top10, 40, 55, 45, 20)
    else:
        conc_score = scale(top10, 55, 90, 20, 0)
    reasons.append(f"Top 10 non-LP wallets hold {top10:.1f}% of supply.")

    score = conc_score

    # Deployer bag: the most common exit-liquidity source.
    if security.creator_percent is not None:
        if security.creator_percent >= 10:
            score -= 25
            reasons.append(f"Deployer holds {security.creator_percent:.1f}% - large overhang.")
        elif security.creator_percent >= 3:
            score -= 10
            reasons.append(f"Deployer holds {security.creator_percent:.1f}%.")
        else:
            reasons.append(f"Deployer holds only {security.creator_percent:.2f}%.")

    # Single-whale check beyond the aggregate.
    biggest = next(
        (h for h in security.top_holders
         if h.percent and not h.is_locked and "burn" not in (h.tag or "").lower()),
        None,
    )
    if biggest is not None and biggest.percent >= 15:
        score -= 12
        reasons.append(f"Single wallet holds {biggest.percent:.1f}%.")

    # Holder count: breadth of ownership.
    holder_count = security.holder_count or (snapshot.holders if snapshot else None)
    if holder_count:
        breadth = log_scale(holder_count, 100, 20_000, 0, 20)
        score += breadth - 10  # centred: 1k holders is roughly neutral
        reasons.append(f"{holder_count:,} holders on record.")
    else:
        reasons.append("Holder count unavailable.")

    confidence = 0.85 if holder_count else 0.65
    score, confidence = _apply_wallet_flow(score, confidence, reasons, wallet_flow)

    return ComponentScore(
        key="holders", label=config.COMPONENT_LABELS["holders"], score=clamp(score),
        weight=weight, reasons=reasons, confidence=confidence,
    )


# ==========================================================================
# Pillar 4 - Mindshare / momentum (15%)
# ==========================================================================
def score_momentum(
    snapshot: TokenSnapshot,
    mindshare: Optional[MindshareReport] = None,
) -> ComponentScore:
    """On-chain momentum, blended with real social mindshare when available.

    Volume and price action are a *proxy* for attention. When Grok has actually
    searched X (``mindshare.is_live``), the real signal replaces part of the
    proxy -- see :data:`src.config.MINDSHARE_WEIGHT_IN_MOMENTUM`.
    """
    reasons: List[str] = []
    weight = config.DEFAULT_WEIGHTS.momentum

    # (a) Turnover: 24h volume vs market cap.  0.5x+ is genuinely hot.
    turnover = snapshot.turnover_24h
    turnover_score = log_scale(max(turnover, 1e-4), 0.01, 1.5)
    reasons.append(f"24h volume is {turnover * 100:.1f}% of market cap.")

    # (b) Price action across timeframes, weighted toward the recent.
    blended_change = (
        snapshot.price_change_1h * 0.4
        + snapshot.price_change_6h * 0.35
        + snapshot.price_change_24h * 0.25
    )
    if blended_change <= -40:
        price_score = 5.0
    elif blended_change < 0:
        price_score = scale(blended_change, -40, 0, 5, 50)
    elif blended_change <= 60:
        price_score = scale(blended_change, 0, 60, 50, 95)
    else:
        # Parabolic moves are momentum, but you are late and buying the wick.
        price_score = clamp(95 - (blended_change - 60) * 0.25, 55, 95)
    reasons.append(
        f"Price {snapshot.price_change_1h:+.1f}% (1h), {snapshot.price_change_6h:+.1f}% (6h), "
        f"{snapshot.price_change_24h:+.1f}% (24h)."
    )
    if blended_change > 60:
        reasons.append("Already extended - entering here means chasing.")

    # (c) Trade count: real participation, not two bots.
    activity_score = log_scale(max(snapshot.txns_24h, 1), 50, 8_000)
    reasons.append(f"{snapshot.txns_24h:,} trades in 24h.")

    # (d) Buy/sell flow balance.
    ratio = snapshot.buy_sell_ratio
    if ratio is None:
        flow_score = 50.0
    else:
        flow_score = scale(ratio, 0.35, 0.65, 10, 95)
        reasons.append(f"Buy share of trades: {ratio * 100:.0f}%.")

    score = (
        turnover_score * 0.35
        + price_score * 0.30
        + activity_score * 0.20
        + flow_score * 0.15
    )

    accel = snapshot.volume_acceleration
    if accel is not None:
        if accel >= 1.6:
            score = min(100.0, score + 6)
            reasons.append(f"Volume accelerating ({accel:.1f}x the 24h run-rate in the last 6h).")
        elif accel <= 0.4:
            score = max(0.0, score - 6)
            reasons.append(f"Volume decaying ({accel:.1f}x the 24h run-rate in the last 6h).")

    # --- (e) real social mindshare, when Grok could see X -----------------
    confidence = 0.8
    from .mindshare import score_from_report      # local import avoids a cycle

    social = score_from_report(mindshare)
    if social is not None and mindshare is not None:
        chain_weight = 1.0 - clamp(config.MINDSHARE_WEIGHT_IN_MOMENTUM, 0.0, 1.0)
        score = score * chain_weight + social * (1.0 - chain_weight)
        source = "live X search" if mindshare.is_live else "model knowledge, not live"
        reasons.append(
            f"X mindshare {social:.0f}/100 ({mindshare.sentiment}, {mindshare.post_volume} volume, "
            f"{source}) blended at {(1 - chain_weight) * 100:.0f}%."
        )
        if mindshare.is_organic is False:
            reasons.append("Discussion reads as coordinated shilling, not organic interest — discounted.")
        for flag in mindshare.red_flags[:2]:
            reasons.append(f"X red flag: {flag}")
        # Live social data raises confidence; stale model knowledge lowers it.
        confidence = 0.9 if mindshare.is_live else 0.7
    elif mindshare is not None and mindshare.error:
        reasons.append(f"X mindshare unavailable ({mindshare.error[:80]}); using on-chain proxies only.")

    return ComponentScore(
        key="momentum", label=config.COMPONENT_LABELS["momentum"], score=clamp(score),
        weight=weight, reasons=reasons, confidence=confidence,
    )


# ==========================================================================
# Pillar 5 - Narrative potential (10%)
# ==========================================================================
def score_narrative(snapshot: TokenSnapshot, narrative: Optional[NarrativeReport] = None) -> ComponentScore:
    """Heuristic meme-ability: presence, branding, ticker quality.

    Intentionally shallow.  ``src/llm.py`` is the seam where a model reads the
    project's socials and replaces this with an actual narrative judgement.
    """
    reasons: List[str] = []
    weight = config.DEFAULT_WEIGHTS.narrative
    score = 35.0  # a nameless token with no presence starts low

    kinds = {s.kind.lower() for s in snapshot.socials}
    if "twitter" in kinds or "x" in kinds:
        score += 18
        reasons.append("Has an X/Twitter presence.")
    else:
        reasons.append("No X/Twitter link - meme coins live or die there.")
    if "telegram" in kinds:
        score += 10
        reasons.append("Active Telegram link.")
    if "website" in kinds:
        score += 8
        reasons.append("Has a website.")
    if "discord" in kinds:
        score += 4
        reasons.append("Has a Discord.")
    if snapshot.image_url:
        score += 5
        reasons.append("Token logo present on DexScreener.")
    if snapshot.description:
        score += 5
        reasons.append("Project description published.")

    # Ticker / name meme-ability.
    text = f"{snapshot.name} {snapshot.symbol}".lower()
    hits = sorted({kw for kw in _MEME_KEYWORDS if kw in text})
    if hits:
        score += min(10, 4 * len(hits))
        reasons.append(f"Recognisable meme themes: {', '.join(hits[:4])}.")

    symbol = (snapshot.symbol or "").strip()
    if symbol and 3 <= len(symbol) <= 6 and re.fullmatch(r"[A-Za-z]+", symbol):
        score += 5
        reasons.append(f"Clean, memorable ticker (${symbol.upper()}).")
    elif len(symbol) > 10:
        score -= 5
        reasons.append("Unwieldy ticker - hard to meme.")

    if snapshot.boosts:
        score += min(8, snapshot.boosts * 0.5)
        reasons.append(f"{snapshot.boosts} active DexScreener boosts (paid promotion).")

    # An LLM narrative, when present, overrides part of the heuristic.
    confidence = 0.45
    if narrative and narrative.source != "heuristic":
        confidence = 0.8
        if narrative.themes:
            reasons.append(f"LLM themes: {', '.join(narrative.themes[:4])}.")
        score = clamp(score * 0.6 + 40 * 0.4 + 5 * len(narrative.bull_case[:3]))

    return ComponentScore(
        key="narrative", label=config.COMPONENT_LABELS["narrative"], score=clamp(score),
        weight=weight, reasons=reasons, confidence=confidence,
    )


# ==========================================================================
# Pillar 6 - Catalyst / listing potential (10%)
# ==========================================================================
def score_catalyst(snapshot: TokenSnapshot, security: Optional[SecurityReport] = None) -> ComponentScore:
    """Room to run: venue spread, size band, freshness, volume acceleration."""
    reasons: List[str] = []
    weight = config.DEFAULT_WEIGHTS.catalyst
    score = 40.0

    # Venue spread - already-everywhere tokens have less listing upside.
    if snapshot.dex_count >= 4:
        score += 6
        reasons.append(f"Trades on {snapshot.dex_count} DEXes - broad access, less listing upside left.")
    elif snapshot.dex_count >= 2:
        score += 12
        reasons.append(f"Trades on {snapshot.dex_count} DEXes - spreading organically.")
    else:
        score += 4
        reasons.append("Single-DEX token - a CEX or second-venue listing would be a real catalyst.")

    # Market-cap band: the classic 'room to 10x' zone for meme coins.
    mcap = snapshot.market_cap
    if mcap <= 0:
        reasons.append("Market cap unknown - upside band cannot be judged.")
    elif mcap < 300_000:
        score += 8
        reasons.append(f"Micro cap ({mcap / 1e3:.0f}k) - maximum upside, maximum fragility.")
    elif mcap <= 5_000_000:
        score += 18
        reasons.append(f"${mcap / 1e6:.2f}M cap - the sweet spot for 5-10x runs.")
    elif mcap <= 25_000_000:
        score += 10
        reasons.append(f"${mcap / 1e6:.1f}M cap - still room, but the easy multiple is gone.")
    else:
        score -= 5
        reasons.append(f"${mcap / 1e6:.0f}M cap - needs serious inflows to move.")

    # Age: old enough to not be a 5-minute rug, young enough to be undiscovered.
    age = snapshot.age_hours
    if age is None:
        reasons.append("Pair age unknown.")
    elif age < 6:
        score -= 5
        reasons.append(f"Only {age:.1f}h old - unproven, highest rug window.")
    elif age <= 24 * 14:
        score += 12
        reasons.append(f"{snapshot.age_label} old - fresh but past the instant-rug window.")
    elif age <= 24 * 90:
        score += 6
        reasons.append(f"{snapshot.age_label} old - established.")
    else:
        reasons.append(f"{snapshot.age_label} old - needs a new narrative to re-rate.")

    accel = snapshot.volume_acceleration
    if accel is not None and accel >= 1.5:
        score += 10
        reasons.append("Volume is accelerating into the last 6h - something is happening now.")

    if snapshot.boosts:
        score += min(6, snapshot.boosts * 0.4)
        reasons.append("Paid promotion active - short-term attention catalyst.")

    if security and security.available and security.owner_renounced and (security.lp_secured_pct or 0) >= 95:
        score += 6
        reasons.append("Renounced + LP secured - the box CEX listing desks tick first.")

    return ComponentScore(
        key="catalyst", label=config.COMPONENT_LABELS["catalyst"], score=clamp(score),
        weight=weight, reasons=reasons, confidence=0.5,
    )


# ==========================================================================
# Composite
# ==========================================================================
def decide(composite: float) -> str:
    """Map a composite score onto the four-way decision."""
    if composite >= config.DECISION_THRESHOLDS["Strong Buy"]:
        return "Strong Buy"
    if composite >= config.DECISION_THRESHOLDS["Buy"]:
        return "Buy"
    if composite >= config.DECISION_THRESHOLDS["Watch"]:
        return "Watch"
    return "Pass"


def build_scorecard(
    snapshot: TokenSnapshot,
    security: Optional[SecurityReport] = None,
    narrative: Optional[NarrativeReport] = None,
    weights: Optional[config.ScoreWeights] = None,
    mindshare: Optional[MindshareReport] = None,
    wallet_flow: Optional[WalletFlowReport] = None,
) -> ScoreCard:
    """Run every pillar and combine into a composite score + decision."""
    weights = weights or config.DEFAULT_WEIGHTS
    weights.validate()
    weight_map = weights.as_dict()

    components = [
        score_security(security),
        score_liquidity(snapshot, security),
        score_holders(security, snapshot, wallet_flow),
        score_momentum(snapshot, mindshare),
        score_narrative(snapshot, narrative),
        score_catalyst(snapshot, security),
    ]
    # Honour caller-supplied weights (the sidebar can tweak them).
    for component in components:
        component.weight = weight_map.get(component.key, component.weight)

    composite = sum(c.weighted for c in components)
    confidence = sum(c.confidence * c.weight for c in components)

    positives: List[str] = []
    risks: List[str] = []
    if security and security.available:
        positives.extend(security.positives)
        risks.extend(security.warnings)

    # Cross-cutting observations the individual pillars can't see.
    if snapshot.liquidity_usd and snapshot.market_cap:
        if snapshot.liquidity_ratio < 0.03:
            risks.append(
                f"Thin float: only {snapshot.liquidity_ratio * 100:.1f}% of the market cap is in the pool."
            )
        elif snapshot.liquidity_ratio >= 0.10:
            positives.append(f"Liquidity is {snapshot.liquidity_ratio * 100:.1f}% of market cap - solid backing.")
    if snapshot.turnover_24h >= 0.5:
        positives.append(f"Turnover of {snapshot.turnover_24h:.2f}x market cap in 24h - real attention.")
    if snapshot.volume_24h < 25_000:
        risks.append("Under $25k of 24h volume - you may not find a bid when you want out.")
    if (snapshot.age_hours or 0) < 24:
        risks.append("Pair is under 24 hours old - the majority of rugs happen in this window.")
    if not snapshot.has_socials:
        risks.append("No socials listed - no community channel to sustain a narrative.")
    if wallet_flow is not None and wallet_flow.available:
        if wallet_flow.quiet_accumulation:
            positives.append(
                "Quiet accumulation: wallets are adding while the price is range-bound."
            )
        if wallet_flow.watchlist_hits:
            for hit in wallet_flow.watchlist_hits:
                label = hit.label or "watchlist wallet"
                if hit.net_tokens > 0:
                    positives.append(f"Smart-money watchlist: {label} is accumulating.")
                else:
                    risks.append(f"Smart-money watchlist: {label} is distributing.")
        risks.extend(wallet_flow.warnings)

    if mindshare is not None and mindshare.available and mindshare.is_live:
        if mindshare.is_organic is False:
            risks.append("X discussion looks coordinated rather than organic.")
        elif mindshare.post_volume in ("high", "viral") and mindshare.sentiment == "bullish":
            positives.append(f"Live X attention is {mindshare.post_volume} and bullish.")
        elif mindshare.post_volume in ("none", "low"):
            risks.append("Almost no one is talking about this on X right now.")
        for flag in mindshare.red_flags:
            risks.append(f"X: {flag}")

    # Hard veto: certain security findings make the score irrelevant.
    vetoed = False
    veto_reason = ""
    decision = decide(composite)
    if security and security.available and security.is_critical:
        vetoed = True
        if security.is_honeypot:
            veto_reason = "Honeypot detected - the position cannot be exited."
        elif security.cannot_sell_all:
            veto_reason = "Contract blocks selling the full balance."
        elif security.hidden_owner:
            veto_reason = "Hidden owner retains full control of the contract."
        else:
            veto_reason = "Contract contains a selfdestruct path."
        decision = "Pass"
        composite = min(composite, 20.0)
        risks.insert(0, veto_reason)

    return ScoreCard(
        composite=round(clamp(composite), 1),
        decision=decision,
        components=components,
        positives=list(dict.fromkeys(positives)),
        risks=list(dict.fromkeys(risks)),
        vetoed=vetoed,
        veto_reason=veto_reason,
        confidence=round(clamp(confidence, 0.0, 1.0), 2),
    )


def quick_score(snapshot: TokenSnapshot) -> Tuple[float, List[str]]:
    """Cheap, security-free score used to rank Scanner rows.

    Scanner mode may evaluate hundreds of tokens; running the security API for
    each would blow the rate limit.  This re-weights the market-data-only
    pillars so the ranking stays meaningful, and the full analysis (with
    security) runs on demand when the user drills in.
    """
    liquidity = score_liquidity(snapshot, None)
    momentum = score_momentum(snapshot)
    narrative = score_narrative(snapshot, None)
    catalyst = score_catalyst(snapshot, None)

    score = (
        liquidity.score * 0.35
        + momentum.score * 0.35
        + narrative.score * 0.12
        + catalyst.score * 0.18
    )

    reasons: List[str] = []
    if snapshot.turnover_24h >= 0.4:
        reasons.append(f"Turnover {snapshot.turnover_24h:.2f}x")
    if snapshot.liquidity_ratio >= 0.08:
        reasons.append(f"Liq/MC {snapshot.liquidity_ratio * 100:.0f}%")
    elif snapshot.liquidity_ratio < 0.03:
        reasons.append("Thin liquidity")
    if snapshot.price_change_24h >= 20:
        reasons.append(f"+{snapshot.price_change_24h:.0f}% 24h")
    elif snapshot.price_change_24h <= -20:
        reasons.append(f"{snapshot.price_change_24h:.0f}% 24h")
    accel = snapshot.volume_acceleration
    if accel is not None and accel >= 1.5:
        reasons.append("Volume accelerating")
    if (snapshot.age_hours or 999) < 24:
        reasons.append("Under 24h old")
    if not snapshot.has_socials:
        reasons.append("No socials")
    return round(clamp(score), 1), reasons


# ==========================================================================
# Risk management
# ==========================================================================
def _volatility_proxy(snapshot: TokenSnapshot) -> float:
    """Rough daily volatility (%) from the available price-change windows."""
    samples = [
        abs(snapshot.price_change_1h) * math.sqrt(24),
        abs(snapshot.price_change_6h) * math.sqrt(4),
        abs(snapshot.price_change_24h),
    ]
    samples = [s for s in samples if s > 0]
    if not samples:
        return 35.0
    return clamp(sum(samples) / len(samples), 8.0, 200.0)


def build_risk_plan(
    snapshot: TokenSnapshot,
    scorecard: ScoreCard,
    portfolio_usd: float,
    risk_profile_key: str = config.DEFAULT_RISK_PROFILE,
    security: Optional[SecurityReport] = None,
) -> RiskPlan:
    """Volatility-aware position sizing with a hard liquidity cap.

    Method:

    1. Pick a stop distance wide enough to survive normal noise for this
       token's realised volatility, clamped by the risk profile.
    2. Size so that being stopped out costs exactly ``risk_per_trade_pct`` of
       the portfolio.
    3. Scale by conviction (composite score) and cap by the profile's max
       position size.
    4. Cap again so the position never exceeds a small share of pool
       liquidity -- because a size you cannot exit is not a position, it's a
       donation.
    """
    profile = config.RISK_PROFILES.get(risk_profile_key, config.RISK_PROFILES[config.DEFAULT_RISK_PROFILE])
    portfolio_usd = max(0.0, float(portfolio_usd or 0.0))
    warnings: List[str] = []
    notes: List[str] = []

    # --- 1. stop distance ---------------------------------------------
    volatility = _volatility_proxy(snapshot)
    stop_pct = clamp(max(profile.base_stop_pct, volatility * 0.6), 12.0, 65.0)
    notes.append(
        f"Realised volatility proxy ~{volatility:.0f}%/day -> {stop_pct:.0f}% stop "
        f"(floor {profile.base_stop_pct:.0f}% from the {profile.label} profile)."
    )

    # --- 2. risk-based size -------------------------------------------
    risk_budget_usd = portfolio_usd * profile.risk_per_trade_pct / 100.0
    raw_position_usd = risk_budget_usd / (stop_pct / 100.0) if stop_pct else 0.0

    # --- 3. conviction & profile caps ---------------------------------
    # Conviction can only shrink the bet, never grow it: the profile's
    # risk-per-trade is a hard ceiling, so a score of 80+ earns the full
    # budget and anything weaker is sized down (40 -> 0.5x, 20 -> 0.25x).
    conviction = clamp(clamp(scorecard.composite, 0, 100) / 80.0, 0.25, 1.0)
    position_usd = raw_position_usd * conviction
    notes.append(f"Conviction multiplier {conviction:.2f}x from a composite score of {scorecard.composite:.0f}.")

    max_position_usd = portfolio_usd * profile.max_position_pct / 100.0
    if position_usd > max_position_usd:
        position_usd = max_position_usd
        notes.append(f"Capped at the {profile.label} maximum of {profile.max_position_pct:.1f}% of portfolio.")

    # --- 4. liquidity cap ---------------------------------------------
    liquidity_capped = False
    liquidity_cap_usd = snapshot.liquidity_usd * profile.max_liquidity_share_pct / 100.0
    if snapshot.liquidity_usd > 0 and position_usd > liquidity_cap_usd:
        warnings.append(
            f"Liquidity-adjusted: size cut from {_usd(position_usd)} to {_usd(liquidity_cap_usd)} so the "
            f"position stays under {profile.max_liquidity_share_pct:.2f}% of the ${snapshot.liquidity_usd:,.0f} pool."
        )
        position_usd = liquidity_cap_usd
        liquidity_capped = True

    # Sub-$50 positions are not worth the gas/spread on most chains.
    if 0 < position_usd < 50:
        warnings.append(
            f"Risk-appropriate size is only {_usd(position_usd)} - below a sensible minimum. "
            "Either skip this or accept sizing above your risk rules."
        )

    # --- derived numbers ----------------------------------------------
    position_pct = (position_usd / portfolio_usd * 100.0) if portfolio_usd else 0.0
    max_loss_usd = position_usd * stop_pct / 100.0
    stop_price = snapshot.price_usd * (1 - stop_pct / 100.0) if snapshot.price_usd else None
    liquidity_share = (position_usd / snapshot.liquidity_usd * 100.0) if snapshot.liquidity_usd else 0.0

    # Constant-product price impact for a swap of size x into a pool of depth L:
    # impact ~ x / (L/2 + x). Approximate, but the right order of magnitude.
    half_pool = snapshot.liquidity_usd / 2.0
    est_slippage = (position_usd / (half_pool + position_usd) * 100.0) if half_pool > 0 else 0.0
    if est_slippage >= 3:
        warnings.append(
            f"Estimated ~{est_slippage:.1f}% price impact on entry (and again on exit) at this size."
        )

    # Take-profit ladder scaled to the stop distance (asymmetric by design:
    # meme trades need >2R winners to survive their hit rate).
    targets = []
    for label, multiple, portion in (("TP1", 1.5, 33), ("TP2", 3.0, 33), ("TP3 (moon bag)", 6.0, 34)):
        gain_pct = stop_pct * multiple
        targets.append({
            "label": label,
            "gain_pct": round(gain_pct, 1),
            "r_multiple": multiple,
            "price": round(snapshot.price_usd * (1 + gain_pct / 100.0), 12) if snapshot.price_usd else None,
            "sell_portion_pct": portion,
            "profit_usd": round(position_usd * gain_pct / 100.0 * portion / 100.0, 2),
        })

    if scorecard.vetoed:
        warnings.insert(0, "Security veto in force - the correct position size is zero.")
        position_usd = 0.0
        position_pct = 0.0
        max_loss_usd = 0.0
    if security and security.available and (security.max_tax_pct or 0) >= 5:
        notes.append(
            f"Round-trip tax of ~{(security.buy_tax_pct or 0) + (security.sell_tax_pct or 0):.1f}% "
            "is a guaranteed loss before price even moves."
        )

    return RiskPlan(
        portfolio_usd=portfolio_usd,
        risk_profile=profile.label,
        position_pct=round(position_pct, 3),
        position_usd=round(position_usd, 2),
        stop_loss_pct=round(stop_pct, 1),
        stop_price=stop_price,
        max_loss_usd=round(max_loss_usd, 2),
        max_loss_pct_of_portfolio=round((max_loss_usd / portfolio_usd * 100.0) if portfolio_usd else 0.0, 3),
        take_profit_targets=targets,
        liquidity_capped=liquidity_capped,
        liquidity_share_pct=round(liquidity_share, 3),
        est_slippage_pct=round(est_slippage, 2),
        warnings=warnings,
        notes=notes,
    )


def _usd(value: float) -> str:
    """Local formatter kept private so scorers.py has no UI dependency."""
    return f"${value:,.2f}"
