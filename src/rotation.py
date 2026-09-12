"""Where liquidity is flowing, and what to do about it.

The question this module answers is not "is this coin good" -- the analyzer
already does that -- but "is this *chain* where the money currently is, and is
my book positioned for where it is going next".

Heat is measured twice per chain, from two independent samples:

* **your bags** -- the tokens you actually hold there, value-weighted
* **the chain** -- a basket of liquid tokens you do *not* hold, discovered live
  from DexScreener, plus DefiLlama's TVL and DEX volume trend

Reporting both is the point. When your Brew positions on BNB are screaming and
the BSC basket is flat, that is an idiosyncratic pump in your tokens; when both
run together, liquidity has genuinely rotated onto the chain. The first says
trim the token, the second says trim the chain -- and the divergence figure is
what separates them.

Nothing in here places a trade. It produces a ranked list and a set of
suggested moves with the arithmetic shown, for you to execute wherever you
trade.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import config, data_fetchers, portfolio_store
from .models import (ChainHeat, PortfolioSnapshot, Position, RotationAction,
                     RotationPlan, TokenSnapshot)
from .utils import TTLCache, clamp, fmt_usd, safe_float, scale, utcnow_iso

logger = logging.getLogger(__name__)

_stats_cache = TTLCache(ttl_seconds=config.CACHE_TTL_CHAIN_STATS)

# Input labels used in ``inputs_used`` / ``missing_inputs``.
INPUT_POSITIONS = "your positions"
INPUT_BASKET = "chain basket (DexScreener)"
INPUT_DEFILLAMA = "chain TVL / DEX volume (DefiLlama)"
INPUT_HISTORY = "stored heat history"


# ==========================================================================
# Component scoring
# ==========================================================================
def _weighted(values: Sequence[Tuple[float, float]]) -> Optional[float]:
    """Weighted mean of ``(value, weight)`` pairs, ignoring zero weights."""
    usable = [(v, w) for v, w in values if w > 0]
    if not usable:
        return None
    total_weight = sum(w for _, w in usable)
    return sum(v * w for v, w in usable) / total_weight if total_weight else None


def score_components(
    samples: Sequence[Tuple[TokenSnapshot, float]],
    liquidity_trend_pct: Optional[float] = None,
) -> Dict[str, float]:
    """Turn weighted token snapshots into 0-100 component scores.

    ``samples`` pairs each snapshot with its weight: position value for your
    own bags, pool liquidity for a chain basket. The mapping ranges are chosen
    so a quiet chain lands near 50 and only genuine acceleration reaches 80+.
    """
    if not samples:
        return {}

    momentum = _weighted([
        # 24h carries the move, 6h says whether it is still going.
        (0.6 * snapshot.price_change_24h + 0.4 * snapshot.price_change_6h, weight)
        for snapshot, weight in samples
    ])
    acceleration = _weighted([
        (snapshot.volume_acceleration, weight)
        for snapshot, weight in samples
        if snapshot.volume_acceleration is not None
    ])
    turnover = _weighted([(snapshot.turnover_24h, weight) for snapshot, weight in samples])
    buy_pressure = _weighted([
        (snapshot.buy_sell_ratio, weight)
        for snapshot, weight in samples
        if snapshot.buy_sell_ratio is not None
    ])

    components: Dict[str, float] = {}
    if momentum is not None:
        components["price_momentum"] = scale(momentum, -25.0, 60.0)
    if acceleration is not None:
        # 1.0 means the last 6h is running at exactly the 24h rate.
        components["volume_acceleration"] = scale(acceleration, 0.4, 2.2)
    if turnover is not None:
        components["turnover"] = scale(turnover, 0.03, 1.2)
    if buy_pressure is not None:
        components["buy_pressure"] = scale(buy_pressure, 0.35, 0.68)
    if liquidity_trend_pct is not None:
        components["liquidity_trend"] = scale(liquidity_trend_pct, -20.0, 25.0)
    return components


def _blend_component_maps(
    portfolio: Dict[str, float],
    market: Dict[str, float],
    share: float,
) -> Dict[str, float]:
    """Merge the two halves component by component.

    A component present on only one side keeps that side's value rather than
    being averaged against a zero it never scored.
    """
    blended: Dict[str, float] = {}
    for key in set(portfolio) | set(market):
        mine = portfolio.get(key)
        theirs = market.get(key)
        if mine is None:
            blended[key] = theirs
        elif theirs is None:
            blended[key] = mine
        else:
            blended[key] = mine * share + theirs * (1 - share)
    return blended


def blend_components(components: Dict[str, float], weights: config.HeatWeights) -> Optional[float]:
    """Weighted blend, renormalized over whichever components are present.

    Renormalizing matters: a chain with no DefiLlama coverage must not score
    lower merely because one input is unavailable -- it scores on what is
    known, and the missing input is reported separately.
    """
    weight_map = weights.as_dict()
    usable = {key: value for key, value in components.items() if key in weight_map}
    if not usable:
        return None
    total_weight = sum(weight_map[key] for key in usable)
    if total_weight <= 0:
        return None
    return clamp(sum(value * weight_map[key] for key, value in usable.items()) / total_weight)


# ==========================================================================
# Chain-wide data
# ==========================================================================
def fetch_defillama_stats(chain: str, use_cache: bool = True) -> Optional[Dict[str, float]]:
    """TVL and DEX volume trend for one chain, or ``None`` when uncovered."""
    slug = config.DEFILLAMA_CHAIN_SLUGS.get(chain)
    if not slug:
        return None

    key = ("llama", slug)
    if use_cache:
        cached = _stats_cache.get(key)
        if cached is not None:
            return cached

    stats: Dict[str, float] = {}
    try:
        chains = data_fetchers._get_json(f"{config.DEFILLAMA_BASE}/v2/chains")
        for entry in chains or []:
            if str(entry.get("name", "")).lower() == slug.lower():
                stats["tvl_usd"] = safe_float(entry.get("tvl"))
                break
    except Exception as exc:  # noqa: BLE001 - a missing input is not a failure
        logger.info("DefiLlama chain TVL unavailable: %s", exc)

    try:
        overview = data_fetchers._get_json(
            f"{config.DEFILLAMA_BASE}/overview/dexs/{slug}",
            params={"excludeTotalDataChart": "true", "excludeTotalDataChartBreakdown": "true"},
        )
        if isinstance(overview, dict):
            stats["dex_volume_24h"] = safe_float(overview.get("total24h"))
            # DefiLlama reports these as percentages already.
            stats["dex_change_1d"] = safe_float(overview.get("change_1d"))
            stats["dex_change_7d"] = safe_float(overview.get("change_7d"))
    except Exception as exc:  # noqa: BLE001
        logger.info("DefiLlama DEX overview unavailable for %s: %s", slug, exc)

    if not stats:
        return None
    if use_cache:
        _stats_cache.set(key, stats)
    return stats


def fetch_chain_basket(chain: str, use_cache: bool = True) -> Tuple[List[TokenSnapshot], List[str]]:
    """A basket of liquid tokens on one chain, discovered live.

    Derived from the same discovery path the scanner uses, so there is no
    hardcoded address list to rot, and chains DefiLlama has never heard of
    (Robinhood Chain) still get a real chain-wide reading.
    """
    key = ("basket", chain)
    if use_cache:
        cached = _stats_cache.get(key)
        if cached is not None:
            return cached, []

    try:
        pairs, warnings = data_fetchers.discover_pairs(chain, use_cache=use_cache)
    except Exception as exc:  # noqa: BLE001
        return [], [f"Chain basket unavailable for {config.get_chain(chain).label}: {exc}"]

    snapshots = [
        snapshot for snapshot in data_fetchers.collapse_pairs_to_tokens(pairs)
        if snapshot.liquidity_usd >= config.BENCHMARK_MIN_LIQUIDITY
    ]
    snapshots.sort(key=lambda s: s.liquidity_usd, reverse=True)
    snapshots = snapshots[: config.BENCHMARK_BASKET_SIZE]
    if use_cache and snapshots:
        _stats_cache.set(key, snapshots)
    return snapshots, warnings


# ==========================================================================
# State machine
# ==========================================================================
def classify_state(
    heat: float,
    history: Sequence[Dict[str, Any]],
) -> Tuple[str, Optional[float], Optional[float]]:
    """Return ``(state, trailing_mean, hours_in_state)``.

    A chain is HOT only when it is high *and* not already rolling over, which
    is what stops the planner trimming into the second day of a decline, and
    COLD only when it is low and not yet turning up -- a chain that has started
    to lift is the interesting one to rotate into.
    """
    recent = _within_lookback(history)
    trailing = (
        sum(safe_float(row.get("heat")) for row in recent) / len(recent) if recent else None
    )

    if trailing is None:
        if heat >= config.HEAT_HOT_THRESHOLD:
            state = "hot"
        elif heat <= config.HEAT_COLD_THRESHOLD:
            state = "cold"
        else:
            state = "heating" if heat >= 50 else "cooling"
        return state, None, None

    trend = heat - trailing
    rising = trend > config.HEAT_TREND_EPSILON
    falling = trend < -config.HEAT_TREND_EPSILON

    if heat >= config.HEAT_HOT_THRESHOLD and not falling:
        state = "hot"
    elif heat <= config.HEAT_COLD_THRESHOLD and not rising:
        state = "cold"
    elif rising:
        state = "heating"
    elif falling:
        state = "cooling"
    elif heat >= config.HEAT_HOT_THRESHOLD:
        state = "hot"
    elif heat <= config.HEAT_COLD_THRESHOLD:
        state = "cold"
    else:
        state = "heating" if heat >= 50 else "cooling"

    return state, trailing, _hours_in_state(state, history)


def _within_lookback(history: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=config.HEAT_TREND_LOOKBACK_HOURS)
    rows: List[Dict[str, Any]] = []
    for row in history:
        moment = _parse_time(row.get("taken_at"))
        if moment is None or moment >= cutoff:
            rows.append(row)
    return rows


def _parse_time(value: Any) -> Optional[datetime]:
    if not value:
        return None
    try:
        moment = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)


def _hours_in_state(state: str, history: Sequence[Dict[str, Any]]) -> Optional[float]:
    """How long the chain has been in this state, walking history backwards."""
    oldest: Optional[datetime] = None
    for row in reversed(list(history)):
        if str(row.get("state")) != state:
            break
        moment = _parse_time(row.get("taken_at"))
        if moment is None:
            break
        oldest = moment
    if oldest is None:
        return None
    return max(0.0, (datetime.now(timezone.utc) - oldest).total_seconds() / 3600.0)


# ==========================================================================
# Heat
# ==========================================================================
def compute_chain_heat(
    chain: str,
    positions: Optional[Sequence[Position]] = None,
    position_changes: Optional[Dict[str, Dict[str, float]]] = None,
    weights: Optional[config.HeatWeights] = None,
    use_cache: bool = True,
    db_path: Optional[Any] = None,
    include_market: bool = True,
) -> ChainHeat:
    """Score one chain, naming every input it did and did not get."""
    weights = weights or config.DEFAULT_HEAT_WEIGHTS
    heat = ChainHeat(chain=chain, taken_at=utcnow_iso())
    changes = position_changes or {}

    # -- your bags ---------------------------------------------------------
    held = [p for p in (positions or []) if p.chain == chain and p.snapshot and p.value_usd > 0]
    portfolio_components: Dict[str, float] = {}
    if held:
        liquidity_trend = _weighted([
            (changes.get(p.key, {}).get("liquidity_delta_pct", 0.0), p.value_usd) for p in held
        ]) if changes else None
        portfolio_components = score_components(
            [(p.snapshot, p.value_usd) for p in held], liquidity_trend_pct=liquidity_trend
        )
        heat.portfolio_heat = blend_components(portfolio_components, weights)
        heat.position_count = len(held)
        heat.inputs_used.append(INPUT_POSITIONS)
    else:
        heat.missing_inputs.append(INPUT_POSITIONS)

    # -- the chain itself --------------------------------------------------
    market_components: Dict[str, float] = {}
    if include_market:
        basket, basket_warnings = fetch_chain_basket(chain, use_cache=use_cache)
        heat.notes.extend(basket_warnings)
        if basket:
            market_components = score_components([(s, s.liquidity_usd) for s in basket])
            heat.basket_size = len(basket)
            heat.inputs_used.append(INPUT_BASKET)
        else:
            heat.missing_inputs.append(INPUT_BASKET)

        llama = fetch_defillama_stats(chain, use_cache=use_cache)
        if llama:
            heat.tvl_usd = llama.get("tvl_usd")
            heat.dex_volume_24h = llama.get("dex_volume_24h")
            trend = llama.get("dex_change_1d")
            if trend is None:
                trend = llama.get("dex_change_7d")
            if trend is not None:
                market_components["chain_tvl_trend"] = scale(trend, -40.0, 60.0)
                heat.inputs_used.append(INPUT_DEFILLAMA)
        else:
            heat.missing_inputs.append(INPUT_DEFILLAMA)
            if chain not in config.DEFILLAMA_CHAIN_SLUGS:
                heat.notes.append(
                    f"DefiLlama does not cover {config.get_chain(chain).label}; heat here is "
                    "measured from DexScreener activity alone."
                )
        heat.market_heat = blend_components(market_components, weights)
    else:
        heat.missing_inputs.extend([INPUT_BASKET, INPUT_DEFILLAMA])

    # -- blend the two halves ----------------------------------------------
    share = clamp(config.HEAT_PORTFOLIO_WEIGHT, 0.0, 1.0)
    if heat.portfolio_heat is not None and heat.market_heat is not None:
        heat.heat = heat.portfolio_heat * share + heat.market_heat * (1 - share)
        heat.components = _blend_component_maps(
            portfolio_components, market_components, share
        )
    elif heat.portfolio_heat is not None:
        heat.heat = heat.portfolio_heat
        heat.components = portfolio_components
    elif heat.market_heat is not None:
        heat.heat = heat.market_heat
        heat.components = market_components
    else:
        heat.heat = 0.0
        heat.confidence = 0.0
        heat.notes.append(
            f"No usable data for {config.get_chain(chain).label} — this is 'unknown', not 'cold'."
        )

    # -- state, from stored history ----------------------------------------
    history = portfolio_store.heat_history(chain, db_path=db_path)
    if history:
        heat.inputs_used.append(INPUT_HISTORY)
        heat.previous_heat = safe_float(history[-1].get("heat"))
    else:
        heat.missing_inputs.append(INPUT_HISTORY)
    heat.state, trailing, hours = classify_state(heat.heat, history)
    heat.hours_in_state = hours
    if trailing is not None:
        heat.previous_heat = trailing

    # Confidence falls with each missing input rather than pretending.
    if heat.confidence > 0:
        penalties = {INPUT_POSITIONS: 0.15, INPUT_BASKET: 0.35,
                     INPUT_DEFILLAMA: 0.1, INPUT_HISTORY: 0.15}
        heat.confidence = clamp(
            1.0 - sum(penalties.get(name, 0.0) for name in heat.missing_inputs), 0.0, 1.0
        )
    return heat


def compute_heats(
    snapshot: Optional[PortfolioSnapshot] = None,
    chains: Optional[Sequence[str]] = None,
    position_changes: Optional[Dict[str, Dict[str, float]]] = None,
    weights: Optional[config.HeatWeights] = None,
    use_cache: bool = True,
    persist: bool = True,
    db_path: Optional[Any] = None,
) -> List[ChainHeat]:
    """Score every rotation chain, hottest first, and store the readings."""
    positions = list(snapshot.positions) if snapshot else []
    targets = list(chains or config.ROTATION_CHAINS)

    heats: List[ChainHeat] = []
    for chain in targets:
        try:
            heat = compute_chain_heat(
                chain, positions=positions, position_changes=position_changes,
                weights=weights, use_cache=use_cache, db_path=db_path,
            )
        except Exception as exc:  # noqa: BLE001 - one bad chain is not a dead tab
            logger.warning("Heat calculation failed for %s: %s", chain, exc)
            heat = ChainHeat(chain=chain, confidence=0.0, taken_at=utcnow_iso(),
                             notes=[f"Heat unavailable: {exc}"])
        heats.append(heat)
        if persist and heat.confidence > 0:
            portfolio_store.record_heat(heat, db_path=db_path)

    heats.sort(key=lambda h: h.heat, reverse=True)
    return heats


# ==========================================================================
# The plan
# ==========================================================================
def _gain_pct(position: Position) -> Optional[float]:
    """Best available read of how far a position is up.

    Cost basis when you have entered one, otherwise the 24h move -- and the
    reason string always says which, so a trim suggestion is never based on a
    number you did not know was being used.
    """
    if position.unrealized_pnl_pct is not None:
        return position.unrealized_pnl_pct
    if position.snapshot:
        return position.snapshot.price_change_24h
    return None


def _gain_source(position: Position) -> str:
    """Name the number a trim is based on, including how the basis was set.

    A trim suggestion should never rest on a figure whose origin is invisible:
    "+180% vs your avg cost (derived, 82% coverage)" can be argued with, while
    a bare "+180%" cannot.
    """
    if position.unrealized_pnl_pct is None:
        return "24h"
    if position.basis_source == "derived":
        coverage = position.basis_coverage_pct
        if coverage is not None and position.basis_is_partial:
            return f"vs your derived avg cost, {coverage:.0f}% coverage"
        return "vs your derived avg cost"
    if position.basis_source == "manual":
        return "vs the avg cost you entered"
    return "vs your avg cost"


def build_rotation_plan(
    snapshot: PortfolioSnapshot,
    heats: Sequence[ChainHeat],
    settings: Optional[config.RotationSettings] = None,
    risk: Optional[config.RiskProfile] = None,
) -> RotationPlan:
    """Turn heat readings plus your book into concrete, sized moves."""
    settings = settings or config.DEFAULT_ROTATION_SETTINGS
    risk = risk or config.RISK_PROFILES[config.DEFAULT_RISK_PROFILE]
    max_position_pct = (
        settings.max_position_pct if settings.max_position_pct is not None
        else risk.max_position_pct
    )

    plan = RotationPlan(
        generated_at=utcnow_iso(),
        portfolio_usd=snapshot.total_usd,
        heats=list(heats),
    )
    if not snapshot.positions:
        plan.notes.append("No positions to act on — sync your wallets first.")
        return plan

    by_chain = {heat.chain: heat for heat in heats}

    # -- 1. what to take off -----------------------------------------------
    for position in snapshot.positions:
        heat = by_chain.get(position.chain)
        allocation = snapshot.allocation_pct(position)
        gain = _gain_pct(position)
        reasons: List[str] = []
        trim_pct = 0.0

        hot_chain = heat is not None and heat.state in ("hot", "heating") and heat.confidence > 0
        if hot_chain and gain is not None and gain >= settings.min_gain_pct_to_trim:
            trim_pct = settings.trim_pct_hot
            state_label = config.HEAT_STATE_LABELS.get(heat.state, heat.state)
            duration = (
                f" for {heat.hours_in_state:.0f}h" if heat.hours_in_state else ""
            )
            reasons.append(
                f"{position.symbol or 'position'} is {gain:+.0f}% ({_gain_source(position)}) "
                f"while {config.get_chain(position.chain).label} reads "
                f"{heat.heat:.0f}/100 {state_label}{duration}"
            )
            divergence = heat.divergence
            if divergence is not None and divergence >= config.HEAT_DIVERGENCE_THRESHOLD:
                reasons.append(
                    f"your holdings there run {divergence:.0f} points hotter than the chain "
                    "basket, so this is your tokens moving rather than the whole chain"
                )

        if allocation > max_position_pct:
            # Trim back to the cap, whichever is larger.
            to_cap = (allocation - max_position_pct) / allocation * 100.0
            if to_cap > trim_pct:
                trim_pct = to_cap
            reasons.append(
                f"position is {allocation:.1f}% of the book against a "
                f"{max_position_pct:.0f}% cap ({risk.label.lower()} profile)"
            )

        if trim_pct <= 0:
            continue

        amount = position.value_usd * trim_pct / 100.0
        warnings: List[str] = []

        # Liquidity guard: never suggest selling more than the pool can take.
        if position.snapshot and position.snapshot.liquidity_usd > 0:
            cap = position.snapshot.liquidity_usd * risk.max_liquidity_share_pct / 100.0
            if amount > cap:
                warnings.append(
                    f"Capped at {fmt_usd(cap)} — {risk.max_liquidity_share_pct:.2f}% of the "
                    f"{fmt_usd(position.snapshot.liquidity_usd)} pool. Selling more in one "
                    "go would move the price against you; scale out instead."
                )
                amount = cap
                trim_pct = amount / position.value_usd * 100.0 if position.value_usd else 0.0

        if amount < settings.min_action_usd:
            continue

        plan.actions.append(RotationAction(
            kind="trim",
            chain=position.chain,
            symbol=position.symbol or position.address[:8],
            address=position.address,
            amount_usd=amount,
            pct_of_position=min(trim_pct, 100.0),
            reason="; ".join(reasons) + ".",
            priority=(heat.heat if heat else 0.0) + allocation,
            warnings=warnings,
        ))

    # -- 2. where it should go ---------------------------------------------
    proceeds = sum(action.amount_usd for action in plan.actions if action.kind == "trim")
    destinations = [
        heat for heat in heats
        # Never rotate into a chain we could not actually read.
        if heat.confidence > 0.3
        and heat.heat < settings.rotate_into_below_heat
        and heat.state in ("cold", "cooling", "heating")
    ]
    # A cold chain that has started to lift is the earliest turn, so it ranks
    # ahead of one still falling at the same heat.
    destinations.sort(key=lambda h: (0 if h.state == "heating" else 1, h.heat))

    if proceeds >= settings.min_action_usd and destinations:
        shares = [0.6, 0.4] if len(destinations) > 1 else [1.0]
        for heat, share in zip(destinations[:2], shares):
            amount = proceeds * share
            if amount < settings.min_action_usd:
                continue
            allocation = snapshot.chain_allocation_pct().get(heat.chain, 0.0)
            turn = (
                "and has started to turn up" if heat.state == "heating"
                else "and is still cooling — scale in rather than all at once"
            )
            # The source is whichever chain is actually funding the move, not
            # simply the first action in the list.
            by_source: Dict[str, float] = {}
            for trim in plan.actions:
                if trim.kind == "trim":
                    by_source[trim.chain] = by_source.get(trim.chain, 0.0) + trim.amount_usd
            source = max(by_source, key=by_source.get) if by_source else ""
            plan.actions.append(RotationAction(
                kind="rotate",
                chain=source,
                dest_chain=heat.chain,
                amount_usd=amount,
                reason=(
                    f"{config.get_chain(heat.chain).label} reads {heat.heat:.0f}/100 "
                    f"({config.HEAT_STATE_LABELS.get(heat.state, heat.state)}) {turn}. "
                    f"You hold {allocation:.1f}% of the book there today."
                ),
                priority=100.0 - heat.heat,
            ))
    elif proceeds >= settings.min_action_usd:
        plan.notes.append(
            "Nothing is cool enough to rotate into right now — every chain we can read is "
            "above the rotation threshold. Holding the proceeds in stables is a position too."
        )

    # -- 3. what to leave alone, and what you are under-exposed to ---------
    for heat in heats:
        divergence = heat.divergence
        if divergence is None or heat.confidence <= 0.3:
            continue
        if divergence <= -config.HEAT_DIVERGENCE_THRESHOLD:
            allocation = snapshot.chain_allocation_pct().get(heat.chain, 0.0)
            plan.notes.append(
                f"{config.get_chain(heat.chain).label} is running {abs(divergence):.0f} points "
                f"hotter than your holdings there ({allocation:.1f}% of the book) — the chain is "
                "moving without you."
            )

    for heat in heats:
        if heat.confidence == 0.0:
            plan.warnings.append(
                f"{config.get_chain(heat.chain).label}: {'; '.join(heat.notes) or 'no data'}"
            )

    plan.actions.sort(key=lambda action: action.priority, reverse=True)
    return plan


def clear_cache() -> None:
    """Drop cached chain statistics (used by the sidebar refresh button)."""
    _stats_cache.clear()
