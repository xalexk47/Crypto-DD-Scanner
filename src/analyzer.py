"""Orchestration: raw addresses in, full :class:`AnalysisResult` objects out.

This is the single entry point the UI (or a script, or a future API) calls.
It sequences the fetchers, the scoring engine, the narrative layer and the
risk calculator, and guarantees a result object even when everything upstream
is on fire.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Sequence, Tuple

from . import config, data_fetchers, llm, llm_analyzers, mindshare as mindshare_mod, scorers
from .models import AnalysisResult, ScanCandidate, TokenSnapshot
from .utils import normalize_address, utcnow_iso

logger = logging.getLogger(__name__)

MAX_PARALLEL_ANALYSES = 4


def analyze_token(
    address: str,
    settings: Optional[config.AppSettings] = None,
    use_cache: bool = True,
    snapshot: Optional[TokenSnapshot] = None,
) -> AnalysisResult:
    """Run the full due-diligence pipeline for one contract address.

    ``snapshot`` lets Scanner mode hand over market data it already fetched,
    saving a round trip when the user drills into a row.
    """
    settings = settings or config.AppSettings()
    result = AnalysisResult(address=address, chain=settings.chain, analyzed_at=utcnow_iso())

    # --- 1. market data --------------------------------------------------
    warnings: List[str] = []
    if snapshot is None:
        snapshot, warnings = data_fetchers.fetch_token_snapshot(address, settings.chain, use_cache=use_cache)
    if snapshot is None:
        result.ok = False
        result.error = warnings[0] if warnings else "No market data found for this address."
        result.data_warnings = warnings
        return result
    result.snapshot = snapshot
    result.chain = snapshot.chain or settings.chain
    result.data_warnings = list(warnings)

    # --- 1b. DexScreener token profile ------------------------------------
    # Fills in the project description and any socials the pair data lacks,
    # which materially improves both the narrative and the LLM payload.
    try:
        profile = data_fetchers.fetch_token_profile(address, result.chain, use_cache=use_cache)
    except Exception as exc:  # noqa: BLE001 - enrichment is strictly optional
        logger.info("Token profile lookup failed for %s: %s", address, exc)
        profile = None
    if profile is not None:
        result.profile = profile
        data_fetchers.enrich_snapshot_with_profile(snapshot, profile)

    # --- 2. security -----------------------------------------------------
    security = data_fetchers.fetch_security_report(address, result.chain, use_cache=use_cache)
    result.security = security
    if not security.available and security.error:
        result.data_warnings.append(security.error)
    if security.holder_count and snapshot.holders is None:
        snapshot.holders = security.holder_count
    if security.total_supply and snapshot.total_supply is None:
        snapshot.total_supply = security.total_supply

    # --- 2b. X / Twitter mindshare via Grok -------------------------------
    # Runs before scoring so the momentum pillar can use real social data
    # instead of only the volume/price proxy.
    if settings.use_x_search:
        report = mindshare_mod.fetch_mindshare(snapshot, use_cache=use_cache)
        result.mindshare = report
        if not report.available and report.error:
            result.data_warnings.append(f"X mindshare: {report.error}")

    # --- 3. narrative (heuristic today, LLM-ready) -----------------------
    provider_name = None if settings.use_llm else "heuristic"
    result.narrative = llm.analyze_narrative(snapshot, security, provider_name=provider_name)

    # --- 4. score --------------------------------------------------------
    scorecard = scorers.build_scorecard(
        snapshot, security, result.narrative, weights=settings.weights, mindshare=result.mindshare,
    )
    result.scorecard = scorecard

    # --- 5. position sizing ---------------------------------------------
    result.risk_plan = scorers.build_risk_plan(
        snapshot,
        scorecard,
        portfolio_usd=settings.portfolio_usd,
        risk_profile_key=settings.risk_profile,
        security=security,
    )

    # --- 6. multi-LLM ensemble (optional) ---------------------------------
    # Runs last so the models can see the deterministic verdict and argue with
    # it. The rules-engine score above is never overwritten -- the blended
    # score lives alongside it in result.ensemble.
    if settings.use_ensemble:
        result.ensemble = llm_analyzers.run_ensemble(
            snapshot,
            security,
            scorecard,
            profile=result.profile,
            mindshare=result.mindshare,
            providers=settings.ensemble_providers,
            blend_weight=settings.blend_weight,
        )
        for note in result.ensemble.notes:
            logger.info("ensemble: %s", note)

    return result


def analyze_many(
    addresses: Sequence[str],
    settings: Optional[config.AppSettings] = None,
    use_cache: bool = True,
) -> List[AnalysisResult]:
    """Analyze several addresses concurrently, preserving input order.

    Concurrency is capped deliberately: the upstream APIs are free and we would
    rather be a good citizen than get rate limited mid-scan.
    """
    settings = settings or config.AppSettings()
    # Dedupe case-insensitively: the same EVM token pasted in checksummed and
    # lower-cased form is one token, not two.
    seen = set()
    unique = []
    for address in addresses:
        key = normalize_address(address)
        if key and key not in seen:
            seen.add(key)
            unique.append(address)
    if not unique:
        return []
    if len(unique) == 1:
        return [analyze_token(unique[0], settings, use_cache=use_cache)]

    workers = min(MAX_PARALLEL_ANALYSES, len(unique))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {addr: pool.submit(analyze_token, addr, settings, use_cache) for addr in unique}
    results = []
    for addr, future in futures.items():
        try:
            results.append(future.result())
        except Exception as exc:  # noqa: BLE001 - one bad token must not kill the batch
            logger.exception("Analysis failed for %s", addr)
            results.append(
                AnalysisResult(
                    address=addr, chain=settings.chain, ok=False,
                    error=f"Unexpected error during analysis: {exc}", analyzed_at=utcnow_iso(),
                )
            )
    return results


# --------------------------------------------------------------------------
# Scanner
# --------------------------------------------------------------------------
def passes_filters(snapshot: TokenSnapshot, filters: config.ScannerFilters) -> bool:
    """Apply the user's scanner filters to a candidate snapshot."""
    if snapshot.market_cap <= 0:
        return False
    if not (filters.min_market_cap <= snapshot.market_cap <= filters.max_market_cap):
        return False
    if snapshot.liquidity_usd < filters.min_liquidity:
        return False
    if snapshot.volume_24h < filters.min_volume_24h:
        return False
    if snapshot.txns_24h < filters.min_txns_24h:
        return False
    if filters.exclude_no_socials and not snapshot.has_socials:
        return False

    age = snapshot.age_hours
    if age is not None:
        if age < filters.min_age_hours:
            return False
        if age > filters.max_age_days * 24:
            return False
    return True


def scan(
    filters: Optional[config.ScannerFilters] = None,
    use_cache: bool = True,
) -> Tuple[List[ScanCandidate], List[str]]:
    """Discover, filter and rank candidate tokens for one chain.

    Returns ``(candidates, warnings)``.  Ranking uses
    :func:`src.scorers.quick_score` (market data only) so a single scan does
    not fire hundreds of security-API calls; the full score runs when the user
    opens a token.
    """
    filters = filters or config.ScannerFilters()
    pairs, warnings = data_fetchers.discover_pairs(filters.chain, use_cache=use_cache)
    if not pairs:
        return [], warnings or ["No pairs discovered for this chain."]

    snapshots = data_fetchers.collapse_pairs_to_tokens(pairs)
    scanned_total = len(snapshots)

    candidates: List[ScanCandidate] = []
    for snapshot in snapshots:
        if not passes_filters(snapshot, filters):
            continue
        score, reasons = scorers.quick_score(snapshot)
        candidates.append(ScanCandidate(snapshot=snapshot, quick_score=score, reasons=reasons))

    candidates.sort(key=lambda c: c.quick_score, reverse=True)
    warnings.append(
        f"Screened {scanned_total} tokens from DexScreener; {len(candidates)} passed your filters."
    )
    return candidates[: filters.max_results], warnings
