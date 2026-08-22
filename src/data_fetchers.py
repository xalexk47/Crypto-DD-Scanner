"""External data access: DexScreener (market) and GoPlus (security).

Design notes
------------
* Every public function degrades gracefully.  A dead upstream produces an
  empty/`available=False` result plus a human-readable warning -- never an
  exception that kills the Streamlit run.
* Responses are normalized into the dataclasses from ``src.models`` right at
  the boundary, so no raw provider JSON leaks into the scoring or UI layers.
* A process-local :class:`~src.utils.TTLCache` fronts every network call, which
  keeps us inside DexScreener's ~300 req/min budget even with an eager UI.

Adding a provider: write a ``fetch_x`` function that returns one of the
normalized dataclasses and wire it into :func:`fetch_security_report` (or
:func:`fetch_token_snapshot`) as a fallback.  Nothing else has to change.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import httpx

from . import config
from .models import HolderEntry, SecurityReport, SocialLink, TokenSnapshot
from .utils import (
    TTLCache,
    chunked,
    dig,
    is_burn_address,
    looks_locked,
    normalize_address,
    safe_bool,
    safe_float,
    safe_int,
    utcnow_iso,
)

logger = logging.getLogger(__name__)

# One cache per upstream so TTLs can differ.
_dex_cache = TTLCache(ttl_seconds=config.CACHE_TTL_TOKEN)
_scan_cache = TTLCache(ttl_seconds=config.CACHE_TTL_SCANNER)
_sec_cache = TTLCache(ttl_seconds=config.CACHE_TTL_SECURITY)

_client_lock = threading.Lock()
_client: Optional[httpx.Client] = None


class FetchError(Exception):
    """Raised internally when an upstream call fails after all retries."""


# --------------------------------------------------------------------------
# HTTP plumbing
# --------------------------------------------------------------------------
def get_client() -> httpx.Client:
    """Lazily create a shared, connection-pooled HTTP client."""
    global _client
    with _client_lock:
        if _client is None or _client.is_closed:
            _client = httpx.Client(
                timeout=httpx.Timeout(config.HTTP_TIMEOUT_SECONDS),
                headers={"User-Agent": config.HTTP_USER_AGENT, "Accept": "application/json"},
                follow_redirects=True,
            )
        return _client


def close_client() -> None:
    """Close the shared client (useful in tests / graceful shutdown)."""
    global _client
    with _client_lock:
        if _client is not None and not _client.is_closed:
            _client.close()
        _client = None


def _get_json(url: str, params: Optional[Dict[str, Any]] = None, retries: Optional[int] = None) -> Any:
    """GET a JSON document with bounded retries and exponential backoff.

    Retries on transport errors, 429 and 5xx.  4xx (other than 429) fails fast
    because retrying a bad request just burns rate limit.
    """
    attempts = (retries if retries is not None else config.HTTP_MAX_RETRIES) + 1
    last_error: Optional[str] = None
    for attempt in range(attempts):
        try:
            response = get_client().get(url, params=params)
            if response.status_code == 200:
                return response.json()
            if response.status_code in (429,) or response.status_code >= 500:
                last_error = f"HTTP {response.status_code}"
            else:
                raise FetchError(f"HTTP {response.status_code} from {url}")
        except FetchError:
            raise
        except (httpx.HTTPError, ValueError) as exc:  # network error or bad JSON
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < attempts - 1:
            time.sleep(0.6 * (2 ** attempt))  # 0.6s, 1.2s, 2.4s ...
    raise FetchError(f"{last_error or 'unknown error'} from {url}")


# ==========================================================================
# DexScreener
# ==========================================================================
def _social_links(info: Dict[str, Any]) -> List[SocialLink]:
    """Flatten DexScreener's ``info.websites`` + ``info.socials`` blocks."""
    links: List[SocialLink] = []
    for site in dig(info, "websites", default=[]) or []:
        url = dig(site, "url", default="")
        if url:
            links.append(SocialLink(kind="website", url=url, label=dig(site, "label", default="Website") or "Website"))
    for social in dig(info, "socials", default=[]) or []:
        url = dig(social, "url", default="")
        kind = (dig(social, "type", default="") or dig(social, "platform", default="") or "link").lower()
        if url:
            links.append(SocialLink(kind=kind, url=url, label=kind.title()))
    return links


def _pair_liquidity(pair: Dict[str, Any]) -> float:
    return safe_float(dig(pair, "liquidity", "usd"))


def pick_primary_pair(pairs: Sequence[Dict[str, Any]], chain: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Choose the pair that best represents a token: deepest liquidity.

    DexScreener returns every pair a token trades in (often across chains).  We
    filter to the requested chain when given, then take the deepest pool -- that
    is the price/liquidity everyone actually quotes.
    """
    if not pairs:
        return None
    candidates = list(pairs)
    if chain:
        chain_matches = [p for p in candidates if (dig(p, "chainId", default="") or "").lower() == chain.lower()]
        if chain_matches:
            candidates = chain_matches
    return max(candidates, key=_pair_liquidity)


def snapshot_from_pair(
    pair: Dict[str, Any],
    all_pairs: Optional[Sequence[Dict[str, Any]]] = None,
    address: Optional[str] = None,
) -> TokenSnapshot:
    """Normalize one DexScreener pair (plus siblings) into a TokenSnapshot."""
    base = dig(pair, "baseToken", default={}) or {}
    info = dig(pair, "info", default={}) or {}
    chain = (dig(pair, "chainId", default="") or "").lower()

    siblings = [
        p for p in (all_pairs or [pair])
        if (dig(p, "chainId", default="") or "").lower() == chain
    ] or [pair]

    return TokenSnapshot(
        address=address or dig(base, "address", default="") or "",
        chain=chain,
        name=dig(base, "name", default="") or "",
        symbol=dig(base, "symbol", default="") or "",
        price_usd=safe_float(dig(pair, "priceUsd")),
        market_cap=safe_float(dig(pair, "marketCap")) or safe_float(dig(pair, "fdv")),
        fdv=safe_float(dig(pair, "fdv")),
        # Liquidity is summed across sibling pools on the same chain: that is
        # the depth an exit order can actually reach.
        liquidity_usd=sum(_pair_liquidity(p) for p in siblings),
        volume_24h=sum(safe_float(dig(p, "volume", "h24")) for p in siblings),
        volume_6h=sum(safe_float(dig(p, "volume", "h6")) for p in siblings),
        volume_1h=sum(safe_float(dig(p, "volume", "h1")) for p in siblings),
        volume_5m=sum(safe_float(dig(p, "volume", "m5")) for p in siblings),
        price_change_5m=safe_float(dig(pair, "priceChange", "m5")),
        price_change_1h=safe_float(dig(pair, "priceChange", "h1")),
        price_change_6h=safe_float(dig(pair, "priceChange", "h6")),
        price_change_24h=safe_float(dig(pair, "priceChange", "h24")),
        txns_24h_buys=sum(safe_int(dig(p, "txns", "h24", "buys")) for p in siblings),
        txns_24h_sells=sum(safe_int(dig(p, "txns", "h24", "sells")) for p in siblings),
        txns_1h_buys=sum(safe_int(dig(p, "txns", "h1", "buys")) for p in siblings),
        txns_1h_sells=sum(safe_int(dig(p, "txns", "h1", "sells")) for p in siblings),
        pair_address=dig(pair, "pairAddress", default="") or "",
        # Oldest sibling pair = the token's real trading age.
        pair_created_at=min(
            (safe_int(dig(p, "pairCreatedAt")) for p in siblings if dig(p, "pairCreatedAt")),
            default=None,
        ),
        dex_id=dig(pair, "dexId", default="") or "",
        dex_count=len({dig(p, "dexId", default="") for p in siblings if dig(p, "dexId")}) or 1,
        pair_count=len(siblings),
        quote_symbol=dig(pair, "quoteToken", "symbol", default="") or "",
        url=dig(pair, "url", default="") or "",
        image_url=dig(info, "imageUrl", default="") or "",
        description=dig(info, "description", default="") or "",
        socials=_social_links(info),
        boosts=safe_int(dig(pair, "boosts", "active")),
        fetched_at=utcnow_iso(),
    )


def fetch_token_pairs(address: str, use_cache: bool = True) -> List[Dict[str, Any]]:
    """Raw DexScreener pairs for a token address (all chains it trades on)."""
    key = ("dex_token", normalize_address(address))
    if use_cache:
        cached = _dex_cache.get(key)
        if cached is not None:
            return cached
    payload = _get_json(f"{config.DEXSCREENER_BASE}/latest/dex/tokens/{address}")
    pairs = (payload or {}).get("pairs") or []
    _dex_cache.set(key, pairs)
    return pairs


def fetch_token_snapshot(
    address: str,
    chain: Optional[str] = None,
    use_cache: bool = True,
) -> Tuple[Optional[TokenSnapshot], List[str]]:
    """Fetch and normalize a token's market data.

    Returns ``(snapshot, warnings)``.  ``snapshot`` is ``None`` when the token
    is unknown to DexScreener or the API is unreachable; ``warnings`` always
    explains why in plain English.
    """
    warnings: List[str] = []
    try:
        pairs = fetch_token_pairs(address, use_cache=use_cache)
    except FetchError as exc:
        logger.warning("DexScreener token fetch failed for %s: %s", address, exc)
        return None, [f"DexScreener unavailable ({exc}). No market data could be loaded."]

    if not pairs:
        return None, ["No DexScreener pairs found for this address - unlisted, brand new, or wrong chain."]

    primary = pick_primary_pair(pairs, chain)
    if primary is None:
        return None, ["No tradeable pair found for this address."]

    actual_chain = (dig(primary, "chainId", default="") or "").lower()
    if chain and actual_chain != chain.lower():
        warnings.append(f"Token not found on {chain}; showing data from {actual_chain} instead.")

    snapshot = snapshot_from_pair(primary, pairs, address=address)
    if snapshot.market_cap <= 0:
        warnings.append("Market cap unavailable from DexScreener (FDV used where possible).")
    if snapshot.liquidity_usd <= 0:
        warnings.append("Reported liquidity is zero - treat every number here with suspicion.")
    return snapshot, warnings


def search_pairs(query: str, use_cache: bool = True) -> List[Dict[str, Any]]:
    """DexScreener full-text pair search (used to seed Scanner mode)."""
    key = ("dex_search", query)
    if use_cache:
        cached = _scan_cache.get(key)
        if cached is not None:
            return cached
    payload = _get_json(f"{config.DEXSCREENER_BASE}/latest/dex/search", params={"q": query})
    pairs = (payload or {}).get("pairs") or []
    _scan_cache.set(key, pairs)
    return pairs


def fetch_boosted_tokens(top: bool = True, use_cache: bool = True) -> List[Dict[str, Any]]:
    """Tokens currently paying for DexScreener boosts.

    Boosts are a paid-promotion signal, not a quality signal -- but they are a
    decent proxy for "someone is actively marketing this right now", which is
    exactly the mindshare edge Scanner mode is hunting for.
    """
    endpoint = "/token-boosts/top/v1" if top else "/token-boosts/latest/v1"
    key = ("dex_boosts", endpoint)
    if use_cache:
        cached = _scan_cache.get(key)
        if cached is not None:
            return cached
    try:
        payload = _get_json(f"{config.DEXSCREENER_BASE}{endpoint}")
    except FetchError as exc:
        logger.info("Boost feed unavailable: %s", exc)
        return []
    items = payload if isinstance(payload, list) else (payload or {}).get("tokens") or []
    _scan_cache.set(key, items)
    return items


def fetch_token_profiles(use_cache: bool = True) -> List[Dict[str, Any]]:
    """Newly created DexScreener token profiles (fresh listings feed)."""
    key = ("dex_profiles",)
    if use_cache:
        cached = _scan_cache.get(key)
        if cached is not None:
            return cached
    try:
        payload = _get_json(f"{config.DEXSCREENER_BASE}/token-profiles/latest/v1")
    except FetchError as exc:
        logger.info("Token profile feed unavailable: %s", exc)
        return []
    items = payload if isinstance(payload, list) else (payload or {}).get("profiles") or []
    _scan_cache.set(key, items)
    return items


def fetch_pairs_for_addresses(chain: str, addresses: Sequence[str], use_cache: bool = True) -> List[Dict[str, Any]]:
    """Batch lookup (up to 30 addresses per call) via the tokens/v1 endpoint."""
    results: List[Dict[str, Any]] = []
    for batch in chunked(addresses, 30):
        joined = ",".join(batch)
        key = ("dex_batch", chain, joined)
        if use_cache:
            cached = _scan_cache.get(key)
            if cached is not None:
                results.extend(cached)
                continue
        try:
            payload = _get_json(f"{config.DEXSCREENER_BASE}/tokens/v1/{chain}/{joined}")
        except FetchError as exc:
            logger.info("Batch pair lookup failed for %s: %s", chain, exc)
            continue
        pairs = payload if isinstance(payload, list) else (payload or {}).get("pairs") or []
        _scan_cache.set(key, pairs)
        results.extend(pairs)
    return results


def discover_pairs(chain: str, use_cache: bool = True) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Gather a broad candidate pool of pairs for one chain.

    DexScreener has no "all pairs on chain" endpoint, so we union three public
    sources: seed search queries, the boosted-token feed and the newest token
    profiles.  Duplicates are collapsed on pair address.
    """
    warnings: List[str] = []
    chain_cfg = config.get_chain(chain)
    chain_id = chain_cfg.dexscreener_id
    by_pair: Dict[str, Dict[str, Any]] = {}

    queries = config.SCANNER_SEED_QUERIES.get(chain_id, [chain_id])
    query_failures = 0
    for query in queries:
        try:
            for pair in search_pairs(query, use_cache=use_cache):
                if (dig(pair, "chainId", default="") or "").lower() != chain_id:
                    continue
                pair_addr = dig(pair, "pairAddress", default="")
                if pair_addr:
                    by_pair.setdefault(pair_addr, pair)
        except FetchError as exc:
            query_failures += 1
            logger.info("Search '%s' failed: %s", query, exc)

    if query_failures == len(queries):
        warnings.append("DexScreener search is unreachable - scanner results may be empty.")

    # Boosted + freshly profiled tokens, resolved to pairs in batches.
    promo_addresses: List[str] = []
    for item in list(fetch_boosted_tokens(use_cache=use_cache)) + list(fetch_token_profiles(use_cache=use_cache)):
        if (dig(item, "chainId", default="") or "").lower() != chain_id:
            continue
        token_addr = dig(item, "tokenAddress", default="")
        if token_addr:
            promo_addresses.append(token_addr)

    if promo_addresses:
        known = {normalize_address(dig(p, "baseToken", "address", default="")) for p in by_pair.values()}
        missing = [a for a in dict.fromkeys(promo_addresses) if normalize_address(a) not in known]
        for pair in fetch_pairs_for_addresses(chain_id, missing[:90], use_cache=use_cache):
            pair_addr = dig(pair, "pairAddress", default="")
            if pair_addr:
                by_pair.setdefault(pair_addr, pair)

    if not by_pair and not warnings:
        warnings.append("No pairs returned by DexScreener for this chain right now.")
    return list(by_pair.values()), warnings


def collapse_pairs_to_tokens(pairs: Sequence[Dict[str, Any]]) -> List[TokenSnapshot]:
    """Group pairs by base token and normalize each group into one snapshot."""
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for pair in pairs:
        addr = normalize_address(dig(pair, "baseToken", "address", default=""))
        if addr:
            grouped.setdefault(addr, []).append(pair)

    snapshots: List[TokenSnapshot] = []
    for addr, group in grouped.items():
        primary = pick_primary_pair(group)
        if primary is None:
            continue
        snapshots.append(snapshot_from_pair(primary, group, address=dig(primary, "baseToken", "address", default=addr)))
    return snapshots


# ==========================================================================
# GoPlus Security
# ==========================================================================
def _goplus_headers() -> Dict[str, str]:
    """GoPlus works unauthenticated; an app key simply raises rate limits.

    Full HMAC auth is a documented follow-up (see README) - we only pass the
    key through when present so nothing breaks for key-less users.
    """
    headers: Dict[str, str] = {}
    if config.GOPLUS_APP_KEY:
        headers["X-API-KEY"] = config.GOPLUS_APP_KEY
    return headers


def _pct(value: Any) -> Optional[float]:
    """GoPlus percentages are fractional strings: ``"0.0512"`` -> ``5.12``."""
    if value in (None, "", "NA"):
        return None
    return safe_float(value) * 100.0


def _tax_pct(value: Any) -> Optional[float]:
    """Taxes come back fractional too (``"0.05"`` == 5%)."""
    if value in (None, "", "NA"):
        return None
    return safe_float(value) * 100.0


def _parse_holders(raw_holders: Iterable[Dict[str, Any]]) -> List[HolderEntry]:
    entries: List[HolderEntry] = []
    for holder in raw_holders or []:
        entries.append(
            HolderEntry(
                address=dig(holder, "address", default="") or "",
                percent=_pct(dig(holder, "percent")) or 0.0,
                tag=dig(holder, "tag", default="") or "",
                is_contract=bool(safe_int(dig(holder, "is_contract"))),
                is_locked=bool(safe_int(dig(holder, "is_locked"))),
            )
        )
    return sorted(entries, key=lambda h: h.percent, reverse=True)


def _lp_security(raw_lp_holders: Iterable[Dict[str, Any]]) -> Tuple[Optional[float], Optional[float]]:
    """Return ``(locked_pct, burned_pct)`` of the LP supply.

    "Burned" means the LP tokens sit at a dead address (irreversible).
    "Locked" means a locker contract holds them, or GoPlus flags is_locked.
    """
    holders = list(raw_lp_holders or [])
    if not holders:
        return None, None
    locked = 0.0
    burned = 0.0
    for holder in holders:
        percent = _pct(dig(holder, "percent")) or 0.0
        address = dig(holder, "address", default="") or ""
        tag = dig(holder, "tag", default="") or ""
        if is_burn_address(address, tag):
            burned += percent
        elif safe_int(dig(holder, "is_locked")) == 1 or looks_locked(tag):
            locked += percent
    return min(locked, 100.0), min(burned, 100.0)


def _adjusted_top10(holders: List[HolderEntry]) -> Optional[float]:
    """Top-10 concentration excluding burn, locker and LP/contract holders.

    Raw top-10 is misleading: the biggest "holder" is usually the LP pool.
    What matters is how much a handful of *sellable* wallets control.
    """
    if not holders:
        return None
    real = [
        h for h in holders[:10]
        if not is_burn_address(h.address, h.tag)
        and not h.is_locked
        and not looks_locked(h.tag)
        and "lp" not in (h.tag or "").lower()
        and "pair" not in (h.tag or "").lower()
        and "pool" not in (h.tag or "").lower()
    ]
    return round(sum(h.percent for h in real), 2)


def _annotate(report: SecurityReport) -> SecurityReport:
    """Turn normalized flags into human-readable warnings/positives."""
    w, p, n = report.warnings, report.positives, report.notes

    if report.is_honeypot:
        w.append("HONEYPOT detected - you will not be able to sell. Do not buy.")
    if report.cannot_sell_all:
        w.append("Contract prevents selling the full balance (partial-sell trap).")
    if report.selfdestruct:
        w.append("Contract contains a selfdestruct path.")
    if report.hidden_owner:
        w.append("Hidden owner detected - renouncement is cosmetic.")
    if report.can_take_back_ownership:
        w.append("Ownership can be reclaimed after renouncing.")
    if report.is_mintable:
        w.append("Supply is mintable - the team can dilute holders at will.")
    if report.transfer_pausable:
        w.append("Transfers can be paused by the owner.")
    if report.is_blacklisted:
        w.append("Contract can blacklist wallets (your address can be frozen out).")
    if report.slippage_modifiable:
        w.append("Tax/slippage can be changed after launch.")
    if report.is_freezable:
        w.append("Freeze authority is still active (Solana) - balances can be frozen.")
    if report.trading_cooldown:
        n.append("Trading cooldown is enforced between transactions.")
    if report.is_whitelisted:
        n.append("Contract has a whitelist mechanism.")
    if report.anti_whale_modifiable:
        n.append("Max-wallet / anti-whale limits are modifiable.")
    if report.external_call:
        n.append("Contract makes external calls - behaviour can change off-contract.")
    if report.is_proxy:
        n.append("Proxy contract - logic is upgradeable.")
    if report.is_open_source is False:
        w.append("Contract source is not verified - nobody can audit what it does.")

    max_tax = report.max_tax_pct
    if max_tax is not None:
        if max_tax >= 25:
            w.append(f"Extreme trading tax ({max_tax:.1f}%).")
        elif max_tax >= 10:
            w.append(f"High trading tax ({max_tax:.1f}%).")
        elif max_tax <= 1:
            p.append("Effectively zero buy/sell tax.")

    secured = report.lp_secured_pct
    if secured is not None:
        if secured >= 95:
            p.append(f"LP is {secured:.0f}% burned/locked.")
        elif secured >= 50:
            n.append(f"LP only {secured:.0f}% burned/locked - the remainder can be pulled.")
        else:
            w.append(f"LP is largely unsecured ({secured:.0f}% burned/locked) - rug risk.")

    if report.owner_renounced:
        p.append("Ownership renounced.")
    elif report.owner_renounced is False:
        n.append("Ownership not renounced - owner privileges remain live.")

    if report.top10_pct_adjusted is not None:
        if report.top10_pct_adjusted >= 50:
            w.append(f"Top 10 non-LP wallets hold {report.top10_pct_adjusted:.1f}% of supply.")
        elif report.top10_pct_adjusted <= 15:
            p.append(f"Healthy distribution - top 10 non-LP wallets hold {report.top10_pct_adjusted:.1f}%.")

    if report.creator_percent is not None and report.creator_percent >= 5:
        w.append(f"Deployer still holds {report.creator_percent:.1f}% of supply.")

    if report.is_open_source:
        p.append("Contract source is verified.")
    return report


def _parse_goplus_evm(address: str, chain: str, data: Dict[str, Any]) -> SecurityReport:
    """Normalize the GoPlus EVM ``token_security`` payload."""
    holders = _parse_holders(dig(data, "holders", default=[]) or [])
    lp_locked, lp_burned = _lp_security(dig(data, "lp_holders", default=[]) or [])
    owner_address = dig(data, "owner_address", default="") or ""
    renounced = None
    if "owner_address" in data:
        renounced = normalize_address(owner_address) in (
            "",
            "0x0000000000000000000000000000000000000000",
            "0x000000000000000000000000000000000000dead",
        )

    report = SecurityReport(
        address=address,
        chain=chain,
        available=True,
        source="goplus",
        is_honeypot=safe_bool(dig(data, "is_honeypot")),
        cannot_sell_all=safe_bool(dig(data, "cannot_sell_all")),
        buy_tax_pct=_tax_pct(dig(data, "buy_tax")),
        sell_tax_pct=_tax_pct(dig(data, "sell_tax")),
        transfer_tax_pct=_tax_pct(dig(data, "transfer_tax")),
        is_open_source=safe_bool(dig(data, "is_open_source")),
        is_proxy=safe_bool(dig(data, "is_proxy")),
        is_mintable=safe_bool(dig(data, "is_mintable")),
        owner_renounced=renounced,
        can_take_back_ownership=safe_bool(dig(data, "can_take_back_ownership")),
        hidden_owner=safe_bool(dig(data, "hidden_owner")),
        selfdestruct=safe_bool(dig(data, "selfdestruct")),
        external_call=safe_bool(dig(data, "external_call")),
        transfer_pausable=safe_bool(dig(data, "transfer_pausable")),
        is_blacklisted=safe_bool(dig(data, "is_blacklisted")),
        is_whitelisted=safe_bool(dig(data, "is_whitelisted")),
        trading_cooldown=safe_bool(dig(data, "trading_cooldown")),
        anti_whale_modifiable=safe_bool(dig(data, "anti_whale_modifiable")),
        slippage_modifiable=safe_bool(dig(data, "slippage_modifiable")),
        is_in_dex=safe_bool(dig(data, "is_in_dex")),
        owner_address=owner_address,
        creator_address=dig(data, "creator_address", default="") or "",
        creator_percent=_pct(dig(data, "creator_percent")),
        owner_percent=_pct(dig(data, "owner_percent")),
        lp_locked_pct=lp_locked,
        lp_burned_pct=lp_burned,
        lp_holder_count=safe_int(dig(data, "lp_holder_count")) or None,
        holder_count=safe_int(dig(data, "holder_count")) or None,
        top_holders=holders,
        top10_pct=round(sum(h.percent for h in holders[:10]), 2) if holders else None,
        top10_pct_adjusted=_adjusted_top10(holders),
        total_supply=safe_float(dig(data, "total_supply")) or None,
        token_name=dig(data, "token_name", default="") or "",
        token_symbol=dig(data, "token_symbol", default="") or "",
    )
    return _annotate(report)


def _parse_goplus_solana(address: str, chain: str, data: Dict[str, Any]) -> SecurityReport:
    """Normalize the GoPlus Solana payload (different shape from EVM)."""
    holders = _parse_holders(dig(data, "holders", default=[]) or [])
    lp_locked, lp_burned = _lp_security(dig(data, "lp_holders", default=[]) or [])
    creators = dig(data, "creators", default=[]) or []

    report = SecurityReport(
        address=address,
        chain=chain,
        available=True,
        source="goplus-solana",
        is_mintable=safe_bool(dig(data, "mintable", "status")),
        is_freezable=safe_bool(dig(data, "freezable", "status")),
        transfer_pausable=safe_bool(dig(data, "transfer_hook", "status")),
        is_blacklisted=safe_bool(dig(data, "default_account_state_upgradable", "status")),
        transfer_tax_pct=safe_float(dig(data, "transfer_fee", "current_fee_rate"), 0.0) or None,
        # On Solana "renounced" == the mint/freeze authorities are revoked.
        owner_renounced=(
            not safe_bool(dig(data, "mintable", "status"), False)
            and not safe_bool(dig(data, "freezable", "status"), False)
        ),
        creator_address=(dig(creators, 0, "address", default="") or ""),
        creator_percent=_pct(dig(creators, 0, "malicious_address")) if creators else None,
        lp_locked_pct=lp_locked,
        lp_burned_pct=lp_burned,
        holder_count=safe_int(dig(data, "holder_count")) or None,
        top_holders=holders,
        top10_pct=round(sum(h.percent for h in holders[:10]), 2) if holders else None,
        top10_pct_adjusted=_adjusted_top10(holders),
        total_supply=safe_float(dig(data, "total_supply")) or None,
        token_name=dig(data, "metadata", "name", default="") or "",
        token_symbol=dig(data, "metadata", "symbol", default="") or "",
    )
    if safe_bool(dig(data, "metadata_mutable", "status")):
        report.notes.append("Token metadata is mutable - name/symbol/image can be changed.")
    return _annotate(report)


def fetch_security_report(address: str, chain: str, use_cache: bool = True) -> SecurityReport:
    """Fetch a token-security report, degrading to ``available=False``.

    Never raises: a missing security feed is a scoring input ("unknown"), not
    an application error.
    """
    chain_cfg = config.get_chain(chain)
    key = ("goplus", chain_cfg.key, normalize_address(address))
    if use_cache:
        cached = _sec_cache.get(key)
        if cached is not None:
            return cached

    if chain_cfg.goplus_solana:
        url = f"{config.GOPLUS_BASE}/api/v1/solana/token_security"
    elif chain_cfg.goplus_id:
        url = f"{config.GOPLUS_BASE}/api/v1/token_security/{chain_cfg.goplus_id}"
    else:
        return SecurityReport(
            address=address, chain=chain_cfg.key, available=False,
            error=f"No security provider configured for {chain_cfg.label}.",
        )

    try:
        payload = _get_json(url, params={"contract_addresses": address})
    except FetchError as exc:
        logger.warning("GoPlus fetch failed for %s: %s", address, exc)
        return SecurityReport(
            address=address, chain=chain_cfg.key, available=False,
            error=f"GoPlus unavailable ({exc}). Security checks were skipped.",
        )

    if safe_int(dig(payload, "code"), 0) != 1:
        message = dig(payload, "message", default="unexpected response") or "unexpected response"
        return SecurityReport(
            address=address, chain=chain_cfg.key, available=False,
            error=f"GoPlus returned no data: {message}",
        )

    result = dig(payload, "result", default={}) or {}
    # GoPlus keys results by lower-cased address; match tolerantly.
    data = result.get(address) or result.get(address.lower()) or result.get(normalize_address(address))
    if not data and len(result) == 1:
        data = next(iter(result.values()))
    if not data:
        return SecurityReport(
            address=address, chain=chain_cfg.key, available=False,
            error="GoPlus has no security record for this token yet (very new or unsupported).",
        )

    parser = _parse_goplus_solana if chain_cfg.goplus_solana else _parse_goplus_evm
    report = parser(address, chain_cfg.key, data)
    _sec_cache.set(key, report)
    return report


# --------------------------------------------------------------------------
# Cache control (exposed in the UI sidebar)
# --------------------------------------------------------------------------
def clear_caches() -> None:
    _dex_cache.clear()
    _scan_cache.clear()
    _sec_cache.clear()
