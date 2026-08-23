"""On-chain wallet flow: who is buying, and are they adding or leaving.

Built on Etherscan V2, whose free tier covers every supported EVM chain from a
single key (5 calls/sec, 100k/day) by passing ``chainid``. Solana is not an EVM
chain and is not covered.

What this can and cannot tell you
---------------------------------
It reads raw ERC-20 transfer logs, so everything here is an **observation about
behaviour**, never a claim about anyone's skill:

* which wallets received tokens from the liquidity pool (bought) and sent them
  back (sold), over what window, and in what size
* whether the early cohort is still holding or has flipped
* whether wallets are net accumulating or distributing right now
* **quiet accumulation**: net accumulation while the price is range-bound,
  which is the pattern that precedes a move rather than following one

It deliberately does **not** claim to compute historical win rates. A real
win-rate needs the price at the moment of every trade a wallet ever made, which
no free explorer API exposes -- that is exactly what paid services like Cielo
sell. Inventing one from transfer logs would produce a confident-looking number
with nothing behind it, which is worse than no number.

The high-precision signal here is your own watchlist: wallets you already trust
(see ``data/smart_money.example.json``). When one of them shows up in a token's
flow, that is worth more than any heuristic in this file.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from . import config
from .models import TokenSnapshot, WalletActivity, WalletFlowReport
from .utils import TTLCache, normalize_address, safe_float, safe_int

logger = logging.getLogger(__name__)

_cache = TTLCache(ttl_seconds=config.CACHE_TTL_WALLET_FLOW)

# Addresses that are never "a holder": mints, burns and the zero address.
_NON_WALLETS = {
    "0x0000000000000000000000000000000000000000",
    "0x000000000000000000000000000000000000dead",
}


# --------------------------------------------------------------------------
# Watchlist
# --------------------------------------------------------------------------
def load_watchlist(path: Optional[Any] = None) -> Dict[str, Any]:
    """Load the user's smart-money list.

    Shape::

        {"wallets": [{"address": "0x..", "label": "whale that caught BRETT"}],
         "x_handles": ["someanalyst"]}

    Missing or malformed files degrade to an empty list rather than an error:
    a watchlist is an enhancement, not a dependency.
    """
    target = path or config.SMART_MONEY_PATH
    empty: Dict[str, Any] = {"wallets": {}, "x_handles": []}
    try:
        with open(target, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except FileNotFoundError:
        return empty
    except Exception as exc:  # noqa: BLE001 - a bad file must not break analysis
        logger.warning("Could not read smart-money watchlist %s: %s", target, exc)
        return empty

    wallets: Dict[str, str] = {}
    for entry in raw.get("wallets") or []:
        if isinstance(entry, str):
            wallets[normalize_address(entry)] = ""
        elif isinstance(entry, dict) and entry.get("address"):
            wallets[normalize_address(entry["address"])] = str(entry.get("label") or "")
    handles = [str(h).strip().lstrip("@") for h in (raw.get("x_handles") or []) if str(h).strip()]
    return {"wallets": wallets, "x_handles": handles}


def parse_watchlist_input(raw: str) -> List[Dict[str, str]]:
    """Parse pasted wallet lines into ``{address, label}`` entries.

    Accepts what people actually paste out of GMGN, Cielo, Arkham or a
    spreadsheet: one entry per line, address first, an optional label after a
    comma, tab or whitespace. Anything that is not a valid address is skipped
    rather than rejected, so one bad row does not lose the paste.
    """
    import re

    from .utils import is_valid_address

    entries: List[Dict[str, str]] = []
    seen = set()
    for line in (raw or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in re.split(r"[,\t]", line, maxsplit=1)]
        address = parts[0].split()[0] if parts[0] else ""
        label = parts[1] if len(parts) > 1 else ""
        if not label and " " in parts[0]:
            address, _, label = parts[0].partition(" ")
        if not is_valid_address(address):
            continue
        key = normalize_address(address)
        if key in seen:
            continue
        seen.add(key)
        entries.append({"address": address, "label": label.strip()})
    return entries


def save_watchlist(
    wallets: List[Dict[str, str]],
    x_handles: List[str],
    path: Optional[Any] = None,
) -> str:
    """Write the watchlist to disk. Returns the path written."""
    target = path or config.SMART_MONEY_PATH
    payload = {
        "_comment": "Managed from the MemeDD Watchlist tab. Gitignored - this stays yours.",
        "wallets": wallets,
        "x_handles": [h.strip().lstrip("@") for h in x_handles if h.strip()],
    }
    target = str(target)
    import os

    os.makedirs(os.path.dirname(target) or ".", exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return target


def watchlist_handles(path: Optional[Any] = None) -> List[str]:
    """X handles the user nominated as smart money (used by the Grok prompt)."""
    return load_watchlist(path).get("x_handles", [])


# --------------------------------------------------------------------------
# Etherscan V2
# --------------------------------------------------------------------------
class EtherscanClient:
    """Thin Etherscan V2 client. One key, every EVM chain, via ``chainid``."""

    def __init__(self, api_key: str = "", base_url: str = "", session: Any = None) -> None:
        self.api_key = api_key or config.ETHERSCAN_API_KEY
        self.base_url = base_url or config.ETHERSCAN_BASE_URL
        self._session = session      # injected in tests

    def unavailable_reason(self, chain: str) -> str:
        chain_cfg = config.get_chain(chain)
        if chain_cfg.etherscan_chain_id is None:
            return (
                f"{chain_cfg.label} is not an Etherscan V2 chain, so wallet-flow "
                "analysis cannot run there."
            )
        if not self.api_key:
            return (
                "No ETHERSCAN_API_KEY set. A free key at etherscan.io/apis covers "
                "every EVM chain (100k calls/day)."
            )
        return ""

    def token_transfers(self, address: str, chain: str, limit: int) -> List[Dict[str, Any]]:
        """Most recent ERC-20 transfers of one token contract.

        Sorted descending so a capped query returns the *newest* activity --
        the opposite order would fill the budget with launch-day history and
        tell us nothing about what is happening now.
        """
        from .data_fetchers import _get_json      # local import avoids a cycle

        chain_cfg = config.get_chain(chain)
        params = {
            "chainid": chain_cfg.etherscan_chain_id,
            "module": "account",
            "action": "tokentx",
            "contractaddress": address,
            "page": 1,
            "offset": max(1, min(limit, 10_000)),
            "sort": "desc",
            "apikey": self.api_key,
        }
        payload = (
            self._session(self.base_url, params) if self._session
            else _get_json(self.base_url, params=params)
        )
        status = str((payload or {}).get("status", ""))
        result = (payload or {}).get("result")

        if status != "1":
            message = (payload or {}).get("message") or ""
            # "No transactions found" is a legitimate empty answer, not a failure.
            if isinstance(result, list) and not result:
                return []
            if "no transactions" in str(message).lower():
                return []
            raise RuntimeError(f"Etherscan: {message or result or 'unknown error'}")
        return result if isinstance(result, list) else []


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------
def _token_units(transfer: Dict[str, Any]) -> float:
    """Convert a raw transfer value using the token's decimals."""
    value = safe_float(transfer.get("value"))
    decimals = safe_int(transfer.get("tokenDecimal"), 18)
    return value / (10 ** decimals) if decimals else value


def identify_pools(transfers: List[Dict[str, Any]], known: Optional[List[str]] = None) -> List[str]:
    """Guess which addresses are liquidity pools.

    A pool sits on one side of a very large share of transfers, because every
    buy and sell routes through it. Counting appearances and taking the
    dominant addresses identifies it without needing a pair registry.
    """
    counts: Dict[str, int] = {}
    for transfer in transfers:
        for side in ("from", "to"):
            addr = normalize_address(transfer.get(side, ""))
            if addr and addr not in _NON_WALLETS:
                counts[addr] = counts.get(addr, 0) + 1
    if not counts:
        return []

    total = len(transfers)
    pools = {normalize_address(a) for a in (known or []) if a}
    # An address touching >25% of all transfers is the pool, not a trader.
    for addr, count in counts.items():
        if total >= 20 and count / total >= 0.25:
            pools.add(addr)
    if not pools:                      # fall back to the single busiest address
        pools.add(max(counts.items(), key=lambda kv: kv[1])[0])
    return sorted(pools)


def build_activity(
    transfers: List[Dict[str, Any]],
    pools: List[str],
) -> Dict[str, WalletActivity]:
    """Fold transfers into per-wallet buy/sell activity.

    Received *from* a pool is a buy; sent *to* a pool is a sell. Wallet-to-wallet
    transfers are ignored -- they move tokens without expressing conviction, and
    counting them as buys is how naive trackers get fooled by self-transfers.
    """
    pool_set = {normalize_address(p) for p in pools}
    wallets: Dict[str, WalletActivity] = {}

    def touch(addr: str, ts: int) -> WalletActivity:
        activity = wallets.get(addr)
        if activity is None:
            activity = WalletActivity(address=addr, first_seen_ts=ts, last_seen_ts=ts)
            wallets[addr] = activity
        activity.first_seen_ts = min(activity.first_seen_ts or ts, ts)
        activity.last_seen_ts = max(activity.last_seen_ts or ts, ts)
        activity.tx_count += 1
        return activity

    for transfer in transfers:
        sender = normalize_address(transfer.get("from", ""))
        receiver = normalize_address(transfer.get("to", ""))
        timestamp = safe_int(transfer.get("timeStamp"))
        units = _token_units(transfer)
        if units <= 0:
            continue

        if sender in pool_set and receiver not in pool_set and receiver not in _NON_WALLETS:
            touch(receiver, timestamp).bought_tokens += units
        elif receiver in pool_set and sender not in pool_set and sender not in _NON_WALLETS:
            touch(sender, timestamp).sold_tokens += units
    return wallets


def analyze_transfers(
    transfers: List[Dict[str, Any]],
    snapshot: TokenSnapshot,
    watchlist: Optional[Dict[str, Any]] = None,
    now: Optional[datetime] = None,
) -> WalletFlowReport:
    """Turn raw transfers into a wallet-flow report. Pure: no network."""
    watchlist = watchlist or {"wallets": {}, "x_handles": []}
    reference = now or datetime.now(timezone.utc)
    now_ts = int(reference.timestamp())

    report = WalletFlowReport(
        chain=snapshot.chain,
        address=snapshot.address,
        available=True,
        transfers_analyzed=len(transfers),
    )
    if not transfers:
        report.available = False
        report.error = "No token transfers returned - the token may be too new or unindexed."
        return report

    pools = identify_pools(transfers, known=[snapshot.pair_address])
    report.pool_addresses = pools
    wallets = build_activity(transfers, pools)
    report.unique_wallets = len(wallets)

    if not wallets:
        report.available = False
        report.error = "No pool trades found in the transfer sample."
        return report

    recent_cutoff = now_ts - int(config.WALLET_FLOW_RECENT_HOURS * 3600)
    recent = [w for w in wallets.values() if (w.last_seen_ts or 0) >= recent_cutoff]
    pool_of_interest = recent or list(wallets.values())

    report.accumulating_wallets = sum(1 for w in pool_of_interest if w.net_tokens > 0)
    report.distributing_wallets = sum(1 for w in pool_of_interest if w.net_tokens < 0)
    report.net_flow_tokens = round(sum(w.net_tokens for w in pool_of_interest), 4)

    if snapshot.total_supply:
        report.net_flow_pct_of_supply = round(
            report.net_flow_tokens / snapshot.total_supply * 100, 4
        )

    # --- early cohort ---------------------------------------------------
    launch_ts = min((w.first_seen_ts or now_ts) for w in wallets.values())
    early_cutoff = launch_ts + int(config.WALLET_FLOW_EARLY_HOURS * 3600)
    early = [w for w in wallets.values() if (w.first_seen_ts or now_ts) <= early_cutoff]
    report.early_buyers = len(early)
    report.early_still_holding = sum(1 for w in early if w.net_tokens > 0)
    report.early_flipped = sum(1 for w in early if w.round_tripped and w.net_tokens <= 0)

    # --- fresh wallets ---------------------------------------------------
    # Wallets whose entire footprint in this token is a single recent buy. A
    # burst of these is a sybil/bot signature, not organic demand.
    single_touch = [w for w in pool_of_interest if w.tx_count == 1 and w.bought_tokens > 0]
    if pool_of_interest:
        report.fresh_wallet_ratio = round(len(single_touch) / len(pool_of_interest), 3)

    # --- leaderboards ----------------------------------------------------
    ranked = sorted(wallets.values(), key=lambda w: w.net_tokens, reverse=True)
    report.top_accumulators = [w for w in ranked if w.net_tokens > 0][:8]
    report.top_distributors = [w for w in reversed(ranked) if w.net_tokens < 0][:8]

    # --- watchlist -------------------------------------------------------
    known_wallets: Dict[str, str] = watchlist.get("wallets", {})
    for addr, activity in wallets.items():
        if addr in known_wallets:
            activity.label = known_wallets[addr] or "watchlist"
            report.watchlist_hits.append(activity)
    report.watchlist_hits.sort(key=lambda w: w.net_tokens, reverse=True)

    # --- verdict ---------------------------------------------------------
    traders = report.accumulating_wallets + report.distributing_wallets
    if traders:
        accumulate_share = report.accumulating_wallets / traders
        if accumulate_share >= 0.60 and report.net_flow_tokens > 0:
            report.accumulation_verdict = "accumulating"
        elif accumulate_share <= 0.40 or report.net_flow_tokens < 0:
            report.accumulation_verdict = "distributing"
        else:
            report.accumulation_verdict = "balanced"

    # --- the headline pattern --------------------------------------------
    # Quiet accumulation: wallets adding while the price goes nowhere. Buying
    # into a rip is chasing; buying into a flat chart is positioning.
    report.consolidating = abs(snapshot.price_change_24h) <= config.WALLET_FLOW_CONSOLIDATION_PCT
    report.quiet_accumulation = bool(
        report.consolidating and report.accumulation_verdict == "accumulating"
    )

    _annotate(report, snapshot)
    return report


def _annotate(report: WalletFlowReport, snapshot: TokenSnapshot) -> None:
    """Attach plain-English notes and warnings."""
    if report.quiet_accumulation:
        report.notes.append(
            f"Wallets are net accumulating while price is flat "
            f"({snapshot.price_change_24h:+.1f}% over 24h) - positioning, not chasing."
        )
    elif report.accumulation_verdict == "accumulating":
        report.notes.append(
            f"{report.accumulating_wallets} wallets accumulating vs "
            f"{report.distributing_wallets} distributing in the last "
            f"{config.WALLET_FLOW_RECENT_HOURS:.0f}h."
        )
    elif report.accumulation_verdict == "distributing":
        report.warnings.append(
            f"More wallets are selling than buying "
            f"({report.distributing_wallets} vs {report.accumulating_wallets}) "
            f"in the last {config.WALLET_FLOW_RECENT_HOURS:.0f}h."
        )

    hold_rate = report.early_hold_rate
    if hold_rate is not None and report.early_buyers >= 5:
        if hold_rate >= 0.6:
            report.notes.append(
                f"{hold_rate * 100:.0f}% of the {report.early_buyers} early buyers are still holding."
            )
        elif hold_rate <= 0.3:
            report.warnings.append(
                f"Only {hold_rate * 100:.0f}% of early buyers still hold - the first cohort took profit."
            )

    if report.fresh_wallet_ratio is not None and report.fresh_wallet_ratio >= 0.8:
        report.warnings.append(
            f"{report.fresh_wallet_ratio * 100:.0f}% of active wallets bought once and never "
            "traded again - consistent with airdrop farming or bots rather than real demand."
        )

    for hit in report.watchlist_hits:
        label = hit.label or "watchlist wallet"
        if hit.net_tokens > 0:
            report.notes.append(f"Watchlist hit: {label} is accumulating ({hit.address[:10]}...).")
        else:
            report.warnings.append(f"Watchlist hit: {label} is distributing ({hit.address[:10]}...).")

    if report.transfers_analyzed >= config.WALLET_FLOW_MAX_TRANSFERS:
        report.warnings.append(
            f"Only the most recent {report.transfers_analyzed:,} transfers were analyzed; "
            "older history is not included."
        )


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def fetch_wallet_flow(
    snapshot: TokenSnapshot,
    client: Optional[EtherscanClient] = None,
    use_cache: bool = True,
    watchlist_path: Optional[Any] = None,
) -> WalletFlowReport:
    """Fetch and analyze wallet flow for a token. Never raises."""
    started = time.monotonic()
    client = client or EtherscanClient()

    reason = client.unavailable_reason(snapshot.chain)
    if reason:
        return WalletFlowReport(
            chain=snapshot.chain, address=snapshot.address, available=False, error=reason
        )

    key = ("wallet_flow", snapshot.chain, normalize_address(snapshot.address))
    if use_cache:
        cached = _cache.get(key)
        if cached is not None:
            return cached

    try:
        transfers = client.token_transfers(
            snapshot.address, snapshot.chain, config.WALLET_FLOW_MAX_TRANSFERS
        )
    except Exception as exc:  # noqa: BLE001 - degrade, never break the analysis
        logger.warning("Wallet-flow fetch failed for %s: %s", snapshot.address, exc)
        return WalletFlowReport(
            chain=snapshot.chain,
            address=snapshot.address,
            available=False,
            error=f"{type(exc).__name__}: {exc}"[:240],
            latency_ms=int((time.monotonic() - started) * 1000),
        )

    report = analyze_transfers(transfers, snapshot, load_watchlist(watchlist_path))
    report.latency_ms = int((time.monotonic() - started) * 1000)
    if report.available:
        _cache.set(key, report)
    return report


def clear_cache() -> None:
    _cache.clear()
