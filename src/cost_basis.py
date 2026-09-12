"""Reconstruct what you paid, from the wallet's own transaction history.

A balance read answers "how many do I hold". This module answers "what did they
cost", using nothing but public data:

1. **Group the wallet's transfers by transaction hash.** A buy is "BREW came
   in, WBNB went out" inside one transaction; a sell is the reverse. What went
   out *is* the price, denominated in the quote token.
2. **Price that leg at the transaction's own timestamp.** Stablecoin legs are
   $1 and need no call at all. Everything else is priced by DefiLlama's coins
   API and cached forever, because a past price cannot change.
3. **Walk the trades forward** on a weighted-average basis: buys pool into one
   average, a sell removes a proportional slice of it and realizes P&L.

Prices are cached in SQLite and nowhere else. A second, in-memory layer would
be marginally faster and would let the two disagree -- a memory hit skipping
the durable write -- which is not a trade worth making for a lookup that is
already sub-millisecond.

Gas is never part of a cost basis here. It is a real cost, but it is paid in
a different asset than most buys, so folding it in would make the average
disagree with every trading app you compare it against -- and supporting it on
one chain but not another would be worse than not supporting it at all.

What this module refuses to do is guess. Tokens that arrived without a purchase
-- an airdrop, a bridge, a send from an address you have not registered -- have
no knowable cost, so they lower the report's **coverage** instead of being
quietly priced at zero. A failed price lookup does the same. An average over
82% of your balance is reported as exactly that, never as the whole story.

All of your registered wallets on a chain are derived together as one book, so
moving a bag between your own addresses cancels out rather than looking like a
sale followed by a repurchase.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from . import balances as balances_mod
from . import config, portfolio_store
from .models import CostBasisReport, Position, Trade, Wallet
from .utils import dig, normalize_address, safe_float, safe_int, utcnow_iso

logger = logging.getLogger(__name__)

# DefiLlama's coins API takes "chain:address"; its chain slugs match the
# DexScreener ids we already store. A chain absent here can still price native
# legs through the canonical ids below.
COINS_CHAIN_SLUGS: Dict[str, str] = {
    "base": "base",
    "ethereum": "ethereum",
    "bsc": "bsc",
    "arbitrum": "arbitrum",
    "solana": "solana",
}

# Native coins are priced by canonical id rather than by their wrapped contract
# on whichever chain the trade happened. ETH is ETH: pricing a Robinhood Chain
# buy through mainnet ETH is both possible and more reliable than hoping the
# L2's wrapped contract has price history.
NATIVE_PRICE_IDS: Dict[str, str] = {
    "ETH": "coingecko:ethereum",
    "BNB": "coingecko:binancecoin",
    "SOL": "coingecko:solana",
}


@dataclass
class Ledger:
    """Normalized history for one chain, merged across your wallets."""

    transfers: List[Dict[str, Any]] = field(default_factory=list)
    native: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    truncated: bool = False
    ok: bool = True


# ==========================================================================
# Fetching
# ==========================================================================
def _normalize_transfer(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Etherscan/Blockscout transfer row -> the shape this module works in."""
    token = row.get("contractAddress") or ""
    tx_hash = row.get("hash") or row.get("transactionHash") or ""
    if not token or not tx_hash:
        return None
    decimals = safe_int(row.get("tokenDecimal"), 18)
    raw = safe_float(row.get("value"))
    return {
        "hash": tx_hash,
        "timestamp": safe_int(row.get("timeStamp")),
        "from": normalize_address(row.get("from") or ""),
        "to": normalize_address(row.get("to") or ""),
        "token": normalize_address(token),
        "symbol": row.get("tokenSymbol") or "",
        "decimals": decimals,
        "quantity": raw / (10 ** max(0, min(decimals, 36))),
        "native": False,
    }


def _normalize_native(row: Dict[str, Any], symbol: str) -> Optional[Dict[str, Any]]:
    """Outer transaction -> a pseudo-transfer of the chain's native coin."""
    tx_hash = row.get("hash") or ""
    value = safe_float(row.get("value"))
    if not tx_hash or value <= 0:
        return None
    # A reverted transaction moved nothing; counting it would invent a trade.
    if str(row.get("isError", "0")) == "1":
        return None
    return {
        "hash": tx_hash,
        "timestamp": safe_int(row.get("timeStamp")),
        "from": normalize_address(row.get("from") or ""),
        "to": normalize_address(row.get("to") or ""),
        "token": "native",
        "symbol": symbol,
        "decimals": 18,
        "quantity": value / (10 ** 18),
        "native": True,
    }


def fetch_evm_ledger(chain: str, wallets: Sequence[Wallet]) -> Ledger:
    """History for every registered wallet on one EVM chain, merged."""
    ledger = Ledger()
    chain_cfg = config.get_chain(chain)
    cap = max(1, min(config.PORTFOLIO_DISCOVERY_TRANSFERS, 10_000))

    blockscout = balances_mod.BlockscoutProvider(chain)
    etherscan = balances_mod.EvmRpcProvider(chain)
    use_etherscan = not etherscan.discovery_reason()

    if not use_etherscan and blockscout.unavailable_reason():
        ledger.ok = False
        ledger.warnings.append(etherscan.discovery_reason())
        return ledger

    for wallet in wallets:
        raw_transfers: List[Dict[str, Any]] = []
        raw_native: List[Dict[str, Any]] = []
        try:
            if use_etherscan:
                raw_transfers = etherscan.token_transfers(wallet.address)
                raw_native = etherscan.native_transactions(wallet.address)
            else:
                raw_transfers = blockscout.token_transfers(wallet.address)
                raw_native = blockscout.native_transactions(wallet.address)
        except RuntimeError as exc:
            ledger.ok = False
            ledger.warnings.append(
                f"History unavailable for {wallet.address[:10]}… on {chain_cfg.label}: {exc}"
            )
            continue

        if len(raw_transfers) >= cap:
            ledger.truncated = True
            ledger.warnings.append(
                f"{chain_cfg.label} history for {wallet.address[:10]}… hit the "
                f"{cap:,}-row limit, so trades older than that are not included."
            )

        for row in raw_transfers:
            normalized = _normalize_transfer(row)
            if normalized:
                ledger.transfers.append(normalized)
        for row in raw_native:
            normalized = _normalize_native(row, chain_cfg.native_symbol)
            if normalized:
                ledger.native.append(normalized)

    return ledger


# --------------------------------------------------------------------------
# Solana
# --------------------------------------------------------------------------
# Solana has no "list this wallet's transfers" endpoint: you list signatures,
# then read each transaction. That is one RPC call per transaction, which is
# why this path is opted into per position rather than swept up automatically.
def _solana_legs(transaction: Dict[str, Any], owner: str) -> List[Dict[str, Any]]:
    """Turn one parsed transaction into transfer legs for this owner.

    Solana states balances before and after, so the *difference* is the trade
    -- no need to interpret instructions, which is both simpler and more
    robust than trying to recognise every DEX program.
    """
    meta = transaction.get("meta") or {}
    if meta.get("err"):
        return []                    # a failed transaction moved nothing

    signature = ""
    signatures = dig(transaction, "transaction", "signatures", default=[]) or []
    if signatures:
        signature = signatures[0]
    timestamp = safe_int(transaction.get("blockTime"))

    legs: List[Dict[str, Any]] = []

    # SPL tokens: match pre/post entries by account index for this owner.
    def by_index(entries: Any) -> Dict[int, Dict[str, Any]]:
        table: Dict[int, Dict[str, Any]] = {}
        for entry in entries or []:
            if entry.get("owner") == owner:
                table[safe_int(entry.get("accountIndex"), -1)] = entry
        return table

    pre = by_index(meta.get("preTokenBalances"))
    post = by_index(meta.get("postTokenBalances"))
    for index in set(pre) | set(post):
        before = safe_float(dig(pre.get(index, {}), "uiTokenAmount", "uiAmount", default=0.0))
        after = safe_float(dig(post.get(index, {}), "uiTokenAmount", "uiAmount", default=0.0))
        delta = after - before
        if abs(delta) <= 0:
            continue
        entry = post.get(index) or pre.get(index) or {}
        mint = entry.get("mint") or ""
        if not mint:
            continue
        legs.append({
            "hash": signature, "timestamp": timestamp,
            "from": "" if delta > 0 else owner,
            "to": owner if delta > 0 else "",
            "token": mint, "symbol": "",
            "decimals": safe_int(dig(entry, "uiTokenAmount", "decimals", default=0)),
            "quantity": abs(delta), "native": False,
        })

    # Native SOL, net of the fee the owner paid.
    accounts = dig(transaction, "transaction", "message", "accountKeys", default=[]) or []
    owner_index = None
    for index, account in enumerate(accounts):
        key = account.get("pubkey") if isinstance(account, dict) else account
        if key == owner:
            owner_index = index
            break
    if owner_index is not None:
        pre_lamports = (meta.get("preBalances") or [])
        post_lamports = (meta.get("postBalances") or [])
        if owner_index < len(pre_lamports) and owner_index < len(post_lamports):
            delta = safe_int(post_lamports[owner_index]) - safe_int(pre_lamports[owner_index])
            # The signer pays the fee whichever way the trade went, so adding
            # it back in both directions keeps gas out of the traded amount --
            # otherwise 3 SOL of proceeds reads as 2.999995 and a buy reads as
            # very slightly more expensive than it was.
            delta += safe_int(meta.get("fee"))
            sol = delta / 1e9
            # Ignore rent-sized dust, which every transaction moves around.
            if abs(sol) > 0.000_01:
                legs.append({
                    "hash": signature, "timestamp": timestamp,
                    "from": "" if sol > 0 else owner,
                    "to": owner if sol > 0 else "",
                    "token": "native", "symbol": "SOL", "decimals": 9,
                    "quantity": abs(sol), "native": True,
                })
    return legs


def fetch_solana_ledger(
    wallets: Sequence[Wallet],
    session: Optional[Callable] = None,
    progress: Optional[Callable[[str, float], None]] = None,
) -> Ledger:
    """History for Solana wallets, one signature page and batch at a time."""
    ledger = Ledger()
    provider = balances_mod.SolanaRpcProvider(session=session)
    reason = provider.unavailable_reason()
    if reason:
        return Ledger(ok=False, warnings=[reason])

    for wallet in wallets:
        signatures: List[str] = []
        before: Optional[str] = None
        while len(signatures) < config.COST_BASIS_MAX_SIGNATURES:
            params: Dict[str, Any] = {"limit": 1000}
            if before:
                params["before"] = before
            try:
                page = provider._rpc("getSignaturesForAddress", [wallet.address, params]) or []
            except Exception as exc:  # noqa: BLE001
                ledger.ok = False
                ledger.warnings.append(f"Solana history unavailable: {exc}")
                break
            page = [entry for entry in page if not entry.get("err")]
            if not page:
                break
            signatures.extend(entry.get("signature") for entry in page if entry.get("signature"))
            before = page[-1].get("signature")
            if len(page) < 1000:
                break

        if len(signatures) >= config.COST_BASIS_MAX_SIGNATURES:
            ledger.truncated = True
            ledger.warnings.append(
                f"Stopped after {config.COST_BASIS_MAX_SIGNATURES:,} Solana transactions for "
                f"{wallet.address[:8]}…; anything older is not included. Raise "
                "MEMEDD_COST_BASIS_MAX_SIGNATURES, ideally with a Helius RPC URL."
            )
            signatures = signatures[: config.COST_BASIS_MAX_SIGNATURES]

        batch_size = max(1, config.COST_BASIS_RPC_BATCH_SIZE)
        total = max(1, len(signatures))
        for offset in range(0, len(signatures), batch_size):
            chunk = signatures[offset:offset + batch_size]
            if progress:
                progress(f"Solana {offset + len(chunk)}/{total}", (offset + len(chunk)) / total)
            payload = [
                {
                    "jsonrpc": "2.0", "id": index, "method": "getTransaction",
                    "params": [signature,
                               {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0}],
                }
                for index, signature in enumerate(chunk)
            ]
            try:
                responses = (
                    session(provider.rpc_url, payload) if session
                    else balances_mod._post_json(provider.rpc_url, payload)
                )
            except Exception as exc:  # noqa: BLE001
                ledger.warnings.append(f"Solana batch failed: {exc}")
                ledger.ok = False
                continue
            if isinstance(responses, dict):
                responses = [responses]
            for response in responses or []:
                transaction = (response or {}).get("result")
                if transaction:
                    ledger.transfers.extend(_solana_legs(transaction, wallet.address))

    return ledger


# ==========================================================================
# Classification
# ==========================================================================
def group_by_tx(rows: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Bucket every leg by the transaction it belongs to."""
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["hash"], []).append(row)
    return grouped


def _leg_totals(
    legs: Sequence[Dict[str, Any]],
    owned: set,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, Dict[str, Any]]]:
    """Sum each token in and out of *your* addresses within one transaction."""
    incoming: Dict[str, float] = {}
    outgoing: Dict[str, float] = {}
    meta: Dict[str, Dict[str, Any]] = {}
    for leg in legs:
        token = leg["token"]
        meta.setdefault(token, leg)
        to_mine = leg["to"] in owned
        from_mine = leg["from"] in owned
        # A transfer between two of your own addresses nets to nothing; it is
        # not a sale and not a purchase.
        if to_mine and from_mine:
            continue
        if to_mine:
            incoming[token] = incoming.get(token, 0.0) + leg["quantity"]
        elif from_mine:
            outgoing[token] = outgoing.get(token, 0.0) + leg["quantity"]
    return incoming, outgoing, meta


def _pick_quote(
    candidates: Dict[str, float],
    meta: Dict[str, Dict[str, Any]],
    chain: str,
    token: str,
) -> Optional[Tuple[str, float]]:
    """Choose which leg represents the money side of a swap.

    Preference order is about *pricing confidence*, not size: a stablecoin leg
    is exact, a native leg is close behind, and an arbitrary token is a guess
    we can only price if DefiLlama happens to know it.
    """
    stables = {addr.lower() for addr in config.STABLECOINS.get(chain, {})}
    options = [(addr, qty) for addr, qty in candidates.items() if addr != token and qty > 0]
    if not options:
        return None

    def rank(item: Tuple[str, float]) -> Tuple[int, float]:
        address, quantity = item
        if address in stables:
            return (0, -quantity)
        if meta.get(address, {}).get("native") or address == normalize_address(
            config.WRAPPED_NATIVE.get(chain, "")
        ):
            return (1, -quantity)
        return (2, -quantity)

    return sorted(options, key=rank)[0]


def classify_transaction(
    legs: Sequence[Dict[str, Any]],
    token: str,
    chain: str,
    owned: set,
) -> Optional[Trade]:
    """Work out what one transaction did to one token, from your side."""
    incoming, outgoing, meta = _leg_totals(legs, owned)
    token = normalize_address(token)
    received = incoming.get(token, 0.0)
    sent = outgoing.get(token, 0.0)
    net = received - sent
    if abs(net) <= 0:
        return None          # internal move, or the token was untouched

    first = legs[0]
    common = {
        "timestamp": first.get("timestamp", 0),
        "tx_hash": first.get("hash", ""),
        "chain": chain,
        "token_address": token,
    }

    if net > 0:
        quote = _pick_quote(outgoing, meta, chain, token)
        if quote is None:
            return Trade(kind="transfer_in", quantity=net, **common)
        address, quantity = quote
        return Trade(
            kind="buy", quantity=net, quote_address=address, quote_quantity=quantity,
            quote_symbol=meta.get(address, {}).get("symbol", ""), **common,
        )

    quote = _pick_quote(incoming, meta, chain, token)
    if quote is None:
        return Trade(kind="transfer_out", quantity=net, **common)
    address, quantity = quote
    return Trade(
        kind="sell", quantity=net, quote_address=address, quote_quantity=quantity,
        quote_symbol=meta.get(address, {}).get("symbol", ""), **common,
    )


def trades_for_token(ledger: Ledger, token: str, chain: str, owned: set) -> List[Trade]:
    """Every trade in the ledger that touched one token, oldest first."""
    grouped = group_by_tx(list(ledger.transfers) + list(ledger.native))
    trades: List[Trade] = []
    for legs in grouped.values():
        trade = classify_transaction(legs, token, chain, owned)
        if trade is not None:
            trades.append(trade)
    trades.sort(key=lambda trade: trade.timestamp)
    return trades


# ==========================================================================
# Historical pricing
# ==========================================================================
def price_bucket(timestamp: int) -> int:
    """Round a timestamp so nearby trades share one cached price."""
    size = max(1, config.COST_BASIS_PRICE_BUCKET_SECONDS)
    return int(timestamp) // size * size


def coin_id(chain: str, address: str, symbol: str = "", native: bool = False) -> Optional[str]:
    """DefiLlama coin identifier for a quote leg, or ``None`` if unpriceable."""
    if native or address == "native":
        return NATIVE_PRICE_IDS.get((symbol or config.get_chain(chain).native_symbol).upper())
    slug = COINS_CHAIN_SLUGS.get(chain)
    if not slug:
        # A chain DefiLlama does not index (Robinhood) can still price the
        # wrapped native coin through its canonical id.
        if normalize_address(address) == normalize_address(config.WRAPPED_NATIVE.get(chain, "")):
            return NATIVE_PRICE_IDS.get(config.get_chain(chain).native_symbol.upper())
        return None
    return f"{slug}:{address}"


def stablecoin_price(chain: str, address: str) -> Optional[float]:
    """$1.00 for a known stablecoin leg — exact, and costs no request."""
    table = {k.lower(): v for k, v in config.STABLECOINS.get(chain, {}).items()}
    return 1.0 if normalize_address(address).lower() in table else None


def fetch_historical_prices(
    wanted: Dict[str, List[int]],
    session: Optional[Callable] = None,
) -> Dict[Tuple[str, int], float]:
    """Batch-fetch prices as of specific timestamps.

    ``wanted`` maps a DefiLlama coin id to the timestamps needed. Returns
    ``{(coin_id, timestamp): price}``; anything the API cannot answer is simply
    absent, and the caller treats that as unpriced rather than as zero.
    """
    if not wanted:
        return {}
    from .data_fetchers import FetchError, _get_json

    payload_arg = json.dumps({coin: sorted(set(stamps)) for coin, stamps in wanted.items()})
    url = f"{config.DEFILLAMA_COINS_BASE}/batchHistorical"
    try:
        payload = (
            session(url, {"coins": payload_arg}) if session
            else _get_json(url, params={"coins": payload_arg})
        )
    except FetchError as exc:
        logger.info("Historical price lookup failed: %s", exc)
        return {}

    prices: Dict[Tuple[str, int], float] = {}
    for coin, entry in ((payload or {}).get("coins") or {}).items():
        for point in (entry or {}).get("prices") or []:
            stamp = safe_int(point.get("timestamp"))
            price = safe_float(point.get("price"))
            if stamp and price > 0:
                prices[(coin, stamp)] = price
    return prices


def price_trades(
    trades: Sequence[Trade],
    chain: str,
    session: Optional[Callable] = None,
    db_path: Optional[Any] = None,
) -> List[str]:
    """Attach a USD value to every buy and sell. Returns notes about failures.

    Order of resort: stablecoin (exact, free) -> local cache (a past price
    cannot change) -> one batched network call for whatever is left.
    """
    notes: List[str] = []
    pending: Dict[str, List[int]] = {}
    lookups: List[Tuple[Trade, str, int]] = []

    for trade in trades:
        if trade.kind not in ("buy", "sell") or not trade.quote_address:
            continue

        stable = stablecoin_price(chain, trade.quote_address)
        if stable is not None:
            trade.quote_price_usd = stable
            trade.usd_value = trade.quote_quantity * stable
            continue

        native = trade.quote_address == "native"
        identifier = coin_id(chain, trade.quote_address, trade.quote_symbol, native=native)
        if not identifier:
            trade.note = "no price source for this quote token"
            continue

        bucket = price_bucket(trade.timestamp)
        cached = portfolio_store.get_cached_price(chain, identifier, bucket, db_path=db_path)
        if cached is not None:
            trade.quote_price_usd = cached
            trade.usd_value = trade.quote_quantity * cached
            continue

        pending.setdefault(identifier, []).append(bucket)
        lookups.append((trade, identifier, bucket))

    if pending:
        fetched = fetch_historical_prices(pending, session=session)
        to_cache: List[Dict[str, Any]] = []
        for trade, identifier, bucket in lookups:
            price = fetched.get((identifier, bucket))
            if price is None:
                # The API answers on its own grid, so fall back to the closest
                # point it returned for this coin -- closest, not merely the
                # first within range, or a trade can be priced hours off.
                nearby = [
                    (abs(stamp - bucket), value)
                    for (coin, stamp), value in fetched.items()
                    if coin == identifier
                    and abs(stamp - bucket) <= config.COST_BASIS_PRICE_BUCKET_SECONDS
                ]
                if nearby:
                    price = min(nearby)[1]
            if price is None:
                trade.note = "price unavailable at that time"
                continue
            trade.quote_price_usd = price
            trade.usd_value = trade.quote_quantity * price
            to_cache.append({"chain": chain, "address": identifier,
                             "bucket": bucket, "price_usd": price})
        if to_cache:
            portfolio_store.cache_prices(to_cache, db_path=db_path)

    unpriced = sum(1 for t in trades if t.kind in ("buy", "sell") and not t.priced)
    if unpriced:
        notes.append(
            f"{unpriced} trade(s) could not be priced at the time they happened; "
            "they are excluded from the average rather than counted as free."
        )
    return notes


# ==========================================================================
# Weighted-average engine
# ==========================================================================
def build_report(
    trades: Sequence[Trade],
    current_quantity: float,
    chain: str,
    token: str,
    symbol: str = "",
    notes: Optional[List[str]] = None,
) -> CostBasisReport:
    """Walk trades oldest-first into a weighted-average basis.

    Two pools are tracked: tokens whose cost is known, and tokens that arrived
    without one. A sale draws proportionally from both, so an unexplained
    airdrop can never masquerade as a free lot that inflates realized profit.
    """
    report = CostBasisReport(
        chain=chain, token_address=normalize_address(token), symbol=symbol,
        trades=list(trades), current_quantity=current_quantity,
        derived_at=utcnow_iso(), notes=list(notes or []),
    )

    explained_qty = 0.0
    explained_cost = 0.0
    unexplained_qty = 0.0
    realized = 0.0
    airdropped = 0.0

    for trade in sorted(trades, key=lambda t: t.timestamp):
        quantity = abs(trade.quantity)
        if quantity <= 0:
            continue

        if trade.kind == "buy":
            if trade.priced:
                explained_qty += quantity
                explained_cost += trade.usd_value or 0.0
            else:
                unexplained_qty += quantity
            continue

        if trade.kind == "transfer_in":
            if config.COST_BASIS_COUNT_AIRDROPS_AS_ZERO:
                explained_qty += quantity        # counted, at a cost of zero
            else:
                unexplained_qty += quantity
            airdropped += quantity
            continue

        # Sells and transfers out draw proportionally from both pools.
        held = explained_qty + unexplained_qty
        if held <= 0:
            continue
        share = min(1.0, quantity / held)
        from_explained = explained_qty * share
        cost_removed = explained_cost * share

        if trade.kind == "sell" and trade.priced and quantity > 0:
            proceeds = (trade.usd_value or 0.0) * (from_explained / quantity)
            realized += proceeds - cost_removed

        explained_qty -= from_explained
        explained_cost -= cost_removed
        unexplained_qty -= unexplained_qty * share

    report.quantity_explained = max(0.0, explained_qty)
    report.total_cost_usd = max(0.0, explained_cost)
    report.realized_pnl_usd = realized
    report.avg_cost_usd = (explained_cost / explained_qty) if explained_qty > 0 else None

    reconstructed = explained_qty + unexplained_qty
    if airdropped > 0 and not config.COST_BASIS_COUNT_AIRDROPS_AS_ZERO:
        report.notes.append(
            f"{airdropped:,.4g} token(s) arrived without a purchase (airdrop, bridge, or a "
            "transfer from an address you have not registered), so they have no cost basis."
        )
    if current_quantity > 0 and reconstructed > 0:
        drift = abs(reconstructed - current_quantity) / current_quantity * 100.0
        if drift > 5:
            report.notes.append(
                f"History accounts for {reconstructed:,.4g} tokens but you hold "
                f"{current_quantity:,.4g} — the difference predates the available history."
            )
    coverage = report.coverage_pct
    if coverage is not None and coverage < config.COST_BASIS_MIN_COVERAGE_PCT:
        report.notes.append(
            f"Only {coverage:.0f}% of your balance has a known cost, so the average covers "
            "that portion and is labelled partial."
        )
    if report.avg_cost_usd is None:
        report.notes.append("No priced purchase found, so there is no cost basis to show.")
    return report


# ==========================================================================
# Public API
# ==========================================================================
def derive_for_chain(
    chain: str,
    positions: Sequence[Position],
    wallets: Sequence[Wallet],
    session: Optional[Callable] = None,
    db_path: Optional[Any] = None,
    progress: Optional[Callable[[str, float], None]] = None,
) -> Dict[str, CostBasisReport]:
    """Derive basis for every position on one chain, from one history fetch.

    One fetch covers every position on the chain: the ledger is the wallet's
    whole history, and each token simply reads its own trades out of it.
    """
    chain_wallets = [w for w in wallets if w.chain == chain]
    targets = [p for p in positions if p.chain == chain]
    if not chain_wallets or not targets:
        return {}

    if config.get_chain(chain).address_kind == "solana":
        ledger = fetch_solana_ledger(chain_wallets, session=session, progress=progress)
    else:
        ledger = fetch_evm_ledger(chain, chain_wallets)
    owned = {normalize_address(w.address) for w in chain_wallets}

    reports: Dict[str, CostBasisReport] = {}
    for position in targets:
        trades = trades_for_token(ledger, position.address, chain, owned)
        notes = list(ledger.warnings)
        if not ledger.ok:
            reports[position.key] = CostBasisReport(
                chain=chain, token_address=position.address, symbol=position.symbol,
                current_quantity=position.quantity, ok=False,
                error="; ".join(ledger.warnings) or "History unavailable.",
                notes=notes, derived_at=utcnow_iso(),
            )
            continue
        notes.extend(price_trades(trades, chain, session=session, db_path=db_path))
        reports[position.key] = build_report(
            trades, position.quantity, chain, position.address,
            symbol=position.symbol, notes=notes,
        )
    return reports


def derive_position(
    position: Position,
    wallets: Optional[Sequence[Wallet]] = None,
    session: Optional[Callable] = None,
    persist: bool = True,
    db_path: Optional[Any] = None,
    progress: Optional[Callable[[str, float], None]] = None,
) -> Optional[CostBasisReport]:
    """Derive one position on request, whatever chain it is on.

    This is the route Solana takes: the caller asked for it, knows it costs a
    call per transaction, and gets progress while it runs.
    """
    registered = list(
        wallets if wallets is not None else portfolio_store.list_wallets(db_path=db_path)
    )
    reports = derive_for_chain(
        position.chain, [position], registered,
        session=session, db_path=db_path, progress=progress,
    )
    report = reports.get(position.key)
    if report and persist and report.ok and (
        report.avg_cost_usd is not None or report.realized_pnl_usd
    ):
        portfolio_store.save_derived_basis(report, db_path=db_path)
    return report


def derive_for_positions(
    positions: Sequence[Position],
    wallets: Optional[Sequence[Wallet]] = None,
    only_missing: bool = True,
    persist: bool = True,
    db_path: Optional[Any] = None,
    progress: Optional[Callable[[str, float], None]] = None,
) -> Dict[str, CostBasisReport]:
    """Derive cost basis across chains, skipping work that is already done.

    ``only_missing`` leaves alone any position that already has a basis, which
    is what makes this safe to run on every sync: historical prices never
    change, so a derived basis is recomputed only on request.
    """
    registered = list(
        wallets if wallets is not None else portfolio_store.list_wallets(db_path=db_path)
    )
    candidates = [
        position for position in positions
        if not (only_missing and position.avg_cost_usd is not None)
    ]
    # Solana history costs one RPC call per transaction, so it is opted into
    # explicitly rather than swept up by an automatic pass.
    evm = [
        position for position in candidates
        if config.get_chain(position.chain).address_kind == "evm"
    ]
    if not evm:
        return {}

    reports: Dict[str, CostBasisReport] = {}
    chains = sorted({position.chain for position in evm})
    for index, chain in enumerate(chains):
        if progress:
            progress(config.get_chain(chain).label, index / max(1, len(chains)))
        try:
            reports.update(derive_for_chain(chain, evm, registered, db_path=db_path))
        except Exception as exc:  # noqa: BLE001 - one chain must not kill the run
            logger.warning("Cost basis derivation failed on %s: %s", chain, exc)

    if persist:
        for report in reports.values():
            if report.ok and (report.avg_cost_usd is not None or report.realized_pnl_usd):
                portfolio_store.save_derived_basis(report, db_path=db_path)
    if progress:
        progress("done", 1.0)
    return reports

