"""Turn wallet balances into a priced, annotated book.

The split is deliberate: :mod:`src.balances` answers "what do I hold", this
module answers "what is it worth, what is it grouped with, and how has it
moved". Pricing reuses the DexScreener batch endpoint the rest of the app
already relies on, so positions are valued exactly the way the analyzer values
a token -- one source of truth, one cache, one rate limit.

What this module will not do is invent a cost basis. A wallet read gives
quantity, never entry price, so ``avg_cost_usd`` stays ``None`` until you type
one in, and P&L reports "basis unknown" rather than a confident zero. The
stored snapshot history covers the gap: every sync is persisted, so
performance *since the first sync* is always available for free.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import balances as balances_mod
from . import config, data_fetchers, portfolio_store
from .models import PortfolioSnapshot, Position, TokenSnapshot, Wallet
from .utils import is_valid_address, normalize_address, safe_float, utcnow_iso

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Wallet input
# --------------------------------------------------------------------------
def parse_wallet_input(raw: str, chain: str) -> List[Wallet]:
    """Parse pasted wallet lines into :class:`Wallet` entries for one chain.

    Same forgiving format as the smart-money watchlist: address first, optional
    label after a comma, tab or space, ``#`` comments and blank lines ignored,
    invalid rows skipped rather than rejecting the whole paste.
    """
    wallets: List[Wallet] = []
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
        # A Solana address pasted under an EVM chain (or the reverse) reads as
        # valid but would be queried against the wrong network entirely.
        if _address_kind(address) != config.get_chain(chain).address_kind:
            continue
        key = normalize_address(address)
        if key in seen:
            continue
        seen.add(key)
        wallets.append(Wallet(address=address, chain=chain, label=label.strip(),
                              added_at=utcnow_iso()))
    return wallets


def _address_kind(address: str) -> Optional[str]:
    from .utils import detect_address_kind
    return detect_address_kind(address)


def wallet_input_errors(raw: str, chain: str) -> List[str]:
    """Lines that were skipped, and why — so a typo is visible, not silent."""
    errors: List[str] = []
    expected = config.get_chain(chain).address_kind
    for line in (raw or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        address = re.split(r"[,\t ]", line, maxsplit=1)[0].strip()
        if not is_valid_address(address):
            errors.append(f"`{address[:24]}` is not a valid address")
        elif _address_kind(address) != expected:
            errors.append(
                f"`{address[:12]}…` looks like a "
                f"{'Solana' if _address_kind(address) == 'solana' else 'EVM'} address, "
                f"but {config.get_chain(chain).label} expects "
                f"{'Solana' if expected == 'solana' else 'EVM'}"
            )
    return errors


# --------------------------------------------------------------------------
# Pricing
# --------------------------------------------------------------------------
def price_tokens(
    chain: str,
    addresses: Sequence[str],
    use_cache: bool = True,
) -> Dict[str, TokenSnapshot]:
    """Price a chain's tokens in batches, keyed by normalized address.

    Reuses ``fetch_pairs_for_addresses`` (30 per call, already TTL-cached) and
    ``collapse_pairs_to_tokens``, so a position is valued off the same primary
    pair the analyzer would pick.
    """
    if not addresses:
        return {}
    chain_id = config.get_chain(chain).dexscreener_id
    pairs = data_fetchers.fetch_pairs_for_addresses(chain_id, list(addresses), use_cache=use_cache)
    return {
        normalize_address(snapshot.address): snapshot
        for snapshot in data_fetchers.collapse_pairs_to_tokens(pairs)
    }


# --------------------------------------------------------------------------
# Position building
# --------------------------------------------------------------------------
def build_positions(
    token_balances: Sequence[balances_mod.TokenBalance],
    meta: Optional[Dict[str, Dict[str, Any]]] = None,
    use_cache: bool = True,
) -> Tuple[List[Position], List[Dict[str, Any]], float, int]:
    """Price balances and fold them into positions.

    Returns ``(positions, unpriced, dust_usd, dust_count)``. Positions below
    the dust threshold are summed into ``dust_usd`` rather than dropped, so the
    total always reconciles with the wallet.
    """
    meta = meta or {}

    # Merge the same token held in several wallets into one position.
    merged: Dict[str, Position] = {}
    for balance in token_balances:
        if balance.quantity <= 0:
            continue
        key = f"{balance.chain}:{normalize_address(balance.address)}"
        position = merged.get(key)
        if position is None:
            position = Position(
                address=balance.address, chain=balance.chain, quantity=0.0,
                symbol=balance.symbol, name=balance.name, source=balance.source,
            )
            merged[key] = position
        position.quantity += balance.quantity
        if balance.wallet and balance.wallet not in position.wallets:
            position.wallets.append(balance.wallet)
        position.symbol = position.symbol or balance.symbol
        position.name = position.name or balance.name

    # Price one chain at a time: the batch endpoint is per-chain.
    by_chain: Dict[str, List[Position]] = {}
    for position in merged.values():
        by_chain.setdefault(position.chain, []).append(position)

    positions: List[Position] = []
    unpriced: List[Dict[str, Any]] = []
    dust_usd = 0.0
    dust_count = 0

    for chain, chain_positions in by_chain.items():
        prices = price_tokens(
            chain, [p.address for p in chain_positions], use_cache=use_cache
        )
        for position in chain_positions:
            snapshot = prices.get(normalize_address(position.address))
            if snapshot is None:
                position.priced = False
                unpriced.append({
                    "chain": chain,
                    "address": position.address,
                    "symbol": position.symbol or "?",
                    "quantity": position.quantity,
                    "reason": "No DexScreener pair — illiquid, unlisted or too new to price.",
                })
                continue
            position.snapshot = snapshot
            position.price_usd = snapshot.price_usd
            position.value_usd = snapshot.price_usd * position.quantity
            position.symbol = snapshot.symbol or position.symbol
            position.name = snapshot.name or position.name

            annotations = meta.get(position.key) or {}
            position.avg_cost_usd = annotations.get("avg_cost_usd")
            position.tag = annotations.get("tag") or ""
            position.note = annotations.get("note") or ""
            position.first_seen = annotations.get("first_seen") or ""
            position.basis_source = annotations.get("basis_source") or ""
            position.basis_coverage_pct = annotations.get("basis_coverage_pct")
            position.realized_pnl_usd = annotations.get("realized_pnl_usd")
            stored_notes = annotations.get("basis_notes")
            if stored_notes:
                try:
                    position.basis_notes = json.loads(stored_notes)
                except (TypeError, ValueError):
                    position.basis_notes = [str(stored_notes)]

            if position.value_usd < config.PORTFOLIO_DUST_USD:
                dust_usd += position.value_usd
                dust_count += 1
                continue
            positions.append(position)

    positions.sort(key=lambda p: p.value_usd, reverse=True)
    return positions, unpriced, dust_usd, dust_count


# --------------------------------------------------------------------------
# Sync
# --------------------------------------------------------------------------
def sync_portfolio(
    wallets: Optional[Sequence[Wallet]] = None,
    manual_tokens: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    use_cache: bool = True,
    persist: bool = True,
    derive_basis: bool = True,
    db_path: Optional[Any] = None,
) -> PortfolioSnapshot:
    """Read every wallet, price the result, and store the snapshot.

    Never raises: a chain that cannot be read contributes a coverage note, and
    the rest of the book still renders.
    """
    registered = list(wallets if wallets is not None else portfolio_store.list_wallets(db_path=db_path))
    snapshot = PortfolioSnapshot(taken_at=utcnow_iso(), wallets_synced=len(registered))
    if not registered:
        snapshot.warnings.append("No wallets registered yet — add one to start tracking.")
        return snapshot

    read = balances_mod.fetch_all_balances(registered, manual_tokens=manual_tokens)
    snapshot.warnings.extend(read.warnings)

    meta = portfolio_store.all_position_meta(db_path=db_path)
    positions, unpriced, dust_usd, dust_count = build_positions(
        read.balances, meta=meta, use_cache=use_cache
    )
    snapshot.positions = positions
    snapshot.unpriced = unpriced
    snapshot.dust_usd = dust_usd
    snapshot.dust_count = dust_count

    # Name every chain that produced nothing, and why, so an unreadable chain
    # is never mistaken for an empty one.
    chains_with_wallets = {wallet.chain for wallet in registered}
    chains_with_positions = {position.chain for position in positions}
    for chain in sorted(chains_with_wallets - chains_with_positions):
        snapshot.coverage_notes[chain] = balances_mod.provider_status(chain)

    if persist:
        first_seen_now = snapshot.taken_at
        for position in positions:
            if not position.first_seen:
                position.first_seen = first_seen_now
                portfolio_store.set_position_meta(
                    position.chain, position.address,
                    first_seen=first_seen_now, db_path=db_path,
                )

    # Basis is derived before the snapshot is written, so the stored payload
    # carries it too -- otherwise a position's first snapshot would record it
    # as having no cost basis when it does.
    if derive_basis:
        attach_cost_basis(snapshot, registered, db_path=db_path)
    if persist:
        portfolio_store.record_snapshot(snapshot, db_path=db_path)
    return snapshot


def attach_cost_basis(
    snapshot: PortfolioSnapshot,
    wallets: Optional[Sequence[Wallet]] = None,
    only_missing: bool = True,
    db_path: Optional[Any] = None,
    progress: Optional[Any] = None,
) -> Dict[str, Any]:
    """Reconstruct cost basis for positions that do not have one yet.

    Deliberately conservative about what it touches: a basis you typed in is
    never replaced, and a derived one is recomputed only when explicitly asked
    for, because the historical prices behind it cannot change.
    """
    from . import cost_basis      # imported here to keep module import cheap

    reports = cost_basis.derive_for_positions(
        snapshot.positions, wallets=wallets, only_missing=only_missing,
        db_path=db_path, progress=progress,
    )
    for position in snapshot.positions:
        report = reports.get(position.key)
        if report is None or not report.ok:
            continue
        if position.avg_cost_usd is None and report.avg_cost_usd is not None:
            position.avg_cost_usd = report.avg_cost_usd
            position.basis_source = "derived"
            position.basis_coverage_pct = report.coverage_pct
        if report.realized_pnl_usd:
            position.realized_pnl_usd = report.realized_pnl_usd
        if report.notes:
            position.basis_notes = list(report.notes)
    return reports


# --------------------------------------------------------------------------
# Derived views
# --------------------------------------------------------------------------
def position_changes(
    snapshot: PortfolioSnapshot,
    previous: Optional[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    """Per-position value delta against the previous stored snapshot.

    This is the "performance since last sync" that fills in for an unknown cost
    basis, and it is what the rotation engine reads liquidity trend from.
    """
    if not previous:
        return {}
    earlier = {
        f"{row.get('chain')}:{normalize_address(row.get('address', ''))}": row
        for row in (previous.get("positions") or [])
    }
    changes: Dict[str, Dict[str, float]] = {}
    for position in snapshot.positions:
        before = earlier.get(position.key)
        if not before:
            continue
        prior_value = safe_float(before.get("value_usd"))
        prior_qty = safe_float(before.get("quantity"))
        prior_liq = safe_float((before.get("snapshot") or {}).get("liquidity_usd"))
        changes[position.key] = {
            "value_delta_usd": position.value_usd - prior_value,
            "value_delta_pct": ((position.value_usd - prior_value) / prior_value * 100.0)
            if prior_value else 0.0,
            "quantity_delta": position.quantity - prior_qty,
            "liquidity_delta_pct": (
                ((position.snapshot.liquidity_usd - prior_liq) / prior_liq * 100.0)
                if prior_liq and position.snapshot else 0.0
            ),
        }
    return changes


def performance_since_first_sync(
    position: Position,
    curve: Sequence[Dict[str, Any]],
) -> Optional[float]:
    """Percent change in a position's price since the earliest snapshot holding it.

    The honest stand-in for P&L when no cost basis was entered: it measures
    what the app has actually observed rather than guessing an entry price.
    """
    for row in curve:
        payload = row.get("payload") if isinstance(row, dict) else None
        if not payload:
            continue
        for stored in payload.get("positions") or []:
            if f"{stored.get('chain')}:{normalize_address(stored.get('address', ''))}" == position.key:
                first_price = safe_float(stored.get("price_usd"))
                if first_price > 0 and position.price_usd > 0:
                    return (position.price_usd - first_price) / first_price * 100.0
                return None
    return None


def totals_by_tag(snapshot: PortfolioSnapshot) -> List[Dict[str, Any]]:
    """Ecosystem roll-up, e.g. every Brew token on BNB as one line."""
    groups: Dict[str, Dict[str, Any]] = {}
    total = snapshot.total_usd
    for position in snapshot.positions:
        label = position.tag or "Untagged"
        group = groups.setdefault(label, {
            "tag": label, "value_usd": 0.0, "positions": 0, "chains": set(),
            "weighted_change_24h": 0.0,
        })
        group["value_usd"] += position.value_usd
        group["positions"] += 1
        group["chains"].add(position.chain)
        if position.snapshot:
            group["weighted_change_24h"] += (
                position.snapshot.price_change_24h * position.value_usd
            )

    rows: List[Dict[str, Any]] = []
    for group in groups.values():
        value = group["value_usd"]
        rows.append({
            "tag": group["tag"],
            "value_usd": value,
            "positions": group["positions"],
            "chains": ", ".join(sorted(group["chains"])),
            "allocation_pct": (value / total * 100.0) if total > 0 else 0.0,
            "change_24h": (group["weighted_change_24h"] / value) if value else 0.0,
        })
    return sorted(rows, key=lambda row: row["value_usd"], reverse=True)
