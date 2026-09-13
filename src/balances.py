"""Read what a wallet actually holds, straight off-chain.

There is no platform API behind this: balances come from public JSON-RPC nodes
and public explorers, so nothing here needs a key with spend authority, and
none of it can move a coin.

Three providers, each with an ``unavailable_reason()`` in the same style as
:class:`src.wallet_flow.EtherscanClient`, so a chain that cannot be read says
why instead of silently reporting an empty wallet:

``BlockscoutProvider``  one unauthenticated call returns every token balance --
                        the only route on chains Etherscan V2 does not index,
                        Robinhood Chain among them.
``EvmRpcProvider``      Etherscan V2 discovers which tokens an address has ever
                        touched, then ``balanceOf`` on a public RPC reads the
                        current balance. Discovery needs the (free) Etherscan
                        key; the balance read needs no key at all.
``SolanaRpcProvider``   ``getTokenAccountsByOwner`` returns mints and amounts in
                        one call, for both the SPL Token and Token-2022
                        programs.

Every provider returns :class:`TokenBalance` rows with a raw quantity. Pricing
is deliberately somewhere else (:mod:`src.portfolio`), so balances stay
testable without a market data round trip.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import httpx

from . import config
from .models import Wallet
from .utils import TTLCache, dig, normalize_address, safe_float, safe_int

logger = logging.getLogger(__name__)

# Transfer history is fetched once per wallet and reused by both token
# discovery and the cost-basis engine.
_ledger_cache = TTLCache(ttl_seconds=config.CACHE_TTL_PORTFOLIO)


def _etherscan_rows(payload: Any, provider: str) -> List[Dict[str, Any]]:
    """Unwrap an Etherscan-shaped response into rows.

    Both Etherscan V2 and Blockscout answer in this shape. An address with no
    history is a legitimate empty answer; anything else is a failure and is
    raised rather than returned as "no transactions".
    """
    result = (payload or {}).get("result")
    if str((payload or {}).get("status", "")) != "1":
        message = (payload or {}).get("message") or ""
        if isinstance(result, list) and not result:
            return []
        if "no transactions" in str(message).lower() or "not found" in str(message).lower():
            return []
        raise RuntimeError(f"{provider}: {message or result or 'unknown error'}")
    return result if isinstance(result, list) else []


def clear_ledger_cache() -> None:
    """Drop cached wallet history (used by the sidebar refresh button)."""
    _ledger_cache.clear()

# ERC-20 function selectors (first 4 bytes of the keccak hash of the signature).
SELECTOR_BALANCE_OF = "0x70a08231"
SELECTOR_DECIMALS = "0x313ce567"

# SPL Token and Token-2022. Newer meme launches increasingly use the latter, so
# querying only the classic program would quietly miss those positions.
SPL_TOKEN_PROGRAM = "TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA"
SPL_TOKEN_2022_PROGRAM = "TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb"

# Chains whose native coin is ETH, for labelling a native balance row.
_NATIVE_DECIMALS = 18
_SOL_DECIMALS = 9


@dataclass
class TokenBalance:
    """One token held by one wallet, before pricing."""

    address: str
    chain: str
    quantity: float
    symbol: str = ""
    name: str = ""
    decimals: Optional[int] = None
    wallet: str = ""
    source: str = ""
    is_native: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BalanceResult:
    """What one provider managed to read for one wallet."""

    balances: List[TokenBalance] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    source: str = ""
    ok: bool = True

    def extend(self, other: "BalanceResult") -> None:
        self.balances.extend(other.balances)
        self.warnings.extend(other.warnings)


# --------------------------------------------------------------------------
# JSON-RPC plumbing
# --------------------------------------------------------------------------
def _post_json(url: str, payload: Any, retries: Optional[int] = None) -> Any:
    """POST a JSON-RPC document, with the same retry policy as ``_get_json``.

    Lives here rather than in ``data_fetchers`` because that module is strictly
    GET-based; the shared, pooled client is still reused.
    """
    from .data_fetchers import FetchError, get_client   # local import avoids a cycle

    attempts = (retries if retries is not None else config.HTTP_MAX_RETRIES) + 1
    last_error: Optional[str] = None
    for attempt in range(attempts):
        try:
            response = get_client().post(url, json=payload)
            if response.status_code == 200:
                return response.json()
            if response.status_code == 429 or response.status_code >= 500:
                last_error = f"HTTP {response.status_code}"
            else:
                raise FetchError(f"HTTP {response.status_code} from {url}")
        except FetchError:
            raise
        except (httpx.HTTPError, ValueError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        if attempt < attempts - 1:
            time.sleep(0.6 * (2 ** attempt))
    raise FetchError(f"{last_error or 'unknown error'} from {url}")


def decode_uint(hex_value: Any) -> Optional[int]:
    """Decode a 32-byte hex word into an int. ``None`` when it is not one.

    An RPC that answers ``0x`` (no contract at that address) must not be read
    as a balance of zero -- that is the difference between "you hold none" and
    "we asked the wrong chain".
    """
    if not isinstance(hex_value, str):
        return None
    raw = hex_value.strip()
    if not raw.startswith("0x") or len(raw) <= 2:
        return None
    try:
        return int(raw, 16)
    except ValueError:
        return None


def units_from_raw(raw: Optional[int], decimals: Optional[int]) -> float:
    """Scale a raw on-chain integer by the token's decimals."""
    if raw is None:
        return 0.0
    places = _NATIVE_DECIMALS if decimals is None else max(0, min(int(decimals), 36))
    return raw / (10 ** places)


# --------------------------------------------------------------------------
# Blockscout: one call, every balance, no key
# --------------------------------------------------------------------------
class BlockscoutProvider:
    """Explorer-backed balances for any chain with a Blockscout instance."""

    def __init__(self, chain: str, base_url: str = "", session: Any = None) -> None:
        self.chain = chain
        self.base_url = (base_url or config.BLOCKSCOUT_BASE_URLS.get(chain, "")).rstrip("/")
        self._session = session       # injected in tests

    def unavailable_reason(self) -> str:
        if not self.base_url:
            label = config.get_chain(self.chain).label
            return f"No Blockscout instance configured for {label}."
        return ""

    def fetch(self, wallet: str) -> BalanceResult:
        result = BalanceResult(source="blockscout")
        reason = self.unavailable_reason()
        if reason:
            return BalanceResult(source="blockscout", ok=False, warnings=[reason])

        from .data_fetchers import FetchError, _get_json

        params = {"module": "account", "action": "tokenlist", "address": wallet}
        try:
            payload = (
                self._session(f"{self.base_url}/api", params) if self._session
                else _get_json(f"{self.base_url}/api", params=params)
            )
        except FetchError as exc:
            return BalanceResult(
                source="blockscout", ok=False,
                warnings=[f"Blockscout ({self.chain}) unreachable: {exc}"],
            )

        rows = (payload or {}).get("result")
        if not isinstance(rows, list):
            message = (payload or {}).get("message") or "unexpected response"
            # An address with no tokens is a legitimate empty answer.
            if "no token" in str(message).lower():
                return result
            return BalanceResult(
                source="blockscout", ok=False,
                warnings=[f"Blockscout ({self.chain}): {message}"],
            )

        for row in rows:
            if str(row.get("type", "")).upper().startswith("ERC-721"):
                continue                      # NFTs are not positions
            if str(row.get("type", "")).upper().startswith("ERC-1155"):
                continue
            decimals = safe_int(row.get("decimals"), 18)
            quantity = units_from_raw(safe_int(row.get("balance"), 0), decimals)
            address = row.get("contractAddress") or ""
            if not address or quantity <= 0:
                continue
            result.balances.append(
                TokenBalance(
                    address=address, chain=self.chain, quantity=quantity,
                    symbol=row.get("symbol") or "", name=row.get("name") or "",
                    decimals=decimals, wallet=wallet, source="blockscout",
                )
            )
        return result


    def token_transfers(self, wallet: str) -> List[Dict[str, Any]]:
        """ERC-20 transfer rows from Blockscout, in the Etherscan shape.

        This is the only route to transaction history on a chain Etherscan V2
        does not index -- Robinhood Chain among them.
        """
        if self.unavailable_reason():
            return []
        key = ("scout_tokentx", self.chain, normalize_address(wallet))
        cached = _ledger_cache.get(key)
        if cached is not None:
            return cached

        from .data_fetchers import FetchError, _get_json

        params = {"module": "account", "action": "tokentx", "address": wallet, "sort": "desc"}
        try:
            payload = (
                self._session(f"{self.base_url}/api", params) if self._session
                else _get_json(f"{self.base_url}/api", params=params)
            )
        except FetchError as exc:
            raise RuntimeError(f"Blockscout ({self.chain}) unreachable: {exc}") from exc

        rows = _etherscan_rows(payload, f"Blockscout ({self.chain})")
        _ledger_cache.set(key, rows)
        return rows

    def native_transactions(self, wallet: str) -> List[Dict[str, Any]]:
        """Outer transactions, for buys paid in the chain's native coin."""
        if self.unavailable_reason():
            return []
        key = ("scout_txlist", self.chain, normalize_address(wallet))
        cached = _ledger_cache.get(key)
        if cached is not None:
            return cached

        from .data_fetchers import FetchError, _get_json

        params = {"module": "account", "action": "txlist", "address": wallet, "sort": "desc"}
        try:
            payload = (
                self._session(f"{self.base_url}/api", params) if self._session
                else _get_json(f"{self.base_url}/api", params=params)
            )
            rows = _etherscan_rows(payload, f"Blockscout ({self.chain})")
        except (FetchError, RuntimeError) as exc:
            logger.info("Blockscout txlist failed on %s: %s", self.chain, exc)
            return []
        _ledger_cache.set(key, rows)
        return rows


# --------------------------------------------------------------------------
# EVM: Etherscan discovery + RPC balances
# --------------------------------------------------------------------------
class EvmRpcProvider:
    """Token balances for one EVM chain, read over public JSON-RPC."""

    def __init__(self, chain: str, rpc_url: str = "", api_key: str = "", session: Any = None) -> None:
        self.chain = chain
        self.rpc_url = rpc_url or config.EVM_RPC_URLS.get(chain, "")
        self.api_key = api_key or config.ETHERSCAN_API_KEY
        self._session = session       # injected in tests: (url, payload) -> response

    def unavailable_reason(self) -> str:
        chain_cfg = config.get_chain(self.chain)
        if not self.rpc_url:
            return (
                f"No RPC endpoint configured for {chain_cfg.label}. Set "
                f"{self.chain.upper()}_RPC_URL in .env to read balances there."
            )
        return ""

    def discovery_reason(self) -> str:
        """Why automatic token discovery cannot run (balances still can)."""
        chain_cfg = config.get_chain(self.chain)
        if chain_cfg.etherscan_chain_id is None:
            return (
                f"{chain_cfg.label} is not an Etherscan V2 chain, so tokens cannot be "
                "discovered automatically there."
            )
        if not self.api_key:
            return (
                "No ETHERSCAN_API_KEY set, so tokens cannot be discovered automatically. "
                "A free key at etherscan.io/apis covers every EVM chain."
            )
        return ""

    # -- discovery ---------------------------------------------------------
    def token_transfers(self, wallet: str) -> List[Dict[str, Any]]:
        """Raw ERC-20 transfer rows for this wallet, newest first.

        Cached, because the cost-basis engine wants exactly the same rows that
        discovery does -- fetching them twice would double the cost of a sync
        for no new information.
        """
        if self.discovery_reason():
            return []
        key = ("tokentx", self.chain, normalize_address(wallet))
        cached = _ledger_cache.get(key)
        if cached is not None:
            return cached

        from .data_fetchers import FetchError, _get_json

        chain_cfg = config.get_chain(self.chain)
        params = {
            "chainid": chain_cfg.etherscan_chain_id,
            "module": "account",
            "action": "tokentx",
            "address": wallet,
            "page": 1,
            "offset": max(1, min(config.PORTFOLIO_DISCOVERY_TRANSFERS, 10_000)),
            "sort": "desc",
            "apikey": self.api_key,
        }
        try:
            payload = (
                self._session(config.ETHERSCAN_BASE_URL, params) if self._session
                else _get_json(config.ETHERSCAN_BASE_URL, params=params)
            )
        except FetchError as exc:
            raise RuntimeError(f"Etherscan discovery failed: {exc}") from exc

        rows = _etherscan_rows(payload, "Etherscan")
        _ledger_cache.set(key, rows)
        return rows

    def native_transactions(self, wallet: str) -> List[Dict[str, Any]]:
        """Outer transactions, which carry the native-coin leg of a swap.

        A buy paid in ETH or BNB produces no ERC-20 transfer for the coin
        spent, so without this the trade looks like it arrived from nowhere.
        """
        if self.discovery_reason():
            return []
        key = ("txlist", self.chain, normalize_address(wallet))
        cached = _ledger_cache.get(key)
        if cached is not None:
            return cached

        from .data_fetchers import FetchError, _get_json

        chain_cfg = config.get_chain(self.chain)
        params = {
            "chainid": chain_cfg.etherscan_chain_id,
            "module": "account",
            "action": "txlist",
            "address": wallet,
            "page": 1,
            "offset": max(1, min(config.PORTFOLIO_DISCOVERY_TRANSFERS, 10_000)),
            "sort": "desc",
            "apikey": self.api_key,
        }
        try:
            payload = (
                self._session(config.ETHERSCAN_BASE_URL, params) if self._session
                else _get_json(config.ETHERSCAN_BASE_URL, params=params)
            )
        except FetchError as exc:
            logger.info("Etherscan txlist failed on %s: %s", self.chain, exc)
            return []
        try:
            rows = _etherscan_rows(payload, "Etherscan")
        except RuntimeError as exc:
            logger.info("Etherscan txlist failed on %s: %s", self.chain, exc)
            return []
        _ledger_cache.set(key, rows)
        return rows

    def discover_tokens(self, wallet: str) -> Dict[str, Dict[str, Any]]:
        """Every ERC-20 this wallet has ever received or sent, with decimals.

        Etherscan's transfer rows carry ``tokenDecimal`` and ``tokenSymbol``,
        so discovery also supplies the metadata a ``balanceOf`` call lacks.
        """
        tokens: Dict[str, Dict[str, Any]] = {}
        for row in self.token_transfers(wallet):
            address = row.get("contractAddress") or ""
            if not address:
                continue
            key = normalize_address(address)
            if key not in tokens:
                tokens[key] = {
                    "address": address,
                    "symbol": row.get("tokenSymbol") or "",
                    "name": row.get("tokenName") or "",
                    "decimals": safe_int(row.get("tokenDecimal"), 18),
                }
        return tokens

    # -- balances ----------------------------------------------------------
    def _rpc(self, method: str, params: List[Any]) -> Any:
        payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
        response = (
            self._session(self.rpc_url, payload) if self._session
            else _post_json(self.rpc_url, payload)
        )
        if isinstance(response, dict) and response.get("error"):
            raise RuntimeError(str(dig(response, "error", "message", default="RPC error")))
        return (response or {}).get("result") if isinstance(response, dict) else None

    def balance_of(self, token: str, wallet: str) -> Optional[int]:
        """Raw ``balanceOf`` result, or ``None`` when the call did not answer."""
        padded = normalize_address(wallet).replace("0x", "").rjust(64, "0")
        data = SELECTOR_BALANCE_OF + padded
        return decode_uint(self._rpc("eth_call", [{"to": token, "data": data}, "latest"]))

    def decimals(self, token: str) -> Optional[int]:
        value = decode_uint(self._rpc("eth_call", [{"to": token, "data": SELECTOR_DECIMALS}, "latest"]))
        return value if value is not None and 0 <= value <= 36 else None

    def native_balance(self, wallet: str) -> Optional[float]:
        raw = decode_uint(self._rpc("eth_getBalance", [wallet, "latest"]))
        return None if raw is None else units_from_raw(raw, _NATIVE_DECIMALS)

    def fetch(self, wallet: str, tokens: Optional[Sequence[Dict[str, Any]]] = None) -> BalanceResult:
        """Balances for the given tokens, or for everything discovered."""
        result = BalanceResult(source="rpc")
        reason = self.unavailable_reason()
        if reason:
            return BalanceResult(source="rpc", ok=False, warnings=[reason])

        candidates: List[Dict[str, Any]] = list(tokens or [])
        if not candidates:
            discovery_reason = self.discovery_reason()
            if discovery_reason:
                return BalanceResult(source="rpc", ok=False, warnings=[discovery_reason])
            try:
                candidates = list(self.discover_tokens(wallet).values())
            except RuntimeError as exc:
                return BalanceResult(source="rpc", ok=False, warnings=[str(exc)])

        # Cap the fan-out: one airdrop-spammed address should not turn a sync
        # into hundreds of RPC round trips.
        if len(candidates) > config.PORTFOLIO_MAX_TOKENS_PER_WALLET:
            result.warnings.append(
                f"{len(candidates)} tokens seen on {config.get_chain(self.chain).label}; "
                f"reading the first {config.PORTFOLIO_MAX_TOKENS_PER_WALLET}. Raise "
                "MEMEDD_PORTFOLIO_MAX_TOKENS to widen this."
            )
            candidates = candidates[: config.PORTFOLIO_MAX_TOKENS_PER_WALLET]

        failures = 0
        for token in candidates:
            address = token.get("address") or ""
            if not address:
                continue
            try:
                raw = self.balance_of(address, wallet)
            except Exception as exc:   # noqa: BLE001 - one dead call, not a dead sync
                failures += 1
                logger.info("balanceOf failed for %s on %s: %s", address, self.chain, exc)
                continue
            if raw is None or raw <= 0:
                continue
            decimals = token.get("decimals")
            if decimals is None:
                try:
                    decimals = self.decimals(address)
                except Exception:  # noqa: BLE001
                    decimals = None
            result.balances.append(
                TokenBalance(
                    address=address, chain=self.chain,
                    quantity=units_from_raw(raw, decimals),
                    symbol=token.get("symbol") or "", name=token.get("name") or "",
                    decimals=decimals, wallet=wallet, source="rpc",
                )
            )

        if failures:
            result.warnings.append(
                f"{failures} balance call(s) failed on {config.get_chain(self.chain).label} — "
                "those positions are missing from this sync, not empty."
            )
            result.ok = failures < len(candidates)

        # Native coin: the dry powder a rotation actually moves.
        try:
            native = self.native_balance(wallet)
        except Exception as exc:  # noqa: BLE001
            native = None
            logger.info("Native balance failed on %s: %s", self.chain, exc)
        if native:
            chain_cfg = config.get_chain(self.chain)
            wrapped = config.WRAPPED_NATIVE.get(self.chain, "")
            result.balances.append(
                TokenBalance(
                    address=wrapped or f"native:{self.chain}", chain=self.chain,
                    quantity=native, symbol=chain_cfg.native_symbol,
                    name=f"{chain_cfg.label} native", decimals=_NATIVE_DECIMALS,
                    wallet=wallet, source="rpc", is_native=True,
                )
            )
            if not wrapped:
                result.warnings.append(
                    f"{chain_cfg.native_symbol} on {chain_cfg.label} has no known wrapped "
                    "contract to price against, so it is listed unpriced."
                )
        return result


# --------------------------------------------------------------------------
# Solana
# --------------------------------------------------------------------------
class SolanaRpcProvider:
    """SPL balances via ``getTokenAccountsByOwner`` — no key, one call each."""

    def __init__(self, rpc_url: str = "", session: Any = None) -> None:
        self.chain = "solana"
        self.rpc_url = rpc_url or config.SOLANA_RPC_URL
        self._session = session

    def unavailable_reason(self) -> str:
        if not self.rpc_url:
            return "No SOLANA_RPC_URL set, so Solana balances cannot be read."
        return ""

    def _rpc(self, method: str, params: List[Any]) -> Any:
        payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
        response = (
            self._session(self.rpc_url, payload) if self._session
            else _post_json(self.rpc_url, payload)
        )
        if isinstance(response, dict) and response.get("error"):
            raise RuntimeError(str(dig(response, "error", "message", default="RPC error")))
        return (response or {}).get("result") if isinstance(response, dict) else None

    def fetch(self, wallet: str) -> BalanceResult:
        result = BalanceResult(source="solana-rpc")
        reason = self.unavailable_reason()
        if reason:
            return BalanceResult(source="solana-rpc", ok=False, warnings=[reason])

        # Both token programs: classic SPL and Token-2022. Querying only the
        # former would quietly miss newer launches.
        totals: Dict[str, Dict[str, Any]] = {}
        reached = 0
        for program in (SPL_TOKEN_PROGRAM, SPL_TOKEN_2022_PROGRAM):
            try:
                accounts = self._rpc(
                    "getTokenAccountsByOwner",
                    [wallet, {"programId": program}, {"encoding": "jsonParsed"}],
                )
            except Exception as exc:  # noqa: BLE001
                logger.info("Solana token accounts failed (%s): %s", program[:8], exc)
                continue
            reached += 1
            for entry in (dig(accounts, "value", default=[]) or []):
                info = dig(entry, "account", "data", "parsed", "info", default={}) or {}
                mint = info.get("mint") or ""
                amount = dig(info, "tokenAmount", "uiAmount", default=None)
                if amount is None:
                    # uiAmount can be null for exotic decimals; fall back to raw.
                    amount = units_from_raw(
                        safe_int(dig(info, "tokenAmount", "amount", default=0)),
                        safe_int(dig(info, "tokenAmount", "decimals", default=0)),
                    )
                quantity = safe_float(amount)
                if not mint or quantity <= 0:
                    continue
                # One mint can sit in several token accounts; they are one position.
                bucket = totals.setdefault(
                    mint,
                    {"quantity": 0.0, "decimals": safe_int(dig(info, "tokenAmount", "decimals", default=0))},
                )
                bucket["quantity"] += quantity

        if not reached:
            return BalanceResult(
                source="solana-rpc", ok=False,
                warnings=["Solana RPC unreachable — no Solana positions in this sync."],
            )

        for mint, bucket in totals.items():
            result.balances.append(
                TokenBalance(
                    address=mint, chain="solana", quantity=bucket["quantity"],
                    decimals=bucket["decimals"], wallet=wallet, source="solana-rpc",
                )
            )

        # Native SOL, priced later against the wrapped-SOL mint.
        try:
            lamports = dig(self._rpc("getBalance", [wallet]), "value", default=None)
        except Exception as exc:  # noqa: BLE001
            lamports = None
            logger.info("Solana getBalance failed: %s", exc)
        if lamports:
            result.balances.append(
                TokenBalance(
                    address=config.WRAPPED_NATIVE["solana"], chain="solana",
                    quantity=units_from_raw(safe_int(lamports), _SOL_DECIMALS),
                    symbol="SOL", name="Solana", decimals=_SOL_DECIMALS,
                    wallet=wallet, source="solana-rpc", is_native=True,
                )
            )
        return result


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def describe_provider(chain: str) -> Tuple[bool, str]:
    """``(ready, plain-language explanation)`` for reading one chain.

    Written for someone deciding whether they still have setting up to do, so
    it says what it means rather than naming the provider and leaving them to
    infer whether that is good news.

    "Ready" means a provider is configured, not that it has answered: only the
    sync itself proves the connection.
    """
    chain_cfg = config.get_chain(chain)

    if chain_cfg.address_kind == "solana":
        reason = SolanaRpcProvider().unavailable_reason()
        if reason:
            return False, reason
        return True, "Ready — reads Solana directly, no key needed."

    blockscout = BlockscoutProvider(chain)
    rpc = EvmRpcProvider(chain)
    if not blockscout.unavailable_reason():
        return True, "Ready — finds your tokens through Blockscout, no key needed."
    if not rpc.unavailable_reason() and not rpc.discovery_reason():
        return True, "Ready — finds your tokens through Etherscan, using your API key."
    if not rpc.unavailable_reason():
        # The chain can be read, but nothing can be *found* on it, which reads
        # as an empty wallet unless we say so.
        return False, rpc.discovery_reason()
    return False, rpc.unavailable_reason()


def provider_status(chain: str) -> str:
    """The explanation alone, for contexts that only show text."""
    return describe_provider(chain)[1]


def fetch_balances_for_wallet(
    wallet: Wallet,
    manual_tokens: Optional[Sequence[Dict[str, Any]]] = None,
) -> BalanceResult:
    """Read one wallet, trying each provider the chain supports in turn.

    Order is deliberate: Blockscout first because it is one unauthenticated
    call that returns balances *and* metadata, then Etherscan discovery plus
    RPC, then whatever token list the user pasted in by hand. A chain is only
    reported as unreadable once every route has been tried.
    """
    chain_cfg = config.get_chain(wallet.chain)
    if chain_cfg.address_kind == "solana":
        return SolanaRpcProvider().fetch(wallet.address)

    attempts: List[BalanceResult] = []

    blockscout = BlockscoutProvider(wallet.chain)
    if not blockscout.unavailable_reason():
        outcome = blockscout.fetch(wallet.address)
        if outcome.ok and outcome.balances:
            # Blockscout has no native-coin row in tokenlist; add it from RPC
            # when we have one, so dry powder is not missing from the book.
            rpc = EvmRpcProvider(wallet.chain)
            if not rpc.unavailable_reason():
                try:
                    native = rpc.native_balance(wallet.address)
                except Exception:  # noqa: BLE001
                    native = None
                wrapped = config.WRAPPED_NATIVE.get(wallet.chain, "")
                if native:
                    outcome.balances.append(
                        TokenBalance(
                            address=wrapped or f"native:{wallet.chain}", chain=wallet.chain,
                            quantity=native, symbol=chain_cfg.native_symbol,
                            name=f"{chain_cfg.label} native", decimals=_NATIVE_DECIMALS,
                            wallet=wallet.address, source="rpc", is_native=True,
                        )
                    )
            return outcome
        attempts.append(outcome)

    rpc = EvmRpcProvider(wallet.chain)
    outcome = rpc.fetch(wallet.address, tokens=manual_tokens)
    if outcome.ok and (outcome.balances or not attempts):
        for earlier in attempts:
            outcome.warnings.extend(earlier.warnings)
        return outcome
    attempts.append(outcome)

    merged = BalanceResult(source="none", ok=False)
    for earlier in attempts:
        merged.extend(earlier)
    if not merged.warnings:
        merged.warnings.append(
            f"No balance provider could read {chain_cfg.label} for "
            f"{wallet.address[:10]}…"
        )
    return merged


def fetch_all_balances(
    wallets: Sequence[Wallet],
    manual_tokens: Optional[Dict[str, List[Dict[str, Any]]]] = None,
) -> BalanceResult:
    """Read every registered wallet and merge the results.

    Failures are collected as warnings rather than raised: one dead RPC must
    not cost you the view of the other three chains.
    """
    merged = BalanceResult(source="multi")
    manual = manual_tokens or {}
    for wallet in wallets:
        try:
            outcome = fetch_balances_for_wallet(wallet, manual_tokens=manual.get(wallet.chain))
        except Exception as exc:  # noqa: BLE001 - a provider must never kill a sync
            logger.warning("Balance sync failed for %s: %s", wallet.address, exc)
            merged.warnings.append(
                f"{config.get_chain(wallet.chain).label} wallet {wallet.address[:10]}… failed: {exc}"
            )
            continue
        merged.extend(outcome)
    return merged
