"""Small, dependency-light helpers shared across the app.

Deliberately framework agnostic: nothing here imports Streamlit, so the same
helpers can be reused from a CLI, a cron job or a future FastAPI service.
"""

from __future__ import annotations

import math
import re
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Hashable, Iterable, List, Optional, Tuple

# --------------------------------------------------------------------------
# Address handling
# --------------------------------------------------------------------------
# The 0x prefix is matched case-insensitively: a spreadsheet or explorer that
# echoes "0X..." is still a valid EVM address, and treating it as unknown
# would leave it uncomparable -- the same holding counted twice.
_EVM_RE = re.compile(r"^0[xX][a-fA-F0-9]{40}$")
# Base58 alphabet (no 0, O, I, l) - Solana mints are 32-44 chars.
_SOLANA_RE = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{32,44}$")

BURN_ADDRESSES = {
    "0x0000000000000000000000000000000000000000",
    "0x000000000000000000000000000000000000dead",
    "0x00000000000000000000000000000000000000ff",
    "11111111111111111111111111111111",
}

_BURN_TAG_HINTS = ("burn", "black hole", "null", "dead")
_LOCK_TAG_HINTS = ("lock", "unicrypt", "team.finance", "pinklock", "vesting")


def detect_address_kind(address: str) -> Optional[str]:
    """Return ``"evm"``, ``"solana"`` or ``None`` for an arbitrary string."""
    addr = (address or "").strip()
    if _EVM_RE.match(addr):
        return "evm"
    if _SOLANA_RE.match(addr):
        return "solana"
    return None


def is_valid_address(address: str) -> bool:
    return detect_address_kind(address) is not None


def normalize_address(address: str) -> str:
    """Normalize for comparison/caching: EVM lower-cased, Solana untouched."""
    addr = (address or "").strip()
    if detect_address_kind(addr) == "evm":
        return addr.lower()
    return addr


def short_address(address: str, head: int = 6, tail: int = 4) -> str:
    addr = (address or "").strip()
    if len(addr) <= head + tail + 1:
        return addr
    return f"{addr[:head]}...{addr[-tail:]}"


def parse_addresses(raw: str) -> Tuple[List[str], List[str]]:
    """Split a free-form textarea into ``(valid, invalid)`` address lists.

    Accepts newline, comma, semicolon or whitespace separated input and also
    tolerates pasted DexScreener / explorer URLs by extracting the last
    address-looking token.
    """
    tokens = re.split(r"[\s,;]+", raw or "")
    valid: List[str] = []
    invalid: List[str] = []
    seen = set()
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        candidate = token
        if "/" in token:  # looks like a URL - grab the last address-ish chunk
            parts = [p for p in token.split("/") if p]
            candidate = next(
                (p.split("?")[0] for p in reversed(parts) if is_valid_address(p.split("?")[0])),
                token,
            )
        if is_valid_address(candidate):
            key = normalize_address(candidate)
            if key not in seen:
                seen.add(key)
                valid.append(candidate)
        else:
            invalid.append(token)
    return valid, invalid


def is_burn_address(address: str, tag: str = "") -> bool:
    addr = normalize_address(address)
    if addr in BURN_ADDRESSES:
        return True
    low = (tag or "").lower()
    return any(hint in low for hint in _BURN_TAG_HINTS)


def looks_locked(tag: str = "") -> bool:
    low = (tag or "").lower()
    return any(hint in low for hint in _LOCK_TAG_HINTS)


# --------------------------------------------------------------------------
# Safe coercion / nested access
# --------------------------------------------------------------------------
def safe_float(value: Any, default: float = 0.0) -> float:
    """Best-effort float conversion that never raises.

    Handles the string-typed numbers that DexScreener and GoPlus return, plus
    ``None``, empty strings and stray ``%``/``,`` characters.
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return default if math.isnan(value) or math.isinf(value) else float(value)
    try:
        cleaned = str(value).strip().replace(",", "").replace("%", "").replace("$", "")
        if cleaned in ("", "-", "null", "None", "NaN"):
            return default
        parsed = float(cleaned)
        return default if math.isnan(parsed) or math.isinf(parsed) else parsed
    except (TypeError, ValueError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    return int(safe_float(value, float(default)))


def safe_bool(value: Any, default: Optional[bool] = None) -> Optional[bool]:
    """Interpret GoPlus-style ``"1"``/``"0"`` flags.

    Returns ``None`` when the field is missing/unknown so callers can tell
    "no" apart from "we don't know".
    """
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("1", "true", "yes", "y"):
        return True
    if text in ("0", "false", "no", "n"):
        return False
    return default


def dig(obj: Any, *path: Any, default: Any = None) -> Any:
    """Walk nested dicts/lists safely: ``dig(d, "a", 0, "b", default=1)``."""
    current = obj
    for key in path:
        if isinstance(current, dict):
            current = current.get(key)
        elif isinstance(current, (list, tuple)) and isinstance(key, int):
            current = current[key] if -len(current) <= key < len(current) else None
        else:
            return default
        if current is None:
            return default
    return current


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def scale(value: float, low: float, high: float, out_low: float = 0.0, out_high: float = 100.0) -> float:
    """Linearly map ``value`` from ``[low, high]`` onto ``[out_low, out_high]``."""
    if high == low:
        return out_high if value >= high else out_low
    ratio = (value - low) / (high - low)
    return clamp(out_low + ratio * (out_high - out_low), min(out_low, out_high), max(out_low, out_high))


def log_scale(value: float, low: float, high: float, out_low: float = 0.0, out_high: float = 100.0) -> float:
    """Like :func:`scale` but on a log10 axis - the right shape for money.

    Liquidity of $10k vs $100k matters far more than $1.0M vs $1.1M.
    """
    if value <= 0:
        return out_low
    low = max(low, 1e-9)
    high = max(high, low * 10)
    return scale(math.log10(max(value, low)), math.log10(low), math.log10(high), out_low, out_high)


def safe_ratio(numerator: float, denominator: float, default: float = 0.0) -> float:
    if not denominator:
        return default
    return numerator / denominator


# --------------------------------------------------------------------------
# Formatting
# --------------------------------------------------------------------------
def fmt_usd(value: Optional[float], compact: bool = True, dashes: str = "n/a") -> str:
    """Format a dollar amount: ``$1.2M``, ``$45.3K``, ``$0.00001234``."""
    if value is None:
        return dashes
    value = safe_float(value, float("nan"))
    if math.isnan(value):
        return dashes
    sign = "-" if value < 0 else ""
    value = abs(value)
    if compact:
        for cutoff, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
            if value >= cutoff:
                return f"{sign}${value / cutoff:,.2f}{suffix}"
    if value == 0:
        return "$0"
    if value < 1:
        # Meme-coin prices live in the decimals: keep ~4 significant digits
        # instead of rounding $0.0142 down to "$0.01".
        decimals = min(12, max(2, 3 - int(math.floor(math.log10(value)))))
        text = f"{value:,.{decimals}f}"
        if "." in text:  # trim trailing zeros, but never below 2 decimals
            whole, frac = text.split(".")
            frac = frac.rstrip("0").ljust(2, "0")
            text = f"{whole}.{frac}"
        return f"{sign}${text}"
    return f"{sign}${value:,.2f}"


def fmt_number(value: Optional[float], dashes: str = "n/a") -> str:
    if value is None:
        return dashes
    value = safe_float(value, float("nan"))
    if math.isnan(value):
        return dashes
    for cutoff, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(value) >= cutoff:
            return f"{value / cutoff:,.2f}{suffix}"
    return f"{value:,.0f}"


def fmt_pct(value: Optional[float], decimals: int = 2, signed: bool = False, dashes: str = "n/a") -> str:
    if value is None:
        return dashes
    value = safe_float(value, float("nan"))
    if math.isnan(value):
        return dashes
    prefix = "+" if signed and value > 0 else ""
    return f"{prefix}{value:,.{decimals}f}%"


def fmt_age(created_at_ms: Optional[int], now: Optional[datetime] = None) -> str:
    """Human age from a millisecond epoch timestamp."""
    hours = age_hours(created_at_ms, now)
    if hours is None:
        return "unknown"
    if hours < 1:
        return f"{int(hours * 60)}m"
    if hours < 48:
        return f"{hours:.1f}h"
    days = hours / 24
    if days < 60:
        return f"{days:.1f}d"
    return f"{days / 30.44:.1f}mo"


def age_hours(created_at_ms: Optional[int], now: Optional[datetime] = None) -> Optional[float]:
    if not created_at_ms:
        return None
    try:
        created = datetime.fromtimestamp(float(created_at_ms) / 1000.0, tz=timezone.utc)
    except (OSError, OverflowError, TypeError, ValueError):
        return None
    reference = now or datetime.now(timezone.utc)
    return max(0.0, (reference - created).total_seconds() / 3600.0)


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def score_color(score: float) -> str:
    """Hex colour for a 0-100 score, used by cards and progress bars."""
    if score >= 78:
        return "#22c55e"   # green
    if score >= 64:
        return "#84cc16"   # lime
    if score >= 48:
        return "#eab308"   # amber
    if score >= 30:
        return "#f97316"   # orange
    return "#ef4444"       # red


def score_emoji(score: float) -> str:
    if score >= 78:
        return "🟢"
    if score >= 64:
        return "🟡"
    if score >= 48:
        return "🟠"
    return "🔴"


def chunked(items: Iterable[Any], size: int) -> Iterable[List[Any]]:
    """Yield ``size``-sized chunks (used for batched API calls)."""
    batch: List[Any] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


# --------------------------------------------------------------------------
# Tiny thread-safe TTL cache
# --------------------------------------------------------------------------
class TTLCache:
    """Minimal in-process TTL cache.

    Streamlit has ``st.cache_data``, but keeping a cache here too means the
    fetchers stay polite to upstream APIs when used outside Streamlit
    (scripts, tests, a future scheduler).
    """

    def __init__(self, ttl_seconds: int = 120, max_entries: int = 512) -> None:
        self.ttl = ttl_seconds
        self.max_entries = max_entries
        self._store: Dict[Hashable, Tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def get(self, key: Hashable) -> Optional[Any]:
        with self._lock:
            entry = self._store.get(key)
            if not entry:
                return None
            expires_at, value = entry
            if expires_at < time.time():
                self._store.pop(key, None)
                return None
            return value

    def set(self, key: Hashable, value: Any, ttl: Optional[int] = None) -> None:
        with self._lock:
            if len(self._store) >= self.max_entries:
                # Cheap eviction: drop the soonest-to-expire entry.
                oldest = min(self._store.items(), key=lambda kv: kv[1][0])[0]
                self._store.pop(oldest, None)
            self._store[key] = (time.time() + (ttl if ttl is not None else self.ttl), value)

    def get_or_set(self, key: Hashable, producer: Callable[[], Any], ttl: Optional[int] = None) -> Any:
        cached = self.get(key)
        if cached is not None:
            return cached
        value = producer()
        if value is not None:
            self.set(key, value, ttl)
        return value

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
