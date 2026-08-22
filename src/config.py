"""Central configuration for MemeDD Dashboard.

Everything that a user might reasonably want to tune -- chains, scoring
weights, scanner filters, API endpoints, timeouts -- lives here so the rest of
the code stays free of magic numbers.

Secrets are read from environment variables (optionally via a local ``.env``
file).  The v1 feature set works with zero API keys; keys only unlock the
optional LLM narrative layer.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

try:  # python-dotenv is optional at runtime
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv missing is not fatal
    pass


# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("MEMEDD_DATA_DIR", PROJECT_ROOT / "data"))
HISTORY_DB_PATH = DATA_DIR / "history.sqlite3"


# --------------------------------------------------------------------------
# Chains
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ChainConfig:
    """Static metadata for a supported chain."""

    key: str                      # internal key, matches DexScreener chainId
    label: str                    # human readable name
    dexscreener_id: str           # DexScreener `chainId`
    goplus_id: Optional[str]      # GoPlus numeric chain id (None for non-EVM)
    address_kind: str             # "evm" | "solana"
    explorer_token_url: str       # format string with {address}
    native_symbol: str = "ETH"
    goplus_solana: bool = False   # use the GoPlus Solana endpoint instead


CHAINS: Dict[str, ChainConfig] = {
    "base": ChainConfig(
        key="base",
        label="Base",
        dexscreener_id="base",
        goplus_id="8453",
        address_kind="evm",
        explorer_token_url="https://basescan.org/token/{address}",
    ),
    "ethereum": ChainConfig(
        key="ethereum",
        label="Ethereum",
        dexscreener_id="ethereum",
        goplus_id="1",
        address_kind="evm",
        explorer_token_url="https://etherscan.io/token/{address}",
    ),
    "solana": ChainConfig(
        key="solana",
        label="Solana",
        dexscreener_id="solana",
        goplus_id=None,
        address_kind="solana",
        explorer_token_url="https://solscan.io/token/{address}",
        native_symbol="SOL",
        goplus_solana=True,
    ),
    "bsc": ChainConfig(
        key="bsc",
        label="BNB Chain",
        dexscreener_id="bsc",
        goplus_id="56",
        address_kind="evm",
        explorer_token_url="https://bscscan.com/token/{address}",
        native_symbol="BNB",
    ),
    "arbitrum": ChainConfig(
        key="arbitrum",
        label="Arbitrum",
        dexscreener_id="arbitrum",
        goplus_id="42161",
        address_kind="evm",
        explorer_token_url="https://arbiscan.io/token/{address}",
    ),
}

DEFAULT_CHAIN = "base"


def get_chain(key: str) -> ChainConfig:
    """Look up a chain config, falling back to the default chain."""
    return CHAINS.get((key or "").lower(), CHAINS[DEFAULT_CHAIN])


def chain_from_dexscreener_id(chain_id: str) -> Optional[ChainConfig]:
    """Reverse lookup: DexScreener chainId -> ChainConfig (None if unknown)."""
    for cfg in CHAINS.values():
        if cfg.dexscreener_id == (chain_id or "").lower():
            return cfg
    return None


# --------------------------------------------------------------------------
# API endpoints & networking
# --------------------------------------------------------------------------
DEXSCREENER_BASE = os.getenv("DEXSCREENER_BASE_URL", "https://api.dexscreener.com")
GOPLUS_BASE = os.getenv("GOPLUS_BASE_URL", "https://api.gopluslabs.io")

HTTP_TIMEOUT_SECONDS = float(os.getenv("MEMEDD_HTTP_TIMEOUT", "12"))
HTTP_MAX_RETRIES = int(os.getenv("MEMEDD_HTTP_RETRIES", "2"))
HTTP_USER_AGENT = os.getenv("MEMEDD_USER_AGENT", "MemeDD-Dashboard/1.0 (+https://github.com)")

# Cache TTLs (seconds).  DexScreener asks for <=300 req/min; caching keeps us
# comfortably inside that even with a chatty UI.
CACHE_TTL_TOKEN = int(os.getenv("MEMEDD_CACHE_TTL_TOKEN", "120"))
CACHE_TTL_SCANNER = int(os.getenv("MEMEDD_CACHE_TTL_SCANNER", "180"))
CACHE_TTL_SECURITY = int(os.getenv("MEMEDD_CACHE_TTL_SECURITY", "600"))


# --------------------------------------------------------------------------
# Optional API keys (LLM layer + authenticated GoPlus)
# --------------------------------------------------------------------------
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
XAI_API_KEY = os.getenv("XAI_API_KEY", "")
GOPLUS_APP_KEY = os.getenv("GOPLUS_APP_KEY", "")
GOPLUS_APP_SECRET = os.getenv("GOPLUS_APP_SECRET", "")

# "none" keeps the app fully offline-capable; see src/llm.py.
LLM_PROVIDER = os.getenv("MEMEDD_LLM_PROVIDER", "none").lower()
# Legacy single-provider override: applies to the narrative path in src/llm.py
# only. The ensemble uses the per-provider model ids below.
LLM_MODEL = os.getenv("MEMEDD_LLM_MODEL", "")


# --------------------------------------------------------------------------
# Multi-LLM ensemble (src/llm_analyzers.py)
# --------------------------------------------------------------------------
# Each provider gets its own model id so the ensemble can mix tiers -- e.g. a
# frontier model for judgement plus a cheaper one for a second opinion.
ANTHROPIC_MODEL = os.getenv("MEMEDD_ANTHROPIC_MODEL", "claude-opus-5")
OPENAI_MODEL = os.getenv("MEMEDD_OPENAI_MODEL", "gpt-4.1")
XAI_MODEL = os.getenv("MEMEDD_XAI_MODEL", "grok-4")

# xAI speaks the OpenAI wire protocol, so the OpenAI SDK drives it with a
# different base URL.
XAI_BASE_URL = os.getenv("XAI_BASE_URL", "https://api.x.ai/v1")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "")  # blank = SDK default

# Providers the ensemble will try, in display order. Any without a key or SDK
# installed are skipped with a reason rather than failing the run.
ENSEMBLE_PROVIDERS = tuple(
    p.strip().lower()
    for p in os.getenv("MEMEDD_ENSEMBLE_PROVIDERS", "xai,anthropic,openai").split(",")
    if p.strip()
)

LLM_TIMEOUT_SECONDS = float(os.getenv("MEMEDD_LLM_TIMEOUT", "75"))
LLM_MAX_TOKENS = int(os.getenv("MEMEDD_LLM_MAX_TOKENS", "16000"))
# Low temperature: we want reproducible scoring, not creative writing.
LLM_TEMPERATURE = float(os.getenv("MEMEDD_LLM_TEMPERATURE", "0.2"))

# How much the LLM consensus moves the final blended score. 0.0 = ignore the
# models entirely, 1.0 = trust them over the deterministic engine.
ENSEMBLE_BLEND_WEIGHT = float(os.getenv("MEMEDD_ENSEMBLE_BLEND_WEIGHT", "0.35"))

# Decision vocabulary shared with the models (snake_case in the JSON schema).
LLM_DECISIONS = ("strong_buy", "buy", "watch", "pass")
LLM_DIMENSIONS = ("security", "liquidity", "holders", "mindshare", "lore", "catalyst")

# Maps the models' snake_case decisions onto the UI's display labels.
DECISION_LABELS = {
    "strong_buy": "Strong Buy",
    "buy": "Buy",
    "watch": "Watch",
    "pass": "Pass",
}


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ScoreWeights:
    """Weights of the composite score.  Must sum to 1.0."""

    security: float = 0.30
    liquidity: float = 0.20
    holders: float = 0.15
    momentum: float = 0.15
    narrative: float = 0.10
    catalyst: float = 0.10

    def as_dict(self) -> Dict[str, float]:
        return {
            "security": self.security,
            "liquidity": self.liquidity,
            "holders": self.holders,
            "momentum": self.momentum,
            "narrative": self.narrative,
            "catalyst": self.catalyst,
        }

    def validate(self) -> None:
        total = sum(self.as_dict().values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Score weights must sum to 1.0 (got {total:.4f})")


DEFAULT_WEIGHTS = ScoreWeights()

# Human-friendly labels used across the UI and exports.
COMPONENT_LABELS: Dict[str, str] = {
    "security": "Security",
    "liquidity": "Liquidity Quality",
    "holders": "Holder Distribution",
    "momentum": "Mindshare / Momentum",
    "narrative": "Narrative Potential",
    "catalyst": "Catalyst / Listing Potential",
}

# Composite score -> decision thresholds.
DECISION_THRESHOLDS = {
    "Strong Buy": 78.0,
    "Buy": 64.0,
    "Watch": 48.0,
}
# Anything below the "Watch" threshold is a Pass.


# --------------------------------------------------------------------------
# Scanner defaults
# --------------------------------------------------------------------------
@dataclass
class ScannerFilters:
    """User-tunable filters for Scanner mode."""

    chain: str = DEFAULT_CHAIN
    min_market_cap: float = 500_000
    max_market_cap: float = 5_000_000
    min_liquidity: float = 50_000
    min_volume_24h: float = 100_000
    min_age_hours: float = 6.0
    max_age_days: float = 365.0
    min_txns_24h: int = 100
    max_results: int = 40
    exclude_no_socials: bool = False

    def describe(self) -> str:
        return (
            f"{get_chain(self.chain).label} | MC ${self.min_market_cap:,.0f}-${self.max_market_cap:,.0f} "
            f"| Liq >= ${self.min_liquidity:,.0f} | Vol24h >= ${self.min_volume_24h:,.0f}"
        )


# Seed queries used to discover candidates on DexScreener's public search.
# DexScreener has no "list all pairs on chain" endpoint, so we fan out over a
# set of broad meme-ish search terms plus the boosted/profile feeds.
SCANNER_SEED_QUERIES: Dict[str, list] = {
    "base": ["base", "WETH base", "USDC base", "meme base", "brett", "degen"],
    "ethereum": ["WETH", "USDC", "pepe", "meme"],
    "solana": ["SOL", "USDC sol", "bonk", "wif", "pump"],
    "bsc": ["WBNB", "USDT bsc", "meme"],
    "arbitrum": ["WETH arbitrum", "USDC arbitrum"],
}


# --------------------------------------------------------------------------
# Risk management defaults
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class RiskProfile:
    """A named risk tolerance preset."""

    key: str
    label: str
    risk_per_trade_pct: float   # % of portfolio you accept losing on one idea
    max_position_pct: float     # hard cap on position as % of portfolio
    base_stop_pct: float        # baseline stop distance before volatility adj.
    max_liquidity_share_pct: float  # position as % of pool liquidity


RISK_PROFILES: Dict[str, RiskProfile] = {
    "conservative": RiskProfile("conservative", "Conservative", 0.50, 2.0, 22.0, 0.50),
    "moderate": RiskProfile("moderate", "Moderate", 1.00, 5.0, 30.0, 1.00),
    "aggressive": RiskProfile("aggressive", "Aggressive", 2.00, 10.0, 40.0, 2.00),
    "degen": RiskProfile("degen", "Degen", 3.50, 15.0, 50.0, 3.00),
}

DEFAULT_RISK_PROFILE = "moderate"
DEFAULT_PORTFOLIO_USD = 10_000.0


@dataclass
class AppSettings:
    """Bundle of everything the analysis pipeline needs from the UI."""

    chain: str = DEFAULT_CHAIN
    portfolio_usd: float = DEFAULT_PORTFOLIO_USD
    risk_profile: str = DEFAULT_RISK_PROFILE
    weights: ScoreWeights = field(default_factory=lambda: DEFAULT_WEIGHTS)
    use_llm: bool = False
    # Multi-model ensemble (Grok + Claude + GPT in parallel).
    use_ensemble: bool = False
    ensemble_providers: tuple = ENSEMBLE_PROVIDERS
    blend_weight: float = ENSEMBLE_BLEND_WEIGHT

    def risk(self) -> RiskProfile:
        return RISK_PROFILES.get(self.risk_profile, RISK_PROFILES[DEFAULT_RISK_PROFILE])
