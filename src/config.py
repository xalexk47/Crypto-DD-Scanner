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
from typing import Dict, Optional, Tuple

try:  # python-dotenv is optional at runtime
    import io

    from dotenv import find_dotenv, load_dotenv

    # Read .env ourselves in universal-newline mode before handing it to
    # python-dotenv. A .env saved by a GUI editor (TextEdit on macOS, Notepad
    # on Windows) can carry classic-Mac CR or Windows CRLF line endings, which
    # the parser reads as one giant line -- every setting silently vanishes and
    # the app behaves as though no keys were configured at all. Python's text
    # mode normalises all three conventions to "\n".
    _dotenv_path = find_dotenv(usecwd=True)
    if _dotenv_path:
        with open(_dotenv_path, "r", encoding="utf-8", errors="replace") as _fh:
            load_dotenv(stream=io.StringIO(_fh.read()))
    else:
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
    # Etherscan V2 chain id for on-chain wallet-flow analysis. Deliberately
    # separate from goplus_id even where the numbers match: the two providers
    # support different chains and will diverge.
    etherscan_chain_id: Optional[str] = None


CHAINS: Dict[str, ChainConfig] = {
    "base": ChainConfig(
        key="base",
        label="Base",
        dexscreener_id="base",
        goplus_id="8453",
        etherscan_chain_id="8453",
        address_kind="evm",
        explorer_token_url="https://basescan.org/token/{address}",
    ),
    "ethereum": ChainConfig(
        key="ethereum",
        label="Ethereum",
        dexscreener_id="ethereum",
        goplus_id="1",
        etherscan_chain_id="1",
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
        etherscan_chain_id="56",
        address_kind="evm",
        explorer_token_url="https://bscscan.com/token/{address}",
        native_symbol="BNB",
    ),
    "arbitrum": ChainConfig(
        key="arbitrum",
        label="Arbitrum",
        dexscreener_id="arbitrum",
        goplus_id="42161",
        etherscan_chain_id="42161",
        address_kind="evm",
        explorer_token_url="https://arbiscan.io/token/{address}",
    ),
    # Robinhood Chain: Arbitrum Orbit L2 (chain id 4663), mainnet since
    # 2026-07-01, ETH gas. DexScreener indexes it; GoPlus does NOT support
    # 4663, so goplus_id stays None and every token here is scored with
    # security marked unavailable (penalised, low confidence) rather than
    # silently assumed safe. See SECURITY_PROVIDER_NOTES below.
    "robinhood": ChainConfig(
        key="robinhood",
        label="Robinhood Chain",
        dexscreener_id="robinhood",
        goplus_id=None,
        address_kind="evm",
        explorer_token_url="https://robinhoodchain.blockscout.com/token/{address}",
    ),
}


# Chains with no contract-security provider, and why. Surfaced in the UI so a
# missing rug check is never mistaken for a clean rug check.
SECURITY_PROVIDER_NOTES: Dict[str, str] = {
    "robinhood": (
        "GoPlus does not support Robinhood Chain (id 4663), so honeypot, tax, "
        "mint-authority, LP-lock and holder-concentration checks cannot run here. "
        "Scores fall back to market data only, with the security pillar penalised "
        "and confidence reduced. Verify contracts manually before trading."
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
XAI_MODEL = os.getenv("MEMEDD_XAI_MODEL", "grok-4.6")

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

# --------------------------------------------------------------------------
# On-chain wallet flow (Etherscan V2)
# --------------------------------------------------------------------------
# One free key covers every EVM chain via the chainid parameter: 5 calls/sec,
# 100k/day. Solana is not an EVM chain and is not covered here.
ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY", "")
ETHERSCAN_BASE_URL = os.getenv("ETHERSCAN_BASE_URL", "https://api.etherscan.io/v2/api")
# Transfers pulled per token. Etherscan caps a single query at 10k rows.
WALLET_FLOW_MAX_TRANSFERS = int(os.getenv("MEMEDD_WALLET_FLOW_MAX_TRANSFERS", "3000"))
# Window treated as "early" when identifying the first-buyer cohort.
WALLET_FLOW_EARLY_HOURS = float(os.getenv("MEMEDD_WALLET_FLOW_EARLY_HOURS", "6"))
# Recent window used for the accumulation/distribution read.
WALLET_FLOW_RECENT_HOURS = float(os.getenv("MEMEDD_WALLET_FLOW_RECENT_HOURS", "24"))
# Price moves smaller than this over 24h count as "consolidation", the state in
# which quiet accumulation is worth flagging.
WALLET_FLOW_CONSOLIDATION_PCT = float(os.getenv("MEMEDD_WALLET_FLOW_CONSOLIDATION_PCT", "12"))
CACHE_TTL_WALLET_FLOW = int(os.getenv("MEMEDD_CACHE_TTL_WALLET_FLOW", "300"))

# Your own smart-money list: wallets and X handles you already trust. Highest
# precision signal in the app, and it costs nothing. See the .example file.
SMART_MONEY_PATH = Path(os.getenv("MEMEDD_SMART_MONEY_PATH", DATA_DIR / "smart_money.json"))

# --- X / Twitter mindshare via Grok live search ----------------------------
# Grok is the only major model with first-party access to X, which is where
# meme-coin mindshare actually forms. xAI retired the old `search_parameters`
# Live Search API on 2026-01-12 (410 Gone); the current mechanism is the
# server-side Agent Tools API, i.e. a tool entry in the `tools` array.
#
# These are configurable because the exact model id and parameter casing move
# faster than this app does -- run `python scripts/check_grok.py` to see what
# your account actually supports, then set these accordingly.
X_SEARCH_ENABLED = os.getenv("MEMEDD_X_SEARCH", "1").strip().lower() not in ("0", "false", "no")
X_SEARCH_MODEL = os.getenv("MEMEDD_X_SEARCH_MODEL", "grok-4.6")
# Confirmed against the live API, the hard way:
#   * /v1/chat/completions accepts tools of type "function" or "live_search"
#   * "live_search" parses but then returns 410: "Live search is deprecated.
#     Please switch to the Agent Tools API"
#   * the Agent Tools API lives on /v1/responses, where "x_search" is valid
# So server-side X search runs through client.responses.create(), not
# chat.completions. The non-live fallback still uses chat.completions.
X_SEARCH_TOOL_TYPE = os.getenv("MEMEDD_X_SEARCH_TOOL", "x_search")
# Restrict Live Search to X only; "web" is available but dilutes a mindshare
# reading with news articles and blog spam.
X_SEARCH_SOURCES = tuple(
    s.strip() for s in os.getenv("MEMEDD_X_SEARCH_SOURCES", "x").split(",") if s.strip()
)
X_SEARCH_MAX_RESULTS = int(os.getenv("MEMEDD_X_SEARCH_MAX_RESULTS", "20"))
# How far back to search, and how many posts to keep as evidence.
X_SEARCH_WINDOW_HOURS = int(os.getenv("MEMEDD_X_SEARCH_WINDOW_HOURS", "48"))
X_SEARCH_MAX_POSTS = int(os.getenv("MEMEDD_X_SEARCH_MAX_POSTS", "6"))
# Live searches are billed per call, so cache them harder than market data.
CACHE_TTL_MINDSHARE = int(os.getenv("MEMEDD_CACHE_TTL_MINDSHARE", "900"))
# Share of the momentum pillar given to social mindshare when it is available.
MINDSHARE_WEIGHT_IN_MOMENTUM = float(os.getenv("MEMEDD_MINDSHARE_WEIGHT", "0.3"))

# A server-side x_search call is an agentic loop -- xAI analyses the query,
# runs searches, reads results and may search again before answering. That
# routinely takes far longer than a plain completion, so it gets its own,
# much longer budget. Timing it out at the normal 75s would fail a call that
# was about to succeed, and bill for the work either way.
X_SEARCH_TIMEOUT_SECONDS = float(os.getenv("MEMEDD_X_SEARCH_TIMEOUT", "240"))

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
    # Robinhood Chain skews tokenized equities / RWA rather than memes, so the
    # seeds lean on the majors and the chain's own names.
    "robinhood": ["robinhood", "HOOD", "WETH", "USDC", "meme robinhood"],
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
    # Query X/Twitter through Grok for real social mindshare.
    use_x_search: bool = False
    # On-chain wallet flow / smart-money analysis.
    use_wallet_flow: bool = False
    # default_factory, not a bare default: a plain default would bind the
    # module value at import time and then ignore any later change to it,
    # which silently freezes whatever the developer's .env said at startup.
    ensemble_providers: Tuple[str, ...] = field(default_factory=lambda: ENSEMBLE_PROVIDERS)
    blend_weight: float = field(default_factory=lambda: ENSEMBLE_BLEND_WEIGHT)

    def risk(self) -> RiskProfile:
        return RISK_PROFILES.get(self.risk_profile, RISK_PROFILES[DEFAULT_RISK_PROFILE])
