"""Realistic API payloads used by the tests.

Shapes mirror the live DexScreener and GoPlus responses so the normalizers are
exercised against the real thing, not a convenient simplification.
"""

from __future__ import annotations

import time
from typing import Any, Dict

NOW_MS = int(time.time() * 1000)
DAY_MS = 86_400_000

TOKEN_ADDRESS = "0x532f27101965dd16442E59d40670FaF5eBB142E4"
PAIR_ADDRESS = "0x36a46dff597c5a444bbc521d26787f57867d2214"


def dexscreener_pair(**overrides: Any) -> Dict[str, Any]:
    """One DexScreener pair object (Base / Uniswap v3 shape)."""
    pair: Dict[str, Any] = {
        "chainId": "base",
        "dexId": "uniswap",
        "url": f"https://dexscreener.com/base/{PAIR_ADDRESS}",
        "pairAddress": PAIR_ADDRESS,
        "labels": ["v3"],
        "baseToken": {"address": TOKEN_ADDRESS, "name": "Brett", "symbol": "BRETT"},
        "quoteToken": {"address": "0x4200000000000000000000000000000000000006", "name": "Wrapped Ether", "symbol": "WETH"},
        "priceNative": "0.0000041",
        "priceUsd": "0.0142",
        "txns": {
            "m5": {"buys": 12, "sells": 9},
            "h1": {"buys": 210, "sells": 180},
            "h6": {"buys": 980, "sells": 870},
            "h24": {"buys": 3120, "sells": 2650},
        },
        "volume": {"h24": 1_850_000.0, "h6": 520_000.0, "h1": 96_000.0, "m5": 8_400.0},
        "priceChange": {"m5": 0.4, "h1": 2.6, "h6": 7.9, "h24": 18.4},
        "liquidity": {"usd": 412_000.0, "base": 14_500_000.0, "quote": 62.4},
        "fdv": 3_400_000.0,
        "marketCap": 3_100_000.0,
        "pairCreatedAt": NOW_MS - 45 * DAY_MS,
        "info": {
            "imageUrl": "https://dd.dexscreener.com/ds-data/tokens/base/brett.png",
            "websites": [{"label": "Website", "url": "https://basedbrett.com"}],
            "socials": [
                {"type": "twitter", "url": "https://x.com/basedbrett"},
                {"type": "telegram", "url": "https://t.me/basedbrett"},
            ],
        },
        "boosts": {"active": 3},
    }
    pair.update(overrides)
    return pair


def dexscreener_token_response(**overrides: Any) -> Dict[str, Any]:
    return {"schemaVersion": "1.0.0", "pairs": [dexscreener_pair(**overrides)]}


def goplus_response(**field_overrides: Any) -> Dict[str, Any]:
    """A clean, healthy GoPlus EVM token_security payload."""
    data: Dict[str, Any] = {
        "buy_tax": "0",
        "sell_tax": "0",
        "cannot_buy": "0",
        "cannot_sell_all": "0",
        "creator_address": "0x9c1a2b3c4d5e6f708192a3b4c5d6e7f809a1b2c3",
        "creator_balance": "0",
        "creator_percent": "0.0043",
        "external_call": "0",
        "hidden_owner": "0",
        "holder_count": "18432",
        "honeypot_with_same_creator": "0",
        "is_blacklisted": "0",
        "is_honeypot": "0",
        "is_in_dex": "1",
        "is_mintable": "0",
        "is_open_source": "1",
        "is_proxy": "0",
        "is_whitelisted": "0",
        "lp_holder_count": "4",
        "lp_total_supply": "1000",
        "owner_address": "0x0000000000000000000000000000000000000000",
        "owner_percent": "0",
        "selfdestruct": "0",
        "slippage_modifiable": "0",
        "token_name": "Brett",
        "token_symbol": "BRETT",
        "total_supply": "10000000000",
        "trading_cooldown": "0",
        "transfer_pausable": "0",
        "anti_whale_modifiable": "0",
        "holders": [
            {"address": "0x36a46dff597c5a444bbc521d26787f57867d2214", "tag": "Uniswap V3 Pair",
             "is_contract": 1, "balance": "1400000000", "percent": "0.14", "is_locked": 0},
            {"address": "0x000000000000000000000000000000000000dead", "tag": "Burn Address",
             "is_contract": 0, "balance": "900000000", "percent": "0.09", "is_locked": 0},
            {"address": "0xaaa1111111111111111111111111111111111111", "tag": "",
             "is_contract": 0, "balance": "310000000", "percent": "0.031", "is_locked": 0},
            {"address": "0xbbb2222222222222222222222222222222222222", "tag": "",
             "is_contract": 0, "balance": "220000000", "percent": "0.022", "is_locked": 0},
            {"address": "0xccc3333333333333333333333333333333333333", "tag": "",
             "is_contract": 0, "balance": "180000000", "percent": "0.018", "is_locked": 0},
        ],
        "lp_holders": [
            {"address": "0x000000000000000000000000000000000000dead", "tag": "Burn Address",
             "is_contract": 0, "balance": "980", "percent": "0.98", "is_locked": 0},
            {"address": "0xddd4444444444444444444444444444444444444", "tag": "",
             "is_contract": 0, "balance": "20", "percent": "0.02", "is_locked": 0},
        ],
    }
    data.update(field_overrides)
    return {"code": 1, "message": "OK", "result": {TOKEN_ADDRESS.lower(): data}}


def goplus_honeypot_response() -> Dict[str, Any]:
    return goplus_response(
        is_honeypot="1", sell_tax="0.99", buy_tax="0.05", is_mintable="1",
        owner_address="0x9c1a2b3c4d5e6f708192a3b4c5d6e7f809a1b2c3",
        is_open_source="0",
        lp_holders=[{"address": "0xddd4444444444444444444444444444444444444", "tag": "",
                     "is_contract": 0, "balance": "1000", "percent": "1.0", "is_locked": 0}],
    )


def token_profile_item(**overrides: Any) -> Dict[str, Any]:
    """One entry from ``/token-profiles/latest/v1``."""
    item: Dict[str, Any] = {
        "url": f"https://dexscreener.com/base/{TOKEN_ADDRESS}",
        "chainId": "base",
        "tokenAddress": TOKEN_ADDRESS,
        "icon": "https://dd.dexscreener.com/ds-data/tokens/base/brett/icon.png",
        "header": "https://dd.dexscreener.com/ds-data/tokens/base/brett/header.png",
        "description": "Brett is Pepe's best friend and the mascot of Base.",
        "links": [
            {"type": "twitter", "url": "https://x.com/basedbrett"},
            {"type": "telegram", "url": "https://t.me/basedbrett"},
            {"label": "Website", "url": "https://basedbrett.com"},
        ],
    }
    item.update(overrides)
    return item


def token_profiles_response() -> list:
    """The feed: the target token plus entries on other chains to filter out."""
    return [
        token_profile_item(),
        token_profile_item(
            chainId="solana",
            tokenAddress="DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263",
            description="A Solana dog.",
        ),
        token_profile_item(
            tokenAddress="0xB000000000000000000000000000000000000002",
            description="Another Base token.",
            links=[],
        ),
    ]
