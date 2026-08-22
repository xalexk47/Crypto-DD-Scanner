"""Tests for the API normalizers, with the network stubbed out."""

from __future__ import annotations

import pytest

from src import config, data_fetchers
from src.data_fetchers import FetchError
from tests.fixtures import (
    TOKEN_ADDRESS,
    dexscreener_pair,
    dexscreener_token_response,
    goplus_honeypot_response,
    goplus_response,
)


@pytest.fixture(autouse=True)
def clear_caches():
    data_fetchers.clear_caches()
    yield
    data_fetchers.clear_caches()


class TestSnapshotNormalization:
    def test_snapshot_from_pair_maps_every_field(self):
        pair = dexscreener_pair()
        snapshot = data_fetchers.snapshot_from_pair(pair, [pair], address=TOKEN_ADDRESS)

        assert snapshot.symbol == "BRETT"
        assert snapshot.chain == "base"
        assert snapshot.price_usd == pytest.approx(0.0142)
        assert snapshot.market_cap == pytest.approx(3_100_000.0)
        assert snapshot.liquidity_usd == pytest.approx(412_000.0)
        assert snapshot.volume_24h == pytest.approx(1_850_000.0)
        assert snapshot.txns_24h == 3120 + 2650
        assert snapshot.boosts == 3
        assert {s.kind for s in snapshot.socials} == {"website", "twitter", "telegram"}
        assert snapshot.turnover_24h == pytest.approx(1_850_000.0 / 3_100_000.0)
        assert snapshot.liquidity_ratio == pytest.approx(412_000.0 / 3_100_000.0)

    def test_sibling_pairs_aggregate_liquidity_and_volume(self):
        primary = dexscreener_pair()
        secondary = dexscreener_pair(
            pairAddress="0xsecond", dexId="aerodrome",
            liquidity={"usd": 88_000.0}, volume={"h24": 150_000.0, "h6": 40_000.0, "h1": 5_000.0},
        )
        snapshot = data_fetchers.snapshot_from_pair(primary, [primary, secondary], address=TOKEN_ADDRESS)

        assert snapshot.liquidity_usd == pytest.approx(500_000.0)
        assert snapshot.volume_24h == pytest.approx(2_000_000.0)
        assert snapshot.pair_count == 2
        assert snapshot.dex_count == 2

    def test_pick_primary_pair_prefers_deepest_pool_on_chain(self):
        thin = dexscreener_pair(pairAddress="0xthin", liquidity={"usd": 1_000.0})
        deep = dexscreener_pair(pairAddress="0xdeep", liquidity={"usd": 900_000.0})
        other_chain = dexscreener_pair(chainId="ethereum", pairAddress="0xeth", liquidity={"usd": 5_000_000.0})

        assert data_fetchers.pick_primary_pair([thin, deep, other_chain], "base")["pairAddress"] == "0xdeep"
        assert data_fetchers.pick_primary_pair([thin, deep, other_chain])["pairAddress"] == "0xeth"
        assert data_fetchers.pick_primary_pair([]) is None


class TestFetchTokenSnapshot:
    def test_happy_path(self, monkeypatch):
        monkeypatch.setattr(data_fetchers, "_get_json", lambda *a, **k: dexscreener_token_response())
        snapshot, warnings = data_fetchers.fetch_token_snapshot(TOKEN_ADDRESS, "base")
        assert snapshot is not None and snapshot.symbol == "BRETT"
        assert warnings == []

    def test_api_failure_degrades_gracefully(self, monkeypatch):
        def boom(*args, **kwargs):
            raise FetchError("HTTP 503")

        monkeypatch.setattr(data_fetchers, "_get_json", boom)
        snapshot, warnings = data_fetchers.fetch_token_snapshot(TOKEN_ADDRESS, "base")
        assert snapshot is None
        assert "DexScreener unavailable" in warnings[0]

    def test_unknown_token_returns_explanation(self, monkeypatch):
        monkeypatch.setattr(data_fetchers, "_get_json", lambda *a, **k: {"pairs": None})
        snapshot, warnings = data_fetchers.fetch_token_snapshot(TOKEN_ADDRESS, "base")
        assert snapshot is None
        assert "No DexScreener pairs" in warnings[0]

    def test_wrong_chain_warns_but_still_returns_data(self, monkeypatch):
        payload = {"pairs": [dexscreener_pair(chainId="ethereum")]}
        monkeypatch.setattr(data_fetchers, "_get_json", lambda *a, **k: payload)
        snapshot, warnings = data_fetchers.fetch_token_snapshot(TOKEN_ADDRESS, "base")
        assert snapshot is not None and snapshot.chain == "ethereum"
        assert any("not found on base" in w for w in warnings)


class TestSecurityNormalization:
    def test_clean_token_parses_to_safe_report(self, monkeypatch):
        monkeypatch.setattr(data_fetchers, "_get_json", lambda *a, **k: goplus_response())
        report = data_fetchers.fetch_security_report(TOKEN_ADDRESS, "base")

        assert report.available is True
        assert report.is_honeypot is False
        assert report.owner_renounced is True
        assert report.is_open_source is True
        assert report.buy_tax_pct == 0.0
        assert report.holder_count == 18432
        assert report.lp_burned_pct == pytest.approx(98.0)
        assert report.lp_secured_pct == pytest.approx(98.0)
        assert not report.is_critical

    def test_top10_excludes_lp_and_burn_wallets(self, monkeypatch):
        monkeypatch.setattr(data_fetchers, "_get_json", lambda *a, **k: goplus_response())
        report = data_fetchers.fetch_security_report(TOKEN_ADDRESS, "base")

        # Raw includes the 14% LP pool and 9% burn; adjusted counts only real wallets.
        assert report.top10_pct == pytest.approx(30.1, abs=0.2)
        assert report.top10_pct_adjusted == pytest.approx(7.1, abs=0.2)

    def test_percentages_convert_from_fractions(self, monkeypatch):
        monkeypatch.setattr(data_fetchers, "_get_json", lambda *a, **k: goplus_response(buy_tax="0.05", sell_tax="0.12"))
        report = data_fetchers.fetch_security_report(TOKEN_ADDRESS, "base")
        assert report.buy_tax_pct == pytest.approx(5.0)
        assert report.sell_tax_pct == pytest.approx(12.0)
        assert report.max_tax_pct == pytest.approx(12.0)
        assert report.creator_percent == pytest.approx(0.43)

    def test_honeypot_payload_flags_critical_and_warns(self, monkeypatch):
        monkeypatch.setattr(data_fetchers, "_get_json", lambda *a, **k: goplus_honeypot_response())
        report = data_fetchers.fetch_security_report(TOKEN_ADDRESS, "base")

        assert report.is_critical is True
        assert any("HONEYPOT" in w for w in report.warnings)
        assert any("mintable" in w.lower() for w in report.warnings)
        assert any("not verified" in w.lower() for w in report.warnings)
        assert report.owner_renounced is False

    def test_goplus_failure_returns_unavailable_not_exception(self, monkeypatch):
        def boom(*args, **kwargs):
            raise FetchError("timeout")

        monkeypatch.setattr(data_fetchers, "_get_json", boom)
        report = data_fetchers.fetch_security_report(TOKEN_ADDRESS, "base")
        assert report.available is False
        assert "GoPlus unavailable" in report.error

    def test_goplus_error_code_returns_unavailable(self, monkeypatch):
        monkeypatch.setattr(data_fetchers, "_get_json", lambda *a, **k: {"code": 0, "message": "rate limited"})
        report = data_fetchers.fetch_security_report(TOKEN_ADDRESS, "base")
        assert report.available is False
        assert "rate limited" in report.error

    def test_chain_without_security_provider_is_handled(self, monkeypatch):
        monkeypatch.setitem(
            config.CHAINS, "novel",
            config.ChainConfig("novel", "Novel", "novel", None, "evm", "https://x/{address}"),
        )
        report = data_fetchers.fetch_security_report(TOKEN_ADDRESS, "novel")
        assert report.available is False
        assert "No security provider" in report.error


class TestDiscovery:
    def test_collapse_pairs_groups_by_base_token(self):
        pair_a1 = dexscreener_pair(pairAddress="0xa1")
        pair_a2 = dexscreener_pair(pairAddress="0xa2", dexId="aerodrome", liquidity={"usd": 50_000.0})
        pair_b = dexscreener_pair(
            pairAddress="0xb1",
            baseToken={"address": "0xB000000000000000000000000000000000000001", "name": "Other", "symbol": "OTHER"},
        )
        snapshots = data_fetchers.collapse_pairs_to_tokens([pair_a1, pair_a2, pair_b])
        assert len(snapshots) == 2
        brett = next(s for s in snapshots if s.symbol == "BRETT")
        assert brett.pair_count == 2

    def test_discover_pairs_filters_to_requested_chain(self, monkeypatch):
        monkeypatch.setattr(
            data_fetchers, "search_pairs",
            lambda query, use_cache=True: [dexscreener_pair(), dexscreener_pair(chainId="ethereum", pairAddress="0xeth")],
        )
        monkeypatch.setattr(data_fetchers, "fetch_boosted_tokens", lambda **k: [])
        monkeypatch.setattr(data_fetchers, "fetch_token_profiles", lambda **k: [])

        pairs, warnings = data_fetchers.discover_pairs("base")
        assert all(p["chainId"] == "base" for p in pairs)
        assert len(pairs) == 1
