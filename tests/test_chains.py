"""Chain registry tests, with a focus on chains lacking a security provider."""

from __future__ import annotations

import pytest

from src import analyzer, config, data_fetchers, scorers
from tests.fixtures import TOKEN_ADDRESS, dexscreener_pair, token_profiles_response


@pytest.fixture(autouse=True)
def clear_caches():
    data_fetchers.clear_caches()
    yield
    data_fetchers.clear_caches()


class TestChainRegistry:
    def test_every_chain_is_internally_consistent(self):
        for key, chain in config.CHAINS.items():
            assert chain.key == key
            assert chain.dexscreener_id
            assert "{address}" in chain.explorer_token_url
            assert chain.address_kind in ("evm", "solana")

    def test_lookup_falls_back_to_the_default_chain(self):
        assert config.get_chain("nonsense").key == config.DEFAULT_CHAIN
        assert config.get_chain("").key == config.DEFAULT_CHAIN
        assert config.get_chain("ROBINHOOD").key == "robinhood"

    def test_reverse_lookup_by_dexscreener_id(self):
        assert config.chain_from_dexscreener_id("robinhood").key == "robinhood"
        assert config.chain_from_dexscreener_id("unknown-chain") is None

    def test_every_chain_has_scanner_seed_queries(self):
        for key in config.CHAINS:
            assert config.SCANNER_SEED_QUERIES.get(config.CHAINS[key].dexscreener_id), key


class TestRobinhoodChain:
    """Robinhood Chain is an Arbitrum Orbit L2 (id 4663) that DexScreener
    indexes but GoPlus does not cover, so it exercises the no-security path."""

    def test_registered_with_the_right_identifiers(self):
        chain = config.get_chain("robinhood")
        assert chain.label == "Robinhood Chain"
        assert chain.dexscreener_id == "robinhood"
        assert chain.native_symbol == "ETH"      # Orbit L2, ETH gas
        assert chain.address_kind == "evm"
        assert chain.explorer_token_url.format(address="0xabc").endswith("/token/0xabc")

    def test_has_no_security_provider(self):
        assert config.get_chain("robinhood").goplus_id is None

    def test_security_lookup_explains_itself_without_calling_out(self, monkeypatch):
        def fail_if_called(*args, **kwargs):
            raise AssertionError("no security request should be made for an unsupported chain")

        monkeypatch.setattr(data_fetchers, "_get_json", fail_if_called)
        report = data_fetchers.fetch_security_report(TOKEN_ADDRESS, "robinhood")

        assert report.available is False
        assert "GoPlus does not support Robinhood Chain" in report.error
        assert "4663" in report.error

    def test_missing_security_is_penalised_not_treated_as_clean(self):
        report = data_fetchers.fetch_security_report(TOKEN_ADDRESS, "robinhood")
        pillar = scorers.score_security(report)

        assert pillar.score == 40.0          # unknown, not a pass mark
        assert pillar.confidence <= 0.3
        assert "GoPlus does not support" in pillar.reasons[0]

    def test_full_analysis_still_works_and_warns(self, monkeypatch):
        pair = dexscreener_pair(chainId="robinhood")

        def route(url, params=None, retries=None):
            if "token-profiles" in url:
                return token_profiles_response()
            return {"pairs": [pair]}

        monkeypatch.setattr(data_fetchers, "_get_json", route)
        result = analyzer.analyze_token(TOKEN_ADDRESS, config.AppSettings(chain="robinhood"))

        assert result.ok is True
        assert result.chain == "robinhood"
        assert result.scorecard is not None
        assert result.risk_plan is not None
        assert any("GoPlus does not support" in w for w in result.data_warnings)

    def test_scores_lower_than_the_same_token_with_security_data(self, monkeypatch):
        """The same market data must not score as well without a rug check."""
        from src.data_fetchers import _parse_goplus_evm, snapshot_from_pair
        from tests.fixtures import goplus_response

        pair = dexscreener_pair()
        snapshot = snapshot_from_pair(pair, [pair], address=TOKEN_ADDRESS)
        verified = _parse_goplus_evm(TOKEN_ADDRESS, "base", next(iter(goplus_response()["result"].values())))
        unverifiable = data_fetchers.fetch_security_report(TOKEN_ADDRESS, "robinhood")

        with_checks = scorers.build_scorecard(snapshot, verified)
        without_checks = scorers.build_scorecard(snapshot, unverifiable)

        assert without_checks.composite < with_checks.composite
        assert without_checks.confidence < with_checks.confidence

    def test_note_is_available_for_the_ui_to_surface(self):
        note = config.SECURITY_PROVIDER_NOTES["robinhood"]
        assert "GoPlus does not support" in note
        assert "Verify contracts manually" in note
