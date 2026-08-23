"""Tests for on-chain wallet-flow analysis."""

from __future__ import annotations

import json
import time

import pytest

from src import config, scorers, wallet_flow as wf
from src.data_fetchers import snapshot_from_pair
from src.models import WalletActivity, WalletFlowReport
from tests.fixtures import TOKEN_ADDRESS, dexscreener_pair

POOL = "0x36a46dff597c5a444bbc521d26787f57867d2214"
NOW = int(time.time())
HOUR = 3600


def transfer(frm: str, to: str, tokens: float, hours_ago: float, decimals: int = 18) -> dict:
    """One Etherscan tokentx row."""
    return {
        "blockNumber": "1",
        "timeStamp": str(int(NOW - hours_ago * HOUR)),
        "hash": f"0x{abs(hash((frm, to, hours_ago))):x}",
        "from": frm,
        "to": to,
        "value": str(int(tokens * (10 ** decimals))),
        "tokenDecimal": str(decimals),
        "contractAddress": TOKEN_ADDRESS,
    }


def wallet(n: int) -> str:
    return "0x" + f"{n:040x}"


@pytest.fixture
def snapshot():
    pair = dexscreener_pair()
    snap = snapshot_from_pair(pair, [pair], address=TOKEN_ADDRESS)
    snap.total_supply = 10_000_000_000.0
    return snap


@pytest.fixture(autouse=True)
def clear_flow_cache():
    wf.clear_cache()
    yield
    wf.clear_cache()


class TestWatchlist:
    def test_loads_wallets_and_handles(self, tmp_path):
        path = tmp_path / "sm.json"
        path.write_text(json.dumps({
            "wallets": [
                {"address": "0xAAA1111111111111111111111111111111111111", "label": "whale"},
                "0xBBB2222222222222222222222222222222222222",
            ],
            "x_handles": ["@analyst", "onchain"],
        }))
        loaded = wf.load_watchlist(path)

        # Addresses normalise to lower case so matching is case-insensitive.
        assert loaded["wallets"]["0xaaa1111111111111111111111111111111111111"] == "whale"
        assert "0xbbb2222222222222222222222222222222222222" in loaded["wallets"]
        assert loaded["x_handles"] == ["analyst", "onchain"]      # @ stripped

    def test_missing_file_is_not_an_error(self, tmp_path):
        assert wf.load_watchlist(tmp_path / "nope.json") == {"wallets": {}, "x_handles": []}

    def test_malformed_file_degrades(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{ not json at all")
        assert wf.load_watchlist(path) == {"wallets": {}, "x_handles": []}

    def test_shipped_example_is_valid(self):
        loaded = wf.load_watchlist("data/smart_money.example.json")
        assert loaded["wallets"] and loaded["x_handles"]


class TestPoolDetection:
    def test_identifies_the_pool_by_dominance(self):
        transfers = [transfer(POOL, wallet(i), 100, 1) for i in range(1, 30)]
        assert wf.identify_pools(transfers) == [POOL.lower()]

    def test_uses_the_known_pair_address_too(self):
        transfers = [transfer(wallet(1), wallet(2), 5, 1)]
        pools = wf.identify_pools(transfers, known=[POOL])
        assert POOL.lower() in pools

    def test_empty_input_is_safe(self):
        assert wf.identify_pools([]) == []


class TestActivityBuilding:
    def test_pool_transfers_become_buys_and_sells(self):
        transfers = [
            transfer(POOL, wallet(1), 100, 5),      # wallet 1 buys 100
            transfer(POOL, wallet(1), 50, 4),       # buys 50 more
            transfer(wallet(1), POOL, 30, 3),       # sells 30
        ]
        activity = wf.build_activity(transfers, [POOL])

        entry = activity[wallet(1)]
        assert entry.bought_tokens == 150
        assert entry.sold_tokens == 30
        assert entry.net_tokens == 120
        assert entry.is_accumulating and entry.round_tripped

    def test_wallet_to_wallet_transfers_are_ignored(self):
        """Moving tokens between your own wallets is not a buy."""
        transfers = [transfer(wallet(1), wallet(2), 500, 1)]
        assert wf.build_activity(transfers, [POOL]) == {}

    def test_burn_address_is_not_a_holder(self):
        transfers = [transfer(POOL, "0x000000000000000000000000000000000000dEaD", 1000, 1)]
        assert wf.build_activity(transfers, [POOL]) == {}

    def test_decimals_are_applied(self):
        transfers = [transfer(POOL, wallet(1), 1.5, 1, decimals=6)]
        assert wf.build_activity(transfers, [POOL])[wallet(1)].bought_tokens == pytest.approx(1.5)


class TestAnalysis:
    def _accumulation(self):
        """Many wallets buying, few selling, all within the recent window."""
        transfers = [transfer(POOL, wallet(i), 1000, 2) for i in range(1, 13)]
        transfers += [transfer(wallet(i), POOL, 200, 1) for i in range(1, 3)]
        transfers += [transfer(POOL, wallet(i), 500, 30) for i in range(1, 13)]
        return transfers

    def _distribution(self):
        transfers = [transfer(POOL, wallet(i), 1000, 40) for i in range(1, 13)]
        transfers += [transfer(wallet(i), POOL, 1200, 2) for i in range(1, 12)]
        return transfers

    def test_accumulation_verdict(self, snapshot):
        report = wf.analyze_transfers(self._accumulation(), snapshot)

        assert report.available
        assert report.accumulation_verdict == "accumulating"
        assert report.accumulating_wallets > report.distributing_wallets
        assert report.net_flow_tokens > 0

    def test_distribution_verdict(self, snapshot):
        report = wf.analyze_transfers(self._distribution(), snapshot)

        assert report.accumulation_verdict == "distributing"
        assert report.net_flow_tokens < 0
        assert any("selling than buying" in w for w in report.warnings)

    def test_quiet_accumulation_needs_both_flat_price_and_buying(self, snapshot):
        """The headline pattern: accumulation while the chart goes nowhere."""
        snapshot.price_change_24h = 2.0                      # consolidating
        flat = wf.analyze_transfers(self._accumulation(), snapshot)
        assert flat.consolidating is True
        assert flat.quiet_accumulation is True
        assert any("positioning, not chasing" in n for n in flat.notes)

        snapshot.price_change_24h = 65.0                     # ripping
        pumping = wf.analyze_transfers(self._accumulation(), snapshot)
        assert pumping.consolidating is False
        assert pumping.quiet_accumulation is False           # buying a rip is chasing

    def test_distribution_never_counts_as_quiet_accumulation(self, snapshot):
        snapshot.price_change_24h = 1.0
        report = wf.analyze_transfers(self._distribution(), snapshot)
        assert report.quiet_accumulation is False

    def test_early_cohort_hold_rate(self, snapshot):
        transfers = [transfer(POOL, wallet(i), 1000, 100) for i in range(1, 11)]   # 10 early buyers
        transfers += [transfer(wallet(i), POOL, 1000, 50) for i in range(1, 8)]    # 7 sold out
        report = wf.analyze_transfers(transfers, snapshot)

        assert report.early_buyers == 10
        assert report.early_still_holding == 3
        assert report.early_hold_rate == pytest.approx(0.3)
        assert any("early buyers still hold" in w for w in report.warnings)

    def test_one_and_done_wallets_are_flagged(self, snapshot):
        """A wall of single-buy wallets reads as farming, not demand."""
        transfers = [transfer(POOL, wallet(i), 10, 2) for i in range(1, 41)]
        report = wf.analyze_transfers(transfers, snapshot)

        assert report.fresh_wallet_ratio == 1.0
        assert any("bought once and never traded again" in w for w in report.warnings)

    def test_net_flow_as_share_of_supply(self, snapshot):
        transfers = [transfer(POOL, wallet(1), 100_000_000, 2)]      # 1% of supply
        report = wf.analyze_transfers(transfers, snapshot)
        assert report.net_flow_pct_of_supply == pytest.approx(1.0, abs=0.01)

    def test_watchlist_hit_is_surfaced(self, snapshot):
        watchlist = {"wallets": {wallet(1): "known whale"}, "x_handles": []}
        transfers = [transfer(POOL, wallet(i), 1000, 2) for i in range(1, 6)]
        report = wf.analyze_transfers(transfers, snapshot, watchlist)

        assert len(report.watchlist_hits) == 1
        assert report.watchlist_hits[0].label == "known whale"
        assert any("known whale is accumulating" in n for n in report.notes)

    def test_watchlist_wallet_selling_is_a_warning(self, snapshot):
        watchlist = {"wallets": {wallet(1): "known whale"}, "x_handles": []}
        transfers = [
            transfer(POOL, wallet(1), 1000, 40),
            transfer(wallet(1), POOL, 1500, 2),
        ]
        report = wf.analyze_transfers(transfers, snapshot, watchlist)
        assert any("known whale is distributing" in w for w in report.warnings)

    def test_empty_transfers_degrade(self, snapshot):
        report = wf.analyze_transfers([], snapshot)
        assert report.available is False
        assert "No token transfers" in report.error


class TestClient:
    def test_unsupported_chain_is_explained(self):
        client = wf.EtherscanClient(api_key="k")
        assert "not an Etherscan V2 chain" in client.unavailable_reason("solana")
        assert "not an Etherscan V2 chain" in client.unavailable_reason("robinhood")
        assert client.unavailable_reason("base") == ""

    def test_missing_key_is_explained(self, monkeypatch):
        monkeypatch.setattr(config, "ETHERSCAN_API_KEY", "")
        assert "No ETHERSCAN_API_KEY" in wf.EtherscanClient().unavailable_reason("base")

    def test_request_carries_the_chain_id(self, snapshot):
        seen = {}

        def session(url, params):
            seen.update(params)
            return {"status": "1", "result": [transfer(POOL, wallet(1), 10, 1)]}

        client = wf.EtherscanClient(api_key="k", session=session)
        client.token_transfers(TOKEN_ADDRESS, "base", 3000)

        assert seen["chainid"] == "8453"          # Base
        assert seen["action"] == "tokentx"
        assert seen["contractaddress"] == TOKEN_ADDRESS
        assert seen["sort"] == "desc"             # newest activity first

    def test_no_transactions_is_an_empty_result_not_an_error(self):
        client = wf.EtherscanClient(
            api_key="k", session=lambda url, params: {"status": "0", "message": "No transactions found", "result": []}
        )
        assert client.token_transfers(TOKEN_ADDRESS, "base", 100) == []

    def test_api_error_raises(self):
        client = wf.EtherscanClient(
            api_key="k", session=lambda url, params: {"status": "0", "message": "Invalid API Key", "result": None}
        )
        with pytest.raises(RuntimeError, match="Invalid API Key"):
            client.token_transfers(TOKEN_ADDRESS, "base", 100)

    def test_fetch_never_raises(self, snapshot):
        def boom(url, params):
            raise RuntimeError("network down")

        report = wf.fetch_wallet_flow(snapshot, client=wf.EtherscanClient(api_key="k", session=boom))
        assert report.available is False
        assert "network down" in report.error

    def test_unsupported_chain_skips_the_call(self, snapshot):
        snapshot.chain = "solana"

        def fail(url, params):
            raise AssertionError("no request should be made for a non-EVM chain")

        report = wf.fetch_wallet_flow(snapshot, client=wf.EtherscanClient(api_key="k", session=fail))
        assert report.available is False
        assert "not an Etherscan V2 chain" in report.error


class TestScoringIntegration:
    def _report(self, **kwargs):
        base = dict(available=True, accumulation_verdict="accumulating",
                    accumulating_wallets=10, distributing_wallets=2)
        base.update(kwargs)
        return WalletFlowReport(**base)

    def test_quiet_accumulation_raises_the_holder_score(self, snapshot):
        baseline = scorers.score_holders(None, snapshot)
        quiet = scorers.score_holders(None, snapshot, self._report(quiet_accumulation=True))
        assert quiet.score > baseline.score

    def test_distribution_lowers_it(self, snapshot):
        baseline = scorers.score_holders(None, snapshot)
        selling = scorers.score_holders(
            None, snapshot,
            self._report(accumulation_verdict="distributing", accumulating_wallets=2,
                         distributing_wallets=11),
        )
        assert selling.score < baseline.score

    def test_farming_pattern_lowers_it(self, snapshot):
        baseline = scorers.score_holders(None, snapshot, self._report())
        farmed = scorers.score_holders(None, snapshot, self._report(fresh_wallet_ratio=0.95))
        assert farmed.score < baseline.score

    def test_watchlist_accumulation_counts_for_more_than_heuristics(self, snapshot):
        plain = scorers.score_holders(None, snapshot, self._report())
        starred = scorers.score_holders(None, snapshot, self._report(
            watchlist_hits=[WalletActivity(address=wallet(1), bought_tokens=500, label="whale")]
        ))
        assert starred.score > plain.score

    def test_scorecard_surfaces_quiet_accumulation_as_a_positive(self, snapshot):
        card = scorers.build_scorecard(
            snapshot, None, wallet_flow=self._report(quiet_accumulation=True)
        )
        assert any("Quiet accumulation" in p for p in card.positives)

    def test_scorecard_surfaces_watchlist_selling_as_a_risk(self, snapshot):
        card = scorers.build_scorecard(snapshot, None, wallet_flow=self._report(
            watchlist_hits=[WalletActivity(address=wallet(1), sold_tokens=900, label="whale")]
        ))
        assert any("whale is distributing" in r for r in card.risks)


class TestEnsembleHandoff:
    def test_flow_reaches_every_model(self, snapshot):
        from src import llm_analyzers as la

        flow = WalletFlowReport(available=True, quiet_accumulation=True,
                                accumulation_verdict="accumulating",
                                accumulating_wallets=9, distributing_wallets=1)
        payload = la.build_analysis_payload(snapshot, None, None, None, None, flow)

        assert payload["onchain_wallet_flow"]["quiet_accumulation"] is True
        assert payload["onchain_wallet_flow"]["wallets_accumulating"] == 9
        # The models must be told this is behaviour, not a skill rating.
        assert "not a claim about anyone's skill" in payload["onchain_wallet_flow"]["note"]
