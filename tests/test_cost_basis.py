"""Tests for cost-basis reconstruction, with every lookup stubbed."""

from __future__ import annotations

import pytest

from src import config, cost_basis, portfolio_store
from src.models import Position, Trade, Wallet

WALLET = "0x532f27101965dd16442e59d40670faf5ebb142e4"
WALLET_2 = "0x4ed4e862860bed51a9570b96d89af5e1b0efefed"
STRANGER = "0x1111111111111111111111111111111111111111"
BREW = "0x1234567890abcdef1234567890abcdef12345678"
USDT_BSC = "0x55d398326f99059ff775485246999027b3197955"
WBNB = "0xbb4cdb9cbd36b01bd1cbaebf2de08d9173bc095c"
OWNED = {WALLET, WALLET_2}

DAY = 86_400
T0 = 1_756_000_000


@pytest.fixture
def db(tmp_path):
    return tmp_path / "basis.sqlite3"


def leg(token, frm, to, quantity, tx="0xtx1", timestamp=T0, symbol="", native=False, decimals=18):
    return {
        "hash": tx, "timestamp": timestamp, "from": frm, "to": to, "token": token,
        "symbol": symbol, "decimals": decimals, "quantity": quantity, "native": native,
    }


def buy_legs(token_qty, quote_token, quote_qty, tx="0xbuy", timestamp=T0, symbol="USDT"):
    """A swap: the quote token leaves the wallet, the token arrives."""
    return [
        leg(quote_token, WALLET, STRANGER, quote_qty, tx, timestamp, symbol,
            native=(quote_token == "native")),
        leg(BREW, STRANGER, WALLET, token_qty, tx, timestamp, "BREW"),
    ]


class TestClassification:
    def test_a_swap_out_of_a_stablecoin_is_a_buy(self):
        trade = cost_basis.classify_transaction(
            buy_legs(1000, USDT_BSC, 250), BREW, "bsc", OWNED)
        assert trade.kind == "buy"
        assert trade.quantity == pytest.approx(1000)
        assert trade.quote_address == USDT_BSC
        assert trade.quote_quantity == pytest.approx(250)

    def test_a_swap_back_out_is_a_sell(self):
        legs = [
            leg(BREW, WALLET, STRANGER, 400, "0xsell"),
            leg(USDT_BSC, STRANGER, WALLET, 300, "0xsell", symbol="USDT"),
        ]
        trade = cost_basis.classify_transaction(legs, BREW, "bsc", OWNED)
        assert trade.kind == "sell"
        assert trade.quantity == pytest.approx(-400)
        assert trade.quote_quantity == pytest.approx(300)

    def test_a_token_arriving_alone_is_a_transfer_in(self):
        legs = [leg(BREW, STRANGER, WALLET, 500, "0xdrop")]
        assert cost_basis.classify_transaction(legs, BREW, "bsc", OWNED).kind == "transfer_in"

    def test_a_token_leaving_alone_is_a_transfer_out(self):
        legs = [leg(BREW, WALLET, STRANGER, 500, "0xout")]
        assert cost_basis.classify_transaction(legs, BREW, "bsc", OWNED).kind == "transfer_out"

    def test_a_move_between_your_own_wallets_is_not_a_trade(self):
        # Otherwise consolidating bags would read as a sale plus a repurchase.
        legs = [leg(BREW, WALLET, WALLET_2, 500, "0xmove")]
        assert cost_basis.classify_transaction(legs, BREW, "bsc", OWNED) is None

    def test_a_stablecoin_leg_is_preferred_over_a_volatile_one(self):
        # Both legs left the wallet; the stable one prices the trade exactly.
        legs = [
            leg(WBNB, WALLET, STRANGER, 0.5, "0xmulti", symbol="WBNB"),
            leg(USDT_BSC, WALLET, STRANGER, 250, "0xmulti", symbol="USDT"),
            leg(BREW, STRANGER, WALLET, 1000, "0xmulti", symbol="BREW"),
        ]
        trade = cost_basis.classify_transaction(legs, BREW, "bsc", OWNED)
        assert trade.quote_address == USDT_BSC

    def test_a_native_leg_is_used_when_there_is_no_stable(self):
        legs = buy_legs(1000, "native", 2.0, symbol="BNB")
        trade = cost_basis.classify_transaction(legs, BREW, "bsc", OWNED)
        assert trade.kind == "buy"
        assert trade.quote_address == "native"

    def test_an_untouched_token_produces_nothing(self):
        legs = [leg(USDT_BSC, WALLET, STRANGER, 10, "0xother")]
        assert cost_basis.classify_transaction(legs, BREW, "bsc", OWNED) is None


class TestPricing:
    def test_a_stablecoin_leg_costs_no_network_call(self, db):
        def explode(url, params):
            raise AssertionError("a stablecoin must never need a price lookup")

        trades = [Trade(kind="buy", timestamp=T0, quantity=1000, quote_address=USDT_BSC,
                        quote_quantity=250.0)]
        cost_basis.price_trades(trades, "bsc", session=explode, db_path=db)
        assert trades[0].usd_value == pytest.approx(250.0)
        assert trades[0].quote_price_usd == pytest.approx(1.0)

    def test_a_native_leg_is_priced_at_its_timestamp(self, db):
        calls = []

        def session(url, params):
            calls.append(params)
            bucket = cost_basis.price_bucket(T0)
            return {"coins": {"coingecko:binancecoin": {
                "prices": [{"timestamp": bucket, "price": 600.0}]}}}

        trades = [Trade(kind="buy", timestamp=T0, quantity=1000, quote_address="native",
                        quote_symbol="BNB", quote_quantity=2.0)]
        cost_basis.price_trades(trades, "bsc", session=session, db_path=db)
        assert trades[0].usd_value == pytest.approx(1200.0)
        assert len(calls) == 1

    def test_prices_are_cached_so_a_second_run_is_free(self, db):
        bucket = cost_basis.price_bucket(T0)

        def session(url, params):
            return {"coins": {"coingecko:binancecoin": {
                "prices": [{"timestamp": bucket, "price": 600.0}]}}}

        first = [Trade(kind="buy", timestamp=T0, quantity=1, quote_address="native",
                       quote_symbol="BNB", quote_quantity=1.0)]
        cost_basis.price_trades(first, "bsc", session=session, db_path=db)

        def explode(url, params):
            raise AssertionError("a past price cannot change, so it must come from cache")

        second = [Trade(kind="buy", timestamp=T0 + 60, quantity=1, quote_address="native",
                        quote_symbol="BNB", quote_quantity=1.0)]
        cost_basis.price_trades(second, "bsc", session=explode, db_path=db)
        assert second[0].usd_value == pytest.approx(600.0)

    def test_an_unpriceable_leg_is_reported_not_zeroed(self, db):
        trades = [Trade(kind="buy", timestamp=T0, quantity=1000,
                        quote_address="0xsomerandomtoken", quote_quantity=5.0)]
        notes = cost_basis.price_trades(
            trades, "bsc", session=lambda url, params: {"coins": {}}, db_path=db)
        assert trades[0].usd_value is None
        assert not trades[0].priced
        assert any("could not be priced" in note for note in notes)

    def test_a_chain_defillama_does_not_index_still_prices_its_native_coin(self):
        # Robinhood Chain is not in the coins API, but ETH is ETH.
        assert cost_basis.coin_id("robinhood", "native", "ETH", native=True) == "coingecko:ethereum"
        assert cost_basis.coin_id("robinhood", "0xsometoken") is None


class TestWeightedAverage:
    def _buy(self, qty, usd, at=T0):
        return Trade(kind="buy", timestamp=at, quantity=qty, quote_quantity=usd,
                     quote_address=USDT_BSC, usd_value=usd, quote_price_usd=1.0)

    def _sell(self, qty, usd, at=T0 + DAY):
        return Trade(kind="sell", timestamp=at, quantity=-qty, quote_quantity=usd,
                     quote_address=USDT_BSC, usd_value=usd, quote_price_usd=1.0)

    def test_two_buys_average_together(self):
        report = cost_basis.build_report(
            [self._buy(1000, 100.0), self._buy(1000, 300.0, T0 + DAY)],
            current_quantity=2000, chain="bsc", token=BREW)
        assert report.avg_cost_usd == pytest.approx(0.2)
        assert report.total_cost_usd == pytest.approx(400.0)
        assert report.coverage_pct == pytest.approx(100.0)

    def test_a_partial_sell_realizes_pnl_and_keeps_the_average(self):
        # Buy 1000 at $0.10, sell 500 for $250 -> $200 realized, average intact.
        report = cost_basis.build_report(
            [self._buy(1000, 100.0), self._sell(500, 250.0)],
            current_quantity=500, chain="bsc", token=BREW)
        assert report.realized_pnl_usd == pytest.approx(200.0)
        assert report.avg_cost_usd == pytest.approx(0.1)
        assert report.total_cost_usd == pytest.approx(50.0)
        assert report.coverage_pct == pytest.approx(100.0)

    def test_an_airdrop_lowers_coverage_rather_than_counting_as_free(self):
        report = cost_basis.build_report(
            [self._buy(800, 80.0),
             Trade(kind="transfer_in", timestamp=T0 + DAY, quantity=200)],
            current_quantity=1000, chain="bsc", token=BREW)
        assert report.avg_cost_usd == pytest.approx(0.1)     # the 800 you bought
        assert report.coverage_pct == pytest.approx(80.0)
        assert report.is_partial
        assert any("without a purchase" in note for note in report.notes)

    def test_airdrops_can_be_counted_as_free_when_you_ask(self, monkeypatch):
        monkeypatch.setattr(config, "COST_BASIS_COUNT_AIRDROPS_AS_ZERO", True)
        report = cost_basis.build_report(
            [self._buy(800, 80.0),
             Trade(kind="transfer_in", timestamp=T0 + DAY, quantity=200)],
            current_quantity=1000, chain="bsc", token=BREW)
        assert report.avg_cost_usd == pytest.approx(0.08)    # $80 over 1000
        assert report.coverage_pct == pytest.approx(100.0)

    def test_selling_an_airdrop_does_not_invent_profit(self):
        # Half the bag has no basis, so only the explained half's proceeds
        # count toward realized P&L.
        report = cost_basis.build_report(
            [self._buy(500, 50.0),
             Trade(kind="transfer_in", timestamp=T0 + 1, quantity=500),
             self._sell(500, 500.0)],
            current_quantity=500, chain="bsc", token=BREW)
        # 250 of the 500 sold came from the explained pool: $250 proceeds
        # against $25 of cost.
        assert report.realized_pnl_usd == pytest.approx(225.0)
        assert report.avg_cost_usd == pytest.approx(0.1)

    def test_an_unpriced_buy_lowers_coverage(self):
        unpriced = Trade(kind="buy", timestamp=T0 + DAY, quantity=500,
                         quote_address="0xrandom", quote_quantity=3.0)
        report = cost_basis.build_report(
            [self._buy(500, 50.0), unpriced],
            current_quantity=1000, chain="bsc", token=BREW)
        assert report.avg_cost_usd == pytest.approx(0.1)
        assert report.coverage_pct == pytest.approx(50.0)

    def test_a_transfer_out_reduces_the_bag_without_realizing_pnl(self):
        report = cost_basis.build_report(
            [self._buy(1000, 100.0),
             Trade(kind="transfer_out", timestamp=T0 + DAY, quantity=-500)],
            current_quantity=500, chain="bsc", token=BREW)
        assert report.realized_pnl_usd == pytest.approx(0.0)
        assert report.avg_cost_usd == pytest.approx(0.1)
        assert report.total_cost_usd == pytest.approx(50.0)

    def test_no_priced_purchase_means_no_basis_at_all(self):
        report = cost_basis.build_report(
            [Trade(kind="transfer_in", timestamp=T0, quantity=1000)],
            current_quantity=1000, chain="bsc", token=BREW)
        assert report.avg_cost_usd is None
        assert report.coverage_pct == pytest.approx(0.0)
        assert any("no cost basis" in note for note in report.notes)

    def test_history_shorter_than_the_bag_is_flagged(self):
        report = cost_basis.build_report(
            [self._buy(100, 10.0)], current_quantity=1000, chain="bsc", token=BREW)
        assert any("predates the available history" in note for note in report.notes)

    def test_selling_more_than_reconstructed_does_not_go_negative(self):
        report = cost_basis.build_report(
            [self._buy(100, 10.0), self._sell(1000, 900.0)],
            current_quantity=0, chain="bsc", token=BREW)
        assert report.quantity_explained >= 0
        assert report.total_cost_usd >= 0


class TestEndToEnd:
    def _ledger(self, monkeypatch, transfers, native=None):
        monkeypatch.setattr(
            cost_basis, "fetch_evm_ledger",
            lambda chain, wallets: cost_basis.Ledger(
                transfers=transfers, native=native or []),
        )

    def test_derives_a_position_from_raw_history(self, monkeypatch, db):
        self._ledger(monkeypatch, [
            *buy_legs(1000, USDT_BSC, 100.0, "0xbuy1", T0),
            *buy_legs(1000, USDT_BSC, 300.0, "0xbuy2", T0 + DAY),
        ])
        position = Position(address=BREW, chain="bsc", quantity=2000, symbol="BREW")
        reports = cost_basis.derive_for_chain(
            "bsc", [position], [Wallet(address=WALLET, chain="bsc")], db_path=db)

        report = reports[position.key]
        assert report.avg_cost_usd == pytest.approx(0.2)
        assert len(report.buys) == 2

    def test_derivation_persists_and_shows_up_as_derived(self, monkeypatch, db):
        self._ledger(monkeypatch, buy_legs(1000, USDT_BSC, 250.0))
        position = Position(address=BREW, chain="bsc", quantity=1000, symbol="BREW")
        cost_basis.derive_for_positions(
            [position], wallets=[Wallet(address=WALLET, chain="bsc")], db_path=db)

        meta = portfolio_store.all_position_meta(db_path=db)[position.key]
        assert meta["avg_cost_usd"] == pytest.approx(0.25)
        assert meta["basis_source"] == "derived"

    def test_a_manual_basis_survives_a_derive_run(self, monkeypatch, db):
        portfolio_store.set_position_meta(
            "bsc", BREW, avg_cost_usd=0.9, basis_source="manual", db_path=db)
        self._ledger(monkeypatch, buy_legs(1000, USDT_BSC, 250.0))
        position = Position(address=BREW, chain="bsc", quantity=1000, symbol="BREW")
        cost_basis.derive_for_positions(
            [position], wallets=[Wallet(address=WALLET, chain="bsc")],
            only_missing=False, db_path=db)

        meta = portfolio_store.all_position_meta(db_path=db)[position.key]
        assert meta["avg_cost_usd"] == pytest.approx(0.9)
        assert meta["basis_source"] == "manual"

    def test_positions_that_already_have_a_basis_are_skipped(self, monkeypatch, db):
        monkeypatch.setattr(
            cost_basis, "fetch_evm_ledger",
            lambda chain, wallets: pytest.fail("should not fetch history for a known basis"),
        )
        position = Position(address=BREW, chain="bsc", quantity=1000, avg_cost_usd=0.5)
        assert cost_basis.derive_for_positions(
            [position], wallets=[Wallet(address=WALLET, chain="bsc")], db_path=db) == {}

    def test_solana_is_not_swept_up_by_the_automatic_pass(self, monkeypatch, db):
        monkeypatch.setattr(
            cost_basis, "fetch_evm_ledger",
            lambda chain, wallets: pytest.fail("Solana must not go through the EVM path"),
        )
        position = Position(address="MintOne", chain="solana", quantity=10)
        assert cost_basis.derive_for_positions(
            [position], wallets=[Wallet(address="MintOne", chain="solana")], db_path=db) == {}

    def test_unreadable_history_is_an_error_not_an_empty_basis(self, monkeypatch, db):
        monkeypatch.setattr(
            cost_basis, "fetch_evm_ledger",
            lambda chain, wallets: cost_basis.Ledger(ok=False, warnings=["no API key"]),
        )
        position = Position(address=BREW, chain="bsc", quantity=1000)
        report = cost_basis.derive_for_chain(
            "bsc", [position], [Wallet(address=WALLET, chain="bsc")], db_path=db)[position.key]
        assert not report.ok
        assert report.avg_cost_usd is None
        assert "no API key" in report.error

    def test_one_broken_chain_does_not_lose_the_others(self, monkeypatch, db):
        def ledger(chain, wallets):
            if chain == "base":
                raise RuntimeError("etherscan down")
            return cost_basis.Ledger(transfers=buy_legs(1000, USDT_BSC, 100.0))

        monkeypatch.setattr(cost_basis, "fetch_evm_ledger", ledger)
        positions = [
            Position(address=BREW, chain="bsc", quantity=1000),
            Position(address=BREW, chain="base", quantity=50),
        ]
        wallets = [Wallet(address=WALLET, chain="bsc"), Wallet(address=WALLET, chain="base")]
        reports = cost_basis.derive_for_positions(positions, wallets=wallets, db_path=db)
        assert "bsc:" + BREW in reports


class TestSolana:
    """Solana states balances before and after, so the diff is the trade."""

    OWNER = "7xKXtg2CW87d97TXJSDpbD5jBkheTqA83TZRuJosgAsU"
    MINT = "MintOneAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"

    def _transaction(self, token_delta, sol_delta, signature="sig1", fee=5000, err=None):
        """One parsed transaction where the owner's balances moved."""
        before_tokens = 100.0
        before_sol = 10.0
        return {
            "blockTime": T0,
            "transaction": {
                "signatures": [signature],
                "message": {"accountKeys": [{"pubkey": self.OWNER}]},
            },
            "meta": {
                "err": err,
                "fee": fee,
                "preBalances": [int(before_sol * 1e9)],
                "postBalances": [int((before_sol + sol_delta) * 1e9) - fee],
                "preTokenBalances": [{
                    "accountIndex": 3, "owner": self.OWNER, "mint": self.MINT,
                    "uiTokenAmount": {"uiAmount": before_tokens, "decimals": 6},
                }],
                "postTokenBalances": [{
                    "accountIndex": 3, "owner": self.OWNER, "mint": self.MINT,
                    "uiTokenAmount": {"uiAmount": before_tokens + token_delta, "decimals": 6},
                }],
            },
        }

    def test_a_buy_is_tokens_in_and_sol_out(self):
        legs = cost_basis._solana_legs(self._transaction(500.0, -2.0), self.OWNER)
        trade = cost_basis.classify_transaction(legs, self.MINT, "solana", {self.OWNER})
        assert trade.kind == "buy"
        assert trade.quantity == pytest.approx(500.0)
        assert trade.quote_address == "native"
        assert trade.quote_quantity == pytest.approx(2.0)

    def test_a_sell_is_tokens_out_and_sol_in(self):
        legs = cost_basis._solana_legs(self._transaction(-500.0, 3.0), self.OWNER)
        trade = cost_basis.classify_transaction(legs, self.MINT, "solana", {self.OWNER})
        assert trade.kind == "sell"
        assert trade.quote_quantity == pytest.approx(3.0)

    def test_the_network_fee_is_not_mistaken_for_a_trade(self):
        # Only a fee moved, so there is nothing to report.
        legs = cost_basis._solana_legs(self._transaction(0.0, 0.0), self.OWNER)
        assert legs == []

    def test_a_failed_transaction_moved_nothing(self):
        legs = cost_basis._solana_legs(
            self._transaction(500.0, -2.0, err={"InstructionError": [0, "Custom"]}), self.OWNER)
        assert legs == []

    def test_another_wallets_balances_are_ignored(self):
        transaction = self._transaction(500.0, -2.0)
        transaction["meta"]["preTokenBalances"][0]["owner"] = "SomeoneElse"
        transaction["meta"]["postTokenBalances"][0]["owner"] = "SomeoneElse"
        legs = cost_basis._solana_legs(transaction, self.OWNER)
        assert all(leg["token"] == "native" for leg in legs)

    def test_the_ledger_pages_signatures_and_batches_transactions(self):
        calls = {"signatures": 0, "batches": 0}

        def session(url, payload):
            if isinstance(payload, list):
                calls["batches"] += 1
                return [
                    {"id": index, "result": self._transaction(500.0, -2.0, f"sig{index}")}
                    for index, _ in enumerate(payload)
                ]
            if payload["method"] == "getSignaturesForAddress":
                calls["signatures"] += 1
                if calls["signatures"] > 1:
                    return {"result": []}
                return {"result": [{"signature": f"sig{i}"} for i in range(3)]}
            return {"result": None}

        ledger = cost_basis.fetch_solana_ledger(
            [Wallet(address=self.OWNER, chain="solana")], session=session)
        assert ledger.ok
        assert calls["batches"] == 1
        assert ledger.transfers

    def test_hitting_the_signature_cap_is_reported(self, monkeypatch):
        monkeypatch.setattr(config, "COST_BASIS_MAX_SIGNATURES", 2)

        def session(url, payload):
            if isinstance(payload, list):
                return [{"id": i, "result": None} for i, _ in enumerate(payload)]
            if payload["method"] == "getSignaturesForAddress":
                return {"result": [{"signature": f"sig{i}"} for i in range(1000)]}
            return {"result": None}

        ledger = cost_basis.fetch_solana_ledger(
            [Wallet(address=self.OWNER, chain="solana")], session=session)
        assert ledger.truncated
        assert any("Stopped after" in warning for warning in ledger.warnings)

    def test_an_unreachable_rpc_is_a_named_failure(self, monkeypatch):
        monkeypatch.setattr(config, "SOLANA_RPC_URL", "")
        ledger = cost_basis.fetch_solana_ledger([Wallet(address=self.OWNER, chain="solana")])
        assert not ledger.ok
        assert "SOLANA_RPC_URL" in ledger.warnings[0]

    def test_derive_position_runs_solana_end_to_end(self, db):
        bucket = cost_basis.price_bucket(T0)

        def session(url, payload):
            if isinstance(payload, list):
                return [{"id": 0, "result": self._transaction(500.0, -2.0)}]
            if isinstance(payload, dict) and payload.get("method") == "getSignaturesForAddress":
                return {"result": [{"signature": "sig0"}]}
            if isinstance(payload, dict) and payload.get("method"):
                return {"result": None}
            # The price lookup arrives as (url, params).
            return {"coins": {"coingecko:solana": {
                "prices": [{"timestamp": bucket, "price": 150.0}]}}}

        position = Position(address=self.MINT, chain="solana", quantity=500.0, symbol="MINT")
        report = cost_basis.derive_position(
            position, wallets=[Wallet(address=self.OWNER, chain="solana")],
            session=session, db_path=db)

        # 2 SOL at $150 for 500 tokens = $0.60 each.
        assert report.avg_cost_usd == pytest.approx(0.6)
        assert portfolio_store.all_position_meta(db_path=db)[position.key]["basis_source"] == "derived"


class TestRegressions:
    def test_the_closest_price_point_wins_not_the_first(self, db):
        bucket = cost_basis.price_bucket(T0)

        def session(url, params):
            # The API answers on its own grid; the far point is listed first.
            return {"coins": {"coingecko:binancecoin": {"prices": [
                {"timestamp": bucket - 3000, "price": 400.0},
                {"timestamp": bucket + 30, "price": 600.0},
            ]}}}

        trades = [Trade(kind="buy", timestamp=T0 + 1, quantity=1, quote_address="native",
                        quote_symbol="BNB", quote_quantity=1.0)]
        cost_basis.price_trades(trades, "bsc", session=session, db_path=db)
        assert trades[0].quote_price_usd == pytest.approx(600.0)

    def test_a_synced_snapshot_records_the_basis_it_derived(self, monkeypatch, db):
        from src import balances as balances_mod
        from src import portfolio

        monkeypatch.setattr(
            balances_mod, "fetch_all_balances",
            lambda wallets, manual_tokens=None: balances_mod.BalanceResult(
                balances=[balances_mod.TokenBalance(
                    address=BREW, chain="bsc", quantity=1000.0, symbol="BREW")]),
        )
        monkeypatch.setattr(
            portfolio, "price_tokens",
            lambda chain, addresses, use_cache=True: {
                BREW: __import__("src.models", fromlist=["TokenSnapshot"]).TokenSnapshot(
                    address=BREW, chain="bsc", symbol="BREW", price_usd=2.0)},
        )
        monkeypatch.setattr(
            cost_basis, "fetch_evm_ledger",
            lambda chain, wallets: cost_basis.Ledger(transfers=buy_legs(1000, USDT_BSC, 250.0)),
        )

        snapshot = portfolio.sync_portfolio(
            wallets=[Wallet(address=WALLET, chain="bsc")], db_path=db)
        assert snapshot.positions[0].avg_cost_usd == pytest.approx(0.25)

        stored = portfolio_store.previous_snapshot(db_path=db)
        assert stored["positions"][0]["avg_cost_usd"] == pytest.approx(0.25)
        assert stored["positions"][0]["basis_source"] == "derived"
