"""Tests for the portfolio SQLite store."""

from __future__ import annotations

import pytest

from src import portfolio_store
from src.models import ChainHeat, PortfolioSnapshot, Position, TokenSnapshot, Wallet

EVM_WALLET = "0x532f27101965dd16442E59d40670FaF5eBB142E4"
BREW = "0x1234567890abcdef1234567890abcdef12345678"


@pytest.fixture
def db(tmp_path):
    return tmp_path / "store.sqlite3"


class TestSchema:
    def test_init_is_idempotent(self, db):
        portfolio_store.init_db(db)
        portfolio_store.init_db(db)
        assert portfolio_store.list_wallets(db_path=db) == []

    def test_a_broken_path_degrades_instead_of_raising(self, tmp_path):
        # Persistence is a convenience; it must never take the dashboard down.
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory")
        unwritable = blocker / "db.sqlite3"
        assert portfolio_store.list_wallets(db_path=unwritable) == []
        assert portfolio_store.add_wallet(EVM_WALLET, "bsc", db_path=unwritable) is False


class TestWallets:
    def test_add_and_list(self, db):
        portfolio_store.add_wallet(EVM_WALLET, "bsc", "main", db_path=db)
        wallets = portfolio_store.list_wallets(db_path=db)
        assert wallets[0].label == "main"
        assert wallets[0].address == EVM_WALLET.lower()

    def test_re_adding_updates_the_label_instead_of_duplicating(self, db):
        portfolio_store.add_wallet(EVM_WALLET, "bsc", "main", db_path=db)
        portfolio_store.add_wallet(EVM_WALLET, "bsc", "renamed", db_path=db)
        wallets = portfolio_store.list_wallets(db_path=db)
        assert len(wallets) == 1 and wallets[0].label == "renamed"

    def test_the_same_address_on_two_chains_is_two_wallets(self, db):
        portfolio_store.add_wallet(EVM_WALLET, "bsc", db_path=db)
        portfolio_store.add_wallet(EVM_WALLET, "base", db_path=db)
        assert len(portfolio_store.list_wallets(db_path=db)) == 2
        assert len(portfolio_store.list_wallets(chain="base", db_path=db)) == 1

    def test_replace_swaps_the_whole_list(self, db):
        portfolio_store.add_wallet(EVM_WALLET, "bsc", db_path=db)
        portfolio_store.replace_wallets(
            [Wallet(address="0xabc", chain="base", label="new")], db_path=db
        )
        wallets = portfolio_store.list_wallets(db_path=db)
        assert len(wallets) == 1 and wallets[0].chain == "base"

    def test_remove(self, db):
        portfolio_store.add_wallet(EVM_WALLET, "bsc", db_path=db)
        portfolio_store.remove_wallet(EVM_WALLET, "bsc", db_path=db)
        assert portfolio_store.list_wallets(db_path=db) == []


class TestPositionMeta:
    def test_partial_updates_keep_the_other_fields(self, db):
        portfolio_store.set_position_meta("bsc", BREW, tag="Brew", db_path=db)
        portfolio_store.set_position_meta("bsc", BREW, avg_cost_usd=0.25, db_path=db)
        meta = portfolio_store.all_position_meta(db_path=db)[f"bsc:{BREW}"]
        assert meta["tag"] == "Brew"
        assert meta["avg_cost_usd"] == pytest.approx(0.25)

    def test_first_seen_is_stamped_once_and_kept(self, db):
        portfolio_store.set_position_meta("bsc", BREW, first_seen="2026-09-01T00:00:00Z", db_path=db)
        portfolio_store.set_position_meta("bsc", BREW, tag="Brew", db_path=db)
        meta = portfolio_store.all_position_meta(db_path=db)[f"bsc:{BREW}"]
        assert meta["first_seen"] == "2026-09-01T00:00:00Z"

    def test_clearing_a_cost_basis_returns_it_to_unknown(self, db):
        portfolio_store.set_position_meta("bsc", BREW, avg_cost_usd=0.25, db_path=db)
        portfolio_store.clear_avg_cost("bsc", BREW, db_path=db)
        meta = portfolio_store.all_position_meta(db_path=db)[f"bsc:{BREW}"]
        assert meta["avg_cost_usd"] is None

    def test_keys_match_the_position_key_format(self, db):
        position = Position(address=BREW, chain="bsc", quantity=1)
        portfolio_store.set_position_meta("bsc", BREW, tag="Brew", db_path=db)
        assert position.key in portfolio_store.all_position_meta(db_path=db)


class TestSnapshots:
    def _snapshot(self, value=500.0, taken_at="2026-09-12T00:00:00+00:00"):
        return PortfolioSnapshot(
            taken_at=taken_at,
            positions=[Position(
                address=BREW, chain="bsc", quantity=100, price_usd=value / 100,
                value_usd=value, symbol="BREW",
                snapshot=TokenSnapshot(address=BREW, chain="bsc", liquidity_usd=1e5),
            )],
        )

    def test_round_trip_keeps_the_full_payload(self, db):
        stored_id = portfolio_store.record_snapshot(self._snapshot(), db_path=db)
        payload = portfolio_store.load_snapshot(stored_id, db_path=db)
        assert payload["total_usd"] == pytest.approx(500.0)
        assert payload["positions"][0]["symbol"] == "BREW"
        assert payload["positions"][0]["snapshot"]["liquidity_usd"] == pytest.approx(1e5)

    def test_an_empty_snapshot_is_not_recorded(self, db):
        assert portfolio_store.record_snapshot(PortfolioSnapshot(), db_path=db) is None

    def test_equity_curve_is_oldest_first(self, db):
        portfolio_store.record_snapshot(self._snapshot(100.0, "2026-09-10T00:00:00+00:00"), db_path=db)
        portfolio_store.record_snapshot(self._snapshot(300.0, "2026-09-11T00:00:00+00:00"), db_path=db)
        curve = portfolio_store.equity_curve(db_path=db)
        assert [row["total_usd"] for row in curve] == [100.0, 300.0]

    def test_previous_snapshot_returns_the_newest(self, db):
        portfolio_store.record_snapshot(self._snapshot(100.0), db_path=db)
        portfolio_store.record_snapshot(self._snapshot(300.0), db_path=db)
        assert portfolio_store.previous_snapshot(db_path=db)["total_usd"] == pytest.approx(300.0)


class TestHeatHistory:
    def test_history_is_oldest_first_and_per_chain(self, db):
        for heat, chain in ((40.0, "bsc"), (80.0, "bsc"), (20.0, "base")):
            portfolio_store.record_heat(
                ChainHeat(chain=chain, heat=heat, state="hot",
                          taken_at="2026-09-12T00:00:00+00:00"),
                db_path=db,
            )
        assert [row["heat"] for row in portfolio_store.heat_history("bsc", db_path=db)] == [40.0, 80.0]
        assert len(portfolio_store.heat_history("base", db_path=db)) == 1

    def test_clear_keeps_wallets_and_annotations(self, db):
        portfolio_store.add_wallet(EVM_WALLET, "bsc", db_path=db)
        portfolio_store.set_position_meta("bsc", BREW, tag="Brew", db_path=db)
        portfolio_store.record_heat(ChainHeat(chain="bsc", heat=50.0, state="cold"), db_path=db)
        portfolio_store.clear_portfolio(db_path=db)

        assert portfolio_store.heat_history("bsc", db_path=db) == []
        assert portfolio_store.list_wallets(db_path=db)          # wallets survive
        assert portfolio_store.all_position_meta(db_path=db)     # so do your tags
