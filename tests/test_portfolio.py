"""Tests for position building, pricing and the derived portfolio views."""

from __future__ import annotations

import pytest

from src import portfolio, portfolio_store
from src.balances import TokenBalance
from src.models import PortfolioSnapshot, Position, Wallet
from tests.fixtures import dexscreener_pair

EVM_WALLET = "0x532f27101965dd16442E59d40670FaF5eBB142E4"
EVM_WALLET_2 = "0x4ed4E862860beD51a9570b96d89aF5E1B0Efefed"
SOL_WALLET = "7xKXtg2CW87d97TXJSDpbD5jBkheTqA83TZRuJosgAsU"
BREW = "0x1234567890abcdef1234567890abcdef12345678"


@pytest.fixture
def db(tmp_path):
    return tmp_path / "portfolio.sqlite3"


def stub_prices(monkeypatch, prices):
    """Replace the DexScreener round trip with fixed per-token snapshots."""
    from src import data_fetchers

    def fake_fetch(chain, addresses, use_cache=True):
        return [
            dexscreener_pair(
                chainId=chain,
                baseToken={"address": address, "name": meta.get("name", "Token"),
                           "symbol": meta["symbol"]},
                priceUsd=str(meta["price"]),
                pairAddress=f"0xpair{index}",
                liquidity={"usd": meta.get("liquidity", 100_000.0)},
                priceChange={"m5": 0, "h1": 0, "h6": 0, "h24": meta.get("change_24h", 0.0)},
            )
            for index, (address, meta) in enumerate(prices.items())
            if address in {a for a in addresses}
        ]

    monkeypatch.setattr(data_fetchers, "fetch_pairs_for_addresses", fake_fetch)


class TestWalletInput:
    def test_parses_address_and_label(self):
        wallets = portfolio.parse_wallet_input(f"{EVM_WALLET}, main bag\n", "base")
        assert len(wallets) == 1
        assert wallets[0].label == "main bag"
        assert wallets[0].chain == "base"

    def test_duplicates_and_comments_are_dropped(self):
        raw = f"# my wallets\n{EVM_WALLET}\n{EVM_WALLET.lower()}\n\n"
        assert len(portfolio.parse_wallet_input(raw, "base")) == 1

    def test_solana_address_is_rejected_on_an_evm_chain(self):
        # Valid address, wrong network: querying it on Base would return an
        # empty wallet rather than an error, which is the worst outcome.
        assert portfolio.parse_wallet_input(SOL_WALLET, "base") == []
        assert len(portfolio.parse_wallet_input(SOL_WALLET, "solana")) == 1

    def test_errors_name_the_bad_line(self):
        errors = portfolio.wallet_input_errors(f"{SOL_WALLET}\nnot-an-address", "base")
        assert any("Solana address" in e for e in errors)
        assert any("not a valid address" in e for e in errors)


class TestBuildPositions:
    def test_prices_and_sorts_positions(self, monkeypatch):
        stub_prices(monkeypatch, {
            BREW: {"symbol": "BREW", "price": 0.5},
            "0xaaa": {"symbol": "SMALL", "price": 0.01},
        })
        positions, unpriced, dust, dust_count = portfolio.build_positions([
            TokenBalance(address=BREW, chain="bsc", quantity=1000),
            TokenBalance(address="0xaaa", chain="bsc", quantity=10_000),
        ])
        assert [p.symbol for p in positions] == ["BREW", "SMALL"]   # value descending
        assert positions[0].value_usd == pytest.approx(500.0)
        assert unpriced == [] and dust == 0.0 and dust_count == 0

    def test_same_token_in_two_wallets_is_one_position(self, monkeypatch):
        stub_prices(monkeypatch, {BREW: {"symbol": "BREW", "price": 2.0}})
        positions, _, _, _ = portfolio.build_positions([
            TokenBalance(address=BREW, chain="bsc", quantity=100, wallet=EVM_WALLET),
            TokenBalance(address=BREW.upper(), chain="bsc", quantity=50, wallet=EVM_WALLET_2),
        ])
        assert len(positions) == 1
        assert positions[0].quantity == pytest.approx(150.0)
        assert len(positions[0].wallets) == 2

    def test_unpriced_tokens_are_reported_not_dropped(self, monkeypatch):
        stub_prices(monkeypatch, {})
        positions, unpriced, _, _ = portfolio.build_positions([
            TokenBalance(address=BREW, chain="bsc", quantity=100, symbol="BREW"),
        ])
        assert positions == []
        assert unpriced[0]["symbol"] == "BREW"
        assert "No DexScreener pair" in unpriced[0]["reason"]

    def test_dust_is_summed_rather_than_hidden(self, monkeypatch):
        stub_prices(monkeypatch, {
            BREW: {"symbol": "BREW", "price": 1.0},
            "0xdust": {"symbol": "DUST", "price": 0.0001},
        })
        positions, _, dust, dust_count = portfolio.build_positions([
            TokenBalance(address=BREW, chain="bsc", quantity=100),
            TokenBalance(address="0xdust", chain="bsc", quantity=1000),   # $0.10
        ])
        assert [p.symbol for p in positions] == ["BREW"]
        assert dust == pytest.approx(0.1)
        assert dust_count == 1
        # And the total still reconciles with the wallet.
        snapshot = PortfolioSnapshot(positions=positions, dust_usd=dust)
        assert snapshot.total_usd == pytest.approx(100.1)

    def test_annotations_are_attached(self, monkeypatch):
        stub_prices(monkeypatch, {BREW: {"symbol": "BREW", "price": 1.0}})
        meta = {f"bsc:{BREW}": {"avg_cost_usd": 0.25, "tag": "Brew", "note": "core",
                                "first_seen": "2026-09-01T00:00:00Z"}}
        positions, _, _, _ = portfolio.build_positions(
            [TokenBalance(address=BREW, chain="bsc", quantity=1000)], meta=meta
        )
        position = positions[0]
        assert position.tag == "Brew"
        assert position.cost_basis_usd == pytest.approx(250.0)
        assert position.unrealized_pnl_usd == pytest.approx(750.0)
        assert position.unrealized_pnl_pct == pytest.approx(300.0)

    def test_missing_basis_reports_unknown_not_zero(self, monkeypatch):
        stub_prices(monkeypatch, {BREW: {"symbol": "BREW", "price": 1.0}})
        positions, _, _, _ = portfolio.build_positions(
            [TokenBalance(address=BREW, chain="bsc", quantity=100)]
        )
        assert positions[0].cost_basis_usd is None
        assert positions[0].unrealized_pnl_usd is None
        assert positions[0].unrealized_pnl_pct is None

    def test_each_chain_is_priced_against_its_own_chain_id(self, monkeypatch):
        seen = []
        from src import data_fetchers

        def fake_fetch(chain, addresses, use_cache=True):
            seen.append(chain)
            return []

        monkeypatch.setattr(data_fetchers, "fetch_pairs_for_addresses", fake_fetch)
        portfolio.build_positions([
            TokenBalance(address=BREW, chain="bsc", quantity=1),
            TokenBalance(address="MintOne", chain="solana", quantity=1),
        ])
        assert set(seen) == {"bsc", "solana"}


class TestSync:
    def test_sync_persists_a_snapshot_and_stamps_first_seen(self, monkeypatch, db):
        stub_prices(monkeypatch, {BREW: {"symbol": "BREW", "price": 1.0}})
        monkeypatch.setattr(
            "src.balances.fetch_all_balances",
            lambda wallets, manual_tokens=None: __import__(
                "src.balances", fromlist=["BalanceResult"]
            ).BalanceResult(
                balances=[TokenBalance(address=BREW, chain="bsc", quantity=100)]
            ),
        )
        wallets = [Wallet(address=EVM_WALLET, chain="bsc")]
        snapshot = portfolio.sync_portfolio(wallets=wallets, db_path=db)

        assert snapshot.total_usd == pytest.approx(100.0)
        assert snapshot.positions[0].first_seen
        assert len(portfolio_store.recent_snapshots(db_path=db)) == 1

    def test_chain_with_no_positions_gets_a_coverage_note(self, monkeypatch, db):
        stub_prices(monkeypatch, {})
        monkeypatch.setattr(
            "src.balances.fetch_all_balances",
            lambda wallets, manual_tokens=None: __import__(
                "src.balances", fromlist=["BalanceResult"]
            ).BalanceResult(balances=[], warnings=["Blockscout (robinhood) unreachable"]),
        )
        snapshot = portfolio.sync_portfolio(
            wallets=[Wallet(address=EVM_WALLET, chain="robinhood")], db_path=db
        )
        # An unreadable chain must never look like an empty wallet.
        assert "robinhood" in snapshot.coverage_notes
        assert snapshot.warnings

    def test_no_wallets_is_a_prompt_not_an_error(self, db):
        snapshot = portfolio.sync_portfolio(wallets=[], db_path=db)
        assert snapshot.positions == []
        assert "No wallets registered" in snapshot.warnings[0]


class TestDerivedViews:
    def test_tag_rollup_groups_an_ecosystem(self):
        from src.models import TokenSnapshot

        def position(symbol, value, tag, change):
            return Position(
                address=f"0x{symbol}", chain="bsc", quantity=1, price_usd=value,
                value_usd=value, symbol=symbol, tag=tag,
                snapshot=TokenSnapshot(address=f"0x{symbol}", chain="bsc",
                                       price_change_24h=change),
            )

        snapshot = PortfolioSnapshot(positions=[
            position("BREW", 600, "Brew", 100.0),
            position("MUG", 400, "Brew", 50.0),
            position("OTHER", 1000, "", 0.0),
        ])
        rows = {row["tag"]: row for row in portfolio.totals_by_tag(snapshot)}
        assert rows["Brew"]["value_usd"] == pytest.approx(1000.0)
        assert rows["Brew"]["positions"] == 2
        assert rows["Brew"]["allocation_pct"] == pytest.approx(50.0)
        # Value-weighted, so the bigger bag moves the group more.
        assert rows["Brew"]["change_24h"] == pytest.approx(80.0)
        assert "Untagged" in rows

    def test_position_changes_against_previous_snapshot(self):
        from src.models import TokenSnapshot

        current = PortfolioSnapshot(positions=[Position(
            address=BREW, chain="bsc", quantity=100, price_usd=2.0, value_usd=200.0,
            snapshot=TokenSnapshot(address=BREW, chain="bsc", liquidity_usd=110_000.0),
        )])
        previous = {"positions": [{
            "chain": "bsc", "address": BREW, "quantity": 100, "value_usd": 100.0,
            "snapshot": {"liquidity_usd": 100_000.0},
        }]}
        change = portfolio.position_changes(current, previous)[f"bsc:{BREW}"]
        assert change["value_delta_usd"] == pytest.approx(100.0)
        assert change["value_delta_pct"] == pytest.approx(100.0)
        assert change["liquidity_delta_pct"] == pytest.approx(10.0)

    def test_position_changes_without_history_is_empty(self):
        assert portfolio.position_changes(PortfolioSnapshot(), None) == {}

    def test_chain_allocation_adds_up(self):
        snapshot = PortfolioSnapshot(positions=[
            Position(address="a", chain="bsc", quantity=1, value_usd=750.0),
            Position(address="b", chain="base", quantity=1, value_usd=250.0),
        ])
        allocation = snapshot.chain_allocation_pct()
        assert allocation["bsc"] == pytest.approx(75.0)
        assert sum(allocation.values()) == pytest.approx(100.0)
