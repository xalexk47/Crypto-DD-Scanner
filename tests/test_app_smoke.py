"""End-to-end tests that actually run the Streamlit app.

These exercise the wiring the unit tests cannot see: session state, the widget
graph, and the fact that every tab renders without raising. The network is
stubbed at the module boundary, so nothing here touches a real API.
"""

from __future__ import annotations

import pathlib

import pytest

from src import (balances, config, cost_basis, data_fetchers, portfolio_store,
                 rotation)
from src.balances import BalanceResult, TokenBalance
from src.models import TokenSnapshot
from tests.fixtures import dexscreener_pair

streamlit_testing = pytest.importorskip("streamlit.testing.v1")
AppTest = streamlit_testing.AppTest

APP = str(pathlib.Path(__file__).resolve().parent.parent / "app.py")
EVM_WALLET = "0x532f27101965dd16442E59d40670FaF5eBB142E4"
BREW = "0x1234567890abcdef1234567890abcdef12345678"


@pytest.fixture
def app(tmp_path, monkeypatch):
    """A fresh app instance with an isolated database and no network."""
    db = tmp_path / "app.sqlite3"
    monkeypatch.setattr(config, "PORTFOLIO_DB_PATH", db)
    monkeypatch.setattr(config, "HISTORY_DB_PATH", tmp_path / "history.sqlite3")

    monkeypatch.setattr(
        balances, "fetch_all_balances",
        lambda wallets, manual_tokens=None: BalanceResult(
            balances=[TokenBalance(address=BREW, chain="bsc", quantity=1000.0,
                                   symbol="BREW", wallet=EVM_WALLET)],
            source="stub",
        ),
    )
    monkeypatch.setattr(
        data_fetchers, "fetch_pairs_for_addresses",
        lambda chain, addresses, use_cache=True: [dexscreener_pair(
            chainId=chain,
            baseToken={"address": BREW, "name": "Brew", "symbol": "BREW"},
            priceUsd="2.0",
        )] if BREW in list(addresses) else [],
    )
    monkeypatch.setattr(
        rotation, "fetch_chain_basket",
        lambda chain, use_cache=True: (
            [TokenSnapshot(address=f"0x{chain}", chain=chain, symbol="BASKET",
                           price_usd=1.0, market_cap=1e6, liquidity_usd=5e5,
                           volume_24h=2e5, volume_6h=1e5,
                           price_change_24h=90.0 if chain == "bsc" else -12.0,
                           price_change_6h=40.0 if chain == "bsc" else -6.0,
                           txns_24h_buys=700 if chain == "bsc" else 300,
                           txns_24h_sells=300 if chain == "bsc" else 700)],
            [],
        ),
    )
    monkeypatch.setattr(rotation, "fetch_defillama_stats", lambda chain, use_cache=True: None)
    monkeypatch.setattr(data_fetchers, "fetch_latest_profiles", lambda chain, use_cache=True: [])

    # One buy in the wallet's history: 1000 BREW paid for with 250 USDT.
    usdt = list(config.STABLECOINS["bsc"])[0]
    monkeypatch.setattr(
        cost_basis, "fetch_evm_ledger",
        lambda chain, wallets: cost_basis.Ledger(transfers=[
            {"hash": "0xbuy", "timestamp": 1_756_000_000, "from": EVM_WALLET.lower(),
             "to": "0xdex", "token": usdt, "symbol": "USDT", "decimals": 18,
             "quantity": 250.0, "native": False},
            {"hash": "0xbuy", "timestamp": 1_756_000_000, "from": "0xdex",
             "to": EVM_WALLET.lower(), "token": BREW, "symbol": "BREW", "decimals": 18,
             "quantity": 1000.0, "native": False},
        ]),
    )
    rotation.clear_cache()

    instance = AppTest.from_file(APP, default_timeout=60)
    instance.run()
    return instance


def test_app_renders_every_tab_without_error(app):
    assert not app.exception
    labels = [tab.label for tab in app.tabs]
    assert labels[:2] == ["💼 Portfolio", "🔄 Rotation"]
    assert len(labels) == 6


def test_portfolio_sync_shows_a_priced_position(app):
    app.text_area(key="wallets_bsc").set_value(f"{EVM_WALLET}, main").run()
    app.button[0].click().run()                   # Save wallets
    assert not app.exception
    assert portfolio_store.list_wallets(db_path=config.PORTFOLIO_DB_PATH)

    sync = next(b for b in app.button if "Sync balances" in b.label)
    sync.click().run()
    assert not app.exception

    # $2 x 1000 tokens, rendered in the headline metric.
    values = [metric.value for metric in app.metric]
    assert any("2,000" in str(value) for value in values)
    # And the sync is persisted, which is what the equity curve is built from.
    assert portfolio_store.recent_snapshots(db_path=config.PORTFOLIO_DB_PATH)


def test_rotation_refresh_produces_a_ranked_plan(app):
    app.text_area(key="wallets_bsc").set_value(EVM_WALLET).run()
    app.button[0].click().run()
    next(b for b in app.button if "Sync balances" in b.label).click().run()
    next(b for b in app.button if "Refresh heat" in b.label).click().run()

    assert not app.exception
    plan = app.session_state["rotation_plan"]
    # BSC is the chain that ripped in the stubbed basket, so it must rank first.
    assert plan.heats[0].chain == "bsc"
    assert plan.heats[0].heat > plan.heats[-1].heat
    assert portfolio_store.heat_history("bsc", db_path=config.PORTFOLIO_DB_PATH)


def test_rotation_works_before_any_wallet_is_synced(app):
    # The chain-wide half needs no wallet at all; the tab must not require one.
    next(b for b in app.button if "Refresh heat" in b.label).click().run()
    assert not app.exception
    plan = app.session_state["rotation_plan"]
    assert plan.heats
    assert plan.actions == []


def test_sync_derives_cost_basis_and_shows_pnl(app):
    app.text_area(key="wallets_bsc").set_value(EVM_WALLET).run()
    app.button[0].click().run()
    next(b for b in app.button if "Sync balances" in b.label).click().run()
    assert not app.exception

    position = app.session_state["portfolio"].positions[0]
    # Bought 1000 for $250, now worth $2 each: $0.25 basis, $1,750 unrealized.
    assert position.avg_cost_usd == pytest.approx(0.25)
    assert position.basis_source == "derived"
    assert position.unrealized_pnl_usd == pytest.approx(1_750.0)
    assert "derived" in position.basis_label

    # And it survives a round trip through the store.
    meta = portfolio_store.all_position_meta(db_path=config.PORTFOLIO_DB_PATH)[position.key]
    assert meta["basis_source"] == "derived"


def test_a_typed_cost_basis_is_not_overwritten_by_a_later_sync(app):
    app.text_area(key="wallets_bsc").set_value(EVM_WALLET).run()
    app.button[0].click().run()
    next(b for b in app.button if "Sync balances" in b.label).click().run()

    portfolio_store.set_position_meta(
        "bsc", BREW, avg_cost_usd=1.5, basis_source="manual",
        db_path=config.PORTFOLIO_DB_PATH)
    next(b for b in app.button if "Sync balances" in b.label).click().run()

    assert not app.exception
    position = app.session_state["portfolio"].positions[0]
    assert position.avg_cost_usd == pytest.approx(1.5)
    assert position.basis_source == "manual"
