"""Tests for the pieces that only matter once the app is hosted."""

from __future__ import annotations

import json
import os

import pytest

from src import config, portfolio_store
from src.models import ChainHeat, PortfolioSnapshot, Position

WALLET = "0x532f27101965dd16442E59d40670FaF5eBB142E4"
BREW = "0x1234567890abcdef1234567890abcdef12345678"


@pytest.fixture
def db(tmp_path):
    return tmp_path / "hosting.sqlite3"


def seeded(db):
    portfolio_store.add_wallet(WALLET, "bsc", "main", db_path=db)
    portfolio_store.set_position_meta(
        "bsc", BREW, avg_cost_usd=0.25, tag="Brew", basis_source="manual", db_path=db)
    portfolio_store.record_snapshot(
        PortfolioSnapshot(
            positions=[Position(address=BREW, chain="bsc", quantity=100, value_usd=500.0)],
            taken_at="2026-09-12T00:00:00+00:00"),
        db_path=db)
    portfolio_store.record_heat(
        ChainHeat(chain="bsc", heat=84.0, state="hot", taken_at="2026-09-12T00:00:00+00:00"),
        db_path=db)
    return db


class TestSecretsBridge:
    """Streamlit Cloud injects st.secrets; there is no .env on a host."""

    def test_secrets_become_environment_variables(self, monkeypatch):
        monkeypatch.delenv("ETHERSCAN_API_KEY", raising=False)

        class FakeStreamlit:
            secrets = {"ETHERSCAN_API_KEY": "from-secrets", "MEMEDD_HEAT_HOT": 70}

        monkeypatch.setitem(__import__("sys").modules, "streamlit", FakeStreamlit)
        config._load_streamlit_secrets()
        assert os.environ["ETHERSCAN_API_KEY"] == "from-secrets"
        assert os.environ["MEMEDD_HEAT_HOT"] == "70"       # coerced to a string

    def test_a_real_environment_variable_wins(self, monkeypatch):
        monkeypatch.setenv("ETHERSCAN_API_KEY", "from-env")

        class FakeStreamlit:
            secrets = {"ETHERSCAN_API_KEY": "from-secrets"}

        monkeypatch.setitem(__import__("sys").modules, "streamlit", FakeStreamlit)
        config._load_streamlit_secrets()
        # A local .env must never be overridden by a stale hosted secret.
        assert os.environ["ETHERSCAN_API_KEY"] == "from-env"

    def test_nested_secrets_are_skipped(self, monkeypatch):
        class FakeStreamlit:
            secrets = {"connections": {"db": "postgres://"}, "FLAT": "kept"}

        monkeypatch.delenv("FLAT", raising=False)
        monkeypatch.setitem(__import__("sys").modules, "streamlit", FakeStreamlit)
        config._load_streamlit_secrets()
        assert os.environ["FLAT"] == "kept"
        assert "connections" not in os.environ

    def test_no_secrets_at_all_is_not_an_error(self, monkeypatch):
        class FakeStreamlit:
            @property
            def secrets(self):
                raise FileNotFoundError("no secrets.toml")

        monkeypatch.setitem(__import__("sys").modules, "streamlit", FakeStreamlit())
        config._load_streamlit_secrets()      # must not raise

    def test_no_streamlit_installed_is_not_an_error(self, monkeypatch):
        # The app must still import and run as a plain Python package when
        # Streamlit is not importable at all.
        import builtins

        real_import = builtins.__import__

        def blocked(name, *args, **kwargs):
            if name == "streamlit":
                raise ImportError("no streamlit here")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocked)
        config._load_streamlit_secrets()       # must not raise


class TestPasswordGate:
    def test_no_password_configured_means_no_gate(self, monkeypatch):
        monkeypatch.setattr(config, "APP_PASSWORD", "")
        import app

        # require_access returns immediately rather than calling st.stop().
        app.require_access()

    def test_a_configured_password_is_compared_safely(self, monkeypatch):
        import hmac

        monkeypatch.setattr(config, "APP_PASSWORD", "correct horse")
        assert hmac.compare_digest("correct horse", config.APP_PASSWORD)
        assert not hmac.compare_digest("correct", config.APP_PASSWORD)


class TestBackupRestore:
    def test_round_trip_preserves_everything(self, db, tmp_path):
        seeded(db)
        blob = json.dumps(portfolio_store.export_state(db_path=db))

        restored_db = tmp_path / "restored.sqlite3"
        counts = portfolio_store.import_state(json.loads(blob), db_path=restored_db)
        assert counts == {"wallets": 1, "position_meta": 1, "snapshots": 1, "chain_heat": 1}

        wallets = portfolio_store.list_wallets(db_path=restored_db)
        assert wallets[0].label == "main"
        meta = portfolio_store.all_position_meta(db_path=restored_db)[f"bsc:{BREW}"]
        assert meta["avg_cost_usd"] == pytest.approx(0.25)
        assert meta["tag"] == "Brew"
        # The pinned basis stays pinned across a restore.
        assert meta["basis_source"] == "manual"
        assert len(portfolio_store.heat_history("bsc", db_path=restored_db)) == 1
        assert len(portfolio_store.equity_curve(db_path=restored_db)) == 1

    def test_restoring_merges_rather_than_deleting(self, db, tmp_path):
        seeded(db)
        state = portfolio_store.export_state(db_path=db)

        target = tmp_path / "target.sqlite3"
        portfolio_store.add_wallet("0xAAA0000000000000000000000000000000000001",
                                   "base", "added later", db_path=target)
        portfolio_store.import_state(state, db_path=target)
        # A wallet added after the backup was taken must survive the restore.
        assert len(portfolio_store.list_wallets(db_path=target)) == 2

    def test_replace_wipes_first_when_asked(self, db, tmp_path):
        seeded(db)
        state = portfolio_store.export_state(db_path=db)

        target = tmp_path / "target.sqlite3"
        portfolio_store.add_wallet("0xAAA0000000000000000000000000000000000001",
                                   "base", db_path=target)
        portfolio_store.import_state(state, replace=True, db_path=target)
        wallets = portfolio_store.list_wallets(db_path=target)
        assert len(wallets) == 1 and wallets[0].chain == "bsc"

    def test_a_newer_backup_version_is_refused(self, db):
        counts = portfolio_store.import_state(
            {"version": portfolio_store.BACKUP_VERSION + 1, "wallets": [
                {"address": WALLET, "chain": "bsc"}]},
            db_path=db)
        assert counts["wallets"] == 0
        assert portfolio_store.list_wallets(db_path=db) == []

    def test_junk_input_is_survived(self, db):
        assert portfolio_store.import_state({}, db_path=db)["wallets"] == 0
        assert portfolio_store.import_state([], db_path=db)["wallets"] == 0
        assert portfolio_store.import_state(
            {"version": 1, "wallets": [{"label": "no address"}]}, db_path=db)["wallets"] == 0

    def test_is_empty_reports_a_fresh_host(self, db):
        assert portfolio_store.is_empty(db_path=db)
        portfolio_store.add_wallet(WALLET, "bsc", db_path=db)
        assert not portfolio_store.is_empty(db_path=db)

    def test_export_of_an_empty_store_is_still_valid(self, db):
        state = portfolio_store.export_state(db_path=db)
        assert state["version"] == portfolio_store.BACKUP_VERSION
        assert state["wallets"] == []
        json.dumps(state)      # must be serialisable


class TestGateInTheRealApp:
    """Drives the actual app, because a gate that half-renders is worthless."""

    APP = str(__import__("pathlib").Path(__file__).resolve().parent.parent / "app.py")

    @pytest.fixture
    def app_test(self, monkeypatch, tmp_path):
        streamlit_testing = pytest.importorskip("streamlit.testing.v1")
        monkeypatch.setattr(config, "PORTFOLIO_DB_PATH", tmp_path / "gate.sqlite3")
        monkeypatch.setattr(config, "HISTORY_DB_PATH", tmp_path / "history.sqlite3")
        from src import data_fetchers

        monkeypatch.setattr(data_fetchers, "fetch_latest_profiles",
                            lambda chain, use_cache=True: [])
        return streamlit_testing.AppTest

    def test_a_locked_app_renders_no_dashboard_at_all(self, app_test, monkeypatch):
        monkeypatch.setattr(config, "APP_PASSWORD", "hunter2-long-enough")
        at = app_test.from_file(self.APP, default_timeout=60).run()
        assert not at.exception
        # Not one tab, not one number: st.stop() runs before any of it.
        assert len(at.tabs) == 0
        assert len(at.text_input) == 1

    def test_the_wrong_password_keeps_it_locked(self, app_test, monkeypatch):
        monkeypatch.setattr(config, "APP_PASSWORD", "hunter2-long-enough")
        at = app_test.from_file(self.APP, default_timeout=60).run()
        at.text_input[0].set_value("nope").run()
        at.button[0].click().run()
        assert len(at.tabs) == 0
        assert at.error

    def test_the_right_password_opens_it(self, app_test, monkeypatch):
        monkeypatch.setattr(config, "APP_PASSWORD", "hunter2-long-enough")
        at = app_test.from_file(self.APP, default_timeout=60).run()
        at.text_input[0].set_value("hunter2-long-enough").run()
        at.button[0].click().run()
        assert not at.exception
        assert len(at.tabs) == 6

    def test_no_password_configured_means_no_gate(self, app_test, monkeypatch):
        monkeypatch.setattr(config, "APP_PASSWORD", "")
        at = app_test.from_file(self.APP, default_timeout=60).run()
        assert not at.exception
        assert len(at.tabs) == 6
