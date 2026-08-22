"""End-to-end tests: analyzer, scanner, history and report export."""

from __future__ import annotations

import json

import pytest

from src import analyzer, config, data_fetchers, history, llm_analyzers, report
from src.data_fetchers import FetchError
from tests.fake_llm import FakeAnthropicClient, FakeOpenAIClient, verdict_json
from tests.fixtures import (
    NOW_MS,
    TOKEN_ADDRESS,
    dexscreener_pair,
    dexscreener_token_response,
    goplus_honeypot_response,
    goplus_response,
    token_profiles_response,
)


@pytest.fixture(autouse=True)
def clear_caches():
    data_fetchers.clear_caches()
    yield
    data_fetchers.clear_caches()


@pytest.fixture
def stub_apis(monkeypatch):
    """Route both upstreams through one fake transport."""

    def route(url, params=None, retries=None):
        if "token-profiles" in url:
            return token_profiles_response()
        if "dexscreener" in url or "/latest/dex/" in url or "/tokens/v1/" in url:
            return dexscreener_token_response()
        return goplus_response()

    monkeypatch.setattr(data_fetchers, "_get_json", route)
    return route


class TestAnalyzeToken:
    def test_full_pipeline_produces_a_complete_report(self, stub_apis):
        result = analyzer.analyze_token(TOKEN_ADDRESS, config.AppSettings(chain="base", portfolio_usd=25_000))

        assert result.ok is True
        assert result.snapshot.symbol == "BRETT"
        assert result.security.available is True
        assert result.scorecard is not None and 0 <= result.composite <= 100
        assert result.decision in ("Strong Buy", "Buy", "Watch", "Pass")
        assert result.risk_plan.position_usd > 0
        assert result.narrative.summary
        assert result.narrative.source == "heuristic"

    def test_holder_count_flows_from_security_into_snapshot(self, stub_apis):
        result = analyzer.analyze_token(TOKEN_ADDRESS)
        assert result.snapshot.holders == 18432

    def test_dead_market_api_returns_failed_result_not_exception(self, monkeypatch):
        def boom(*args, **kwargs):
            raise FetchError("HTTP 500")

        monkeypatch.setattr(data_fetchers, "_get_json", boom)
        result = analyzer.analyze_token(TOKEN_ADDRESS)
        assert result.ok is False
        assert "DexScreener unavailable" in result.error

    def test_security_outage_still_produces_a_score(self, monkeypatch):
        def route(url, params=None, retries=None):
            if "gopluslabs" in url:
                raise FetchError("timeout")
            return dexscreener_token_response()

        monkeypatch.setattr(data_fetchers, "_get_json", route)
        result = analyzer.analyze_token(TOKEN_ADDRESS)

        assert result.ok is True
        assert result.security.available is False
        assert result.scorecard is not None
        assert any("GoPlus" in w for w in result.data_warnings)
        # Unknown security must drag the score down, not be treated as clean.
        assert result.scorecard.component("security").score < 50

    def test_honeypot_ends_in_pass_with_zero_size(self, monkeypatch):
        def route(url, params=None, retries=None):
            if "gopluslabs" in url:
                return goplus_honeypot_response()
            return dexscreener_token_response()

        monkeypatch.setattr(data_fetchers, "_get_json", route)
        result = analyzer.analyze_token(TOKEN_ADDRESS)

        assert result.decision == "Pass"
        assert result.scorecard.vetoed is True
        assert result.risk_plan.position_usd == 0.0

    def test_analyze_many_handles_a_batch(self, stub_apis):
        addresses = [TOKEN_ADDRESS, "0x4ed4E862860beD51a9570b96d89aF5E1B0Efefed"]
        results = analyzer.analyze_many(addresses, config.AppSettings())
        assert len(results) == 2
        assert all(r.ok for r in results)

    def test_analyze_many_deduplicates_case_insensitively(self, stub_apis):
        results = analyzer.analyze_many([TOKEN_ADDRESS, TOKEN_ADDRESS.lower()], config.AppSettings())
        assert len(results) == 1
        assert analyzer.analyze_many([], config.AppSettings()) == []


class TestScanner:
    def test_filters_by_market_cap_liquidity_and_volume(self):
        from src.data_fetchers import snapshot_from_pair

        pair = dexscreener_pair()
        snapshot = snapshot_from_pair(pair, [pair], address=TOKEN_ADDRESS)
        filters = config.ScannerFilters()

        assert analyzer.passes_filters(snapshot, filters) is True

        snapshot.market_cap = 50_000_000
        assert analyzer.passes_filters(snapshot, filters) is False

        snapshot.market_cap = 3_000_000
        snapshot.liquidity_usd = 1_000
        assert analyzer.passes_filters(snapshot, filters) is False

        snapshot.liquidity_usd = 400_000
        snapshot.volume_24h = 500
        assert analyzer.passes_filters(snapshot, filters) is False

    def test_age_bounds_are_enforced(self):
        from src.data_fetchers import snapshot_from_pair

        pair = dexscreener_pair(pairCreatedAt=NOW_MS - 60 * 60 * 1000)  # 1h old
        snapshot = snapshot_from_pair(pair, [pair], address=TOKEN_ADDRESS)
        assert analyzer.passes_filters(snapshot, config.ScannerFilters(min_age_hours=6)) is False
        assert analyzer.passes_filters(snapshot, config.ScannerFilters(min_age_hours=0.5)) is True

    def test_scan_ranks_candidates_by_score(self, monkeypatch):
        hot = dexscreener_pair(
            pairAddress="0xhot",
            baseToken={"address": "0xA000000000000000000000000000000000000001", "name": "Hot", "symbol": "HOT"},
            volume={"h24": 2_500_000.0, "h6": 900_000.0, "h1": 200_000.0},
            priceChange={"m5": 1, "h1": 8.0, "h6": 25.0, "h24": 60.0},
        )
        cold = dexscreener_pair(
            pairAddress="0xcold",
            baseToken={"address": "0xB000000000000000000000000000000000000002", "name": "Cold", "symbol": "COLD"},
            volume={"h24": 120_000.0, "h6": 8_000.0, "h1": 900.0},
            priceChange={"m5": 0, "h1": -2.0, "h6": -9.0, "h24": -22.0},
        )
        monkeypatch.setattr(data_fetchers, "discover_pairs", lambda chain, use_cache=True: ([hot, cold], []))

        candidates, warnings = analyzer.scan(config.ScannerFilters(chain="base"))
        assert [c.snapshot.symbol for c in candidates] == ["HOT", "COLD"]
        assert candidates[0].quick_score > candidates[1].quick_score
        assert any("passed your filters" in w for w in warnings)

    def test_scan_respects_max_results(self, monkeypatch):
        pairs = [
            dexscreener_pair(
                pairAddress=f"0x{index:040x}",
                baseToken={"address": f"0x{index + 1000:040x}", "name": f"T{index}", "symbol": f"T{index}"},
            )
            for index in range(10)
        ]
        monkeypatch.setattr(data_fetchers, "discover_pairs", lambda chain, use_cache=True: (pairs, []))
        candidates, _ = analyzer.scan(config.ScannerFilters(max_results=3))
        assert len(candidates) == 3

    def test_scan_with_no_pairs_explains_itself(self, monkeypatch):
        monkeypatch.setattr(data_fetchers, "discover_pairs", lambda chain, use_cache=True: ([], ["upstream down"]))
        candidates, warnings = analyzer.scan()
        assert candidates == []
        assert warnings == ["upstream down"]


class TestReportExport:
    def test_markdown_contains_the_key_sections(self, stub_apis):
        result = analyzer.analyze_token(TOKEN_ADDRESS)
        markdown = report.to_markdown(result)

        for heading in ("# BRETT", "## Market", "## Score breakdown", "## Security",
                        "## Narrative", "## Risk plan"):
            assert heading in markdown, heading
        assert "not financial advice" in markdown

    def test_json_export_round_trips(self, stub_apis):
        result = analyzer.analyze_token(TOKEN_ADDRESS)
        payload = json.loads(report.to_json(result))

        assert payload["snapshot"]["symbol"] == "BRETT"
        assert payload["scorecard"]["composite"] == result.composite
        assert len(payload["scorecard"]["components"]) == 6
        assert payload["risk_plan"]["stop_loss_pct"] > 0

    def test_failed_analysis_still_exports(self, monkeypatch):
        def boom(*args, **kwargs):
            raise FetchError("nope")

        monkeypatch.setattr(data_fetchers, "_get_json", boom)
        result = analyzer.analyze_token(TOKEN_ADDRESS)
        assert "Analysis failed" in report.to_markdown(result)
        assert json.loads(report.to_json(result))["ok"] is False

    def test_filename_is_filesystem_safe(self, stub_apis):
        result = analyzer.analyze_token(TOKEN_ADDRESS)
        assert report.filename_for(result, "md") == "memedd_BRETT_base.md"


class TestHistory:
    def test_records_and_reads_back(self, stub_apis, tmp_path):
        db = tmp_path / "history.sqlite3"
        result = analyzer.analyze_token(TOKEN_ADDRESS)

        row_id = history.record(result, db_path=db)
        assert row_id is not None

        rows = history.recent(db_path=db)
        assert len(rows) == 1
        assert rows[0]["symbol"] == "BRETT"
        assert rows[0]["composite"] == result.composite

        payload = history.load(row_id, db_path=db)
        assert payload["snapshot"]["symbol"] == "BRETT"

    def test_failed_analyses_are_not_recorded(self, tmp_path, monkeypatch):
        def boom(*args, **kwargs):
            raise FetchError("nope")

        monkeypatch.setattr(data_fetchers, "_get_json", boom)
        db = tmp_path / "history.sqlite3"
        assert history.record(analyzer.analyze_token(TOKEN_ADDRESS), db_path=db) is None

    def test_clear_empties_the_table(self, stub_apis, tmp_path):
        db = tmp_path / "history.sqlite3"
        history.record(analyzer.analyze_token(TOKEN_ADDRESS), db_path=db)
        history.clear(db_path=db)
        assert history.recent(db_path=db) == []


class TestProfileEnrichmentInPipeline:
    def test_profile_is_attached_and_folded_into_the_snapshot(self, stub_apis):
        result = analyzer.analyze_token(TOKEN_ADDRESS)

        assert result.profile is not None
        assert result.profile.description.startswith("Brett is Pepe's")
        # The fixture pair already has socials; the profile must not duplicate them.
        twitter = [link for link in result.snapshot.socials if link.kind == "twitter"]
        assert len(twitter) == 1

    def test_profile_outage_does_not_break_analysis(self, monkeypatch):
        def route(url, params=None, retries=None):
            if "token-profiles" in url:
                raise FetchError("HTTP 500")
            if "gopluslabs" in url:
                return goplus_response()
            return dexscreener_token_response()

        monkeypatch.setattr(data_fetchers, "_get_json", route)
        result = analyzer.analyze_token(TOKEN_ADDRESS)

        assert result.ok is True
        assert result.profile is None
        assert result.scorecard is not None


class TestEnsembleInPipeline:
    def _settings(self, **kwargs):
        return config.AppSettings(use_ensemble=True, **kwargs)

    def test_ensemble_is_skipped_unless_enabled(self, stub_apis):
        assert analyzer.analyze_token(TOKEN_ADDRESS, config.AppSettings()).ensemble is None

    def test_ensemble_runs_and_attaches_to_the_result(self, stub_apis, monkeypatch):
        clients = {
            "xai": FakeOpenAIClient(content=verdict_json(score=78, decision="buy")),
            "anthropic": FakeAnthropicClient(content=verdict_json(score=72, decision="buy")),
            "openai": FakeOpenAIClient(content=verdict_json(score=66, decision="watch")),
        }
        real_run = llm_analyzers.run_ensemble
        monkeypatch.setattr(
            analyzer.llm_analyzers, "run_ensemble",
            lambda *args, **kwargs: real_run(*args, clients=clients, **kwargs),
        )
        result = analyzer.analyze_token(TOKEN_ADDRESS, self._settings())

        assert result.ensemble is not None and result.ensemble.ok
        assert result.ensemble.consensus.model_count == 3
        assert result.ensemble.blended_score is not None
        # The deterministic scorecard must remain untouched by the models.
        assert result.scorecard.composite == analyzer.analyze_token(
            TOKEN_ADDRESS, config.AppSettings()
        ).scorecard.composite

    def test_models_receive_the_deterministic_score_to_argue_with(self, stub_apis, monkeypatch):
        client = FakeOpenAIClient(content=verdict_json())
        real_run = llm_analyzers.run_ensemble
        monkeypatch.setattr(
            analyzer.llm_analyzers, "run_ensemble",
            lambda *args, **kwargs: real_run(*args, clients={"xai": client}, **kwargs),
        )
        analyzer.analyze_token(TOKEN_ADDRESS, self._settings(ensemble_providers=("xai",)))

        prompt = client.calls[0]["messages"][1]["content"]
        assert "platform_deterministic_score" in prompt
        assert "dexscreener_profile" in prompt

    def test_no_keys_produces_an_explained_empty_ensemble(self, stub_apis, monkeypatch):
        for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "XAI_API_KEY"):
            monkeypatch.setattr(config, name, "")
        result = analyzer.analyze_token(TOKEN_ADDRESS, self._settings())

        assert result.ok is True                      # analysis still succeeds
        assert result.ensemble is not None and result.ensemble.ok is False
        assert any("No LLM providers" in note for note in result.ensemble.notes)


class TestEnsembleExport:
    @pytest.fixture
    def result_with_ensemble(self, stub_apis, monkeypatch):
        clients = {
            "xai": FakeOpenAIClient(content=verdict_json(score=80, decision="buy",
                                                         rug_flags=["Deployer holds a large bag"])),
            "anthropic": FakeAnthropicClient(content=verdict_json(score=55, decision="watch",
                                                                  rug_flags=["Deployer holds a large bag"])),
        }
        real_run = llm_analyzers.run_ensemble
        monkeypatch.setattr(
            analyzer.llm_analyzers, "run_ensemble",
            lambda *args, **kwargs: real_run(*args, clients=clients, **kwargs),
        )
        return analyzer.analyze_token(
            TOKEN_ADDRESS, config.AppSettings(use_ensemble=True, ensemble_providers=("xai", "anthropic")),
        )

    def test_markdown_includes_the_ensemble_section(self, result_with_ensemble):
        markdown = report.to_markdown(result_with_ensemble)

        assert "## Multi-LLM ensemble" in markdown
        assert "## Project profile (DexScreener)" in markdown
        assert "xai/grok-4" in markdown
        assert "Rug flags raised by 2+ models" in markdown
        assert "Disagreement between models" in markdown

    def test_json_round_trips_the_ensemble(self, result_with_ensemble):
        payload = json.loads(report.to_json(result_with_ensemble))

        assert payload["ensemble"]["model_count"] == 2
        assert len(payload["ensemble"]["verdicts"]) == 2
        assert payload["ensemble"]["consensus"]["corroborated_rug_flags"]
        assert payload["profile"]["description"]

    def test_history_stores_a_result_carrying_an_ensemble(self, result_with_ensemble, tmp_path):
        db = tmp_path / "history.sqlite3"
        row_id = history.record(result_with_ensemble, db_path=db)
        payload = history.load(row_id, db_path=db)
        assert payload["ensemble"]["consensus"]["overall_score"] > 0
