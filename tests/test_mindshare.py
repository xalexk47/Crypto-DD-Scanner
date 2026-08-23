"""Tests for Grok-powered X/Twitter mindshare."""

from __future__ import annotations

import json

import pytest

from src import config, mindshare as ms, scorers
from src.data_fetchers import snapshot_from_pair
from src.models import MindshareReport, TokenSnapshot
from tests.fake_llm import FakeResponsesClient
from tests.fixtures import TOKEN_ADDRESS, dexscreener_pair


def payload(**overrides):
    data = {
        "sentiment": "bullish", "sentiment_score": 0.6, "mindshare_score": 72,
        "post_volume": "high", "trend": "accelerating", "is_organic": True,
        "summary": "Steady organic chatter from mid-size Base accounts.",
        "themes": ["base season", "frog meta"],
        "notable_accounts": ["@basedtrader", "cryptowhale"],
        "sample_posts": [
            {"handle": "@basedtrader", "text": "BRETT still the cleanest Base meme",
             "url": "https://x.com/basedtrader/status/1", "engagement": 420},
        ],
        "red_flags": [],
    }
    data.update(overrides)
    return data


def payload_json(**overrides) -> str:
    return json.dumps(payload(**overrides))


@pytest.fixture
def snapshot():
    pair = dexscreener_pair()
    return snapshot_from_pair(pair, [pair], address=TOKEN_ADDRESS)


@pytest.fixture(autouse=True)
def clear_mindshare_cache():
    ms.clear_cache()
    yield
    ms.clear_cache()


class TestQueryBuilding:
    def test_query_includes_ticker_and_contract(self, snapshot):
        query = ms.build_query(snapshot)
        assert "$BRETT" in query
        # The contract address is the disambiguator: every chain has three PEPEs.
        assert TOKEN_ADDRESS in query

    def test_distinct_project_name_is_included(self):
        token = TokenSnapshot(address="0xabc", chain="base", name="Based Brett", symbol="BRETT")
        assert "Based Brett" in ms.build_query(token)

    def test_name_matching_the_ticker_is_not_repeated(self, snapshot):
        """name "Brett" and symbol "BRETT" are one term, not two."""
        assert ms.build_query(snapshot).lower().count("brett") == 1

    def test_prompt_warns_about_ticker_collisions(self, snapshot):
        prompt = ms.build_prompt(snapshot, 48, 6)
        assert "collision" in prompt.lower()
        assert TOKEN_ADDRESS in prompt
        assert "48 hours" in prompt

    def test_query_falls_back_to_the_address(self):
        bare = TokenSnapshot(address="0xabc", chain="base")
        assert ms.build_query(bare) == "0xabc"


class TestParsing:
    def test_parses_a_full_response(self):
        report = ms.parse_mindshare(payload(), "q", "grok-4.1-fast", is_live=True)

        assert report.available and report.is_live
        assert report.source == "x_search"
        assert report.sentiment == "bullish"
        assert report.mindshare_score == 72
        assert report.post_volume == "high"
        assert report.sample_posts[0].handle == "basedtrader"   # @ stripped
        assert report.sample_posts[0].engagement == 420
        assert report.notable_accounts == ["basedtrader", "cryptowhale"]

    def test_out_of_range_values_are_clamped(self):
        report = ms.parse_mindshare(
            payload(sentiment_score=9, mindshare_score=900), "q", "m", is_live=True
        )
        assert report.sentiment_score == 1.0
        assert report.mindshare_score == 100.0

    def test_unknown_enum_values_degrade_to_unknown(self):
        report = ms.parse_mindshare(
            payload(sentiment="euphoric", post_volume="lots", trend="sideways"), "q", "m", is_live=True
        )
        assert report.sentiment == "unknown"
        assert report.post_volume == "unknown"
        assert report.trend == "unknown"

    def test_non_live_answers_carry_a_loud_warning(self):
        report = ms.parse_mindshare(payload(), "q", "m", is_live=False)

        assert report.is_live is False
        assert report.source == "model_knowledge"
        assert any("not current activity" in w for w in report.warnings)

    def test_malformed_posts_are_skipped(self):
        report = ms.parse_mindshare(
            payload(sample_posts=["nonsense", {}, {"handle": "ok", "text": "real post"}]),
            "q", "m", is_live=True,
        )
        assert len(report.sample_posts) == 1
        assert report.sample_posts[0].text == "real post"


class TestSocialScoring:
    def _report(self, **kwargs):
        base = dict(available=True, is_live=True, sentiment="bullish", sentiment_score=0.5,
                    mindshare_score=70, post_volume="high", trend="steady", is_organic=True)
        base.update(kwargs)
        return MindshareReport(**base)

    def test_no_report_scores_nothing(self):
        assert ms.score_from_report(None) is None
        assert ms.score_from_report(MindshareReport(available=False)) is None

    def test_viral_organic_beats_quiet(self):
        viral = ms.score_from_report(self._report(post_volume="viral", mindshare_score=95))
        quiet = ms.score_from_report(self._report(post_volume="none", mindshare_score=5,
                                                  sentiment="quiet", sentiment_score=0.0))
        assert viral > quiet

    def test_coordinated_shilling_is_halved_not_rewarded(self):
        organic = ms.score_from_report(self._report(is_organic=True))
        bots = ms.score_from_report(self._report(is_organic=False))
        assert bots < organic / 1.8

    def test_red_flags_reduce_the_score(self):
        clean = ms.score_from_report(self._report())
        flagged = ms.score_from_report(self._report(red_flags=["Impersonating a known project"]))
        assert flagged < clean

    def test_trend_shifts_the_score(self):
        rising = ms.score_from_report(self._report(trend="accelerating"))
        fading = ms.score_from_report(self._report(trend="fading"))
        assert rising > fading

    def test_stale_data_is_pulled_toward_neutral(self):
        live = ms.score_from_report(self._report(is_live=True, mindshare_score=95, post_volume="viral"))
        stale = ms.score_from_report(self._report(is_live=False, mindshare_score=95, post_volume="viral"))
        assert abs(stale - 50) < abs(live - 50)


class TestMomentumIntegration:
    def _card(self, snapshot, mindshare=None):
        return scorers.score_momentum(snapshot, mindshare)

    def test_live_bullish_mindshare_raises_momentum(self, snapshot):
        hot = MindshareReport(available=True, is_live=True, sentiment="bullish", sentiment_score=0.8,
                              mindshare_score=92, post_volume="viral", trend="accelerating", is_organic=True)
        assert self._card(snapshot, hot).score > self._card(snapshot).score

    def test_bot_driven_hype_lowers_momentum(self, snapshot):
        """High volume plus bullish tone must not help when it is coordinated."""
        bots = MindshareReport(available=True, is_live=True, sentiment="bullish", sentiment_score=0.8,
                               mindshare_score=90, post_volume="high", trend="steady",
                               is_organic=False, red_flags=["Reply-spam from new accounts"])
        assert self._card(snapshot, bots).score < self._card(snapshot).score

    def test_silence_lowers_momentum(self, snapshot):
        quiet = MindshareReport(available=True, is_live=True, sentiment="quiet", sentiment_score=0.0,
                                mindshare_score=3, post_volume="none", trend="fading", is_organic=True)
        assert self._card(snapshot, quiet).score < self._card(snapshot).score

    def test_live_data_raises_confidence_and_stale_lowers_it(self, snapshot):
        live = MindshareReport(available=True, is_live=True, mindshare_score=60, post_volume="moderate")
        stale = MindshareReport(available=True, is_live=False, mindshare_score=60, post_volume="moderate")

        assert self._card(snapshot, live).confidence > self._card(snapshot).confidence
        assert self._card(snapshot, stale).confidence < self._card(snapshot).confidence

    def test_failure_is_explained_in_the_reasons(self, snapshot):
        failed = MindshareReport(available=False, error="No XAI_API_KEY set")
        reasons = " ".join(self._card(snapshot, failed).reasons)
        assert "X mindshare unavailable" in reasons

    def test_scorecard_threads_mindshare_through(self, snapshot):
        hot = MindshareReport(available=True, is_live=True, sentiment="bullish", sentiment_score=0.9,
                              mindshare_score=95, post_volume="viral", trend="accelerating", is_organic=True)
        card = scorers.build_scorecard(snapshot, None, mindshare=hot)
        assert any("Live X attention" in p for p in card.positives)

    def test_coordinated_shilling_surfaces_as_a_risk(self, snapshot):
        bots = MindshareReport(available=True, is_live=True, sentiment="bullish", mindshare_score=80,
                               post_volume="high", is_organic=False, red_flags=["Paid callers"])
        card = scorers.build_scorecard(snapshot, None, mindshare=bots)
        assert any("coordinated" in r for r in card.risks)
        assert any("Paid callers" in r for r in card.risks)


class TestGrokClient:
    def test_live_search_goes_through_the_responses_endpoint(self, snapshot):
        """x_search lives on /v1/responses, not chat.completions.

        Confirmed against the live API: chat.completions rejects x_search
        outright, and its own live_search alternative returns 410 deprecated.
        """
        client = FakeResponsesClient(content=payload_json())
        report = ms.GrokMindshareClient(api_key="k", client=client).fetch(snapshot)

        assert report.available and report.is_live
        assert len(client.responses_calls) == 1
        sent = client.responses_calls[0]
        assert sent["model"] == config.X_SEARCH_MODEL
        assert sent["tools"][0]["type"] == "x_search"
        assert "input" in sent            # Responses API uses input, not messages

    def test_tool_type_is_x_search(self):
        assert config.X_SEARCH_TOOL_TYPE == "x_search"
        for variant in ms.GrokMindshareClient(api_key="k").search_tool_variants(48):
            assert variant["type"] == "x_search"

    def test_variants_run_richest_to_minimal(self):
        variants = ms.GrokMindshareClient(api_key="k").search_tool_variants(48)

        assert len(variants) >= 2
        # The documented minimal form must be last, as the always-valid option.
        assert variants[-1] == {"type": "x_search"}
        first = json.dumps(variants[0])
        assert "from_date" in first and "to_date" in first

    def test_an_sdk_without_the_responses_endpoint_degrades(self, snapshot):
        """An older openai SDK must fall back, not crash."""
        client = FakeResponsesClient(content=payload_json(), no_responses_endpoint=True)
        report = ms.GrokMindshareClient(api_key="k", client=client).fetch(snapshot)

        assert report.available is True
        assert report.is_live is False
        assert report.source == "model_knowledge"

    def test_falls_back_to_non_live_and_labels_it(self, snapshot):
        """When every tool shape is rejected, the answer must not claim to be live."""
        variants = ms.GrokMindshareClient(api_key="k").search_tool_variants(48)
        tool_attempts = len(variants) * 2      # each shape, with and without the schema

        client = FakeResponsesClient(content=payload_json(), fail_modes=tool_attempts)
        report = ms.GrokMindshareClient(api_key="k", client=client).fetch(snapshot)

        assert report.available is True
        assert report.is_live is False               # must not claim live data
        assert report.source == "model_knowledge"
        # Live attempts all went to /v1/responses; the fallback did not.
        assert len(client.responses_calls) == tool_attempts
        assert "tools" not in client.calls[tool_attempts]
        assert any("not current activity" in w for w in report.warnings)

    def test_a_rejected_shape_moves_on_to_the_next(self, snapshot):
        """One bad shape must not sink the feature - the next shape is tried."""
        client = FakeResponsesClient(content=payload_json(), fail_modes=1)
        report = ms.GrokMindshareClient(api_key="k", client=client).fetch(snapshot)

        assert report.is_live is True                # still live, via shape 2
        assert len(client.responses_calls) == 2

    def test_total_failure_returns_an_actionable_error(self, snapshot):
        client = FakeResponsesClient(error=RuntimeError("410 Gone"))
        report = ms.GrokMindshareClient(api_key="k", client=client).fetch(snapshot)

        assert report.available is False
        assert "410 Gone" in report.error
        assert "check_grok.py" in report.error

    def test_missing_key_is_reported_without_calling_out(self, snapshot, monkeypatch):
        monkeypatch.setattr(config, "XAI_API_KEY", "")
        report = ms.GrokMindshareClient().fetch(snapshot)

        assert report.available is False
        assert "No XAI_API_KEY" in report.error

    def test_disabled_by_env_flag(self, snapshot, monkeypatch):
        monkeypatch.setattr(config, "X_SEARCH_ENABLED", False)
        monkeypatch.setattr(config, "XAI_API_KEY", "k")
        assert "disabled" in ms.GrokMindshareClient().unavailable_reason()

    def test_search_window_becomes_an_iso_date_range(self):
        richest = ms.GrokMindshareClient(api_key="k").search_tool_variants(48)[0]

        assert len(richest["from_date"]) == 10 and richest["from_date"].count("-") == 2
        assert richest["from_date"] <= richest["to_date"]


class TestEnsembleHandoff:
    def test_x_findings_are_shared_with_every_model(self, snapshot):
        from src import llm_analyzers as la

        report = ms.parse_mindshare(payload(), "q", "grok", is_live=True)
        built = la.build_analysis_payload(snapshot, None, None, None, report)

        assert built["x_mindshare"]["data_is_live"] is True
        assert built["x_mindshare"]["sentiment"] == "bullish"
        assert built["x_mindshare"]["sample_posts"][0]["handle"] == "basedtrader"

    def test_stale_data_is_flagged_to_the_models(self, snapshot):
        from src import llm_analyzers as la

        report = ms.parse_mindshare(payload(), "q", "grok", is_live=False)
        built = la.build_analysis_payload(snapshot, None, None, None, report)
        assert "caveat" in built["x_mindshare"]
        assert "not a live X search" in built["x_mindshare"]["caveat"]
