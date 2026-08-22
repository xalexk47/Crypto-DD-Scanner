"""Tests for the multi-LLM ensemble: transport, repair, consensus, blending."""

from __future__ import annotations

import json

import pytest

from src import config, llm_analyzers as la, scorers
from src.data_fetchers import _parse_goplus_evm, snapshot_from_pair
from src.models import LLMVerdict
from tests.fake_llm import (
    FakeAnthropicClient,
    FakeOpenAIClient,
    SlowAnthropicClient,
    SlowClient,
    verdict_json,
    verdict_payload,
)
from tests.fixtures import TOKEN_ADDRESS, dexscreener_pair, goplus_honeypot_response, goplus_response


@pytest.fixture
def snapshot():
    pair = dexscreener_pair()
    return snapshot_from_pair(pair, [pair], address=TOKEN_ADDRESS)


@pytest.fixture
def security():
    return _parse_goplus_evm(TOKEN_ADDRESS, "base", next(iter(goplus_response()["result"].values())))


@pytest.fixture
def scorecard(snapshot, security):
    return scorers.build_scorecard(snapshot, security)


def verdict(provider="xai", **kwargs) -> LLMVerdict:
    """Build a normalized verdict straight from the coercion path."""
    return la.coerce_verdict(verdict_payload(**kwargs), provider, f"{provider}-model")


# ==========================================================================
# Schema & payload
# ==========================================================================
class TestSchema:
    def test_schema_matches_the_documented_contract(self):
        props = la.VERDICT_SCHEMA["properties"]
        assert set(la.VERDICT_SCHEMA["required"]) == {
            "overall_score", "decision", "confidence", "dimension_scores",
            "lore_summary", "key_positives", "key_risks", "rug_flags", "rationale",
        }
        assert props["decision"]["enum"] == ["strong_buy", "buy", "watch", "pass"]
        assert set(props["dimension_scores"]["properties"]) == {
            "security", "liquidity", "holders", "mindshare", "lore", "catalyst"
        }

    def test_schema_is_strict_mode_compatible(self):
        """Vendor strict modes require additionalProperties:false and full required lists."""

        def check(node):
            if node.get("type") == "object":
                assert node.get("additionalProperties") is False
                assert set(node.get("required", [])) == set(node.get("properties", {}))
                for child in node.get("properties", {}).values():
                    check(child)

        check(la.VERDICT_SCHEMA)

    def test_schema_avoids_unsupported_numeric_bounds(self):
        """Strict structured outputs ignore/reject minimum/maximum - ranges are enforced in code."""
        blob = json.dumps(la.VERDICT_SCHEMA)
        assert "minimum" not in blob and "maximum" not in blob


class TestPayload:
    def test_payload_carries_market_security_and_deterministic_score(self, snapshot, security, scorecard):
        payload = la.build_analysis_payload(snapshot, security, scorecard)

        assert payload["token"]["symbol"] == "BRETT"
        assert payload["market"]["liquidity_usd"] == pytest.approx(412_000.0)
        assert payload["security"]["lp_secured_pct"] == pytest.approx(98.0)
        assert payload["security"]["top10_pct_excluding_lp_and_burn"] is not None
        assert payload["platform_deterministic_score"]["composite_0_100"] == scorecard.composite
        assert len(payload["platform_deterministic_score"]["pillars"]) == 6

    def test_missing_security_is_explicit_not_silent(self, snapshot):
        payload = la.build_analysis_payload(snapshot, None, None)
        assert payload["security"] is None
        assert "security_unavailable_reason" in payload

    def test_prompt_is_deterministic_for_the_same_token(self, snapshot, security, scorecard):
        """Same payload for every provider is what makes verdicts comparable."""
        first = la.build_user_prompt(la.build_analysis_payload(snapshot, security, scorecard))
        second = la.build_user_prompt(la.build_analysis_payload(snapshot, security, scorecard))
        assert first == second
        assert "TOKEN DATA:" in first


# ==========================================================================
# Parsing and repair
# ==========================================================================
class TestExtractJson:
    def test_plain_json(self):
        assert la.extract_json('{"a": 1}') == {"a": 1}

    def test_strips_markdown_fences(self):
        assert la.extract_json('```json\n{"a": 1}\n```') == {"a": 1}

    def test_recovers_json_embedded_in_prose(self):
        assert la.extract_json('Sure! Here you go:\n{"a": 1}\nHope that helps.') == {"a": 1}

    @pytest.mark.parametrize("bad", ["", "   ", "no json here", "[1, 2, 3]"])
    def test_rejects_unusable_responses(self, bad):
        with pytest.raises(ValueError):
            la.extract_json(bad)


class TestCoerceVerdict:
    def test_clean_response_passes_through_unrepaired(self):
        result = la.coerce_verdict(verdict_payload(score=82, decision="buy", confidence=0.8), "xai", "grok-4")
        assert result.ok and not result.repaired
        assert result.overall_score == 82
        assert result.decision == "buy"
        assert result.decision_label == "Buy"
        assert len(result.dimension_scores) == 6

    def test_percentage_confidence_is_rescaled(self):
        result = la.coerce_verdict(verdict_payload(confidence=85), "xai", "m")
        assert result.confidence == pytest.approx(0.85)
        assert result.repaired and any("percentage" in w for w in result.warnings)

    def test_dimension_scores_on_a_0_100_scale_are_rescaled(self):
        payload = verdict_payload(dimensions={dim: 90 for dim in la.DIMENSIONS})
        result = la.coerce_verdict(payload, "openai", "m")
        assert all(value == 9.0 for value in result.dimension_scores.values())
        assert result.repaired

    def test_decision_aliases_are_normalized(self):
        for raw, expected in (("STRONG BUY", "strong_buy"), ("Strong-Buy", "strong_buy"),
                              ("hold", "watch"), ("avoid", "pass"), ("accumulate", "buy")):
            assert la.coerce_verdict(verdict_payload(decision=raw), "x", "m").decision == expected

    def test_unknown_decision_is_derived_from_the_score(self):
        result = la.coerce_verdict(verdict_payload(score=85, decision="yolo"), "x", "m")
        assert result.decision == "strong_buy"
        assert result.repaired and any("Unrecognised decision" in w for w in result.warnings)

    def test_missing_dimensions_are_filled_and_flagged(self):
        result = la.coerce_verdict(verdict_payload(score=60, dimensions={"security": 8}), "x", "m")
        assert result.dimension_scores["security"] == 8.0
        assert result.dimension_scores["liquidity"] == 6.0     # from the overall score
        assert result.repaired

    def test_values_are_clamped_to_their_ranges(self):
        result = la.coerce_verdict(
            {"overall_score": 900, "decision": "buy", "confidence": -3,
             "dimension_scores": {dim: -5 for dim in la.DIMENSIONS}}, "x", "m",
        )
        assert result.overall_score == 100.0
        assert result.confidence == 0.0
        # Negative scores clamp to 0. They must NOT be mistaken for missing
        # values and back-filled from the (here maxed-out) overall score.
        assert all(value == 0.0 for value in result.dimension_scores.values())

    def test_list_fields_tolerate_junk(self):
        result = la.coerce_verdict(
            verdict_payload(rug_flags=[{"text": "Mintable supply"}, "", "N/A", "Unlocked LP"]), "x", "m",
        )
        assert result.rug_flags == ["Mintable supply", "Unlocked LP"]

    def test_decision_from_score_matches_the_rules_engine_thresholds(self):
        assert la.decision_from_score(90) == "strong_buy"
        assert la.decision_from_score(70) == "buy"
        assert la.decision_from_score(55) == "watch"
        assert la.decision_from_score(10) == "pass"


# ==========================================================================
# Individual analyzers
# ==========================================================================
class TestAnalyzers:
    def test_xai_targets_the_documented_base_url(self):
        analyzer = la.XAIAnalyzer(api_key="k")
        assert analyzer.base_url == "https://api.x.ai/v1"
        assert analyzer.provider == "xai"

    def test_openai_and_xai_share_the_wire_protocol(self):
        assert issubclass(la.XAIAnalyzer, la.OpenAICompatibleAnalyzer)
        assert issubclass(la.OpenAIAnalyzer, la.OpenAICompatibleAnalyzer)

    def test_openai_requests_strict_json_schema_first(self, snapshot, security):
        client = FakeOpenAIClient(content=verdict_json())
        analyzer = la.OpenAIAnalyzer(api_key="k", client=client)
        result = analyzer.analyze(la.build_analysis_payload(snapshot, security))

        assert result.ok
        response_format = client.calls[0]["response_format"]
        assert response_format["type"] == "json_schema"
        assert response_format["json_schema"]["strict"] is True
        assert response_format["json_schema"]["schema"] == la.VERDICT_SCHEMA

    def test_openai_falls_back_through_json_modes(self, snapshot):
        client = FakeOpenAIClient(content=verdict_json(), fail_modes=2)
        result = la.OpenAIAnalyzer(api_key="k", client=client).analyze(la.build_analysis_payload(snapshot))

        assert result.ok
        assert len(client.calls) == 3
        assert client.calls[1]["response_format"] == {"type": "json_object"}
        assert "response_format" not in client.calls[2]

    def test_anthropic_uses_structured_output_first(self, snapshot, security):
        client = FakeAnthropicClient(content=verdict_json())
        result = la.AnthropicAnalyzer(api_key="k", client=client).analyze(
            la.build_analysis_payload(snapshot, security)
        )

        assert result.ok
        output_config = client.calls[0]["output_config"]
        assert output_config["format"]["type"] == "json_schema"
        assert output_config["format"]["schema"] == la.VERDICT_SCHEMA

    def test_anthropic_sends_no_sampling_params(self, snapshot):
        """Current Claude models reject temperature/top_p with a 400."""
        client = FakeAnthropicClient(content=verdict_json())
        la.AnthropicAnalyzer(api_key="k", client=client).analyze(la.build_analysis_payload(snapshot))
        for call in client.calls:
            assert "temperature" not in call
            assert "top_p" not in call

    def test_anthropic_falls_back_to_strict_tool_use(self, snapshot):
        client = FakeAnthropicClient(content=verdict_json(), fail_modes=1, as_tool_use=True)
        result = la.AnthropicAnalyzer(api_key="k", client=client).analyze(la.build_analysis_payload(snapshot))

        assert result.ok
        tool = client.calls[1]["tools"][0]
        assert tool["strict"] is True
        assert tool["input_schema"] == la.VERDICT_SCHEMA
        assert client.calls[1]["tool_choice"] == {"type": "tool", "name": "submit_verdict"}

    def test_provider_failure_returns_a_failed_verdict_not_an_exception(self, snapshot):
        client = FakeOpenAIClient(error=RuntimeError("401 unauthorized"))
        result = la.OpenAIAnalyzer(api_key="k", client=client).analyze(la.build_analysis_payload(snapshot))

        assert result.ok is False
        assert "401 unauthorized" in result.error
        assert result.provider == "openai"

    def test_unparseable_response_fails_cleanly(self, snapshot):
        client = FakeOpenAIClient(content="I'm sorry, I can't help with that.")
        result = la.OpenAIAnalyzer(api_key="k", client=client).analyze(la.build_analysis_payload(snapshot))
        assert result.ok is False
        assert "no JSON object" in result.error

    def test_latency_is_recorded(self, snapshot):
        result = la.OpenAIAnalyzer(api_key="k", client=FakeOpenAIClient(content=verdict_json())).analyze(
            la.build_analysis_payload(snapshot)
        )
        assert result.latency_ms >= 0

    def test_status_explains_why_a_provider_is_unavailable(self, monkeypatch):
        monkeypatch.setattr(config, "OPENAI_API_KEY", "")
        status = la.OpenAIAnalyzer().status()
        assert status.ready is False
        assert "No API key" in status.reason

    def test_build_analyzers_reports_skips_and_dedupes_aliases(self, monkeypatch):
        monkeypatch.setattr(config, "XAI_API_KEY", "")
        ready, skipped = la.build_analyzers(
            ["xai", "nonsense", "anthropic", "claude"],
            clients={"anthropic": FakeAnthropicClient(content=verdict_json())},
        )
        assert [a.provider for a in ready] == ["anthropic"]   # "claude" is the same provider
        assert "xai" in skipped and "No API key" in skipped["xai"]
        assert "Unknown provider" in skipped["nonsense"]


# ==========================================================================
# Consensus
# ==========================================================================
class TestConsensus:
    def test_confidence_weighted_average(self):
        verdicts = [
            verdict("xai", score=80, confidence=1.0, decision="buy"),
            verdict("openai", score=40, confidence=0.0, decision="watch"),
        ]
        consensus = la.build_consensus(verdicts)
        # The 0-confidence model is floored at 0.15, not silenced entirely.
        assert 70 < consensus.overall_score < 80

    def test_unanimous_models_produce_high_agreement(self):
        verdicts = [verdict(p, score=76, confidence=0.8, decision="buy") for p in ("xai", "anthropic", "openai")]
        consensus = la.build_consensus(verdicts)

        assert consensus.agreement == pytest.approx(1.0)
        assert consensus.unanimous
        assert consensus.decision == "buy"
        assert consensus.score_spread == 0.0
        assert consensus.dissent == []

    def test_disagreement_lowers_confidence_and_is_surfaced(self):
        verdicts = [
            verdict("xai", score=90, confidence=0.9, decision="strong_buy"),
            verdict("anthropic", score=30, confidence=0.9, decision="pass"),
        ]
        consensus = la.build_consensus(verdicts)

        assert consensus.agreement < 0.5
        assert consensus.confidence < 0.5      # despite both models being 90% sure
        assert consensus.score_spread == 60.0
        assert any("spread" in note for note in consensus.dissent)
        assert any("split on the call" in note for note in consensus.dissent)

    def test_decision_is_held_to_the_more_conservative_of_vote_and_score(self):
        # Both vote buy, but their weighted average lands in watch territory.
        verdicts = [
            verdict("xai", score=50, confidence=0.8, decision="buy"),
            verdict("anthropic", score=52, confidence=0.8, decision="buy"),
        ]
        consensus = la.build_consensus(verdicts)
        assert consensus.decision == "watch"
        assert any("conservative" in note for note in consensus.dissent)

    def test_ties_break_conservative(self):
        verdicts = [
            verdict("xai", score=70, confidence=0.5, decision="buy"),
            verdict("anthropic", score=70, confidence=0.5, decision="watch"),
        ]
        assert la.build_consensus(verdicts).decision == "watch"

    def test_rug_flags_raised_by_two_models_are_corroborated(self):
        verdicts = [
            verdict("xai", rug_flags=["Deployer holds 20% of supply", "Unlocked LP"]),
            verdict("anthropic", rug_flags=["deployer holds 20% of supply."]),   # same, differently cased
            verdict("openai", rug_flags=[]),
        ]
        consensus = la.build_consensus(verdicts)

        assert len(consensus.corroborated_rug_flags) == 1
        assert "Deployer holds 20%" in consensus.corroborated_rug_flags[0]
        assert "Unlocked LP" in consensus.rug_flags
        assert "Unlocked LP" not in consensus.corroborated_rug_flags
        assert any("only one model" in note for note in consensus.dissent)

    def test_repeated_points_are_merged_and_ranked_by_support(self):
        verdicts = [
            verdict("xai", positives=["LP is burned", "Strong community"]),
            verdict("anthropic", positives=["LP is burned"]),
            verdict("openai", positives=["Renounced"]),
        ]
        consensus = la.build_consensus(verdicts)
        assert consensus.key_positives[0] == "LP is burned"       # 2 models
        assert len(consensus.key_positives) == 3                   # deduped

    def test_single_model_confidence_is_capped(self):
        solo = la.build_consensus([verdict("xai", confidence=1.0)])
        assert solo.model_count == 1
        assert solo.confidence <= 0.7
        assert "Single-model verdict" in solo.rationale

    def test_failed_verdicts_are_excluded(self):
        verdicts = [
            verdict("xai", score=80),
            LLMVerdict(provider="openai", model="gpt", ok=False, error="timeout"),
        ]
        consensus = la.build_consensus(verdicts)
        assert consensus.model_count == 1

    def test_no_usable_verdicts_returns_none(self):
        assert la.build_consensus([]) is None
        assert la.build_consensus([LLMVerdict(provider="x", model="m", ok=False)]) is None


# ==========================================================================
# Blending with the deterministic engine
# ==========================================================================
class TestBlend:
    def test_weight_moves_the_score_between_the_two_engines(self, snapshot, security):
        card = scorers.build_scorecard(snapshot, security)
        consensus = la.build_consensus([verdict("xai", score=40, confidence=0.8, decision="watch")])

        at_zero, _, _ = la.blend_with_deterministic(card, consensus, 0.0)
        at_one, _, _ = la.blend_with_deterministic(card, consensus, 1.0)
        halfway, _, _ = la.blend_with_deterministic(card, consensus, 0.5)

        assert at_zero == pytest.approx(card.composite, abs=0.1)
        assert at_one == pytest.approx(40.0, abs=0.1)
        assert halfway == pytest.approx((card.composite + 40) / 2, abs=0.1)

    def test_security_veto_beats_an_enthusiastic_ensemble(self, snapshot):
        honeypot = _parse_goplus_evm(
            TOKEN_ADDRESS, "base", next(iter(goplus_honeypot_response()["result"].values()))
        )
        card = scorers.build_scorecard(snapshot, honeypot)
        consensus = la.build_consensus([verdict("xai", score=95, confidence=1.0, decision="strong_buy")])

        score, decision, notes = la.blend_with_deterministic(card, consensus, 0.9)
        assert decision == "pass"
        assert score <= 20
        assert any("veto overrides" in note for note in notes)

    def test_models_can_lower_but_not_rescue_the_decision(self, snapshot, security):
        card = scorers.build_scorecard(snapshot, security)   # a Strong Buy on clean data
        bearish = la.build_consensus([verdict("xai", score=10, confidence=0.9, decision="pass")])
        _, lowered, _ = la.blend_with_deterministic(card, bearish, 0.9)
        assert lowered in ("pass", "watch")

        weak_card = scorers.build_scorecard(snapshot, None)   # no security data -> weaker score
        bullish = la.build_consensus([verdict("xai", score=100, confidence=1.0, decision="strong_buy")])
        _, raised, notes = la.blend_with_deterministic(weak_card, bullish, 0.9)
        assert la.DECISION_RANK[raised] <= la.DECISION_RANK[la._decision_key(weak_card.decision)]

    def test_missing_inputs_are_handled(self, scorecard):
        assert la.blend_with_deterministic(None, None) == (None, "", [])
        assert la.blend_with_deterministic(scorecard, None)[0] is None


# ==========================================================================
# Parallel runner
# ==========================================================================
class TestRunEnsemble:
    def _clients(self, **scores):
        return {
            "xai": FakeOpenAIClient(content=verdict_json(score=scores.get("xai", 80), decision="buy")),
            "anthropic": FakeAnthropicClient(content=verdict_json(score=scores.get("anthropic", 70), decision="buy")),
            "openai": FakeOpenAIClient(content=verdict_json(score=scores.get("openai", 60), decision="watch")),
        }

    def test_all_three_providers_run_and_combine(self, snapshot, security, scorecard):
        result = la.run_ensemble(snapshot, security, scorecard, clients=self._clients())

        assert [v.provider for v in result.verdicts] == ["xai", "anthropic", "openai"]
        assert all(v.ok for v in result.verdicts)
        assert result.consensus.model_count == 3
        assert result.blended_score is not None
        assert result.elapsed_ms >= 0

    def test_every_model_receives_the_identical_payload(self, snapshot, security, scorecard):
        clients = self._clients()
        la.run_ensemble(snapshot, security, scorecard, clients=clients)

        xai_prompt = clients["xai"].calls[0]["messages"][1]["content"]
        openai_prompt = clients["openai"].calls[0]["messages"][1]["content"]
        anthropic_prompt = clients["anthropic"].calls[0]["messages"][0]["content"]
        assert xai_prompt == openai_prompt == anthropic_prompt

        xai_system = clients["xai"].calls[0]["messages"][0]["content"]
        assert xai_system == clients["anthropic"].calls[0]["system"] == la.ANALYST_SYSTEM_PROMPT

    def test_one_provider_failing_does_not_sink_the_run(self, snapshot, security, scorecard):
        clients = self._clients()
        clients["openai"] = FakeOpenAIClient(error=RuntimeError("503 upstream"))
        result = la.run_ensemble(snapshot, security, scorecard, clients=clients)

        assert len(result.successful) == 2
        assert len(result.failed) == 1
        assert result.consensus.model_count == 2
        assert any("503 upstream" in note for note in result.notes)

    def test_timeout_marks_the_slow_provider_failed(self, snapshot, security, scorecard):
        clients = self._clients()
        clients["anthropic"] = SlowAnthropicClient(delay=5.0, content=verdict_json())
        result = la.run_ensemble(snapshot, security, scorecard, clients=clients, timeout=0.4)

        anthropic_verdict = next(v for v in result.verdicts if v.provider == "anthropic")
        assert anthropic_verdict.ok is False
        assert "Timed out" in anthropic_verdict.error
        assert len(result.successful) == 2      # the fast ones still counted

    def test_providers_run_in_parallel_not_sequentially(self, snapshot, security, scorecard):
        """Three 0.5s models must finish in well under the 1.5s a serial run would cost."""
        clients = {
            "xai": SlowClient(delay=0.5, content=verdict_json()),
            "openai": SlowClient(delay=0.5, content=verdict_json()),
            "anthropic": SlowAnthropicClient(delay=0.5, content=verdict_json()),
        }
        result = la.run_ensemble(snapshot, security, scorecard, clients=clients, timeout=5)

        assert len(result.successful) == 3
        assert result.elapsed_ms < 1200

    def test_no_configured_providers_explains_itself(self, snapshot, monkeypatch):
        for name in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "XAI_API_KEY"):
            monkeypatch.setattr(config, name, "")
        result = la.run_ensemble(snapshot)

        assert result.ok is False
        assert result.consensus is None
        assert any("No LLM providers are configured" in note for note in result.notes)

    def test_provider_subset_is_honoured(self, snapshot, security, scorecard):
        clients = self._clients()
        result = la.run_ensemble(snapshot, security, scorecard, providers=["xai"], clients=clients)
        assert [v.provider for v in result.verdicts] == ["xai"]
        assert clients["openai"].calls == []

    def test_result_serializes_for_export(self, snapshot, security, scorecard):
        result = la.run_ensemble(snapshot, security, scorecard, clients=self._clients())
        payload = json.loads(json.dumps(result.to_dict(), default=str))

        assert payload["model_count"] == 3
        assert payload["consensus"]["decision_label"]
        assert len(payload["verdicts"]) == 3
