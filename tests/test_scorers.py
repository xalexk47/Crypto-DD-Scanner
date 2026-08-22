"""Tests for the scoring engine and risk calculator."""

from __future__ import annotations

import pytest

from src import config, scorers
from src.data_fetchers import _parse_goplus_evm, snapshot_from_pair
from src.models import SecurityReport, TokenSnapshot
from tests.fixtures import TOKEN_ADDRESS, dexscreener_pair, goplus_honeypot_response, goplus_response


def make_snapshot(**overrides) -> TokenSnapshot:
    pair = dexscreener_pair()
    snapshot = snapshot_from_pair(pair, [pair], address=TOKEN_ADDRESS)
    for key, value in overrides.items():
        setattr(snapshot, key, value)
    return snapshot


def make_security(honeypot: bool = False) -> SecurityReport:
    payload = goplus_honeypot_response() if honeypot else goplus_response()
    data = next(iter(payload["result"].values()))
    return _parse_goplus_evm(TOKEN_ADDRESS, "base", data)


class TestSecurityScore:
    def test_clean_token_scores_high(self):
        component = scorers.score_security(make_security())
        assert component.score >= 85
        assert component.confidence > 0.8

    def test_honeypot_scores_zero(self):
        component = scorers.score_security(make_security(honeypot=True))
        assert component.score == 0.0

    def test_missing_data_is_penalised_not_neutral(self):
        component = scorers.score_security(None)
        assert component.score == 40.0
        assert component.confidence < 0.5

    def test_each_dangerous_flag_lowers_the_score(self):
        baseline = scorers.score_security(make_security()).score
        for field in ("is_mintable", "transfer_pausable", "is_blacklisted", "can_take_back_ownership"):
            report = make_security()
            setattr(report, field, True)
            assert scorers.score_security(report).score < baseline, field

    def test_unlocked_lp_is_punished(self):
        report = make_security()
        report.lp_burned_pct, report.lp_locked_pct = 0.0, 0.0
        assert scorers.score_security(report).score < scorers.score_security(make_security()).score - 20

    def test_high_tax_is_punished_progressively(self):
        scores = []
        for tax in (0.0, 8.0, 15.0, 40.0):
            report = make_security()
            report.buy_tax_pct = report.sell_tax_pct = tax
            scores.append(scorers.score_security(report).score)
        assert scores == sorted(scores, reverse=True)


class TestLiquidityScore:
    def test_deep_balanced_pool_scores_well(self):
        component = scorers.score_liquidity(make_snapshot(), make_security())
        assert component.score >= 70

    def test_zero_liquidity_scores_zero(self):
        assert scorers.score_liquidity(make_snapshot(liquidity_usd=0.0)).score == 0.0

    def test_thin_liquidity_ratio_drags_score_down(self):
        thin = make_snapshot(liquidity_usd=20_000.0, market_cap=5_000_000.0)
        deep = make_snapshot(liquidity_usd=500_000.0, market_cap=5_000_000.0)
        assert scorers.score_liquidity(thin).score < scorers.score_liquidity(deep).score

    def test_wash_trading_turnover_is_penalised(self):
        healthy = make_snapshot(volume_24h=800_000.0)          # ~2x pool
        washed = make_snapshot(volume_24h=25_000_000.0)        # ~60x pool
        assert scorers.score_liquidity(washed).score < scorers.score_liquidity(healthy).score

    def test_unlocked_lp_discounts_liquidity_quality(self):
        report = make_security()
        report.lp_burned_pct, report.lp_locked_pct = 0.0, 5.0
        assert scorers.score_liquidity(make_snapshot(), report).score < scorers.score_liquidity(make_snapshot(), make_security()).score


class TestHolderScore:
    def test_distributed_supply_scores_well(self):
        assert scorers.score_holders(make_security(), make_snapshot()).score >= 75

    def test_concentration_lowers_score(self):
        report = make_security()
        report.top10_pct_adjusted = 65.0
        assert scorers.score_holders(report).score < 35

    def test_deployer_bag_is_penalised(self):
        report = make_security()
        report.creator_percent = 15.0
        assert scorers.score_holders(report).score < scorers.score_holders(make_security()).score

    def test_missing_holder_data_yields_low_confidence(self):
        component = scorers.score_holders(None)
        assert component.confidence <= 0.3


class TestMomentumScore:
    def test_hot_token_beats_dead_token(self):
        hot = make_snapshot(volume_24h=3_000_000.0, price_change_1h=6.0, price_change_24h=45.0)
        dead = make_snapshot(volume_24h=2_000.0, price_change_1h=-3.0, price_change_6h=-8.0,
                             price_change_24h=-25.0, txns_24h_buys=10, txns_24h_sells=40)
        assert scorers.score_momentum(hot).score > scorers.score_momentum(dead).score

    def test_parabolic_move_is_discounted_versus_steady_climb(self):
        steady = make_snapshot(price_change_1h=5.0, price_change_6h=20.0, price_change_24h=40.0)
        parabolic = make_snapshot(price_change_1h=180.0, price_change_6h=600.0, price_change_24h=2000.0)
        steady_price = next(r for r in scorers.score_momentum(steady).reasons if "Price" in r)
        assert steady_price  # sanity
        assert scorers.score_momentum(parabolic).score < scorers.score_momentum(steady).score + 5

    def test_volume_acceleration_is_rewarded(self):
        flat = make_snapshot(volume_6h=462_500.0)     # exactly on run-rate
        heating = make_snapshot(volume_6h=1_200_000.0)
        assert scorers.score_momentum(heating).score > scorers.score_momentum(flat).score


class TestNarrativeAndCatalyst:
    def test_socials_and_meme_name_raise_narrative(self):
        rich = make_snapshot()
        bare = make_snapshot(socials=[], image_url="", description="", name="Token", symbol="TKNXYZABCDEF", boosts=0)
        assert scorers.score_narrative(rich).score > scorers.score_narrative(bare).score

    def test_sweet_spot_market_cap_scores_best(self):
        sweet = scorers.score_catalyst(make_snapshot(market_cap=3_000_000.0)).score
        large = scorers.score_catalyst(make_snapshot(market_cap=80_000_000.0)).score
        assert sweet > large


class TestComposite:
    def test_clean_token_is_a_buy(self):
        card = scorers.build_scorecard(make_snapshot(), make_security())
        assert card.composite >= 60
        assert card.decision in ("Buy", "Strong Buy")
        assert not card.vetoed

    def test_honeypot_is_vetoed_to_pass(self):
        card = scorers.build_scorecard(make_snapshot(), make_security(honeypot=True))
        assert card.vetoed is True
        assert card.decision == "Pass"
        assert card.composite <= 20
        assert "Honeypot" in card.risks[0]

    def test_components_cover_every_configured_pillar(self):
        card = scorers.build_scorecard(make_snapshot(), make_security())
        assert {c.key for c in card.components} == set(config.COMPONENT_LABELS)
        assert sum(c.weight for c in card.components) == pytest.approx(1.0)

    def test_custom_weights_are_applied(self):
        weights = config.ScoreWeights(security=0.7, liquidity=0.1, holders=0.1, momentum=0.05,
                                      narrative=0.025, catalyst=0.025)
        card = scorers.build_scorecard(make_snapshot(), make_security(), weights=weights)
        assert card.component("security").weight == pytest.approx(0.7)

    def test_weights_must_sum_to_one(self):
        with pytest.raises(ValueError):
            config.ScoreWeights(security=0.9, liquidity=0.9, holders=0.1, momentum=0.1,
                                narrative=0.1, catalyst=0.1).validate()

    def test_decision_thresholds(self):
        assert scorers.decide(90) == "Strong Buy"
        assert scorers.decide(70) == "Buy"
        assert scorers.decide(55) == "Watch"
        assert scorers.decide(20) == "Pass"

    def test_missing_security_lowers_confidence(self):
        with_sec = scorers.build_scorecard(make_snapshot(), make_security())
        without = scorers.build_scorecard(make_snapshot(), None)
        assert without.confidence < with_sec.confidence
        assert without.composite < with_sec.composite

    def test_quick_score_needs_no_security_data(self):
        score, reasons = scorers.quick_score(make_snapshot())
        assert 0 <= score <= 100
        assert isinstance(reasons, list)


class TestRiskPlan:
    def _plan(self, portfolio=10_000.0, profile="moderate", **snapshot_overrides):
        snapshot = make_snapshot(**snapshot_overrides)
        card = scorers.build_scorecard(snapshot, make_security())
        return scorers.build_risk_plan(snapshot, card, portfolio, profile, make_security())

    def test_max_loss_respects_the_risk_budget(self):
        plan = self._plan()
        profile = config.RISK_PROFILES["moderate"]
        # Conviction scaling and caps can only reduce risk, never exceed the budget.
        assert plan.max_loss_pct_of_portfolio <= profile.risk_per_trade_pct + 1e-6

    def test_loss_equals_position_times_stop(self):
        plan = self._plan()
        assert plan.max_loss_usd == pytest.approx(plan.position_usd * plan.stop_loss_pct / 100, rel=1e-3)

    def test_position_never_exceeds_profile_cap(self):
        for key, profile in config.RISK_PROFILES.items():
            plan = self._plan(profile=key)
            assert plan.position_pct <= profile.max_position_pct + 1e-6, key

    def test_riskier_profiles_size_up(self):
        sizes = [self._plan(profile=key).position_usd for key in
                 ("conservative", "moderate", "aggressive", "degen")]
        assert sizes == sorted(sizes)

    def test_thin_liquidity_caps_the_position(self):
        plan = self._plan(portfolio=5_000_000.0, liquidity_usd=40_000.0)
        assert plan.liquidity_capped is True
        assert plan.position_usd <= 40_000.0 * config.RISK_PROFILES["moderate"].max_liquidity_share_pct / 100 + 1e-6
        assert any("Liquidity-adjusted" in w for w in plan.warnings)

    def test_volatile_token_gets_a_wider_stop(self):
        calm = self._plan(price_change_1h=0.5, price_change_6h=1.0, price_change_24h=2.0)
        wild = self._plan(price_change_1h=25.0, price_change_6h=60.0, price_change_24h=140.0)
        assert wild.stop_loss_pct > calm.stop_loss_pct

    def test_stop_price_sits_below_spot(self):
        plan = self._plan()
        assert plan.stop_price < make_snapshot().price_usd

    def test_take_profit_ladder_is_asymmetric(self):
        plan = self._plan()
        assert [t["r_multiple"] for t in plan.take_profit_targets] == [1.5, 3.0, 6.0]
        assert all(t["gain_pct"] > plan.stop_loss_pct for t in plan.take_profit_targets)

    def test_veto_forces_zero_size(self):
        snapshot = make_snapshot()
        card = scorers.build_scorecard(snapshot, make_security(honeypot=True))
        plan = scorers.build_risk_plan(snapshot, card, 10_000.0, "moderate", make_security(honeypot=True))
        assert plan.position_usd == 0.0
        assert plan.max_loss_usd == 0.0

    def test_zero_portfolio_does_not_divide_by_zero(self):
        plan = self._plan(portfolio=0.0)
        assert plan.position_usd == 0.0
        assert plan.max_loss_pct_of_portfolio == 0.0
