"""Tests for the chain heat index and the trim/rotate planner."""

from __future__ import annotations

import pytest

from src import config, portfolio_store, rotation
from src.models import ChainHeat, PortfolioSnapshot, Position, TokenSnapshot

BREW = "0x1234567890abcdef1234567890abcdef12345678"


def snapshot_for(
    address=BREW, chain="bsc", change_24h=0.0, change_6h=0.0, volume_24h=100_000.0,
    volume_6h=25_000.0, market_cap=1_000_000.0, liquidity=200_000.0, buys=500, sells=500,
    symbol="BREW",
) -> TokenSnapshot:
    return TokenSnapshot(
        address=address, chain=chain, symbol=symbol, price_usd=1.0,
        market_cap=market_cap, liquidity_usd=liquidity, volume_24h=volume_24h,
        volume_6h=volume_6h, price_change_24h=change_24h, price_change_6h=change_6h,
        txns_24h_buys=buys, txns_24h_sells=sells,
    )


def position_for(value=1000.0, **kwargs) -> Position:
    market = snapshot_for(**kwargs)
    return Position(
        address=market.address, chain=market.chain, quantity=value, price_usd=1.0,
        value_usd=value, symbol=market.symbol, snapshot=market,
    )


@pytest.fixture
def db(tmp_path):
    return tmp_path / "rotation.sqlite3"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Chain-wide inputs are off unless a test opts in."""
    monkeypatch.setattr(rotation, "fetch_chain_basket", lambda chain, use_cache=True: ([], []))
    monkeypatch.setattr(rotation, "fetch_defillama_stats", lambda chain, use_cache=True: None)
    rotation.clear_cache()


class TestComponents:
    def test_a_quiet_chain_scores_mid_range(self):
        components = rotation.score_components([(snapshot_for(), 1.0)])
        blended = rotation.blend_components(components, config.DEFAULT_HEAT_WEIGHTS)
        assert 25 <= blended <= 60

    def test_a_ripping_chain_scores_high(self):
        hot = snapshot_for(change_24h=120.0, change_6h=60.0, volume_24h=900_000.0,
                           volume_6h=500_000.0, buys=900, sells=200)
        components = rotation.score_components([(hot, 1.0)])
        assert rotation.blend_components(components, config.DEFAULT_HEAT_WEIGHTS) > 75

    def test_a_bleeding_chain_scores_low(self):
        cold = snapshot_for(change_24h=-35.0, change_6h=-20.0, volume_24h=20_000.0,
                            volume_6h=2_000.0, buys=150, sells=600)
        components = rotation.score_components([(cold, 1.0)])
        assert rotation.blend_components(components, config.DEFAULT_HEAT_WEIGHTS) < 25

    def test_weighting_follows_position_size(self):
        big = snapshot_for(change_24h=100.0, change_6h=50.0)
        small = snapshot_for(change_24h=-50.0, change_6h=-25.0)
        weighted = rotation.score_components([(big, 9_000.0), (small, 1_000.0)])
        even = rotation.score_components([(big, 1.0), (small, 1.0)])
        assert weighted["price_momentum"] > even["price_momentum"]

    def test_blend_renormalizes_over_present_components(self):
        # A chain missing DefiLlama coverage must not be penalised for it: the
        # score is computed over what is known.
        partial = rotation.blend_components({"price_momentum": 80.0}, config.DEFAULT_HEAT_WEIGHTS)
        assert partial == pytest.approx(80.0)

    def test_no_samples_yields_no_components(self):
        assert rotation.score_components([]) == {}
        assert rotation.blend_components({}, config.DEFAULT_HEAT_WEIGHTS) is None


class TestStateMachine:
    def test_no_history_classifies_on_level_alone(self):
        assert rotation.classify_state(85.0, [])[0] == "hot"
        assert rotation.classify_state(20.0, [])[0] == "cold"

    def test_rising_from_a_low_base_is_heating_not_cold(self):
        # The earliest turn is exactly what a rotation wants to catch.
        history = [{"heat": 30.0, "state": "cold", "taken_at": "2026-09-12T10:00:00+00:00"}]
        state, trailing, _ = rotation.classify_state(48.0, history)
        assert state == "heating"
        assert trailing == pytest.approx(30.0)

    def test_high_but_rolling_over_is_cooling_not_hot(self):
        history = [{"heat": 90.0, "state": "hot", "taken_at": "2026-09-12T10:00:00+00:00"}]
        assert rotation.classify_state(60.0, history)[0] == "cooling"

    def test_still_high_and_steady_stays_hot(self):
        history = [{"heat": 80.0, "state": "hot", "taken_at": "2026-09-12T10:00:00+00:00"}]
        assert rotation.classify_state(82.0, history)[0] == "hot"

    def test_low_and_falling_is_cold(self):
        history = [{"heat": 45.0, "state": "cooling", "taken_at": "2026-09-12T10:00:00+00:00"}]
        assert rotation.classify_state(30.0, history)[0] == "cold"


class TestChainHeat:
    def test_positions_only_heat_reports_the_missing_inputs(self, db):
        heat = rotation.compute_chain_heat(
            "bsc", positions=[position_for(change_24h=90.0, change_6h=40.0)], db_path=db
        )
        assert heat.portfolio_heat is not None
        assert heat.market_heat is None
        assert rotation.INPUT_BASKET in heat.missing_inputs
        assert heat.confidence < 1.0            # and it says so
        assert heat.divergence is None

    def test_portfolio_and_market_are_blended_and_reported_separately(self, monkeypatch, db):
        monkeypatch.setattr(
            rotation, "fetch_chain_basket",
            lambda chain, use_cache=True: ([snapshot_for(change_24h=0.0, address="0xbasket")], []),
        )
        heat = rotation.compute_chain_heat(
            "bsc", positions=[position_for(change_24h=150.0, change_6h=80.0)], db_path=db
        )
        assert heat.portfolio_heat > heat.market_heat
        # The gap is the whole point: your bags ran, the chain did not.
        assert heat.divergence > 0
        assert min(heat.market_heat, heat.portfolio_heat) <= heat.heat <= max(
            heat.market_heat, heat.portfolio_heat)

    def test_a_chain_with_no_data_is_unknown_not_cold(self, db):
        heat = rotation.compute_chain_heat("robinhood", positions=[], db_path=db)
        assert heat.confidence == 0.0
        assert "not 'cold'" in " ".join(heat.notes)

    def test_defillama_trend_feeds_the_chain_component(self, monkeypatch, db):
        monkeypatch.setattr(
            rotation, "fetch_chain_basket",
            lambda chain, use_cache=True: ([snapshot_for(address="0xbasket")], []),
        )
        monkeypatch.setattr(
            rotation, "fetch_defillama_stats",
            lambda chain, use_cache=True: {"tvl_usd": 5e9, "dex_volume_24h": 1e9,
                                           "dex_change_1d": 55.0},
        )
        heat = rotation.compute_chain_heat("bsc", positions=[], db_path=db)
        assert heat.components["chain_tvl_trend"] > 80
        assert heat.tvl_usd == pytest.approx(5e9)
        assert rotation.INPUT_DEFILLAMA in heat.inputs_used

    def test_heats_are_persisted_and_ranked(self, monkeypatch, db):
        def basket(chain, use_cache=True):
            change = 120.0 if chain == "bsc" else -10.0
            return ([snapshot_for(chain=chain, address=f"0x{chain}", change_24h=change,
                                  change_6h=change / 2)], [])

        monkeypatch.setattr(rotation, "fetch_chain_basket", basket)
        heats = rotation.compute_heats(chains=["base", "bsc"], db_path=db)
        assert heats[0].chain == "bsc"                       # hottest first
        assert portfolio_store.heat_history("bsc", db_path=db)

    def test_one_broken_chain_does_not_kill_the_others(self, monkeypatch, db):
        def basket(chain, use_cache=True):
            if chain == "base":
                raise RuntimeError("dexscreener down")
            return ([snapshot_for(chain=chain, address=f"0x{chain}")], [])

        monkeypatch.setattr(rotation, "fetch_chain_basket", basket)
        heats = {h.chain: h for h in rotation.compute_heats(chains=["base", "bsc"], db_path=db)}
        assert heats["bsc"].confidence > 0
        assert heats["base"].confidence < 1.0


class TestRotationPlan:
    def _heats(self, hot_chain="bsc", cold_chain="base"):
        return [
            ChainHeat(chain=hot_chain, heat=84.0, state="hot", hours_in_state=36.0,
                      portfolio_heat=90.0, market_heat=80.0, confidence=1.0),
            ChainHeat(chain=cold_chain, heat=31.0, state="heating", portfolio_heat=30.0,
                      market_heat=32.0, confidence=1.0),
        ]

    def test_trims_a_winner_on_a_hot_chain_and_rotates_to_the_cold_one(self):
        snapshot = PortfolioSnapshot(positions=[
            position_for(value=5_000.0, change_24h=180.0, liquidity=5_000_000.0),
        ])
        # The concentration rule is exercised separately; here only the
        # hot-chain rule should be able to fire.
        plan = rotation.build_rotation_plan(
            snapshot, self._heats(), settings=config.RotationSettings(max_position_pct=100.0)
        )

        trims = plan.actions_of("trim")
        rotates = plan.actions_of("rotate")
        assert len(trims) == 1
        assert trims[0].pct_of_position == pytest.approx(config.DEFAULT_ROTATION_SETTINGS.trim_pct_hot)
        assert trims[0].amount_usd == pytest.approx(1_250.0)
        assert "84/100" in trims[0].reason
        assert rotates and rotates[0].dest_chain == "base"
        assert plan.total_trim_usd == pytest.approx(1_250.0)

    def test_a_flat_position_on_a_hot_chain_is_left_alone(self):
        snapshot = PortfolioSnapshot(positions=[
            position_for(value=5_000.0, change_24h=2.0, liquidity=5_000_000.0),
        ])
        plan = rotation.build_rotation_plan(
            snapshot, self._heats(), settings=config.RotationSettings(max_position_pct=100.0)
        )
        assert plan.actions_of("trim") == []

    def test_an_overweight_position_is_trimmed_back_to_the_cap(self):
        # 60% of the book against a 5% cap (moderate profile) -> trim to the cap.
        snapshot = PortfolioSnapshot(positions=[
            position_for(value=6_000.0, change_24h=0.0, liquidity=9_000_000.0),
            position_for(value=4_000.0, change_24h=0.0, chain="base", address="0xb",
                         liquidity=9_000_000.0),
        ])
        heats = [ChainHeat(chain="bsc", heat=45.0, state="cooling", confidence=1.0),
                 ChainHeat(chain="base", heat=40.0, state="cooling", confidence=1.0)]
        plan = rotation.build_rotation_plan(snapshot, heats)
        trim = plan.actions_of("trim")[0]
        remaining = 6_000.0 - trim.amount_usd
        assert remaining / snapshot.total_usd * 100 == pytest.approx(5.0, abs=0.5)
        assert "cap" in trim.reason

    def test_trim_is_capped_by_pool_liquidity(self):
        # A big position in a thin pool: the plan must not suggest dumping it.
        snapshot = PortfolioSnapshot(positions=[
            position_for(value=50_000.0, change_24h=300.0, liquidity=100_000.0),
        ])
        plan = rotation.build_rotation_plan(
            snapshot, self._heats(), settings=config.RotationSettings(max_position_pct=100.0))
        trim = plan.actions_of("trim")[0]
        risk = config.RISK_PROFILES["moderate"]
        assert trim.amount_usd == pytest.approx(100_000.0 * risk.max_liquidity_share_pct / 100)
        assert trim.warnings and "pool" in trim.warnings[0]

    def test_never_rotates_into_a_chain_it_could_not_read(self):
        heats = [
            ChainHeat(chain="bsc", heat=84.0, state="hot", confidence=1.0),
            ChainHeat(chain="robinhood", heat=0.0, state="cold", confidence=0.0),
        ]
        snapshot = PortfolioSnapshot(positions=[
            position_for(value=5_000.0, change_24h=180.0, liquidity=5_000_000.0),
        ])
        plan = rotation.build_rotation_plan(snapshot, heats)
        assert plan.actions_of("rotate") == []
        assert any("Nothing is cool enough" in note for note in plan.notes)
        assert any("robinhood" in w.lower() or "Robinhood" in w for w in plan.warnings)

    def test_actions_below_the_minimum_are_not_suggested(self):
        snapshot = PortfolioSnapshot(positions=[
            position_for(value=60.0, change_24h=180.0, liquidity=500_000.0),
        ])
        plan = rotation.build_rotation_plan(
            snapshot, self._heats(), settings=config.RotationSettings(max_position_pct=100.0)
        )
        assert plan.actions == []           # a $15 trim is not worth the spread

    def test_cost_basis_is_preferred_over_the_24h_move(self):
        position = position_for(value=1_000.0, change_24h=1.0, liquidity=5_000_000.0)
        position.avg_cost_usd = 0.1        # price is 1.0, so +900%
        plan = rotation.build_rotation_plan(
            PortfolioSnapshot(positions=[position]), self._heats(),
            settings=config.RotationSettings(max_position_pct=100.0))
        assert "vs your avg cost" in plan.actions_of("trim")[0].reason

    def test_divergence_flags_a_chain_moving_without_you(self):
        heats = [ChainHeat(chain="base", heat=70.0, state="heating",
                           portfolio_heat=40.0, market_heat=75.0, confidence=1.0)]
        snapshot = PortfolioSnapshot(positions=[position_for(chain="base", value=500.0)])
        plan = rotation.build_rotation_plan(snapshot, heats)
        assert any("moving without you" in note for note in plan.notes)

    def test_empty_book_produces_no_actions(self):
        plan = rotation.build_rotation_plan(PortfolioSnapshot(), self._heats())
        assert plan.actions == []
        assert "sync your wallets" in plan.notes[0]

    def test_a_trim_never_exceeds_the_position(self):
        snapshot = PortfolioSnapshot(positions=[
            position_for(value=9_000.0, change_24h=400.0, liquidity=50_000_000.0),
        ])
        plan = rotation.build_rotation_plan(
            snapshot, self._heats(),
            settings=config.RotationSettings(trim_pct_hot=90.0, max_position_pct=1.0),
        )
        trim = plan.actions_of("trim")[0]
        assert trim.pct_of_position <= 100.0
        assert trim.amount_usd <= snapshot.positions[0].value_usd

    def test_risk_profile_cap_is_respected(self):
        snapshot = PortfolioSnapshot(positions=[
            position_for(value=3_000.0, change_24h=0.0, liquidity=9_000_000.0),
            position_for(value=7_000.0, change_24h=0.0, chain="base", address="0xb",
                         liquidity=9_000_000.0),
        ])
        heats = [ChainHeat(chain="bsc", heat=45.0, state="cooling", confidence=1.0),
                 ChainHeat(chain="base", heat=44.0, state="cooling", confidence=1.0)]
        conservative = rotation.build_rotation_plan(
            snapshot, heats, risk=config.RISK_PROFILES["conservative"])
        degen = rotation.build_rotation_plan(snapshot, heats, risk=config.RISK_PROFILES["degen"])
        # A 2% cap trims far more than a 15% one.
        assert conservative.total_trim_usd > degen.total_trim_usd


class TestBlending:
    def test_a_component_only_one_side_scored_is_not_averaged_against_zero(self):
        blended = rotation._blend_component_maps(
            {"price_momentum": 80.0}, {"chain_tvl_trend": 20.0}, share=0.4
        )
        assert blended["price_momentum"] == pytest.approx(80.0)
        assert blended["chain_tvl_trend"] == pytest.approx(20.0)

    def test_shared_components_use_the_portfolio_share(self):
        blended = rotation._blend_component_maps(
            {"price_momentum": 100.0}, {"price_momentum": 0.0}, share=0.4
        )
        assert blended["price_momentum"] == pytest.approx(40.0)

    def test_rotation_names_the_chain_funding_the_move(self):
        heats = [
            ChainHeat(chain="bsc", heat=84.0, state="hot", confidence=1.0),
            ChainHeat(chain="solana", heat=70.0, state="hot", confidence=1.0),
            ChainHeat(chain="base", heat=30.0, state="heating", confidence=1.0),
        ]
        snapshot = PortfolioSnapshot(positions=[
            position_for(value=1_000.0, change_24h=200.0, liquidity=9_000_000.0),
            position_for(value=9_000.0, change_24h=200.0, chain="solana", address="Mint",
                         liquidity=9_000_000.0),
        ])
        plan = rotation.build_rotation_plan(
            snapshot, heats, settings=config.RotationSettings(max_position_pct=100.0))
        # Solana contributes the larger trim, so it is the source named.
        assert plan.actions_of("rotate")[0].chain == "solana"


class TestActionCopy:
    """The plan is read by a person, so it must name things the way the UI does."""

    def test_headlines_use_chain_labels_not_internal_keys(self):
        from src.models import RotationAction

        trim = RotationAction(kind="trim", chain="bsc", reason="", symbol="BREW",
                              pct_of_position=38.0)
        rotate = RotationAction(kind="rotate", chain="bsc", dest_chain="robinhood",
                                reason="", amount_usd=3_401.0)
        assert trim.headline == "Trim 38% of BREW on BNB Chain"
        assert "BNB Chain → Robinhood Chain" in rotate.headline
        assert "bsc" not in rotate.headline

    def test_a_cold_destination_is_not_described_as_still_cooling(self):
        heats = [
            ChainHeat(chain="bsc", heat=84.0, state="hot", confidence=1.0),
            ChainHeat(chain="base", heat=22.0, state="cold", confidence=1.0),
        ]
        snapshot = PortfolioSnapshot(positions=[
            position_for(value=5_000.0, change_24h=200.0, liquidity=9_000_000.0)])
        plan = rotation.build_rotation_plan(
            snapshot, heats, settings=config.RotationSettings(max_position_pct=100.0))
        reason = plan.actions_of("rotate")[0].reason
        assert "still cooling" not in reason
        assert "quiet for a while" in reason
