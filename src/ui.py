"""Reusable Streamlit rendering components.

Kept out of ``app.py`` so the entry point stays a thin controller and the
visual language (cards, score bars, badges) is defined in exactly one place.
"""

from __future__ import annotations

from typing import Iterable, List, Optional

import streamlit as st

from . import config
from .models import (AnalysisResult, EnsembleResult, MindshareReport, ScoreCard,
                     SecurityReport, TokenProfile, TokenSnapshot)
from .utils import fmt_number, fmt_pct, fmt_usd, score_color, score_emoji, short_address

# --------------------------------------------------------------------------
# Global styling
# --------------------------------------------------------------------------
CUSTOM_CSS = """
<style>
:root {
  --mdd-bg: #0b0f14;
  --mdd-card: #131a22;
  --mdd-card-2: #182029;
  --mdd-border: #243040;
  --mdd-text: #e6edf3;
  --mdd-muted: #8b98a9;
  --mdd-accent: #22d3ee;
}

/* Tighten Streamlit's default vertical rhythm so more fits on a phone. */
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1400px; }

.mdd-hero {
  background: linear-gradient(135deg, #131a22 0%, #0f1720 55%, #101b26 100%);
  border: 1px solid var(--mdd-border);
  border-radius: 16px;
  padding: 1.1rem 1.3rem;
  margin-bottom: 1rem;
}
.mdd-hero h1 { margin: 0; font-size: 1.65rem; letter-spacing: -0.02em; }
.mdd-hero p { margin: .35rem 0 0; color: var(--mdd-muted); font-size: .92rem; }

.mdd-card {
  background: var(--mdd-card);
  border: 1px solid var(--mdd-border);
  border-radius: 14px;
  padding: 1rem 1.15rem;
  margin-bottom: .85rem;
}
.mdd-card h3 { margin: 0 0 .6rem; font-size: 1.02rem; color: var(--mdd-text); }

.mdd-token-head { display: flex; align-items: center; gap: .85rem; flex-wrap: wrap; }
.mdd-token-logo { width: 46px; height: 46px; border-radius: 50%; border: 1px solid var(--mdd-border); object-fit: cover; }
.mdd-token-title { font-size: 1.35rem; font-weight: 700; margin: 0; line-height: 1.2; }
.mdd-token-sub { color: var(--mdd-muted); font-size: .82rem; font-family: ui-monospace, SFMono-Regular, monospace; word-break: break-all; }

.mdd-badge {
  display: inline-block; padding: .22rem .6rem; border-radius: 999px;
  font-size: .74rem; font-weight: 600; border: 1px solid var(--mdd-border);
  background: var(--mdd-card-2); color: var(--mdd-muted); margin-right: .35rem;
}
.mdd-decision {
  display: inline-block; padding: .45rem 1.1rem; border-radius: 999px;
  font-weight: 800; font-size: 1.02rem; letter-spacing: .01em;
}
.mdd-score-num { font-size: 2.6rem; font-weight: 800; line-height: 1; letter-spacing: -0.03em; }
.mdd-score-den { font-size: 1rem; color: var(--mdd-muted); font-weight: 500; }

.mdd-bar-wrap { margin: .5rem 0 .85rem; }
.mdd-bar-label { display: flex; justify-content: space-between; font-size: .84rem; margin-bottom: .28rem; gap: .5rem; }
.mdd-bar-label .mdd-bar-name { color: var(--mdd-text); font-weight: 600; }
.mdd-bar-label .mdd-bar-val { color: var(--mdd-muted); font-variant-numeric: tabular-nums; white-space: nowrap; }
.mdd-bar-track { height: 9px; background: #0d141c; border: 1px solid var(--mdd-border); border-radius: 999px; overflow: hidden; }
.mdd-bar-fill { height: 100%; border-radius: 999px; }

.mdd-kv { display: flex; justify-content: space-between; padding: .3rem 0; border-bottom: 1px dashed #1f2a36; font-size: .89rem; gap: .75rem; }
.mdd-kv:last-child { border-bottom: none; }
/* Direct-child selectors only: a nested flag <span> must keep its own colour. */
.mdd-kv > span:first-child { color: var(--mdd-muted); }
.mdd-kv > span:last-child { font-weight: 600; text-align: right; font-variant-numeric: tabular-nums; }

.mdd-flag-ok   { color: #22c55e; font-weight: 600; }
.mdd-flag-bad  { color: #ef4444; font-weight: 700; }
.mdd-flag-warn { color: #eab308; font-weight: 600; }
.mdd-flag-unk  { color: var(--mdd-muted); font-weight: 500; }

.mdd-note { color: var(--mdd-muted); font-size: .82rem; }
.mdd-disclaimer { color: var(--mdd-muted); font-size: .76rem; line-height: 1.5; border-top: 1px solid var(--mdd-border); padding-top: .8rem; margin-top: 1.4rem; }

.mdd-model-head { display: flex; align-items: baseline; justify-content: space-between; gap: .5rem; margin-bottom: .1rem; }
.mdd-model-name { font-weight: 700; font-size: .95rem; }
.mdd-model-meta { color: var(--mdd-muted); font-size: .72rem; font-family: ui-monospace, SFMono-Regular, monospace; }
.mdd-model-score { font-size: 1.9rem; font-weight: 800; line-height: 1.1; letter-spacing: -0.02em; }
.mdd-quote { border-left: 3px solid var(--mdd-border); padding: .1rem 0 .1rem .8rem; color: var(--mdd-text); font-size: .9rem; margin: .4rem 0; }
.mdd-agree-track { height: 7px; background: #0d141c; border: 1px solid var(--mdd-border); border-radius: 999px; overflow: hidden; margin-top: .3rem; }
.mdd-agree-fill { height: 100%; border-radius: 999px; }
.mdd-fail { color: #f97316; font-size: .82rem; }
.mdd-post { background: var(--mdd-card-2); border: 1px solid var(--mdd-border); border-radius: 10px;
            padding: .6rem .75rem; margin-bottom: .5rem; }
.mdd-post-handle { color: var(--mdd-accent); font-weight: 600; font-size: .82rem; }
.mdd-post-text { font-size: .87rem; margin-top: .25rem; line-height: 1.45; }
.mdd-post-meta { color: var(--mdd-muted); font-size: .74rem; margin-top: .3rem; }
.mdd-live { background: #052e16; color: #22c55e; border-color: #22c55e55; }
.mdd-stale { background: #2e2405; color: #eab308; border-color: #eab30855; }

/* Phones: stop Streamlit metric labels wrapping into unreadable slivers. */
@media (max-width: 640px) {
  .block-container { padding-left: .7rem; padding-right: .7rem; }
  .mdd-score-num { font-size: 2.1rem; }
  .mdd-hero h1 { font-size: 1.3rem; }
  [data-testid="stMetricValue"] { font-size: 1.05rem; }
}
</style>
"""


def inject_css() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="mdd-hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------
# Primitives
# --------------------------------------------------------------------------
def score_bar_html(label: str, score: float, caption: str = "") -> str:
    """HTML for a coloured 0-100 progress bar with a right-aligned caption.

    Returned as a string (rather than rendered) so several bars can be emitted
    inside one ``st.markdown`` call -- Streamlit closes unbalanced HTML at the
    end of every call, so a wrapper card must be written in a single block.
    """
    color = score_color(score)
    width = max(0.0, min(100.0, score))
    return (
        '<div class="mdd-bar-wrap">'
        '<div class="mdd-bar-label">'
        f'<span class="mdd-bar-name">{label}</span>'
        f'<span class="mdd-bar-val">{caption or f"{score:.0f}/100"}</span>'
        "</div>"
        '<div class="mdd-bar-track">'
        f'<div class="mdd-bar-fill" style="width:{width:.1f}%;background:{color};"></div>'
        "</div></div>"
    )


def score_bar(label: str, score: float, caption: str = "") -> None:
    """Render a single score bar."""
    st.markdown(score_bar_html(label, score, caption), unsafe_allow_html=True)


def kv_rows(pairs: Iterable[tuple]) -> None:
    """Compact key/value list used inside cards."""
    rows = "".join(
        f'<div class="mdd-kv"><span>{key}</span><span>{value}</span></div>' for key, value in pairs
    )
    st.markdown(f'<div class="mdd-card">{rows}</div>', unsafe_allow_html=True)


def decision_badge(decision: str, composite: float) -> str:
    colors = {
        "Strong Buy": ("#052e16", "#22c55e"),
        "Buy": ("#1a2e05", "#84cc16"),
        "Watch": ("#2e2405", "#eab308"),
        "Pass": ("#2e0505", "#ef4444"),
    }
    bg, fg = colors.get(decision, ("#1f2937", "#9ca3af"))
    return (
        f'<span class="mdd-decision" style="background:{bg};color:{fg};'
        f'border:1px solid {fg}55;">{score_emoji(composite)} {decision}</span>'
    )


def _flag(value: Optional[bool], good_when_false: bool = True, unknown_text: str = "unknown") -> str:
    """Render a tri-state security flag with the right colour semantics."""
    if value is None:
        return f'<span class="mdd-flag-unk">{unknown_text}</span>'
    is_good = (not value) if good_when_false else bool(value)
    css = "mdd-flag-ok" if is_good else "mdd-flag-bad"
    text = ("No" if not value else "Yes") if good_when_false else ("Yes" if value else "No")
    return f'<span class="{css}">{text}</span>'


# --------------------------------------------------------------------------
# Report sections
# --------------------------------------------------------------------------
def render_token_header(snapshot: TokenSnapshot, chain_key: str) -> None:
    chain_cfg = config.get_chain(chain_key)
    logo = (
        f'<img class="mdd-token-logo" src="{snapshot.image_url}" alt="">'
        if snapshot.image_url else ""
    )
    badges = [f'<span class="mdd-badge">{chain_cfg.label}</span>']
    if snapshot.dex_id:
        badges.append(f'<span class="mdd-badge">{snapshot.dex_id}</span>')
    badges.append(f'<span class="mdd-badge">{snapshot.age_label} old</span>')
    if snapshot.quote_symbol:
        badges.append(f'<span class="mdd-badge">vs {snapshot.quote_symbol}</span>')
    if snapshot.boosts:
        badges.append(f'<span class="mdd-badge">⚡ {snapshot.boosts} boosts</span>')

    st.markdown(
        f"""
        <div class="mdd-card">
          <div class="mdd-token-head">
            {logo}
            <div>
              <div class="mdd-token-title">{snapshot.symbol or "?"} · {snapshot.name or "Unknown token"}</div>
              <div class="mdd-token-sub">{snapshot.address}</div>
            </div>
          </div>
          <div style="margin-top:.6rem;">{"".join(badges)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    links: List[str] = []
    if snapshot.url:
        links.append(f"[DexScreener]({snapshot.url})")
    links.append(f"[Explorer]({chain_cfg.explorer_token_url.format(address=snapshot.address)})")
    for social in snapshot.socials[:5]:
        links.append(f"[{social.label or social.kind.title()}]({social.url})")
    st.caption(" · ".join(links))


def render_market_metrics(snapshot: TokenSnapshot) -> None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Price", fmt_usd(snapshot.price_usd, compact=False), fmt_pct(snapshot.price_change_24h, 1, True))
    col2.metric("Market cap", fmt_usd(snapshot.market_cap))
    col3.metric("Liquidity", fmt_usd(snapshot.liquidity_usd))
    col4.metric("24h volume", fmt_usd(snapshot.volume_24h))

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("FDV", fmt_usd(snapshot.fdv))
    col6.metric("Vol / MCap", f"{snapshot.turnover_24h:.2f}x")
    col7.metric("Holders", fmt_number(snapshot.holders) if snapshot.holders else "n/a")
    col8.metric("Age", snapshot.age_label)

    col9, col10, col11, col12 = st.columns(4)
    col9.metric("1h", fmt_pct(snapshot.price_change_1h, 1, True))
    col10.metric("6h", fmt_pct(snapshot.price_change_6h, 1, True))
    col11.metric("24h trades", fmt_number(snapshot.txns_24h))
    ratio = snapshot.buy_sell_ratio
    col12.metric("Buy share", fmt_pct(ratio * 100, 0) if ratio is not None else "n/a")


def render_scorecard(card: ScoreCard) -> None:
    left, right = st.columns([1, 2], gap="large")
    with left:
        color = score_color(card.composite)
        st.markdown(
            f"""
            <div class="mdd-card" style="text-align:center;">
              <div class="mdd-score-num" style="color:{color};">{card.composite:.0f}<span class="mdd-score-den">/100</span></div>
              <div style="margin-top:.7rem;">{decision_badge(card.decision, card.composite)}</div>
              <div class="mdd-note" style="margin-top:.7rem;">Confidence {card.confidence * 100:.0f}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if card.vetoed:
            st.error(f"**Security veto:** {card.veto_reason}", icon="🛑")

    with right:
        bars = "".join(
            score_bar_html(
                component.label,
                component.score,
                caption=f"{component.score:.0f}/100 · weight {component.weight * 100:.0f}% · +{component.weighted:.1f}",
            )
            for component in card.components
        )
        st.markdown(
            f'<div class="mdd-card"><h3>Score breakdown</h3>{bars}</div>',
            unsafe_allow_html=True,
        )

    with st.expander("Why each pillar scored the way it did"):
        for component in card.components:
            st.markdown(f"**{component.label}** — {component.score:.0f}/100 "
                        f"(confidence {component.confidence * 100:.0f}%)")
            for reason in component.reasons:
                st.markdown(f"- {reason}")
            st.markdown("")


def render_security(security: Optional[SecurityReport]) -> None:
    st.markdown("#### 🛡️ Rug & security checks")
    if security is None or not security.available:
        st.warning(
            (security.error if security and security.error else "No security data available.")
            + " Scores treat unknown contract capabilities as risk.",
            icon="⚠️",
        )
        return

    left, right = st.columns(2, gap="medium")
    with left:
        st.markdown(
            f"""
            <div class="mdd-card">
              <h3>Contract</h3>
              <div class="mdd-kv"><span>Honeypot</span><span>{_flag(security.is_honeypot)}</span></div>
              <div class="mdd-kv"><span>Can sell all</span><span>{_flag(security.cannot_sell_all)}</span></div>
              <div class="mdd-kv"><span>Source verified</span><span>{_flag(security.is_open_source, good_when_false=False)}</span></div>
              <div class="mdd-kv"><span>Ownership renounced</span><span>{_flag(security.owner_renounced, good_when_false=False)}</span></div>
              <div class="mdd-kv"><span>Mintable supply</span><span>{_flag(security.is_mintable)}</span></div>
              <div class="mdd-kv"><span>Transfers pausable</span><span>{_flag(security.transfer_pausable)}</span></div>
              <div class="mdd-kv"><span>Blacklist function</span><span>{_flag(security.is_blacklisted)}</span></div>
              <div class="mdd-kv"><span>Upgradeable proxy</span><span>{_flag(security.is_proxy)}</span></div>
              <div class="mdd-kv"><span>Buy tax</span><span>{fmt_pct(security.buy_tax_pct, 1)}</span></div>
              <div class="mdd-kv"><span>Sell tax</span><span>{fmt_pct(security.sell_tax_pct, 1)}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        secured = security.lp_secured_pct
        lp_text = fmt_pct(secured, 1) if secured is not None else "unknown"
        st.markdown(
            f"""
            <div class="mdd-card">
              <h3>Liquidity & holders</h3>
              <div class="mdd-kv"><span>LP burned/locked</span><span>{lp_text}</span></div>
              <div class="mdd-kv"><span>&nbsp;&nbsp;· burned</span><span>{fmt_pct(security.lp_burned_pct, 1)}</span></div>
              <div class="mdd-kv"><span>&nbsp;&nbsp;· locked</span><span>{fmt_pct(security.lp_locked_pct, 1)}</span></div>
              <div class="mdd-kv"><span>Top 10 (all)</span><span>{fmt_pct(security.top10_pct, 1)}</span></div>
              <div class="mdd-kv"><span>Top 10 (excl. LP/burn)</span><span>{fmt_pct(security.top10_pct_adjusted, 1)}</span></div>
              <div class="mdd-kv"><span>Deployer holds</span><span>{fmt_pct(security.creator_percent, 2)}</span></div>
              <div class="mdd-kv"><span>Holders</span><span>{fmt_number(security.holder_count) if security.holder_count else "n/a"}</span></div>
              <div class="mdd-kv"><span>LP holders</span><span>{fmt_number(security.lp_holder_count) if security.lp_holder_count else "n/a"}</span></div>
              <div class="mdd-kv"><span>Data source</span><span>{security.source}</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    for warning in security.warnings:
        st.error(warning, icon="🚨")
    if security.notes:
        with st.expander(f"Other contract notes ({len(security.notes)})"):
            for note in security.notes:
                st.markdown(f"- {note}")

    if security.top_holders:
        with st.expander(f"Top holders ({len(security.top_holders)})"):
            import pandas as pd

            frame = pd.DataFrame(
                [
                    {
                        "Address": short_address(h.address, 10, 6),
                        "%": round(h.percent, 3),
                        "Tag": h.tag or "",
                        "Contract": "yes" if h.is_contract else "no",
                        "Locked": "yes" if h.is_locked else "no",
                    }
                    for h in security.top_holders
                ]
            )
            st.dataframe(frame, use_container_width=True, hide_index=True)


def render_narrative(result: AnalysisResult) -> None:
    narrative = result.narrative
    if narrative is None:
        return
    label = narrative.source if narrative.source != "heuristic" else "heuristic placeholder"
    st.markdown(f"#### 📖 Lore & narrative <span class='mdd-badge'>{label}</span>", unsafe_allow_html=True)
    st.markdown(f'<div class="mdd-card">{narrative.summary}</div>', unsafe_allow_html=True)
    if narrative.themes:
        st.markdown(
            " ".join(f'<span class="mdd-badge">{theme}</span>' for theme in narrative.themes),
            unsafe_allow_html=True,
        )
    left, right = st.columns(2, gap="medium")
    with left:
        st.markdown("**Bull case**")
        for item in narrative.bull_case or ["Nothing compelling found."]:
            st.markdown(f"- {item}")
    with right:
        st.markdown("**Bear case**")
        for item in narrative.bear_case or ["No specific bear points found."]:
            st.markdown(f"- {item}")
    if narrative.mindshare_notes:
        with st.expander("Mindshare notes"):
            for note in narrative.mindshare_notes:
                st.markdown(f"- {note}")


def render_risk_plan(result: AnalysisResult) -> None:
    plan = result.risk_plan
    if plan is None:
        return
    st.markdown(f"#### 🎯 Risk management <span class='mdd-badge'>{plan.risk_profile}</span>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    # delta_color="off": these sub-labels are context, not period-over-period
    # changes, so Streamlit's red/green arrows would be misleading here.
    col1.metric("Suggested position", fmt_usd(plan.position_usd, compact=False),
                f"{plan.position_pct:.2f}% of portfolio", delta_color="off")
    col2.metric("Stop loss", f"-{plan.stop_loss_pct:.1f}%",
                fmt_usd(plan.stop_price, compact=False) if plan.stop_price else "n/a", delta_color="off")
    col3.metric("Max loss if stopped", fmt_usd(plan.max_loss_usd, compact=False),
                f"{plan.max_loss_pct_of_portfolio:.2f}% of portfolio", delta_color="off")
    col4.metric("Entry price impact", f"~{plan.est_slippage_pct:.2f}%",
                f"{plan.liquidity_share_pct:.2f}% of pool", delta_color="off")

    for warning in plan.warnings:
        st.warning(warning, icon="⚠️")

    if plan.take_profit_targets:
        import pandas as pd

        frame = pd.DataFrame(
            [
                {
                    "Target": t["label"],
                    "R": f"{t['r_multiple']:.1f}R",
                    "Gain": f"+{t['gain_pct']:.0f}%",
                    "Price": fmt_usd(t["price"], compact=False),
                    "Sell": f"{t['sell_portion_pct']}%",
                    "Profit": fmt_usd(t["profit_usd"], compact=False),
                }
                for t in plan.take_profit_targets
            ]
        )
        st.dataframe(frame, use_container_width=True, hide_index=True)

    with st.expander("How this size was calculated"):
        for note in plan.notes:
            st.markdown(f"- {note}")
        st.markdown(
            "- Sizing rule: risk budget ÷ stop distance, scaled by conviction, "
            "then capped by both your max position size and a share of pool liquidity."
        )


def render_pros_cons(card: ScoreCard) -> None:
    left, right = st.columns(2, gap="medium")
    with left:
        with st.expander(f"✅ Positives ({len(card.positives)})", expanded=bool(card.positives)):
            for item in card.positives or ["Nothing notable in this token's favour."]:
                st.markdown(f"- {item}")
    with right:
        with st.expander(f"🚩 Key risks ({len(card.risks)})", expanded=bool(card.risks)):
            for item in card.risks or ["No specific risks flagged - which is itself unusual."]:
                st.markdown(f"- {item}")


def render_profile(profile: Optional[TokenProfile]) -> None:
    """Show the DexScreener token profile (project-supplied description/links)."""
    if profile is None or not profile.has_content:
        return
    links = " · ".join(f"[{link.label or link.kind.title()}]({link.url})" for link in profile.links[:6])
    st.markdown("#### 🪪 Project profile <span class='mdd-badge'>dexscreener</span>", unsafe_allow_html=True)
    if profile.description:
        st.markdown(f'<div class="mdd-card"><div class="mdd-quote">{profile.description}</div></div>',
                    unsafe_allow_html=True)
    else:
        st.caption("Profile claimed, but no description published.")
    if links:
        st.caption(links)


def _agreement_bar(agreement: float) -> str:
    color = score_color(agreement * 100)
    return (
        f'<div class="mdd-agree-track"><div class="mdd-agree-fill" '
        f'style="width:{max(0.0, min(1.0, agreement)) * 100:.0f}%;background:{color};"></div></div>'
    )


_SENTIMENT_COLORS = {
    "bullish": "#22c55e", "mixed": "#eab308", "bearish": "#ef4444",
    "quiet": "#8b98a9", "unknown": "#8b98a9",
}
_VOLUME_BARS = {"none": 5, "low": 25, "moderate": 55, "high": 80, "viral": 100}


def render_mindshare(mindshare: Optional[MindshareReport]) -> None:
    """Render the X/Twitter mindshare panel from Grok's search."""
    if mindshare is None:
        return

    st.markdown("#### 𝕏 Mindshare <span class='mdd-badge'>grok</span>", unsafe_allow_html=True)

    if not mindshare.available:
        st.info(
            (mindshare.error or "X mindshare was not requested.")
            + "  Run `python scripts/check_grok.py` to diagnose.",
            icon="𝕏",
        )
        return

    # The live/stale distinction is the most important thing on this panel:
    # stale model knowledge must never read as current sentiment.
    if mindshare.is_live:
        badge = '<span class="mdd-badge mdd-live">● LIVE X SEARCH</span>'
    else:
        badge = '<span class="mdd-badge mdd-stale">⚠ NOT LIVE — model knowledge</span>'

    color = _SENTIMENT_COLORS.get(mindshare.sentiment, "#8b98a9")
    organic = (
        "organic" if mindshare.is_organic
        else ("coordinated / bots" if mindshare.is_organic is False else "unclear")
    )

    left, right = st.columns([1, 2], gap="large")
    with left:
        st.markdown(
            f"""
            <div class="mdd-card" style="text-align:center;">
              <div>{badge}</div>
              <div class="mdd-score-num" style="color:{color};margin-top:.6rem;">
                {mindshare.mindshare_score:.0f}<span class="mdd-score-den">/100</span>
              </div>
              <div class="mdd-note">attention right now</div>
              <div style="margin-top:.7rem;">
                <span class="mdd-badge" style="color:{color};border-color:{color}55;">
                  {mindshare.sentiment.title()}
                </span>
                <span class="mdd-badge">{mindshare.post_volume} volume</span>
              </div>
              <div class="mdd-note" style="margin-top:.6rem;">
                trend: {mindshare.trend} · {organic}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        bars = score_bar_html(
            "Post volume", _VOLUME_BARS.get(mindshare.post_volume, 0),
            caption=mindshare.post_volume,
        ) + score_bar_html(
            "Sentiment", (mindshare.sentiment_score + 1) / 2 * 100,
            caption=f"{mindshare.sentiment_score:+.2f}",
        ) + score_bar_html(
            "Attention", mindshare.mindshare_score, caption=f"{mindshare.mindshare_score:.0f}/100",
        )
        summary = mindshare.summary or "No summary returned."
        st.markdown(
            f'<div class="mdd-card"><h3>What X is saying</h3>'
            f'<div class="mdd-quote">{summary}</div>{bars}</div>',
            unsafe_allow_html=True,
        )

    if mindshare.themes:
        st.markdown(
            " ".join(f'<span class="mdd-badge">{theme}</span>' for theme in mindshare.themes),
            unsafe_allow_html=True,
        )

    for flag in mindshare.red_flags:
        st.error(f"X red flag: {flag}", icon="🚩")
    if mindshare.is_organic is False:
        st.warning(
            "This discussion reads as coordinated rather than organic — the momentum "
            "score has been discounted accordingly.",
            icon="🤖",
        )
    for warning in mindshare.warnings:
        st.warning(warning, icon="⚠️")

    if mindshare.sample_posts:
        with st.expander(f"Sample posts ({len(mindshare.sample_posts)})", expanded=True):
            for post in mindshare.sample_posts:
                meta = []
                if post.engagement:
                    meta.append(f"{post.engagement:,} engagements")
                if post.posted_at:
                    meta.append(post.posted_at)
                link = f' · <a href="{post.url}" target="_blank">open</a>' if post.url else ""
                st.markdown(
                    f"""
                    <div class="mdd-post">
                      <div class="mdd-post-handle">@{post.handle or "unknown"}</div>
                      <div class="mdd-post-text">{post.text}</div>
                      <div class="mdd-post-meta">{" · ".join(meta)}{link}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            st.caption(
                "Posts are reported by the model from its search. Spot-check anything "
                "you intend to act on — models can paraphrase or misattribute."
            )

    if mindshare.notable_accounts:
        st.caption("Notable accounts: " + ", ".join(f"@{h}" for h in mindshare.notable_accounts))
    footer = [f"Query: `{mindshare.query}`", f"{mindshare.latency_ms / 1000:.1f}s", mindshare.model]
    st.caption(" · ".join(x for x in footer if x))


def render_ensemble(ensemble: Optional[EnsembleResult], deterministic: Optional[ScoreCard] = None) -> None:
    """Render the multi-model panel: consensus, per-model cards, dissent."""
    if ensemble is None:
        return

    st.markdown("#### 🤖 Multi-LLM ensemble")

    if not ensemble.ok:
        st.info(
            "No model verdicts. "
            + (" ".join(ensemble.notes) if ensemble.notes else "Add an API key to .env to enable the ensemble."),
            icon="🤖",
        )
        return

    consensus = ensemble.consensus
    if consensus is not None:
        left, right = st.columns([1, 2], gap="large")
        with left:
            color = score_color(consensus.overall_score)
            blended_block = ""
            if ensemble.blended_score is not None:
                blend_color = score_color(ensemble.blended_score)
                blended_block = f"""
                  <div style="border-top:1px solid var(--mdd-border);margin-top:.9rem;padding-top:.7rem;">
                    <div class="mdd-note">BLENDED WITH RULES ENGINE</div>
                    <div style="font-size:1.6rem;font-weight:800;color:{blend_color};line-height:1.2;">
                      {ensemble.blended_score:.0f}<span class="mdd-score-den">/100</span>
                    </div>
                    <div class="mdd-note">
                      {config.DECISION_LABELS.get(ensemble.blended_decision, ensemble.blended_decision)}
                      · {ensemble.blend_weight * 100:.0f}% LLM weight
                    </div>
                  </div>
                """
            st.markdown(
                f"""
                <div class="mdd-card" style="text-align:center;">
                  <div class="mdd-note">CONSENSUS OF {consensus.model_count} MODEL(S)</div>
                  <div class="mdd-score-num" style="color:{color};">{consensus.overall_score:.0f}<span class="mdd-score-den">/100</span></div>
                  <div style="margin-top:.7rem;">{decision_badge(consensus.decision_label, consensus.overall_score)}</div>
                  <div class="mdd-note" style="margin-top:.7rem;">
                    Confidence {consensus.confidence * 100:.0f}% · agreement {consensus.agreement * 100:.0f}%
                  </div>
                  {_agreement_bar(consensus.agreement)}
                  {blended_block}
                </div>
                """,
                unsafe_allow_html=True,
            )

        with right:
            bars = "".join(
                score_bar_html(
                    config.COMPONENT_LABELS.get(dim, dim.title()),
                    consensus.dimension_scores.get(dim, 0.0) * 10,   # 0-10 -> 0-100
                    caption=f"{consensus.dimension_scores.get(dim, 0.0):.1f}/10",
                )
                for dim in config.LLM_DIMENSIONS
            )
            st.markdown(
                f'<div class="mdd-card"><h3>Consensus dimensions</h3>{bars}</div>',
                unsafe_allow_html=True,
            )

        if consensus.corroborated_rug_flags:
            st.error(
                "**Rug flags raised by more than one model:** "
                + "; ".join(consensus.corroborated_rug_flags),
                icon="🚨",
            )
        elif consensus.rug_flags:
            st.warning(
                "**Rug flags (single model each — verify):** " + "; ".join(consensus.rug_flags),
                icon="⚠️",
            )

        for note in consensus.dissent:
            st.warning(note, icon="⚖️")

        if deterministic is not None:
            gap = consensus.overall_score - deterministic.composite
            if abs(gap) >= 20:
                direction = "more bullish than" if gap > 0 else "more bearish than"
                st.info(
                    f"The models are {abs(gap):.0f} points {direction} the rules engine "
                    f"({consensus.overall_score:.0f} vs {deterministic.composite:.0f}). "
                    "Worth reading both rationales before acting.",
                    icon="🔍",
                )

    # -- per-model cards ------------------------------------------------
    st.markdown("##### Individual model verdicts")
    verdicts = ensemble.verdicts
    columns = st.columns(min(3, max(1, len(verdicts))), gap="medium")
    for index, verdict in enumerate(verdicts):
        with columns[index % len(columns)]:
            if not verdict.ok:
                st.markdown(
                    f"""
                    <div class="mdd-card">
                      <div class="mdd-model-head"><span class="mdd-model-name">{verdict.provider}</span></div>
                      <div class="mdd-model-meta">{verdict.model}</div>
                      <div class="mdd-fail" style="margin-top:.6rem;">Failed: {verdict.error}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                continue
            color = score_color(verdict.overall_score)
            st.markdown(
                f"""
                <div class="mdd-card">
                  <div class="mdd-model-head">
                    <span class="mdd-model-name">{verdict.provider}</span>
                    <span class="mdd-model-meta">{verdict.latency_ms / 1000:.1f}s</span>
                  </div>
                  <div class="mdd-model-meta">{verdict.model}</div>
                  <div class="mdd-model-score" style="color:{color};margin-top:.5rem;">{verdict.overall_score:.0f}</div>
                  <div style="margin-top:.4rem;">
                    <span class="mdd-badge">{verdict.decision_label}</span>
                    <span class="mdd-badge">conf {verdict.confidence * 100:.0f}%</span>
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.expander(f"{verdict.provider} detail"):
                if verdict.lore_summary:
                    st.markdown(f'<div class="mdd-quote">{verdict.lore_summary}</div>', unsafe_allow_html=True)
                st.markdown(
                    " ".join(
                        f'<span class="mdd-badge">{config.COMPONENT_LABELS.get(dim, dim)} '
                        f'{verdict.dimension_scores.get(dim, 0):.0f}/10</span>'
                        for dim in config.LLM_DIMENSIONS
                    ),
                    unsafe_allow_html=True,
                )
                if verdict.rationale:
                    st.markdown(f"**Rationale.** {verdict.rationale}")
                if verdict.key_positives:
                    st.markdown("**Positives**")
                    for item in verdict.key_positives:
                        st.markdown(f"- {item}")
                if verdict.key_risks:
                    st.markdown("**Risks**")
                    for item in verdict.key_risks:
                        st.markdown(f"- {item}")
                if verdict.rug_flags:
                    st.markdown("**Rug flags**")
                    for item in verdict.rug_flags:
                        st.markdown(f"- 🚩 {item}")
                if verdict.warnings:
                    st.caption("Response repairs: " + "; ".join(verdict.warnings))

    if consensus is not None:
        with st.expander("Merged positives, risks and consensus rationale"):
            st.markdown(f"**Consensus rationale.** {consensus.rationale}")
            left, right = st.columns(2, gap="medium")
            with left:
                st.markdown("**Positives (most corroborated first)**")
                for item in consensus.key_positives or ["None offered."]:
                    st.markdown(f"- {item}")
            with right:
                st.markdown("**Risks (most corroborated first)**")
                for item in consensus.key_risks or ["None offered."]:
                    st.markdown(f"- {item}")

    footer = [f"Ran in {ensemble.elapsed_ms / 1000:.1f}s"]
    if ensemble.skipped:
        footer.append(
            "skipped: " + ", ".join(f"{name} ({reason})" for name, reason in ensemble.skipped.items())
        )
    st.caption(" · ".join(footer))


def disclaimer() -> None:
    st.markdown(
        '<div class="mdd-disclaimer">MemeDD Dashboard is automated research tooling, not financial '
        'advice. Data comes from third-party APIs (DexScreener, GoPlus) that can be wrong, stale or '
        'gamed. Meme coins are adversarial and frequently fraudulent — verify everything yourself and '
        'never risk money you cannot afford to lose entirely.</div>',
        unsafe_allow_html=True,
    )
