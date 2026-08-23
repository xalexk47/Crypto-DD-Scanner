"""MemeDD Dashboard - meme-coin due diligence and scanning for Base & friends.

Run with::

    streamlit run app.py

This module is deliberately thin: it wires Streamlit widgets to the pipeline in
``src/`` and delegates all rendering to ``src.ui``.  Every piece of business
logic lives in importable, testable modules.
"""

from __future__ import annotations

import logging
from typing import List

import pandas as pd
import streamlit as st

from src import config, data_fetchers, history, llm, llm_analyzers, mindshare, report, ui
from src.analyzer import analyze_many, analyze_token, scan
from src.models import AnalysisResult, ScanCandidate
from src.utils import fmt_usd, parse_addresses, score_emoji, short_address

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

st.set_page_config(
    page_title="MemeDD Dashboard",
    page_icon="🧪",
    layout="wide",
    # "auto" keeps the sidebar open on desktop but collapses it on phones,
    # where an expanded sidebar covers the entire viewport.
    initial_sidebar_state="auto",
)


# ==========================================================================
# Cached pipeline wrappers
# ==========================================================================
# Streamlit's cache sits on top of the fetchers' own TTL cache: this one keeps
# reruns instant, the inner one keeps us polite to the upstream APIs.
@st.cache_data(ttl=config.CACHE_TTL_TOKEN, show_spinner=False)
def cached_analyze(
    address: str, chain: str, portfolio_usd: float, risk_profile: str, use_llm: bool,
    use_ensemble: bool, providers: tuple, blend_weight: float, use_x_search: bool, nonce: int
) -> AnalysisResult:
    """Analyze one address.  ``nonce`` busts the cache on a manual refresh."""
    settings = config.AppSettings(
        chain=chain, portfolio_usd=portfolio_usd, risk_profile=risk_profile, use_llm=use_llm,
        use_ensemble=use_ensemble, ensemble_providers=providers, blend_weight=blend_weight,
        use_x_search=use_x_search,
    )
    return analyze_token(address, settings)


@st.cache_data(ttl=config.CACHE_TTL_TOKEN, show_spinner=False)
def cached_analyze_many(
    addresses: List[str], chain: str, portfolio_usd: float, risk_profile: str, use_llm: bool,
    use_ensemble: bool, providers: tuple, blend_weight: float, use_x_search: bool, nonce: int
) -> List[AnalysisResult]:
    settings = config.AppSettings(
        chain=chain, portfolio_usd=portfolio_usd, risk_profile=risk_profile, use_llm=use_llm,
        use_ensemble=use_ensemble, ensemble_providers=providers, blend_weight=blend_weight,
        use_x_search=use_x_search,
    )
    return analyze_many(addresses, settings)


@st.cache_data(ttl=config.CACHE_TTL_SCANNER, show_spinner=False)
def cached_scan(
    chain: str,
    min_market_cap: float,
    max_market_cap: float,
    min_liquidity: float,
    min_volume_24h: float,
    min_age_hours: float,
    max_age_days: float,
    min_txns_24h: int,
    exclude_no_socials: bool,
    max_results: int,
    nonce: int,
):
    filters = config.ScannerFilters(
        chain=chain,
        min_market_cap=min_market_cap,
        max_market_cap=max_market_cap,
        min_liquidity=min_liquidity,
        min_volume_24h=min_volume_24h,
        min_age_hours=min_age_hours,
        max_age_days=max_age_days,
        min_txns_24h=min_txns_24h,
        exclude_no_socials=exclude_no_socials,
        max_results=max_results,
    )
    return scan(filters)


@st.cache_data(ttl=config.CACHE_TTL_SCANNER, show_spinner=False)
def cached_profiles(chain: str, nonce: int):
    """Latest DexScreener token profiles for one chain."""
    return data_fetchers.fetch_latest_profiles(chain)


# ==========================================================================
# Session state
# ==========================================================================
def init_state() -> None:
    defaults = {
        "results": [],            # List[AnalysisResult] currently displayed
        "scan_results": [],       # List[ScanCandidate]
        "scan_warnings": [],
        "pending_address": "",    # set when drilling in from Scanner/History
        "cache_nonce": 0,         # bumped to force a refetch
        "active_tab": "analyzer",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


# ==========================================================================
# Sidebar
# ==========================================================================
def render_sidebar() -> config.AppSettings:
    with st.sidebar:
        st.markdown("### ⚙️ Settings")

        chain_keys = list(config.CHAINS.keys())
        chain = st.selectbox(
            "Chain",
            chain_keys,
            index=chain_keys.index(config.DEFAULT_CHAIN),
            format_func=lambda key: config.CHAINS[key].label,
            help="Base is the default target. Everything else works too — the pipeline is chain-agnostic.",
        )
        # Some chains have no contract-security provider. Say so up front rather
        # than letting a blank rug-check section imply a clean bill of health.
        if chain in config.SECURITY_PROVIDER_NOTES:
            st.warning(config.SECURITY_PROVIDER_NOTES[chain], icon="⚠️")

        st.markdown("#### Risk profile")
        portfolio_usd = st.number_input(
            "Portfolio size (USD)",
            min_value=0.0,
            value=float(config.DEFAULT_PORTFOLIO_USD),
            step=500.0,
            help="Used to size positions. Nothing is stored or transmitted.",
        )
        profile_keys = list(config.RISK_PROFILES.keys())
        risk_profile = st.select_slider(
            "Risk tolerance",
            options=profile_keys,
            value=config.DEFAULT_RISK_PROFILE,
            format_func=lambda key: config.RISK_PROFILES[key].label,
        )
        profile = config.RISK_PROFILES[risk_profile]
        st.caption(
            f"Risking **{profile.risk_per_trade_pct:.2f}%** of portfolio per idea · "
            f"max position **{profile.max_position_pct:.1f}%** · "
            f"baseline stop **{profile.base_stop_pct:.0f}%** · "
            f"max **{profile.max_liquidity_share_pct:.2f}%** of pool liquidity."
        )

        st.markdown("#### Narrative engine")
        st.caption(llm.llm_status())
        use_llm = st.toggle(
            "Use LLM for lore analysis",
            value=False,
            help="Single-model narrative. Requires an API key in .env; falls back to heuristics automatically.",
        )

        st.markdown("#### 𝕏 Mindshare (Grok)")
        x_reason = mindshare.GrokMindshareClient().unavailable_reason()
        st.caption(("✅ Grok X search ready" if not x_reason else f"⚪ {x_reason}"))
        use_x_search = st.toggle(
            "Query X via Grok",
            value=False,
            disabled=bool(x_reason),
            help=(
                "Searches X for live discussion of the token and blends real mindshare "
                "into the momentum score. Billed per search, so results are cached."
                if not x_reason else x_reason
            ),
        )

        st.markdown("#### 🤖 Multi-LLM ensemble")
        statuses = llm_analyzers.provider_statuses()
        ready_providers = [s_.provider for s_ in statuses if s_.ready]
        for status in statuses:
            label = {"xai": "Grok (xAI)", "anthropic": "Claude", "openai": "GPT"}[status.provider]
            st.caption(("✅ " if status.ready else "⚪ ") + f"**{label}** — "
                       + ("ready" if status.ready else status.reason))

        use_ensemble = st.toggle(
            "Run ensemble on analysis",
            value=False,
            disabled=not ready_providers,
            help=(
                "Sends the same payload to every ready model in parallel and combines the verdicts."
                if ready_providers
                else "Add ANTHROPIC_API_KEY / OPENAI_API_KEY / XAI_API_KEY to .env to enable."
            ),
        )
        selected_providers = tuple(
            st.multiselect(
                "Models to query",
                options=ready_providers,
                default=ready_providers,
                format_func=lambda key: {"xai": "Grok", "anthropic": "Claude", "openai": "GPT"}[key],
                disabled=not use_ensemble,
            )
        ) if ready_providers else tuple()
        blend_weight = st.slider(
            "LLM weight in blended score",
            0.0, 1.0, float(config.ENSEMBLE_BLEND_WEIGHT), 0.05,
            disabled=not use_ensemble,
            help="0 = rules engine only, 1 = models only. The security veto always wins regardless.",
        )
        if use_ensemble and not selected_providers:
            st.caption("⚠️ Pick at least one model, or the ensemble will be skipped.")

        with st.expander("Advanced: score weights"):
            st.caption("Weights are normalised to 100%. Defaults follow the spec.")
            weights_raw = {
                key: st.slider(
                    label, 0, 60,
                    int(getattr(config.DEFAULT_WEIGHTS, key) * 100),
                    step=5, key=f"weight_{key}",
                )
                for key, label in config.COMPONENT_LABELS.items()
            }
            total = sum(weights_raw.values()) or 1
            weights = config.ScoreWeights(**{key: value / total for key, value in weights_raw.items()})
            if total != 100:
                st.caption(f"Raw total {total}% — normalised back to 100%.")

        st.divider()
        if st.button("🔄 Refresh data (clear caches)", use_container_width=True):
            st.cache_data.clear()
            data_fetchers.clear_caches()
            mindshare.clear_cache()
            st.session_state["cache_nonce"] += 1
            st.toast("Caches cleared — next request hits the APIs live.")

        st.caption(
            f"Market data cached {config.CACHE_TTL_TOKEN}s · scanner {config.CACHE_TTL_SCANNER}s · "
            f"security {config.CACHE_TTL_SECURITY}s."
        )

    return config.AppSettings(
        chain=chain,
        portfolio_usd=portfolio_usd,
        risk_profile=risk_profile,
        weights=weights,
        use_llm=use_llm,
        use_x_search=bool(use_x_search),
        use_ensemble=bool(use_ensemble and selected_providers),
        ensemble_providers=selected_providers,
        blend_weight=blend_weight,
    )


# ==========================================================================
# Report rendering
# ==========================================================================
def render_result(result: AnalysisResult, settings: config.AppSettings) -> None:
    """Render one full token report."""
    if not result.ok or result.snapshot is None:
        st.error(f"**{short_address(result.address)}** — {result.error}", icon="❌")
        return

    ui.render_token_header(result.snapshot, result.chain)
    ui.render_market_metrics(result.snapshot)
    st.markdown("")

    if result.scorecard:
        ui.render_scorecard(result.scorecard)
        ui.render_pros_cons(result.scorecard)

    st.markdown("")
    ui.render_security(result.security)
    st.markdown("")
    ui.render_profile(result.profile)
    ui.render_narrative(result)
    st.markdown("")
    ui.render_mindshare(result.mindshare)
    st.markdown("")
    if result.ensemble is not None:
        ui.render_ensemble(result.ensemble, result.scorecard)
        st.markdown("")
    ui.render_risk_plan(result)

    for warning in result.data_warnings:
        st.info(warning, icon="ℹ️")

    col1, col2, _ = st.columns([1, 1, 3])
    col1.download_button(
        "⬇️ Markdown",
        data=report.to_markdown(result),
        file_name=report.filename_for(result, "md"),
        mime="text/markdown",
        use_container_width=True,
        key=f"md_{result.address}",
    )
    col2.download_button(
        "⬇️ JSON",
        data=report.to_json(result),
        file_name=report.filename_for(result, "json"),
        mime="application/json",
        use_container_width=True,
        key=f"json_{result.address}",
    )


def run_analysis(addresses: List[str], settings: config.AppSettings) -> None:
    """Analyze addresses, store them in session state and persist to history."""
    if not addresses:
        return
    label = addresses[0] if len(addresses) == 1 else f"{len(addresses)} tokens"
    spinner = f"Fetching market data, running security checks and scoring {label}…"
    if settings.use_x_search:
        spinner = f"Analyzing {label} and searching X via Grok…"
    if settings.use_ensemble:
        spinner = (
            f"Analyzing {label} and querying "
            f"{len(settings.ensemble_providers)} model(s) in parallel…"
        )
    with st.spinner(spinner):
        args = (
            settings.chain, settings.portfolio_usd, settings.risk_profile, settings.use_llm,
            settings.use_ensemble, tuple(settings.ensemble_providers), settings.blend_weight,
            settings.use_x_search, st.session_state["cache_nonce"],
        )
        if len(addresses) == 1:
            results = [cached_analyze(addresses[0], *args)]
        else:
            results = cached_analyze_many(addresses, *args)
    for result in results:
        history.record(result)
    st.session_state["results"] = results


# ==========================================================================
# Tab: CA Analyzer
# ==========================================================================
def tab_analyzer(settings: config.AppSettings) -> None:
    st.markdown("#### 🔍 Contract address analyzer")
    st.caption(
        "Paste one or more contract addresses — newline, comma or space separated. "
        "DexScreener links work too."
    )

    prefill = st.session_state.pop("pending_address", "") or ""
    raw = st.text_area(
        "Contract address(es)",
        value=prefill,
        height=110,
        placeholder="0x532f27101965dd16442E59d40670FaF5eBB142E4\n0x4ed4E862860beD51a9570b96d89aF5E1B0Efefed",
        label_visibility="collapsed",
        key="ca_input",
    )

    col1, col2 = st.columns([1, 4])
    analyze_clicked = col1.button("🚀 Analyze", type="primary", use_container_width=True)

    addresses, invalid = parse_addresses(raw)
    if invalid:
        col2.caption(f"Ignoring {len(invalid)} unrecognised entr{'y' if len(invalid) == 1 else 'ies'}: "
                     + ", ".join(short_address(item) for item in invalid[:5]))

    # Auto-run when we arrived here from a Scanner/History drill-in.
    if (analyze_clicked or (prefill and not st.session_state["results"])) and addresses:
        run_analysis(addresses[:10], settings)
    elif analyze_clicked and not addresses:
        st.warning("No valid contract address found. EVM addresses look like `0x…40 hex chars`.", icon="⚠️")

    results: List[AnalysisResult] = st.session_state.get("results", [])
    if not results:
        st.info(
            "Paste a contract address above, or head to **Scanner** to find candidates automatically.",
            icon="👋",
        )
        return

    st.divider()
    if len(results) > 1:
        summary = pd.DataFrame(
            [
                {
                    "": score_emoji(r.composite),
                    "Symbol": (r.snapshot.symbol if r.snapshot else "?") or "?",
                    "Score": r.composite,
                    "Decision": r.decision,
                    "LLM": (
                        round(r.ensemble.consensus.overall_score, 1)
                        if r.ensemble and r.ensemble.consensus else None
                    ),
                    "MCap": fmt_usd(r.snapshot.market_cap) if r.snapshot else "n/a",
                    "Liquidity": fmt_usd(r.snapshot.liquidity_usd) if r.snapshot else "n/a",
                    "Address": short_address(r.address, 8, 6),
                    "Status": "ok" if r.ok else r.error[:60],
                }
                for r in results
            ]
        ).sort_values("Score", ascending=False)
        st.markdown("##### Batch summary")
        st.dataframe(summary, use_container_width=True, hide_index=True)
        st.divider()

        for result in sorted(results, key=lambda r: r.composite, reverse=True):
            title = f"{score_emoji(result.composite)} {result.display_name} — {result.composite:.0f}/100 · {result.decision}"
            with st.expander(title, expanded=len(results) <= 3):
                render_result(result, settings)
    else:
        render_result(results[0], settings)


# ==========================================================================
# Tab: Scanner
# ==========================================================================
def tab_scanner(settings: config.AppSettings) -> None:
    st.markdown("#### 📡 Scanner")
    st.caption(
        "Sweeps DexScreener search, boosted tokens and new token profiles for the selected chain, "
        "then filters and ranks what comes back. Results are cached for a few minutes."
    )

    with st.expander("Filters", expanded=True):
        col1, col2, col3 = st.columns(3)
        min_mc, max_mc = col1.slider(
            "Market cap (USD)",
            min_value=50_000, max_value=50_000_000,
            value=(500_000, 5_000_000), step=50_000,
            format="$%d",
        )
        min_liq = col2.number_input("Min liquidity (USD)", min_value=0, value=50_000, step=10_000)
        min_vol = col3.number_input("Min 24h volume (USD)", min_value=0, value=100_000, step=25_000)

        col4, col5, col6 = st.columns(3)
        min_age = col4.number_input("Min age (hours)", min_value=0.0, value=6.0, step=1.0)
        max_age = col5.number_input("Max age (days)", min_value=1.0, value=365.0, step=7.0)
        min_txns = col6.number_input("Min 24h trades", min_value=0, value=100, step=50)

        col7, col8 = st.columns(2)
        exclude_no_socials = col7.checkbox("Require socials", value=False)
        max_results = col8.slider("Max results", 5, 100, 40, step=5)

    if st.button("🔎 Run scan", type="primary"):
        with st.spinner(f"Scanning {config.get_chain(settings.chain).label} for candidates…"):
            candidates, warnings = cached_scan(
                settings.chain, float(min_mc), float(max_mc), float(min_liq), float(min_vol),
                float(min_age), float(max_age), int(min_txns), bool(exclude_no_socials),
                int(max_results), st.session_state["cache_nonce"],
            )
        st.session_state["scan_results"] = candidates
        st.session_state["scan_warnings"] = warnings

    candidates: List[ScanCandidate] = st.session_state.get("scan_results", [])
    for warning in st.session_state.get("scan_warnings", []):
        st.caption(f"ℹ️ {warning}")

    if not candidates:
        st.info(
            "Run a scan to populate the table. If nothing comes back, loosen the filters — "
            "the $500k–$5M band with real volume is genuinely narrow.",
            icon="📡",
        )
        return

    frame = pd.DataFrame([candidate.to_row() for candidate in candidates])
    st.markdown(f"##### {len(candidates)} candidates on {config.get_chain(settings.chain).label}")
    st.dataframe(
        frame,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.0f"),
            # %g keeps significant digits without a wall of trailing zeros.
            "Price": st.column_config.NumberColumn("Price", format="$%.6g"),
            "MCap": st.column_config.NumberColumn("MCap", format="$%.0f"),
            "Liq": st.column_config.NumberColumn("Liq", format="$%.0f"),
            "Vol 24h": st.column_config.NumberColumn("Vol 24h", format="$%.0f"),
            "1h %": st.column_config.NumberColumn("1h %", format="%.1f%%"),
            "24h %": st.column_config.NumberColumn("24h %", format="%.1f%%"),
            "Address": st.column_config.TextColumn("Address", width="small"),
        },
    )

    st.markdown("##### Deep-analyze a candidate")
    options = list(range(len(candidates)))
    selected = st.selectbox(
        "Token",
        options,
        format_func=lambda index: (
            f"{candidates[index].quick_score:.0f} · {candidates[index].snapshot.symbol or '?'} "
            f"· {fmt_usd(candidates[index].snapshot.market_cap)} mcap "
            f"· {short_address(candidates[index].snapshot.address)}"
        ),
        label_visibility="collapsed",
    )
    col1, col2 = st.columns([1, 4])
    if col1.button("🔬 Full analysis", type="primary", use_container_width=True):
        address = candidates[selected].snapshot.address
        run_analysis([address], settings)
        st.session_state["pending_address"] = address

    # A full report requested from this tab renders inline, so the user never
    # loses their scan results.
    results = st.session_state.get("results", [])
    if results and st.session_state.get("pending_address"):
        st.divider()
        render_result(results[0], settings)
        st.caption("This report is also waiting for you in the **CA Analyzer** tab.")

    csv = frame.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Export scan as CSV", csv, "memedd_scan.csv", "text/csv")

    render_latest_profiles(settings)


def render_latest_profiles(settings: config.AppSettings) -> None:
    """Newest DexScreener token profiles for the selected chain.

    A different discovery angle from the scanner table: these are projects that
    just claimed their DexScreener page, so they skew brand new and pre-volume.
    No market filters apply here - that is the point.
    """
    chain_label = config.get_chain(settings.chain).label
    with st.expander(f"🆕 Latest token profiles on {chain_label}"):
        st.caption(
            "Live from DexScreener's `/token-profiles/latest/v1` feed — projects that just published "
            "a profile. Newest first, unfiltered: treat as a lead list, not a buy list."
        )
        with st.spinner("Loading latest profiles…"):
            profiles = cached_profiles(settings.chain, st.session_state["cache_nonce"])

        if not profiles:
            st.info(f"No recent profiles for {chain_label} in the current feed.", icon="🪪")
            return

        for index, profile in enumerate(profiles[:12]):
            col1, col2 = st.columns([5, 1])
            with col1:
                links = " · ".join(
                    f"[{link.label or link.kind.title()}]({link.url})" for link in profile.links[:4]
                )
                header = f"**`{short_address(profile.address, 10, 6)}`**"
                if profile.url:
                    header = f"**[{short_address(profile.address, 10, 6)}]({profile.url})**"
                st.markdown(header)
                if profile.description:
                    st.caption(profile.description[:260] + ("…" if len(profile.description) > 260 else ""))
                else:
                    st.caption("_No description published._")
                if links:
                    st.caption(links)
            with col2:
                if st.button("Analyze", key=f"profile_{index}", use_container_width=True):
                    run_analysis([profile.address], settings)
                    st.session_state["pending_address"] = profile.address
                    st.success("Analyzed — see the **CA Analyzer** tab.")
            st.divider()


# ==========================================================================
# Tab: History
# ==========================================================================
def tab_history(settings: config.AppSettings) -> None:
    st.markdown("#### 🕘 Recent analyses")
    st.caption("Stored locally in `data/history.sqlite3`. Nothing leaves your machine.")

    rows = history.recent(limit=100)
    if not rows:
        st.info("No analyses recorded yet.", icon="🗒️")
        return

    frame = pd.DataFrame(rows)
    display = frame[
        ["analyzed_at", "symbol", "name", "chain", "composite", "decision", "market_cap", "liquidity", "address"]
    ].rename(
        columns={
            "analyzed_at": "When", "symbol": "Symbol", "name": "Name", "chain": "Chain",
            "composite": "Score", "decision": "Decision", "market_cap": "MCap",
            "liquidity": "Liquidity", "address": "Address",
        }
    )
    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100, format="%.0f"),
            "MCap": st.column_config.NumberColumn("MCap", format="$%.0f"),
            "Liquidity": st.column_config.NumberColumn("Liquidity", format="$%.0f"),
        },
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        options = list(range(len(rows)))
        selected = st.selectbox(
            "Re-analyze a token",
            options,
            format_func=lambda index: (
                f"{rows[index]['symbol'] or '?'} · {rows[index]['composite']:.0f}/100 · "
                f"{rows[index]['analyzed_at'][:16]}"
            ),
        )
        if st.button("🔁 Re-run with fresh data"):
            st.session_state["cache_nonce"] += 1
            run_analysis([rows[selected]["address"]], settings)
            st.session_state["pending_address"] = rows[selected]["address"]
            st.success("Re-analyzed — open the **CA Analyzer** tab for the full report.")
    with col2:
        st.markdown("&nbsp;", unsafe_allow_html=True)
        if st.button("🗑️ Clear history", use_container_width=True):
            history.clear()
            st.rerun()


# ==========================================================================
# Main
# ==========================================================================
def main() -> None:
    init_state()
    ui.inject_css()
    settings = render_sidebar()

    ui.hero(
        "🧪 MemeDD Dashboard",
        f"Meme-coin due diligence on {config.get_chain(settings.chain).label} — "
        "security, liquidity, distribution, momentum and position sizing in one pass.",
    )

    analyzer_tab, scanner_tab, history_tab = st.tabs(["🔍 CA Analyzer", "📡 Scanner", "🕘 History"])
    with analyzer_tab:
        tab_analyzer(settings)
    with scanner_tab:
        tab_scanner(settings)
    with history_tab:
        tab_history(settings)

    ui.disclaimer()


if __name__ == "__main__":
    main()
