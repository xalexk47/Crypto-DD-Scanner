"""Export an :class:`AnalysisResult` as JSON or Markdown.

Used by the download buttons in the UI, and handy for piping a report into a
Telegram bot, a notebook or an LLM prompt later on.
"""

from __future__ import annotations

import json
from typing import List, Optional

from . import config
from .models import AnalysisResult
from .utils import fmt_number, fmt_pct, fmt_usd, score_emoji, short_address

DISCLAIMER = (
    "This report is automated research output, not financial advice. Meme coins are "
    "adversarial, illiquid and frequently fraudulent. Verify every number yourself and "
    "never risk more than you can afford to lose entirely."
)


def to_json(result: AnalysisResult, indent: int = 2) -> str:
    """Full machine-readable dump of the analysis."""
    return json.dumps(result.to_dict(), indent=indent, default=str)


def to_markdown(result: AnalysisResult) -> str:
    """Human-readable report suitable for pasting into notes or a group chat."""
    lines: List[str] = []
    add = lines.append

    if not result.ok or not result.snapshot:
        add(f"# Analysis failed: {short_address(result.address)}")
        add("")
        add(result.error or "Unknown error.")
        return "\n".join(lines)

    s = result.snapshot
    chain_label = config.get_chain(result.chain).label

    add(f"# {s.symbol or '?'} - {s.name or 'Unknown token'}")
    add("")
    add(f"`{result.address}` on **{chain_label}**")
    add("")
    if result.scorecard:
        card = result.scorecard
        add(f"## {score_emoji(card.composite)} Decision: **{card.decision}** ({card.composite:.1f}/100)")
        add("")
        if card.vetoed:
            add(f"> **SECURITY VETO:** {card.veto_reason}")
            add("")
        add(f"_Analysis confidence: {card.confidence * 100:.0f}% (how much verified data backed this score)._")
        add("")

    # --- market ---------------------------------------------------------
    add("## Market")
    add("")
    add("| Metric | Value |")
    add("| --- | --- |")
    add(f"| Price | {fmt_usd(s.price_usd, compact=False)} |")
    add(f"| Market cap | {fmt_usd(s.market_cap)} |")
    add(f"| FDV | {fmt_usd(s.fdv)} |")
    add(f"| Liquidity | {fmt_usd(s.liquidity_usd)} |")
    add(f"| 24h volume | {fmt_usd(s.volume_24h)} |")
    add(f"| Volume / mcap | {s.turnover_24h:.2f}x |")
    add(f"| Liquidity / mcap | {fmt_pct(s.liquidity_ratio * 100, 1)} |")
    add(f"| 1h / 6h / 24h | {fmt_pct(s.price_change_1h, 1, True)} / "
        f"{fmt_pct(s.price_change_6h, 1, True)} / {fmt_pct(s.price_change_24h, 1, True)} |")
    add(f"| 24h trades | {fmt_number(s.txns_24h)} "
        f"({s.txns_24h_buys:,} buys / {s.txns_24h_sells:,} sells) |")
    add(f"| Holders | {fmt_number(s.holders) if s.holders else 'n/a'} |")
    add(f"| Age | {s.age_label} |")
    add(f"| Venues | {s.pair_count} pool(s) across {s.dex_count} DEX(es) |")
    add("")

    # --- score breakdown ------------------------------------------------
    if result.scorecard:
        add("## Score breakdown")
        add("")
        add("| Pillar | Weight | Score | Weighted |")
        add("| --- | --- | --- | --- |")
        for component in result.scorecard.components:
            add(
                f"| {component.label} | {component.weight * 100:.0f}% | "
                f"{component.score:.0f}/100 | {component.weighted:.1f} |"
            )
        add(f"| **Composite** | 100% | | **{result.scorecard.composite:.1f}** |")
        add("")
        for component in result.scorecard.components:
            add(f"**{component.label}** ({component.score:.0f}/100)")
            add("")
            for reason in component.reasons:
                add(f"- {reason}")
            add("")

    # --- security -------------------------------------------------------
    sec = result.security
    add("## Security")
    add("")
    if sec and sec.available:
        def flag(value: Optional[bool], good_when_false: bool = True) -> str:
            if value is None:
                return "unknown"
            if good_when_false:
                return "NO" if not value else "**YES**"
            return "YES" if value else "**NO**"

        add(f"- Honeypot: {flag(sec.is_honeypot)}")
        add(f"- Buy / sell tax: {fmt_pct(sec.buy_tax_pct, 1)} / {fmt_pct(sec.sell_tax_pct, 1)}")
        add(f"- Mintable supply: {flag(sec.is_mintable)}")
        add(f"- Ownership renounced: {flag(sec.owner_renounced, good_when_false=False)}")
        add(f"- Source verified: {flag(sec.is_open_source, good_when_false=False)}")
        add(f"- LP burned/locked: {fmt_pct(sec.lp_secured_pct, 1)}")
        add(f"- Top 10 non-LP holders: {fmt_pct(sec.top10_pct_adjusted, 1)}")
        add(f"- Deployer holdings: {fmt_pct(sec.creator_percent, 2)}")
        add(f"- Source: {sec.source}")
    else:
        add(f"- No security data available. {sec.error if sec else ''}".rstrip())
    add("")

    # --- narrative ------------------------------------------------------
    if result.narrative:
        n = result.narrative
        add(f"## Narrative ({n.source}{'/' + n.model if n.model else ''})")
        add("")
        add(n.summary)
        add("")
        if n.themes:
            add(f"**Themes:** {', '.join(n.themes)}")
            add("")
        for title, items in (("Bull case", n.bull_case), ("Bear case", n.bear_case),
                             ("Mindshare", n.mindshare_notes)):
            if items:
                add(f"**{title}**")
                add("")
                for item in items:
                    add(f"- {item}")
                add("")

    # --- project profile -------------------------------------------------
    if result.profile is not None and result.profile.has_content:
        add("## Project profile (DexScreener)")
        add("")
        if result.profile.description:
            add(f"> {result.profile.description}")
            add("")
        for link in result.profile.links[:8]:
            add(f"- {link.label or link.kind.title()}: {link.url}")
        add("")

    # --- X mindshare -------------------------------------------------------
    mindshare = result.mindshare
    if mindshare is not None and mindshare.available:
        add("## X / Twitter mindshare (Grok)")
        add("")
        if mindshare.is_live:
            add("**Source: live X search.**")
        else:
            add("**Source: model knowledge — NOT a live search. Do not read as current sentiment.**")
        add("")
        add(f"- Sentiment: **{mindshare.sentiment}** ({mindshare.sentiment_score:+.2f})")
        add(f"- Attention: **{mindshare.mindshare_score:.0f}/100**")
        add(f"- Post volume: **{mindshare.post_volume}**, trend **{mindshare.trend}**")
        organic = {True: "organic", False: "coordinated / bots", None: "unclear"}[mindshare.is_organic]
        add(f"- Discussion reads as: **{organic}**")
        add("")
        if mindshare.summary:
            add(f"> {mindshare.summary}")
            add("")
        if mindshare.themes:
            add(f"**Themes:** {', '.join(mindshare.themes)}")
            add("")
        if mindshare.notable_accounts:
            add(f"**Notable accounts:** {', '.join('@' + h for h in mindshare.notable_accounts)}")
            add("")
        if mindshare.red_flags:
            add("**Red flags**")
            add("")
            for flag in mindshare.red_flags:
                add(f"- 🚩 {flag}")
            add("")
        if mindshare.sample_posts:
            add("**Sample posts**")
            add("")
            for post in mindshare.sample_posts:
                meta = f" ({post.engagement:,} engagements)" if post.engagement else ""
                link = f" — {post.url}" if post.url else ""
                add(f"- **@{post.handle or 'unknown'}**{meta}: {post.text}{link}")
            add("")
            add("_Posts are as reported by the model; spot-check before acting on them._")
            add("")
        for warning in mindshare.warnings:
            add(f"> {warning}")
            add("")
    elif mindshare is not None and mindshare.error:
        add("## X / Twitter mindshare")
        add("")
        add(f"Unavailable: {mindshare.error}")
        add("")

    # --- multi-LLM ensemble ----------------------------------------------
    ensemble = result.ensemble
    if ensemble is not None and ensemble.ok and ensemble.consensus is not None:
        consensus = ensemble.consensus
        add(f"## Multi-LLM ensemble ({consensus.model_count} model(s))")
        add("")
        add(f"**Consensus: {consensus.decision_label} — {consensus.overall_score:.0f}/100** "
            f"(confidence {consensus.confidence * 100:.0f}%, agreement {consensus.agreement * 100:.0f}%)")
        add("")
        if ensemble.blended_score is not None:
            add(f"Blended with the rules engine: **{ensemble.blended_score:.0f}/100** "
                f"({config.DECISION_LABELS.get(ensemble.blended_decision, ensemble.blended_decision)}, "
                f"{ensemble.blend_weight * 100:.0f}% LLM weight)")
            add("")

        add("| Model | Score | Decision | Confidence | Latency |")
        add("| --- | ---: | --- | ---: | ---: |")
        for verdict in ensemble.verdicts:
            if verdict.ok:
                add(f"| {verdict.label} | {verdict.overall_score:.0f} | {verdict.decision_label} | "
                    f"{verdict.confidence * 100:.0f}% | {verdict.latency_ms / 1000:.1f}s |")
            else:
                add(f"| {verdict.label} | — | failed | — | {verdict.error} |")
        add("")

        add("| Dimension | " + " | ".join(v.provider for v in ensemble.successful) + " | Consensus |")
        add("| --- | " + " | ".join("---:" for _ in ensemble.successful) + " | ---: |")
        for dim in config.LLM_DIMENSIONS:
            cells = " | ".join(f"{v.dimension_scores.get(dim, 0):.1f}" for v in ensemble.successful)
            add(f"| {config.COMPONENT_LABELS.get(dim, dim.title())} | {cells} | "
                f"{consensus.dimension_scores.get(dim, 0):.1f} |")
        add("")

        if consensus.corroborated_rug_flags:
            add("**Rug flags raised by 2+ models**")
            add("")
            for flag in consensus.corroborated_rug_flags:
                add(f"- 🚩 {flag}")
            add("")
        single = [f for f in consensus.rug_flags if f not in consensus.corroborated_rug_flags]
        if single:
            add("**Rug flags raised by one model (unverified)**")
            add("")
            for flag in single:
                add(f"- {flag}")
            add("")

        if consensus.dissent:
            add("**Disagreement between models**")
            add("")
            for note in consensus.dissent:
                add(f"- {note}")
            add("")

        if consensus.lore_summary:
            add(f"**Narrative.** {consensus.lore_summary}")
            add("")
        if consensus.rationale:
            add(f"**Consensus rationale.** {consensus.rationale}")
            add("")

        for title, items in (("Model positives", consensus.key_positives), ("Model risks", consensus.key_risks)):
            if items:
                add(f"**{title}**")
                add("")
                for item in items:
                    add(f"- {item}")
                add("")

        for verdict in ensemble.successful:
            add(f"<details><summary>{verdict.label} full verdict</summary>")
            add("")
            if verdict.lore_summary:
                add(verdict.lore_summary)
                add("")
            if verdict.rationale:
                add(f"_{verdict.rationale}_")
                add("")
            for label, items in (("Positives", verdict.key_positives), ("Risks", verdict.key_risks),
                                 ("Rug flags", verdict.rug_flags)):
                if items:
                    add(f"{label}:")
                    add("")
                    for item in items:
                        add(f"- {item}")
                    add("")
            add("</details>")
            add("")
    elif ensemble is not None:
        add("## Multi-LLM ensemble")
        add("")
        add("No model verdicts were produced.")
        add("")
        for note in ensemble.notes:
            add(f"- {note}")
        add("")

    # --- positives / risks ----------------------------------------------
    if result.scorecard:
        if result.scorecard.positives:
            add("## Positives")
            add("")
            for item in result.scorecard.positives:
                add(f"- {item}")
            add("")
        if result.scorecard.risks:
            add("## Key risks")
            add("")
            for item in result.scorecard.risks:
                add(f"- {item}")
            add("")

    # --- risk plan ------------------------------------------------------
    plan = result.risk_plan
    if plan:
        add(f"## Risk plan ({plan.risk_profile}, portfolio {fmt_usd(plan.portfolio_usd)})")
        add("")
        add(f"- Suggested position: **{fmt_usd(plan.position_usd, compact=False)}** "
            f"({plan.position_pct:.2f}% of portfolio)")
        add(f"- Stop loss: **-{plan.stop_loss_pct:.1f}%**"
            + (f" (~{fmt_usd(plan.stop_price, compact=False)})" if plan.stop_price else ""))
        add(f"- Max loss if stopped: **{fmt_usd(plan.max_loss_usd, compact=False)}** "
            f"({plan.max_loss_pct_of_portfolio:.2f}% of portfolio)")
        add(f"- Position vs pool: {plan.liquidity_share_pct:.2f}% of liquidity, "
            f"~{plan.est_slippage_pct:.2f}% estimated entry impact")
        add("")
        if plan.take_profit_targets:
            add("| Target | Gain | Price | Sell | Profit |")
            add("| --- | --- | --- | --- | --- |")
            for target in plan.take_profit_targets:
                add(
                    f"| {target['label']} ({target['r_multiple']:.1f}R) | +{target['gain_pct']:.0f}% | "
                    f"{fmt_usd(target['price'], compact=False)} | {target['sell_portion_pct']}% | "
                    f"{fmt_usd(target['profit_usd'], compact=False)} |"
                )
            add("")
        for warning in plan.warnings:
            add(f"> {warning}")
            add("")
        for note in plan.notes:
            add(f"- _{note}_")
        add("")

    if result.data_warnings:
        add("## Data warnings")
        add("")
        for warning in result.data_warnings:
            add(f"- {warning}")
        add("")

    add("---")
    add("")
    add(f"_Generated by MemeDD Dashboard at {result.analyzed_at}._")
    add("")
    add(f"_{DISCLAIMER}_")
    return "\n".join(lines)


def filename_for(result: AnalysisResult, extension: str) -> str:
    """Suggested download filename, e.g. ``memedd_BPEPE_base.md``."""
    symbol = (result.snapshot.symbol if result.snapshot else "") or short_address(result.address)
    safe = "".join(ch for ch in symbol if ch.isalnum() or ch in "-_") or "token"
    return f"memedd_{safe}_{result.chain}.{extension}"
