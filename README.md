# 🧪 MemeDD Dashboard

**Multi-chain portfolio management, liquidity-rotation tracking and meme-coin due
diligence** across Base, Solana, BNB Chain and Robinhood Chain.

Register your wallet addresses and the app reads what you hold straight off-chain,
prices it live, groups it by ecosystem, and scores every chain on how hot it is
right now — so when one chain's liquidity surge starts cooling you can see it and
rotate before it does. Paste a contract address instead and you get the full
due-diligence report: security/rug checks, liquidity quality, holder distribution,
momentum, narrative, a 0–100 composite score and a position-sizing plan.

Works out of the box on free public APIs — **no API keys required**.

> **Read-only by design.** The app takes public wallet *addresses*, nothing more.
> It never asks for a private key, seed phrase or exchange login, holds no key
> with spend authority, and cannot place a trade. Every suggestion it makes is
> arithmetic shown on screen for you to act on yourself.

---

## What it does

### 🔍 CA Analyzer
Paste one or more contract addresses (newline, comma or space separated —
DexScreener URLs work too) and get, per token:

- **Market snapshot** — price, market cap, FDV, liquidity, 24h volume, turnover,
  price change across 1h/6h/24h, trade counts, buy/sell split, holders, age.
- **Rug & security checks** — honeypot, sellability, buy/sell tax, mintable supply,
  ownership renounced, hidden owner, pausable transfers, blacklist, proxy,
  source verification, **LP burned/locked %**, deployer holdings and
  **top-10 holder concentration excluding LP and burn wallets**.
- **Lore & narrative** — a readable narrative summary with themes, bull case and
  bear case. Heuristic by default; swap in Claude/GPT/Grok with one env var.
- **X / Twitter mindshare** *(optional)* — Grok searches X live for discussion of
  the token and returns sentiment, post volume, notable accounts and real sample
  posts, which feed the momentum score. See below.
- **Multi-LLM ensemble** *(optional)* — the same payload sent to Grok, Claude and
  GPT **in parallel**, each returning strict JSON, combined into a consensus
  verdict with an explicit agreement score and dissent notes. See below.
- **Project profile** — description, icon and links pulled from DexScreener's
  token-profiles feed, folded into both the report and the LLM payload.
- **Composite score (0–100)** with a full breakdown and per-pillar reasoning.
- **Decision** — Strong Buy / Buy / Watch / Pass, with a hard security veto that
  forces Pass on honeypots regardless of how good everything else looks.
- **Risk management** — suggested position size in $ and % of portfolio,
  volatility-aware stop-loss distance, max loss if stopped out, take-profit
  ladder, estimated price impact and a liquidity-adjusted size warning.
- **Export** — download any report as Markdown or JSON.

Chains: **Base** (default), Ethereum, Solana, BNB Chain, Arbitrum and
**Robinhood Chain** — switchable in the sidebar.

### 💼 Portfolio
Register your wallet addresses per chain; the app discovers and prices everything
you hold. No trade ledger to maintain, no CSV to export from anywhere.

- **Balances read off-chain** — Blockscout's token list where available, otherwise
  Etherscan V2 discovery plus `balanceOf` on a public RPC, and
  `getTokenAccountsByOwner` (both SPL Token and Token-2022) on Solana. Your native
  ETH/BNB/SOL counts too: that is the dry powder a rotation actually moves.
- **Allocation by chain and by ecosystem** — tag positions (e.g. `Brew`) and the
  whole cluster rolls up as one line with its own value-weighted 24h move.
- **P&L, honestly** — a wallet read gives quantity, never entry price, so avg cost
  is an editable column. Leave it blank and the app says *basis unknown* rather
  than showing a confident zero; fill it in and P&L becomes real.
- **Equity curve** — every sync is stored locally, so portfolio history builds
  itself from the first sync onward.
- **Nothing hidden** — dust is summed rather than dropped, unpriceable holdings are
  listed with the reason, and a chain that could not be read says so instead of
  looking like an empty wallet.

### 🔄 Rotation
The question this tab answers is not "is this coin good" but "is this *chain* where
the money currently is, and is my book positioned for where it goes next".

Each chain gets a **0–100 heat index**, measured twice from two independent samples:

| Half | Sample | What it tells you |
| --- | --- | --- |
| **Your bags** | the tokens you hold there, value-weighted | how your book is performing |
| **The chain** | a live basket of liquid tokens you *don't* hold, plus DefiLlama TVL and DEX volume | whether the chain itself is moving |

Reporting both is the entire point. When your Brew positions on BNB are screaming
and the BSC basket is flat, that is an idiosyncratic pump in your tokens — trim the
token. When both run together, liquidity has genuinely rotated onto the chain —
trim the chain. The **divergence** figure is what separates the two, and it is
shown on every chain card.

Heat feeds a four-state machine — 🔥 Hot / 📈 Heating / 📉 Cooling / 🧊 Cold —
judged against that chain's own stored history, so a chain is only *hot* while it
is both high **and** not already rolling over, and only *cold* while it is low and
not yet turning up. A cold chain that has started to lift is the earliest turn, and
it ranks ahead of one still falling at the same heat.

From that comes a **concrete plan**, with the arithmetic shown:

> Trim 25% of BREW on BNB Chain — **$1,250**
> BREW is +180% (vs your avg cost) while BNB Chain reads 84/100 🔥 Hot for 36h;
> your holdings there run 22 points hotter than the chain basket, so this is your
> tokens moving rather than the whole chain.
>
> Rotate $750 from BNB Chain → Base
> Base reads 31/100 (📈 Heating) and has started to turn up. You hold 8.4% of the
> book there today.

Guardrails come from your sidebar risk profile: a trim is capped at that profile's
share of pool liquidity (so the plan never suggests dumping more than the pool can
absorb, and says so when it clamps), positions over the max-position cap get
trimmed back to it regardless of heat, and moves too small to beat the spread are
not suggested at all. Nothing is ever proposed into a chain the app could not
actually read.

### 📡 Scanner
Sweeps DexScreener for candidates on the selected chain, filters them
(market cap $500k–$5M by default, plus minimum liquidity, volume, trade count
and age), ranks them by a market-data-only quick score, and renders an
interactive, sortable table. Pick any row and run the full analysis on it.
Results are cached for a few minutes to stay well inside API rate limits.

### 🕘 History
Every completed analysis is stored locally in SQLite (`data/history.sqlite3`)
so you can see what you looked at, what it scored, and re-run it with fresh data.

---

## Scoring model

| Pillar | Weight | What drives it |
| --- | ---: | --- |
| Security | 30% | Honeypot, taxes, mint/pause/blacklist powers, renouncement, LP lock, source verification |
| Liquidity Quality | 20% | Absolute pool depth, liquidity-to-market-cap ratio, turnover sanity, LP security |
| Holder Distribution | 15% | Top-10 non-LP concentration, deployer bag, single-whale check, holder count |
| Mindshare / Momentum | 15% | Volume/market-cap turnover, blended price action, trade count, buy/sell flow, volume acceleration |
| Narrative Potential | 10% | Socials, branding, ticker quality, meme-theme heuristics (LLM-upgradeable) |
| Catalyst / Listing Potential | 10% | Venue spread, market-cap band, age, volume acceleration, listing-readiness |

**Decision thresholds:** ≥78 Strong Buy · ≥64 Buy · ≥48 Watch · below that Pass.

Two design choices worth knowing:

1. **Missing data is not good news.** When a provider returns nothing, the pillar
   moves toward a penalised score *and* lowers the report's confidence, rather
   than silently scoring 0 or 100. Every report shows its confidence level.
2. **Critical security findings veto the score.** A confirmed honeypot,
   unsellable balance, hidden owner or selfdestruct forces `Pass` and a zero
   position size no matter what the other five pillars say.

Weights are adjustable live in the sidebar (**Advanced: score weights**) and are
re-normalised to 100% automatically.

---

## Risk model

Position sizing is not a fixed percentage — it is derived, in this order:

1. **Stop distance** from a realised-volatility proxy (blended 1h/6h/24h moves),
   floored by your risk profile and clamped to 12–65%.
2. **Risk-based size** so that being stopped out costs exactly your profile's
   risk-per-trade budget.
3. **Conviction scaling** — a weak composite score only ever *shrinks* the bet
   (the risk budget is a ceiling, never exceeded), then the profile's hard max
   position cap applies.
4. **Liquidity cap** — the position is cut so it never exceeds a small share of
   pool liquidity, because a size you cannot exit is not a position.

| Profile | Risk/trade | Max position | Baseline stop | Max share of pool |
| --- | ---: | ---: | ---: | ---: |
| Conservative | 0.50% | 2% | 22% | 0.50% |
| Moderate | 1.00% | 5% | 30% | 1.00% |
| Aggressive | 2.00% | 10% | 40% | 2.00% |
| Degen | 3.50% | 15% | 50% | 3.00% |

---

---

## On-chain wallet flow & smart money

Optional, needs a **free** [Etherscan API key](https://etherscan.io/apis) — one key
covers Base, Ethereum, Arbitrum and BSC (5 calls/sec, 100k/day). Toggle
**Analyze on-chain wallet flow** in the sidebar.

Reads raw ERC-20 transfer logs and works out who is actually buying:

- **Quiet accumulation** — the headline signal: wallets net accumulating *while
  the price is range-bound*. Buying a rip is chasing; buying a flat chart is
  positioning. Requires both conditions, so a pump never triggers it.
- **Accumulation vs distribution** — wallet counts and net flow as a share of supply
- **Early-buyer cohort** — who bought first, how many still hold, how many flipped
- **One-and-done wallets** — a wall of wallets that bought once and never traded
  again is farming or bots, and *lowers* the score rather than reading as demand
- **Your watchlist** — wallets you nominated, flagged by name when they appear

Buys and sells are counted only when tokens move **to or from the liquidity
pool**. Wallet-to-wallet transfers are ignored: they move tokens without
expressing conviction, and counting them is how naive trackers get fooled by
self-transfers.

### What this deliberately does not claim

It does **not** compute historical win rates. A real win-rate needs the price at
the moment of every trade a wallet ever made, across every token it touched —
which no free explorer API exposes, and is exactly what paid services like Cielo
($199/mo) sell. Deriving one from transfer logs would produce a confident-looking
number with nothing behind it, which is worse than no number. Everything here is
an **observation about behaviour**, never a claim about anyone's skill.

### The watchlist is the sharp edge — and the bridge to GMGN/Cielo

**GMGN** has no self-serve public API: access is whitelist-only (you submit a
transaction address, invite code and IP) and rate limited to 2 req/sec, and the
third-party scrapers around it are fragile and paid. **Cielo**'s wallet PnL is
behind the $199/mo Whale plan. Neither can be integrated the way Etherscan can.

The watchlist is how you use them anyway. Their real value is *discovery* —
ranking wallets by realised win rate, something free data cannot reproduce.
So do the discovery there, paste the wallets you rate into the **⭐ Watchlist**
tab, and this app does the *monitoring* from free Etherscan data: it flags those
wallets by name whenever they appear in any token's flow, and hands your X
handles to Grok so the mindshare panel reports on those specific accounts.

Paid discovery, free monitoring — and no subscription needed for the part that
runs on every token you analyze.



```bash
cp data/smart_money.example.json data/smart_money.json
```

```json
{
  "wallets": [{ "address": "0x...", "label": "caught BRETT at 200k" }],
  "x_handles": ["someanalyst"]
}
```

Wallets you already trust — exported from Cielo, Arkham, Nansen or your own
notes — get flagged by name whenever they appear in a token's flow, and weigh
more in the score than any heuristic here. The X handles are handed to Grok,
which then reports specifically whether *those accounts* have posted about the
token rather than generic chatter. `data/smart_money.json` is gitignored.

Wallet flow feeds the **Holder Distribution** pillar, and is included in the
payload every ensemble model sees. It runs on Etherscan V2 chains only — Solana
and Robinhood Chain fall back to the same clearly-labelled "unavailable" path as
the security checks.

---

## X / Twitter mindshare (Grok)

Optional, off by default, needs `XAI_API_KEY`. Toggle **Query X via Grok** in the
sidebar.

Every other signal in this app is on-chain or market data. Meme-coin mindshare is
neither — it forms on X, often hours before it reaches volume. Grok is the only
major model with first-party X access, which is what earns it a place here beyond
being a third ensemble opinion.

**Verify your setup first:**

```bash
python scripts/check_grok.py
```

That checks your key, lists the model ids your account can actually call, and
tries the X search tool — printing the exact fix for whatever fails.

### What it returns

Sentiment and a −1..+1 score, an attention rating out of 100, post volume
(`none`→`viral`), trend, whether the discussion looks **organic or coordinated**,
recurring themes, notable accounts, real sample posts with engagement, and red
flags (impersonation, giveaway scams, reply-spam).

### How it changes the score

It replaces part of the momentum pillar — until now that pillar used volume and
price as a *proxy* for attention. Default 30% social / 70% on-chain
(`MEMEDD_MINDSHARE_WEIGHT`). Three judgement calls are baked in:

- **Coordinated shilling lowers the score.** High volume plus bullish tone
  *reduces* momentum when `is_organic` is false — a naive implementation would
  reward exactly the pattern you want to avoid.
- **Attention isn't approval.** A hated token people are arguing about still has
  more mindshare than one nobody mentions; sentiment tilts the score rather than
  setting it.
- **Live and stale are never blended.** If the search tool is unavailable the app
  falls back to a plain Grok call, labels it `is_live=false`, pulls its influence
  toward neutral, and says so in the UI and the report. A model's recollection of
  a ticker is not what X is saying today.

Grok's findings are also injected into the ensemble payload, so Claude and GPT
reason about the social data too rather than Grok alone having seen it.

### A note on the API

Server-side X search runs on xAI's **Agent Tools API** — `client.responses.create()`
against `/v1/responses` — not on chat completions. Established against the live
API, since the docs and the endpoints disagree:

| Attempt | Result |
| --- | --- |
| `x_search` in `chat.completions` `tools` | 422 — `unknown variant`, only `function` or `live_search` accepted there |
| `live_search` in `chat.completions` `tools` | parses, then **410 — "Live search is deprecated. Please switch to the Agent Tools API"** |
| `x_search` on `/v1/responses` | ✅ the supported path |

Because the schema moves, the app doesn't bet on one payload shape. It tries the
documented x_search shapes (richest first, the bare `{"type": "x_search"}` last)
and keeps the first the API accepts; a rejected shape fails at request
validation, before any model runs, so a miss costs no tokens. The call then
degrades through a chain:

| Attempt | Result |
| --- | --- |
| `x_search` on `/v1/responses` + strict schema | live data, structured |
| `x_search` on `/v1/responses`, no schema | live data, JSON repaired on parse |
| `chat.completions`, no tools | model knowledge, clearly labelled not-live |
| all failed | unavailable, with the error and a pointer to `check_grok.py` |

The model id, tool name and sources are env vars (`MEMEDD_X_SEARCH_MODEL`,
`MEMEDD_X_SEARCH_TOOL`, `MEMEDD_X_SEARCH_SOURCES`), so if xAI changes any of
them you can fix it in `.env` without touching code. Model ids move fast —
`grok-4.6` at time of writing; run `scripts/check_grok.py` to list what your own
account can call.

Searches are billed per call, so results are cached for 15 minutes by default.

---

## Multi-LLM ensemble

Optional, off by default, and unlocked by adding any of `XAI_API_KEY`,
`ANTHROPIC_API_KEY` or `OPENAI_API_KEY` to `.env`. Toggle it in the sidebar.

```
                 ┌── Grok  (xAI, api.x.ai/v1) ──┐
 token payload ──┼── Claude (Anthropic) ────────┼──> consensus + agreement
   (identical)   └── GPT   (OpenAI) ────────────┘     + dissent + blended score
```

Every model receives **the same structured payload and the same prompt**, in
parallel, so the wall-clock cost is the slowest model rather than their sum.
Each returns strict JSON against one shared schema:

```json
{
  "overall_score": 0-100,
  "decision": "strong_buy" | "buy" | "watch" | "pass",
  "confidence": 0.0-1.0,
  "dimension_scores": {
    "security": 0-10, "liquidity": 0-10, "holders": 0-10,
    "mindshare": 0-10, "lore": 0-10, "catalyst": 0-10
  },
  "lore_summary": "2-4 sentence narrative assessment",
  "key_positives": ["...", "..."],
  "key_risks": ["...", "..."],
  "rug_flags": ["...", "..."],
  "rationale": "concise paragraph"
}
```

### How strict JSON is enforced

Each vendor gets its native mechanism, with fallbacks so an older model or SDK
degrades instead of failing:

| Provider | Primary | Fallback 1 | Fallback 2 |
| --- | --- | --- | --- |
| Anthropic | `output_config.format` json_schema | strict tool call | prompt-only JSON |
| OpenAI / xAI | `response_format` json_schema (strict) | `json_object` | prompt-only JSON |

Whatever arrives is normalized by `coerce_verdict()`, which **repairs** rather
than rejects the usual model errors — dimensions returned 0-100 instead of
0-10, confidence as a percentage, `"STRONG BUY"` instead of `strong_buy`, a
missing dimension. Every repair is recorded on the verdict, and shown in the UI.

### What the ensemble adds over one model

The point is not a smoother average — it is **agreement as a signal**:

- **Consensus** — confidence-weighted mean score and dimension scores.
- **Agreement (0-100%)** — from the score spread and whether the decisions
  actually match. Three models at 90/50/20 produce a *low-confidence* consensus,
  not a confident 53.
- **Dissent notes** — score spreads ≥25 points, split decisions, and cases where
  the vote and the averaged score disagree, all surfaced explicitly.
- **Corroborated rug flags** — a flag raised independently by 2+ models is
  separated from one model's hunch, and shown in red rather than amber.
- **Conservative reconciliation** — the consensus decision is the *more
  conservative* of the weighted vote and the score's own bucket.

### Blending with the rules engine

The deterministic score is never overwritten. The ensemble produces a separate
blended number (default 35% LLM, adjustable in the sidebar), under two rules
that are not negotiable:

1. **The security veto wins outright.** A model can be talked out of a honeypot
   by a good story; the rules engine cannot.
2. **Models can talk a score down, never rescue one.** The blended decision is
   held to the more conservative of the blend and the deterministic call.

### Cost note

An ensemble run is three frontier-model calls per token. That is fine for
deep-diving a shortlist and expensive for scanning — which is why Scanner mode
ranks on market data only and the ensemble runs on demand, per token.

---

## DexScreener token profiles

`/token-profiles/latest/v1` is the feed of projects that just published a
DexScreener profile. It is wired in at two points:

- **Enrichment** — on every analysis, the token is looked up in the feed and any
  description, icon and social links it carries are folded into the snapshot
  (without overwriting pair data). This is usually the only source of a project
  description, so it materially improves both the heuristic narrative and the
  LLM payload.
- **Discovery** — Scanner mode has a *Latest token profiles* panel listing recent
  profiles for the selected chain, each with a one-click deep analysis. These
  skew brand new and pre-volume, so no market filters apply — treat it as a lead
  list, not a buy list.

The feed is fetched once and cached, so per-token lookups are effectively free.

---

## Install

Requires **Python 3.9+**. Developed and tested on 3.11; it also runs on the
3.9 that ships with macOS — no version-specific syntax is used.

```bash
git clone https://github.com/xalexk47/Crypto-DD-Scanner.git
cd Crypto-DD-Scanner

python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Then open http://localhost:8501. The dark theme comes from `.streamlit/config.toml`,
so launch from the project root.

Run the test suite with:

```bash
python -m pytest tests/ -q
```

---

## API keys

**None are required.** The v1 feature set runs entirely on free public endpoints:

| Data | Provider | Key needed |
| --- | --- | --- |
| Pairs, price, liquidity, volume, socials | [DexScreener](https://docs.dexscreener.com/api/reference) | No |
| Token security / rug checks | [GoPlus Security](https://docs.gopluslabs.io/reference/api-overview) | No (a key only raises rate limits) |
| Lore & narrative | Built-in heuristics | No |
| Project profile / new launches | [DexScreener token profiles](https://docs.dexscreener.com/api/reference) | No |
| Multi-LLM ensemble | xAI / Anthropic / OpenAI | Yes — optional, any subset |
| Wallet balances (EVM) | Public JSON-RPC + [Blockscout](https://docs.blockscout.com/devs/apis) | No |
| Wallet balances (Solana) | Public Solana RPC | No (a Helius/QuickNode URL avoids throttling) |
| Token discovery on Base / BNB Chain | [Etherscan V2](https://docs.etherscan.io/etherscan-v2) | Free key — without it, Blockscout covers discovery |
| Chain TVL & DEX volume | [DefiLlama](https://defillama.com/docs/api) | No |

No key in this table can move a coin. RPC and explorer endpoints are read-only,
and the app has no code path that signs a transaction.

To add keys, copy the template and edit it:

```bash
cp .env.example .env
```

`.env` is git-ignored. Every variable is documented inline in `.env.example`, and
config resolution lives in `src/config.py` — nothing reads `os.environ` directly.

---

## Enabling the LLM layers

There are two independent LLM features:

| Feature | Module | What it does |
| --- | --- | --- |
| Narrative | `src/llm.py` | One model writes the lore/narrative section |
| Ensemble | `src/llm_analyzers.py` | Three models score the token in parallel |

Install the SDKs you have keys for and add the keys to `.env`:

```bash
pip install anthropic openai     # openai also drives Grok via api.x.ai/v1
```

```bash
# .env — any subset works
XAI_API_KEY=xai-...
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...

# For the single-model narrative section:
MEMEDD_LLM_PROVIDER=anthropic    # anthropic | openai | xai | none
```

Then use the sidebar: **Use LLM for lore analysis** for the narrative, and
**Run ensemble on analysis** for the multi-model verdict. The sidebar lists each
provider's live status (ready, no key, or SDK not installed), and both features
degrade safely — a failing provider produces a recorded error, never a crash,
and the deterministic analysis is unaffected.

### Adding your own provider

For the **ensemble**, subclass an analyzer and register it. Everything else —
parallel execution, repair, consensus, blending — comes for free:

```python
class MyAnalyzer(OpenAICompatibleAnalyzer):   # any OpenAI-wire endpoint
    provider = "mine"
    label = "MyModel"
    base_url = "https://api.example.com/v1"

ANALYZERS["mine"] = MyAnalyzer
```

For the **narrative**, subclass `_TransportProvider` in `src/llm.py` (it reuses
the same vendor transports) and register it in `PROVIDERS`. Nothing downstream
changes — the scorer, UI and exporters only ever see a `NarrativeReport` or an
`EnsembleResult`.

---

## Project structure

```
app.py                  Streamlit entry point (thin controller: widgets -> pipeline)
src/
  config.py             Chains, weights, thresholds, risk profiles, env vars
  models.py             Typed dataclasses that flow through the pipeline
  data_fetchers.py      DexScreener + GoPlus clients and normalizers
  scorers.py            Six scoring pillars, composite, veto logic, risk calculator
  analyzer.py           Orchestration: analyze_token / analyze_many / scan
  llm.py                Narrative layer (single model, prose output)
  llm_analyzers.py      Multi-LLM ensemble: strict JSON, parallel, consensus
  mindshare.py          X/Twitter mindshare via Grok's server-side search
  wallet_flow.py        On-chain wallet flow + smart-money watchlist (Etherscan)
  balances.py           Wallet balance providers (Blockscout / EVM RPC / Solana RPC)
  portfolio.py          Positions, pricing, allocation, ecosystem tags, P&L
  portfolio_store.py    Local SQLite: wallets, annotations, snapshots, heat history
  rotation.py           Chain heat index, state machine, trim/rotate planner
  history.py            Local SQLite history
  report.py             Markdown / JSON export
  ui.py                 Reusable Streamlit components + CSS
  utils.py              Formatting, address parsing, safe coercion, TTL cache
  scripts/check_grok.py Diagnose your Grok key, models and X search access
tests/                  361 unit + end-to-end tests (network, RPC and LLMs stubbed)
.streamlit/config.toml  Dark theme
```

The data flow is one direction, with normalization at the boundary:

```
raw API JSON -> TokenSnapshot / SecurityReport / TokenProfile
             -> ScoreCard -> RiskPlan -> [LLM ensemble] -> UI / export
```

No provider JSON ever reaches the scoring or UI layers, so swapping a data source
means writing one normalizer and nothing else.

---

## Adding a chain

Add an entry to `CHAINS` in `src/config.py`:

```python
"blast": ChainConfig(
    key="blast",
    label="Blast",
    dexscreener_id="blast",        # DexScreener chainId
    goplus_id="81457",             # GoPlus chain id, or None
    address_kind="evm",
    explorer_token_url="https://blastscan.io/token/{address}",
),
```

Optionally add seed search terms to `SCANNER_SEED_QUERIES` so Scanner mode has
something to sweep. Base, Ethereum, Solana, BNB Chain, Arbitrum and Robinhood
Chain ship enabled.

### Chains without a security provider

Set `goplus_id=None` when GoPlus does not cover the chain, and add an entry to
`SECURITY_PROVIDER_NOTES` explaining what cannot be checked. The app then:

- skips the security call entirely (no pointless request),
- scores the security pillar at 40/100 with 0.25 confidence — unknown is
  penalised, never treated as clean,
- shows your note in the sidebar the moment the chain is selected, and again in
  place of the rug-check panel, so a blank section is never mistaken for a pass,
- sizes positions down accordingly, since conviction feeds the risk calculator.

**Robinhood Chain** is the shipped example. It is an Arbitrum Orbit L2
(chain id 4663, ETH gas, mainnet since 1 July 2026) that DexScreener indexes but
GoPlus does not support, so market data, scoring, narrative, the LLM ensemble and
position sizing all work there — only the automated rug checks cannot run.
Verify contracts by hand before trading on it.

---

## Known limitations

- **Scanner discovery is search-based.** DexScreener has no "list every pair on a
  chain" endpoint, so the scanner unions its public search, boosted-token and
  new-profile feeds. It is a wide net, not an exhaustive index — tokens nobody
  is searching for or promoting may not surface.
- **Security coverage varies.** GoPlus has no record for very new tokens; those
  reports come back `unavailable` and are scored conservatively rather than skipped.
- **Robinhood Chain has no automated rug checks at all.** GoPlus does not support
  chain 4663, so honeypot, tax, mint-authority, LP-lock and holder-concentration
  checks cannot run there. Everything else works; the security pillar is
  penalised and the gap is stated in the UI.
- **Narrative is heuristic until you add a key.** Without an LLM key the
  narrative reads metadata, not community sentiment. It says so in every report.
- **The ensemble judges the payload, not the chain.** The models see only what
  the fetchers collected. They cannot check a contract themselves, and three
  models agreeing on incomplete data is still incomplete data — which is why
  missing inputs lower confidence rather than being scored as clean.
- **Cross-model deduplication is textual.** Two models phrasing the same risk
  differently count as two points, not one; only near-identical wording merges.
- **Price impact is an approximation.** Slippage uses a constant-product estimate
  (`x / (L/2 + x)`), which is the right order of magnitude but not a quote — v3
  concentrated liquidity in particular can behave very differently.
- **Cost basis cannot be read from a wallet.** No free API returns the USD price
  of each historical buy, so avg cost is something you type in. Until you do, the
  app reports *basis unknown* and falls back to performance since the first sync
  rather than inventing an entry price.
- **Heat needs history to read a trend.** The first refresh can only score the
  level; states (heating vs cooling) sharpen as stored readings accumulate, which
  is why every refresh is persisted locally.
- **Robinhood Chain has thinner rotation data.** DefiLlama does not cover chain
  4663, so its heat comes from DexScreener activity alone. The chain card names
  the missing input and lowers its own confidence rather than scoring it as cold.
- **A chain that cannot be read is "unknown", not "cold".** Heat confidence drops
  with every missing input, and the planner will not rotate into a chain whose
  confidence is too low — an unreachable RPC must never read as a buying
  opportunity.
- **Solana's public RPC throttles.** A large wallet may need a Helius or
  QuickNode URL in `SOLANA_RPC_URL` to sync reliably.

---

## Roadmap

- Optional lot ledger on top of the existing tables, for true realised P&L
- Alerting when a chain changes heat state or a position breaches its cap
- Per-model cost/latency tracking and a cheap-model tier for scanning
- Mindshare history, to score attention *trend* rather than a point reading
- Wallet/bundle clustering to catch sybil "holder counts"
- Historical score tracking and alerting on score changes
- Backtesting the scoring model against realised returns

---

## Disclaimer

MemeDD Dashboard is automated research tooling, **not financial advice**. Data
comes from third-party APIs that can be wrong, stale or deliberately gamed. A
high score is not a safety guarantee — meme coins are adversarial and frequently
fraudulent. Verify everything yourself and never risk money you cannot afford to
lose entirely.
