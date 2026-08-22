# 🧪 MemeDD Dashboard

Meme-coin **due diligence and scanning** for Base (and any other chain you point it at).
Paste a contract address, get a full report: security/rug checks, liquidity quality,
holder distribution, momentum, narrative, a 0–100 composite score, a Buy/Watch/Pass
decision and a concrete position-sizing plan for *your* portfolio.

Works out of the box on free public APIs — **no API keys required**.

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
- **Composite score (0–100)** with a full breakdown and per-pillar reasoning.
- **Decision** — Strong Buy / Buy / Watch / Pass, with a hard security veto that
  forces Pass on honeypots regardless of how good everything else looks.
- **Risk management** — suggested position size in $ and % of portfolio,
  volatility-aware stop-loss distance, max loss if stopped out, take-profit
  ladder, estimated price impact and a liquidity-adjusted size warning.
- **Export** — download any report as Markdown or JSON.

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

## Install

Requires **Python 3.11+**.

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

To add keys, copy the template and edit it:

```bash
cp .env.example .env
```

`.env` is git-ignored. Every variable is documented inline in `.env.example`, and
config resolution lives in `src/config.py` — nothing reads `os.environ` directly.

---

## Adding LLM analysis (Claude / GPT / Grok)

The narrative layer is already abstracted behind a provider interface in
`src/llm.py`, and scaffolding for all three vendors ships in the box. To turn it on:

```bash
pip install anthropic            # or: pip install openai  (also used for Grok)
```

```bash
# .env
MEMEDD_LLM_PROVIDER=anthropic    # anthropic | openai | xai
MEMEDD_LLM_MODEL=claude-sonnet-4-5
ANTHROPIC_API_KEY=sk-ant-...
```

Then flick **Use LLM for lore analysis** in the sidebar. The sidebar always shows
which engine is actually active, and any provider failure degrades silently back
to the heuristic narrative — a flaky LLM can never take the dashboard down.

To add your own provider, implement the `NarrativeProvider` protocol and register
it in `PROVIDERS`:

```python
class MyProvider:
    name = "mine"

    def available(self) -> bool:
        return bool(my_api_key)

    def analyze(self, snapshot, security) -> NarrativeReport:
        raw = call_my_model(build_narrative_prompt(snapshot, security))
        return parse_llm_json(raw, model="my-model", source="mine")

PROVIDERS["mine"] = MyProvider
```

Nothing downstream changes — the scorer, UI and exporters only ever see a
`NarrativeReport`.

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
  llm.py                Narrative layer + LLM provider seam (Claude/GPT/Grok)
  history.py            Local SQLite history
  report.py             Markdown / JSON export
  ui.py                 Reusable Streamlit components + CSS
  utils.py              Formatting, address parsing, safe coercion, TTL cache
tests/                  94 unit + end-to-end tests (network fully stubbed)
.streamlit/config.toml  Dark theme
```

The data flow is one direction, with normalization at the boundary:

```
raw API JSON -> TokenSnapshot / SecurityReport -> ScoreCard -> RiskPlan -> UI / export
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
something to sweep. Base, Ethereum, Solana, BNB Chain and Arbitrum ship enabled.

---

## Known limitations

- **Scanner discovery is search-based.** DexScreener has no "list every pair on a
  chain" endpoint, so the scanner unions its public search, boosted-token and
  new-profile feeds. It is a wide net, not an exhaustive index — tokens nobody
  is searching for or promoting may not surface.
- **Security coverage varies.** GoPlus has no record for very new tokens; those
  reports come back `unavailable` and are scored conservatively rather than skipped.
- **Narrative is heuristic until you add a key.** The v1 narrative reads metadata,
  not community sentiment. It says so in every report.
- **Price impact is an approximation.** Slippage uses a constant-product estimate
  (`x / (L/2 + x)`), which is the right order of magnitude but not a quote — v3
  concentrated liquidity in particular can behave very differently.

---

## Roadmap

- Grok-powered live X/Twitter mindshare scoring
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
