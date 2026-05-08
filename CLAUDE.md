# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the web app (opens at http://localhost:8501)
streamlit run app.py

# Run all tests (no network required — uses synthetic data)
python smoke_test.py
```

There is no linter configured. The test suite is `smoke_test.py` only.

## Architecture

The app is a Buffett-style stock analyser for US tickers. Data flows through three layers:

**`edgar.py`** — SEC EDGAR XBRL fetcher. Hits the public SEC API (no key needed) to pull 10+ years of annual fundamentals (revenue, FCF, EPS, equity, etc.) keyed by `CONCEPTS` dict. Merges across concept-name aliases to handle companies that changed reporting labels mid-history (e.g. ASC 606 revenue renames). Results are module-level cached by CIK.

**`analyzer.py`** — Core logic layer. `fetch(ticker)` calls yfinance for current price/ratios/quarterly data, then calls `edgar.fetch_history()` to backfill annual histories beyond yfinance's ~4-year window. All results populate a `Fundamentals` dataclass (everything is `Optional` — missing data is normal and handled gracefully downstream). Higher-level functions consume a `Fundamentals` instance:
- `auto_assumptions(f)` — picks sensible DCF growth/discount/terminal rates from the data
- `dcf_intrinsic_value(...)` — two-stage DCF returning per-share intrinsic value
- `quality_score(f)` — weighted score across ROE, D/E, EPS growth, FCF consistency, etc.
- `recommendation(mos, score)` — BUY/HOLD/SELL verdict from margin of safety + quality score
- `generate_insights(f, ...)` — human-readable bullet-point analysis
- `ten_year_summary(f)` — DataFrame with key metrics per year for the table view

`smart_fetch(query)` is the public entry point: resolves a free-text query or ticker via `yf.Search`, then calls `fetch()`. ASX support was removed — this is a US-only tool despite what the README says.

**`app.py`** — Streamlit UI. Manages a light/dark theme toggle via `st.session_state` (theme swap requires a full page rerender). Builds all Plotly charts inline. Calls `smart_fetch` → analysis functions → renders cards, charts, sensitivity heatmap, 10-year table, and qualitative checklist. The DCF assumption sliders in the sidebar override `auto_assumptions` output.

**`smoke_test.py`** — Exercises every code path using three synthetic `Fundamentals` fixtures (quality growth stock, stable value stock, minimal/empty stock). No network calls. Also mimics the chart-building and formatting code from `app.py` to catch regressions.

## Key design constraints

- All `Fundamentals` fields are `Optional`. Every function that consumes them must guard against `None` — never assume a field is populated.
- EDGAR data supplements yfinance; it is not a fallback. yfinance provides current price, ratios, and quarterly data; EDGAR provides the long annual history.
- The `ALIASES` dict in `analyzer.py` maps internal field names to yfinance DataFrame row labels. The `CONCEPTS` dict in `edgar.py` maps them to SEC us-gaap concept names. Both must stay in sync if new fields are added.
