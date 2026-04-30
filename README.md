# Stock Analyser

A Buffett-style stock analyser for **US** and **ASX** tickers. Runs as a **web app** in Chrome / Edge — live prices, three core fundamentals, two-stage DCF, margin of safety, and a clear **BUY / HOLD / SELL** verdict with charts.

## What it shows

- **3 fundamentals Buffett checks**: ROE (≥15%), Debt/Equity (≤0.5), EPS growth (≥8%/yr)
- **DCF intrinsic value** from latest free cash flow with adjustable growth, discount, and terminal-growth assumptions
- **Margin of safety** vs the live price → BUY ≥25%, HOLD ≥0%, SELL <0%
- **Charts**: price vs intrinsic, EPS history, ROE history, Debt/Equity history, MoS gauge
- **Insights** + qualitative checklist (moat, management, recession test)

## Open it in Chrome / Edge

Pick whichever fits — all three give you a URL you paste in your browser.

### Option 1 — Streamlit Community Cloud (free, public URL, recommended)

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Click **New app**, pick `Sanz1679/stock-analyser`, branch `main`, file `app.py`.
3. Click **Deploy**. You'll get a `https://<name>.streamlit.app` URL — open it in Chrome/Edge.

No install, works from any device.

### Option 2 — Run locally (one command)

```bash
pip install -r requirements.txt
streamlit run app.py
```

Streamlit auto-opens `http://localhost:8501` in your default browser. If it doesn't, paste that URL into Chrome/Edge.

### Option 3 — Docker (any host)

```bash
docker build -t stock-analyser .
docker run -p 8501:8501 stock-analyser
```

Then open `http://localhost:8501`. Deployable to any Docker host (Render, Railway, Fly.io, AWS, etc).

## Usage

- Pick **US** or **ASX**, type a ticker, hit **Analyse**.
- Examples: `AAPL`, `MSFT`, `KO` (US) · `CBA`, `BHP`, `WES` (ASX — `.AX` added automatically).
- Tweak DCF assumptions in the sidebar.

## Files

- `app.py` — Streamlit UI + Plotly charts
- `analyzer.py` — yfinance data, Buffett scoring, DCF
- `requirements.txt`, `Dockerfile`, `Procfile`, `.streamlit/config.toml`

## Notes

- Data via [yfinance](https://github.com/ranaroussi/yfinance). Some tickers may have gaps; the app degrades gracefully.
- DCF defaults: growth 8%, discount 10%, terminal 2.5%.
- Research tool, not financial advice.
