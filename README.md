# Stock Analyser

A Buffett-style stock analyser for **US** and **ASX** tickers. Live prices, three core fundamentals, two-stage DCF, margin of safety, and a clear **BUY / HOLD / SELL** verdict — with charts.

## What it shows

- **3 fundamentals Buffett checks**: ROE (≥15%), Debt/Equity (≤0.5), EPS growth (≥8%/yr)
- **DCF intrinsic value** from latest free cash flow with adjustable growth, discount, and terminal-growth assumptions
- **Margin of safety** vs the live price → BUY ≥25%, HOLD ≥0%, SELL <0%
- **Charts**: price vs intrinsic, EPS history, ROE history, Debt/Equity history, MoS gauge
- **Insights** + a checklist of qualitative factors (moat, management, recession test)

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Examples: `AAPL`, `MSFT`, `KO` (US) · `CBA`, `BHP`, `WES` (ASX — `.AX` is added automatically).

## Files

- `app.py` — Streamlit UI + Plotly charts
- `analyzer.py` — yfinance data, Buffett scoring, DCF
- `requirements.txt`

## Notes

- Data via [yfinance](https://github.com/ranaroussi/yfinance). Some tickers may have gaps; the app degrades gracefully.
- DCF defaults: growth 8%, discount 10%, terminal 2.5%. Tweak in the sidebar.
- This is a research tool, not financial advice.
