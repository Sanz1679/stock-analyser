"""Streamlit UI for the Buffett-style stock analyser."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analyzer import (
    Fundamentals,
    buffett_score,
    dcf_intrinsic_value,
    fetch,
    margin_of_safety,
    normalize_ticker,
    recommendation,
)

st.set_page_config(page_title="Buffett Stock Analyser", page_icon="📈", layout="wide")

st.markdown(
    """
    <style>
      .verdict { font-size: 2.2rem; font-weight: 700; padding: 0.6rem 1rem;
                 border-radius: 8px; color: white; text-align: center; }
      .metric-pill { background: #f3f4f6; padding: 0.35rem 0.7rem;
                     border-radius: 999px; font-size: 0.9rem; margin-right: 0.4rem; }
      div[data-testid="stMetricValue"] { font-size: 1.4rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📈 Buffett Stock Analyser")
st.caption("Live fundamentals · DCF · margin of safety. US & ASX tickers.")

with st.sidebar:
    st.header("Inputs")
    market = st.radio("Market", ["US", "ASX"], horizontal=True)
    placeholder = "AAPL" if market == "US" else "CBA"
    ticker_input = st.text_input("Ticker", placeholder, help="e.g. AAPL, MSFT (US) · CBA, BHP (ASX)")
    st.divider()
    st.subheader("DCF assumptions")
    growth = st.slider("Growth rate (yr 1-10)", 2.0, 20.0, 8.0, 0.5,
                       help="Buffett uses conservative growth — 8% is a reasonable default.")
    discount = st.slider("Discount rate", 6.0, 15.0, 10.0, 0.5,
                         help="Buffett's hurdle rate is ~10%.")
    terminal = st.slider("Terminal growth", 1.0, 4.0, 2.5, 0.25,
                         help="Long-run GDP-like growth.")
    run = st.button("Analyse", type="primary", use_container_width=True)


def fmt_money(x, currency="USD"):
    if x is None:
        return "—"
    sym = {"USD": "$", "AUD": "A$"}.get(currency, "")
    if abs(x) >= 1e12:
        return f"{sym}{x/1e12:.2f}T"
    if abs(x) >= 1e9:
        return f"{sym}{x/1e9:.2f}B"
    if abs(x) >= 1e6:
        return f"{sym}{x/1e6:.2f}M"
    return f"{sym}{x:,.2f}"


def year_index(series: pd.Series) -> list[str]:
    return [str(d.year) if hasattr(d, "year") else str(d) for d in series.index]


def render(f: Fundamentals, growth: float, discount: float, terminal: float):
    intrinsic = dcf_intrinsic_value(
        latest_fcf=f.fcf_latest,
        shares_outstanding=f.shares_outstanding,
        growth_rate=growth / 100,
        discount_rate=discount / 100,
        terminal_growth=terminal / 100,
        net_cash=f.net_cash or 0,
    )
    mos = margin_of_safety(intrinsic, f.price)
    score, notes = buffett_score(f)
    verdict, color, reason = recommendation(mos, score)

    # Header row
    st.subheader(f"{f.name} · {f.ticker}")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Price", fmt_money(f.price, f.currency))
    c2.metric("Intrinsic (DCF)", fmt_money(intrinsic, f.currency))
    c3.metric("Margin of Safety", f"{mos:.1f}%" if mos is not None else "—")
    c4.metric("Market Cap", fmt_money(f.market_cap, f.currency))

    st.markdown(
        f'<div class="verdict" style="background:{color}">{verdict}</div>',
        unsafe_allow_html=True,
    )
    st.caption(reason)

    # 3 Buffett fundamentals
    st.markdown("### Buffett's 3 fundamentals")
    f1, f2, f3 = st.columns(3)
    f1.metric("ROE (avg)", f"{f.roe_avg:.1f}%" if f.roe_avg is not None else "—",
              help="Return on Equity — Buffett wants >=15% sustained.")
    f2.metric("Debt / Equity", f"{f.de_latest:.2f}" if f.de_latest is not None else "—",
              help="Lower is safer. Buffett prefers <=0.5.")
    f3.metric("EPS growth (CAGR)", f"{f.eps_growth:.1f}%" if f.eps_growth is not None else "—",
              help="Consistent earnings growth >= 8%/yr is strong.")

    st.markdown(f"**Quality score: {score}/3**")
    for n in notes:
        st.markdown(f"- {n}")

    # Charts
    st.markdown("### Charts")
    g1, g2 = st.columns(2)

    with g1:
        if not f.price_history.empty and intrinsic is not None and f.price is not None:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=f.price_history.index, y=f.price_history["Close"],
                name="Price", line=dict(color="#2563eb", width=2)))
            fig.add_hline(y=intrinsic, line_dash="dash", line_color="#16a34a",
                          annotation_text=f"Intrinsic {fmt_money(intrinsic, f.currency)}",
                          annotation_position="top right")
            fig.add_hline(y=intrinsic * 0.75, line_dash="dot", line_color="#f59e0b",
                          annotation_text="Buy zone (-25%)", annotation_position="bottom right")
            fig.update_layout(title="Price vs Intrinsic Value", height=350,
                              margin=dict(l=10, r=10, t=40, b=10), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Price/intrinsic chart unavailable.")

    with g2:
        if not f.eps_history.empty:
            fig = go.Figure(go.Bar(
                x=year_index(f.eps_history), y=f.eps_history.values,
                marker_color="#2563eb"))
            fig.update_layout(title="EPS history", height=350,
                              margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("EPS history unavailable.")

    g3, g4 = st.columns(2)

    with g3:
        if not f.roe_history.empty:
            fig = go.Figure(go.Bar(
                x=year_index(f.roe_history), y=f.roe_history.values,
                marker_color=["#16a34a" if v >= 15 else "#f59e0b" if v >= 10 else "#dc2626"
                              for v in f.roe_history.values]))
            fig.add_hline(y=15, line_dash="dash", line_color="#16a34a",
                          annotation_text="Buffett threshold (15%)")
            fig.update_layout(title="ROE % history", height=350,
                              margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("ROE history unavailable.")

    with g4:
        if not f.de_history.empty:
            fig = go.Figure(go.Bar(
                x=year_index(f.de_history), y=f.de_history.values,
                marker_color=["#16a34a" if v <= 0.5 else "#f59e0b" if v <= 1.0 else "#dc2626"
                              for v in f.de_history.values]))
            fig.add_hline(y=0.5, line_dash="dash", line_color="#16a34a",
                          annotation_text="Safe (0.5)")
            fig.update_layout(title="Debt / Equity history", height=350,
                              margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Debt/equity history unavailable.")

    # Margin of safety gauge
    if mos is not None:
        gauge_val = max(min(mos, 80), -80)
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=gauge_val,
            number={"suffix": "%"},
            title={"text": "Margin of Safety"},
            gauge={
                "axis": {"range": [-80, 80]},
                "bar": {"color": color},
                "steps": [
                    {"range": [-80, 0], "color": "#fee2e2"},
                    {"range": [0, 25], "color": "#fef3c7"},
                    {"range": [25, 80], "color": "#dcfce7"},
                ],
                "threshold": {"line": {"color": "black", "width": 3}, "value": 25},
            },
        ))
        fig.update_layout(height=280, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    # Insights
    st.markdown("### Insights")
    insights: list[str] = []
    if f.fcf_growth is not None:
        insights.append(f"Free cash flow growing ~{f.fcf_growth:.1f}%/yr — your DCF assumption is "
                        f"{'aggressive' if growth > f.fcf_growth + 2 else 'conservative' if growth < f.fcf_growth - 2 else 'aligned'}.")
    if f.de_latest is not None and f.de_latest > 1.0:
        insights.append("High leverage — earnings sensitive to interest rates.")
    if f.roe_avg is not None and f.roe_avg >= 20:
        insights.append("Exceptional ROE — likely a wide-moat business.")
    if mos is not None and mos < 0 and score >= 2:
        insights.append("Quality business but priced for perfection — wait for a pullback.")
    if mos is not None and mos > 25 and score < 2:
        insights.append("Cheap on DCF but quality flags weak — could be a value trap.")
    if not insights:
        insights.append("No standout signals — judge against peers and your own thesis.")
    for i in insights:
        st.markdown(f"- {i}")

    with st.expander("What else to consider"):
        st.markdown(
            """
- **Moat** — what stops competitors? Brand, network effects, switching costs, scale.
- **Management** — capital allocation track record, insider ownership, candor in shareholder letters.
- **Owner earnings** — Buffett prefers FCF minus maintenance capex over reported EPS.
- **Recession test** — how did earnings hold up in 2008 / 2020?
- **Concentration & cyclicality** — customer concentration, commodity exposure.
- **Position sizing** — even a great buy shouldn't be your only holding.
- **DCF sensitivity** — try ±2% growth and ±1% discount and see if the verdict flips.
            """
        )


if run:
    ticker = normalize_ticker(ticker_input, market)
    with st.spinner(f"Fetching {ticker}…"):
        try:
            f = fetch(ticker)
        except Exception as e:
            st.error(f"Could not load {ticker}: {e}")
            st.stop()
    if f.price is None:
        st.error(f"No price data for {ticker}. Check the ticker symbol.")
        st.stop()
    render(f, growth, discount, terminal)
else:
    st.info("Enter a ticker on the left and hit **Analyse**.")
