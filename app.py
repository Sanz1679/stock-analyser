"""Terminal-style Buffett stock analyser — tabs, cards, tranches, charts."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analyzer import (
    Fundamentals,
    buffett_score,
    dcf_intrinsic_value,
    dcf_sensitivity,
    fcf_yield,
    fetch,
    graham_number,
    margin_of_safety,
    normalize_ticker,
    price_zone,
    quality_checklist,
    recommendation,
    reverse_dcf_growth,
    tranches,
)

st.set_page_config(page_title="Buffett Terminal", page_icon="◉", layout="wide",
                   initial_sidebar_state="expanded")

# ─── CSS / theme ────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700;800&display=swap');

html, body, [class*="css"], .stApp, p, span, div, label, button, input, textarea {
    font-family: 'JetBrains Mono', 'Courier New', monospace !important;
}
.stApp {
    background: radial-gradient(ellipse at top, #0d1f12 0%, #0a0e0a 50%, #050805 100%) !important;
}
section[data-testid="stSidebar"] {
    background: #0a0e0a !important;
    border-right: 1px solid #22c55e22;
}
h1, h2, h3, h4 { color: #d4d4d4 !important; letter-spacing: 0.02em; }

/* Mastercard-style metric card */
.tcard {
    background: linear-gradient(135deg, #131816 0%, #0d1311 100%);
    border: 1px solid #22c55e22;
    border-left: 3px solid #22c55e;
    border-radius: 14px;
    padding: 1.1rem 1.2rem;
    box-shadow: 0 8px 24px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.03);
    position: relative;
    overflow: hidden;
    margin-bottom: 0.75rem;
    height: 100%;
}
.tcard::after {
    content: '';
    position: absolute; top: -50%; right: -30%;
    width: 200px; height: 200px; border-radius: 50%;
    background: radial-gradient(circle, rgba(34,197,94,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.tcard.amber { border-left-color: #fbbf24; }
.tcard.amber::after { background: radial-gradient(circle, rgba(251,191,36,0.10) 0%, transparent 70%); }
.tcard.red { border-left-color: #ef4444; }
.tcard.red::after { background: radial-gradient(circle, rgba(239,68,68,0.10) 0%, transparent 70%); }
.tcard.blue { border-left-color: #3b82f6; }
.tcard.blue::after { background: radial-gradient(circle, rgba(59,130,246,0.10) 0%, transparent 70%); }

.tlabel {
    color: #6b7280; font-size: 0.7rem; text-transform: uppercase;
    letter-spacing: 0.15em; margin-bottom: 0.45rem; font-weight: 500;
}
.tvalue { font-size: 1.9rem; font-weight: 700; line-height: 1.1; color: #d4d4d4; }
.tvalue.green { color: #22c55e; }
.tvalue.amber { color: #fbbf24; }
.tvalue.red { color: #ef4444; }
.tsub { color: #9ca3af; font-size: 0.78rem; margin-top: 0.4rem; }

/* Verdict hero */
.verdict-hero {
    background: linear-gradient(135deg, var(--vc1) 0%, var(--vc2) 100%);
    border-radius: 18px;
    padding: 1.6rem 2rem;
    text-align: center;
    box-shadow: 0 12px 32px rgba(0,0,0,0.5);
    border: 1px solid rgba(255,255,255,0.08);
    margin: 0.5rem 0 1rem 0;
}
.verdict-label {
    font-size: 0.75rem; letter-spacing: 0.25em; color: rgba(255,255,255,0.7);
    margin-bottom: 0.4rem; text-transform: uppercase;
}
.verdict-text { font-size: 2.6rem; font-weight: 800; color: white; line-height: 1; }
.verdict-reason { color: rgba(255,255,255,0.85); margin-top: 0.6rem; font-size: 0.9rem; }

/* Header bar */
.tickerbar {
    display: flex; align-items: baseline; gap: 1rem;
    border-bottom: 1px solid #22c55e22; padding-bottom: 0.6rem; margin-bottom: 1rem;
}
.tickersym { font-size: 1.8rem; font-weight: 800; color: #22c55e; }
.tickername { color: #9ca3af; font-size: 0.95rem; }
.tprompt::before { content: "> "; color: #22c55e; font-weight: 700; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.25rem; border-bottom: 1px solid #22c55e22; background: transparent;
}
.stTabs [data-baseweb="tab"] {
    background: transparent; color: #6b7280;
    border-radius: 8px 8px 0 0; padding: 0.5rem 1.1rem;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 500; letter-spacing: 0.05em;
}
.stTabs [aria-selected="true"] { color: #22c55e !important; background: #131816 !important; }

/* Inputs */
.stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
    background: #0d1311 !important; border-color: #22c55e44 !important; color: #d4d4d4 !important;
}
.stButton button[kind="primary"] {
    background: #22c55e !important; color: #0a0e0a !important; border: none !important;
    font-weight: 700 !important; letter-spacing: 0.1em;
}
.stSlider [data-baseweb="slider"] { color: #22c55e; }

/* Tranche table */
.tranche-row {
    display: grid; grid-template-columns: 0.5fr 1fr 1fr 1fr 1fr 1fr;
    padding: 0.7rem 1rem; border-bottom: 1px solid #22c55e15;
    align-items: center; gap: 0.5rem;
}
.tranche-row.head { color: #6b7280; font-size: 0.7rem; text-transform: uppercase;
                    letter-spacing: 0.12em; border-bottom: 1px solid #22c55e44; }
.tranche-row.active { background: rgba(34,197,94,0.06); }
.badge-active { background: #22c55e; color: #0a0e0a; padding: 0.2rem 0.6rem;
                border-radius: 4px; font-size: 0.7rem; font-weight: 700; letter-spacing: 0.1em; }
.badge-wait   { background: #1f2937; color: #9ca3af; padding: 0.2rem 0.6rem;
                border-radius: 4px; font-size: 0.7rem; letter-spacing: 0.1em; }

/* Checklist rows */
.chk-pass { color: #22c55e; }
.chk-fail { color: #ef4444; }
hr { border-color: #22c55e22 !important; }
</style>
""", unsafe_allow_html=True)

# ─── helpers ────────────────────────────────────────────────────────────────
def fmt_money(x, currency="USD"):
    if x is None: return "—"
    sym = {"USD": "$", "AUD": "A$"}.get(currency, "")
    if abs(x) >= 1e12: return f"{sym}{x/1e12:.2f}T"
    if abs(x) >= 1e9:  return f"{sym}{x/1e9:.2f}B"
    if abs(x) >= 1e6:  return f"{sym}{x/1e6:.2f}M"
    return f"{sym}{x:,.2f}"

def fmt_pct(x, decimals=1):
    return f"{x:.{decimals}f}%" if x is not None else "—"

def card(label, value, sub="", color="green"):
    cls = {"green":"", "amber":"amber", "red":"red", "blue":"blue", "neutral":""}.get(color, "")
    val_color = {"green":"green","amber":"amber","red":"red","blue":"green","neutral":""}.get(color, "")
    return f"""
    <div class="tcard {cls}">
      <div class="tlabel">{label}</div>
      <div class="tvalue {val_color}">{value}</div>
      <div class="tsub">{sub}</div>
    </div>
    """

def color_for(value, good_above=None, good_below=None):
    if value is None: return "neutral"
    if good_above is not None:
        if value >= good_above: return "green"
        if value >= good_above * 0.66: return "amber"
        return "red"
    if good_below is not None:
        if value <= good_below: return "green"
        if value <= good_below * 1.5: return "amber"
        return "red"
    return "neutral"

def year_index(series: pd.Series) -> list[str]:
    return [str(d.year) if hasattr(d, "year") else str(d) for d in series.index]

def dark_layout(title, height=320):
    return dict(title=title, height=height, template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=10, r=10, t=40, b=10), font=dict(family="JetBrains Mono"))

# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="tprompt"><b>BUFFETT TERMINAL</b></div>', unsafe_allow_html=True)
    st.caption("v2 · live yfinance feed")
    st.divider()
    market = st.radio("MARKET", ["US", "ASX"], horizontal=True)
    placeholder = "AAPL" if market == "US" else "CBA"
    ticker_input = st.text_input("TICKER", placeholder)
    st.divider()
    st.markdown("**DCF ASSUMPTIONS**")
    growth = st.slider("Growth %/yr (yr 1-10)", 2.0, 20.0, 8.0, 0.5)
    discount = st.slider("Discount rate %", 6.0, 15.0, 10.0, 0.5)
    terminal = st.slider("Terminal growth %", 1.0, 4.0, 2.5, 0.25)
    st.divider()
    st.markdown("**POSITION SIZING**")
    budget = st.number_input("Total budget", min_value=100.0, value=10000.0, step=500.0)
    t1 = st.slider("Tranche 1 discount %", 5, 30, 15)
    t2 = st.slider("Tranche 2 discount %", 20, 45, 30)
    t3 = st.slider("Tranche 3 discount %", 35, 60, 45)
    st.divider()
    run = st.button("► ANALYSE", type="primary", use_container_width=True)


# ─── Render ─────────────────────────────────────────────────────────────────
def render_header(f: Fundamentals, mos, verdict, color, reason):
    change = None
    if f.price and f.prev_close:
        change = (f.price - f.prev_close) / f.prev_close * 100
    chg_str = f"<span style='color:{'#22c55e' if (change or 0)>=0 else '#ef4444'}'>{'+' if (change or 0)>=0 else ''}{change:.2f}%</span>" if change is not None else ""

    st.markdown(f"""
    <div class="tickerbar">
      <span class="tickersym">{f.ticker}</span>
      <span class="tickername">{f.name} · {f.sector or 'n/a'}</span>
      <span style="margin-left:auto; font-size:1.4rem; color:#d4d4d4;">{fmt_money(f.price, f.currency)} {chg_str}</span>
    </div>
    """, unsafe_allow_html=True)

    # Verdict hero
    grad = {"#22c55e": ("#16a34a", "#15803d"),
            "#fbbf24": ("#d97706", "#b45309"),
            "#ef4444": ("#dc2626", "#991b1b"),
            "#888888": ("#374151", "#1f2937")}[color]
    st.markdown(f"""
    <div class="verdict-hero" style="--vc1:{grad[0]}; --vc2:{grad[1]}">
      <div class="verdict-label">Today's verdict</div>
      <div class="verdict-text">{verdict}</div>
      <div class="verdict-reason">{reason}</div>
    </div>
    """, unsafe_allow_html=True)


def render_overview(f: Fundamentals, intrinsic, mos, verdict, color, score, t1, t2, t3, budget):
    fcfy = fcf_yield(f.market_cap, f.fcf_latest)

    # 4-card top row
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(card("INTRINSIC (DCF)", fmt_money(intrinsic, f.currency),
                     f"{fmt_pct(mos)} margin of safety", color="green" if mos and mos >= 25 else "amber" if mos and mos >= 0 else "red"),
                unsafe_allow_html=True)
    c2.markdown(card("MARKET CAP", fmt_money(f.market_cap, f.currency),
                     f"P/E {f.pe_ratio:.1f}" if f.pe_ratio else "P/E —", color="blue"),
                unsafe_allow_html=True)
    c3.markdown(card("FCF YIELD", fmt_pct(fcfy),
                     f"FCF {fmt_money(f.fcf_latest, f.currency)}",
                     color=color_for(fcfy, good_above=5)),
                unsafe_allow_html=True)
    c4.markdown(card("DIVIDEND YIELD", fmt_pct(f.dividend_yield) if f.dividend_yield else "—",
                     f"Payout {fmt_pct(f.payout_ratio)}" if f.payout_ratio else "No payout data",
                     color="blue"),
                unsafe_allow_html=True)

    st.markdown("### " + "PRICE ZONE")
    pz = price_zone(f.price_history, f.price, f.fifty_two_high, f.fifty_two_low)
    if pz:
        fig = go.Figure()
        fig.add_shape(type="rect", x0=pz["low"], x1=pz["low"]+(pz["high"]-pz["low"])*0.33,
                      y0=0, y1=1, fillcolor="#22c55e22", line_width=0)
        fig.add_shape(type="rect", x0=pz["low"]+(pz["high"]-pz["low"])*0.33, x1=pz["low"]+(pz["high"]-pz["low"])*0.66,
                      y0=0, y1=1, fillcolor="#fbbf2422", line_width=0)
        fig.add_shape(type="rect", x0=pz["low"]+(pz["high"]-pz["low"])*0.66, x1=pz["high"],
                      y0=0, y1=1, fillcolor="#ef444422", line_width=0)
        # Markers
        markers = [("52w low", pz["low"], "#22c55e"),
                   ("52w high", pz["high"], "#ef4444"),
                   ("Current", pz["current"], "#ffffff")]
        if intrinsic: markers.append(("Intrinsic", intrinsic, "#3b82f6"))
        if f.fifty_day_avg: markers.append(("50d MA", f.fifty_day_avg, "#fbbf24"))
        if f.two_hundred_day_avg: markers.append(("200d MA", f.two_hundred_day_avg, "#a855f7"))
        for name, val, col in markers:
            fig.add_trace(go.Scatter(x=[val], y=[0.5], mode="markers+text",
                                     marker=dict(size=14, color=col, line=dict(color="white", width=1)),
                                     text=[name], textposition="top center",
                                     textfont=dict(color=col, size=11), showlegend=False,
                                     hovertemplate=f"{name}: %{{x:.2f}}<extra></extra>"))
        fig.update_layout(**dark_layout("", height=200), yaxis=dict(visible=False, range=[0,1]),
                          xaxis=dict(title="", showgrid=False))
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Position in 52w range: **{pz['position_pct']:.0f}%** "
                   f"(0% = at low, 100% = at high). Green band = value zone.")
    else:
        st.info("Price-zone data unavailable.")

    # Tranche table
    st.markdown("### POSITION SIZING — BUY TRANCHES")
    if not intrinsic:
        st.warning("Need a valid DCF intrinsic value to compute tranches.")
        return
    tr = tranches(intrinsic, f.price,
                  discounts=(t1/100, t2/100, t3/100),
                  allocations=(0.25, 0.35, 0.40),
                  budget=budget)
    head = """<div class="tranche-row head">
      <div>#</div><div>Trigger price</div><div>Discount</div>
      <div>Allocation</div><div>$ to deploy</div><div>Status</div></div>"""
    rows = [head]
    cur = "USD" if not f.currency else f.currency
    for i, t in enumerate(tr, 1):
        active_cls = "active" if t["active"] else ""
        badge = '<span class="badge-active">● ACTIVE</span>' if t["active"] \
                else f'<span class="badge-wait">wait {t["distance_pct"]:+.1f}%</span>'
        rows.append(f"""<div class="tranche-row {active_cls}">
          <div><b>{i}</b></div>
          <div>{fmt_money(t["trigger"], cur)}</div>
          <div>-{t["discount_pct"]:.0f}%</div>
          <div>{t["allocation_pct"]:.0f}%</div>
          <div>{fmt_money(t["dollars"], cur)}</div>
          <div>{badge}</div>
        </div>""")
    st.markdown("".join(rows), unsafe_allow_html=True)
    st.caption("Tranches let you scale in: deploy more capital as the discount widens. "
               "Adjust thresholds in the sidebar.")


def render_fundamentals(f: Fundamentals):
    st.markdown("### CORE BUFFETT METRICS")
    c1, c2, c3 = st.columns(3)
    c1.markdown(card("ROE (avg)", fmt_pct(f.roe_avg),
                     "Buffett wants ≥15%", color=color_for(f.roe_avg, good_above=15)),
                unsafe_allow_html=True)
    c2.markdown(card("DEBT / EQUITY", f"{f.de_latest:.2f}" if f.de_latest is not None else "—",
                     "Lower is safer (≤0.5)", color=color_for(f.de_latest, good_below=0.5)),
                unsafe_allow_html=True)
    c3.markdown(card("EPS GROWTH (CAGR)", fmt_pct(f.eps_growth),
                     "Strong if ≥8%/yr", color=color_for(f.eps_growth, good_above=8)),
                unsafe_allow_html=True)

    st.markdown("### EXTENDED METRICS")
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(card("FCF GROWTH", fmt_pct(f.fcf_growth), "5y CAGR",
                     color=color_for(f.fcf_growth, good_above=8)), unsafe_allow_html=True)
    c2.markdown(card("REVENUE GROWTH", fmt_pct(f.revenue_growth), "5y CAGR",
                     color=color_for(f.revenue_growth, good_above=5)), unsafe_allow_html=True)
    c3.markdown(card("OPERATING MARGIN", fmt_pct(f.operating_margin), "Pricing power",
                     color=color_for(f.operating_margin, good_above=15)), unsafe_allow_html=True)
    c4.markdown(card("EARNINGS STABILITY", f"{f.earnings_stability:.0f}/100" if f.earnings_stability else "—",
                     "Higher = more predictable",
                     color=color_for(f.earnings_stability, good_above=60)), unsafe_allow_html=True)

    st.markdown("### OWNER EARNINGS")
    if f.owner_earnings_latest is not None:
        oe_yield = fcf_yield(f.market_cap, f.owner_earnings_latest)
        c1, c2 = st.columns(2)
        c1.markdown(card("OWNER EARNINGS (latest)", fmt_money(f.owner_earnings_latest, f.currency),
                         "NI + D&A − Capex (Buffett's preferred FCF)",
                         color=color_for(f.owner_earnings_latest, good_above=0)), unsafe_allow_html=True)
        c2.markdown(card("OWNER-EARNINGS YIELD", fmt_pct(oe_yield),
                         "Higher = better cash return per $ invested",
                         color=color_for(oe_yield, good_above=5)), unsafe_allow_html=True)
    else:
        st.info("Owner-earnings data unavailable.")


def render_valuation(f: Fundamentals, intrinsic, growth, discount, terminal, mos):
    st.markdown("### VALUATION CROSS-CHECK")
    g_num = graham_number(f.eps_ttm, f.book_value_per_share)
    g_mos = ((g_num - f.price) / g_num * 100) if (g_num and f.price) else None
    rev_g = reverse_dcf_growth(f.price, f.fcf_latest, f.shares_outstanding,
                               discount/100, terminal/100, 10, f.net_cash or 0)

    c1, c2, c3 = st.columns(3)
    c1.markdown(card("DCF INTRINSIC", fmt_money(intrinsic, f.currency),
                     f"MoS {fmt_pct(mos)}",
                     color="green" if mos and mos >= 25 else "amber" if mos and mos >= 0 else "red"),
                unsafe_allow_html=True)
    c2.markdown(card("GRAHAM NUMBER", fmt_money(g_num, f.currency),
                     f"MoS {fmt_pct(g_mos)}" if g_mos is not None else "Needs +ve EPS & BVPS",
                     color="green" if g_mos and g_mos >= 25 else "amber" if g_mos and g_mos >= 0 else "red"),
                unsafe_allow_html=True)
    c3.markdown(card("MARKET-IMPLIED GROWTH", fmt_pct(rev_g),
                     "Reverse-DCF: growth needed to justify price",
                     color="green" if rev_g and rev_g <= 8 else "amber" if rev_g and rev_g <= 15 else "red"),
                unsafe_allow_html=True)
    if rev_g is not None:
        st.caption(f"At today's price the market is pricing in **{rev_g:.1f}%/yr** FCF growth. "
                   f"If you believe actual growth will exceed that, the stock is undervalued.")

    # Sensitivity
    st.markdown("### DCF SENSITIVITY — INTRINSIC PER SHARE")
    if f.fcf_latest and f.shares_outstanding:
        g_grid = [0.04, 0.06, 0.08, 0.10, 0.12]
        d_grid = [0.08, 0.09, 0.10, 0.11, 0.12]
        sens = dcf_sensitivity(f.fcf_latest, f.shares_outstanding, g_grid, d_grid,
                               terminal/100, 10, f.net_cash or 0)
        if sens is not None:
            sens_disp = sens.applymap(lambda v: f"{v:,.2f}" if v is not None else "—")
            st.markdown("Rows = discount rate · Columns = growth rate")
            sens_for_style = sens.copy()
            try:
                styled = sens_for_style.style.background_gradient(cmap="Greens", axis=None) \
                    .format("{:,.2f}")
                st.dataframe(styled, use_container_width=True)
            except Exception:
                st.dataframe(sens_disp, use_container_width=True)
            st.caption("Try ±2% growth × ±1% discount. If verdict flips inside the grid, your DCF isn't robust.")
    else:
        st.info("Sensitivity grid unavailable (no FCF or shares data).")


def render_charts(f: Fundamentals, intrinsic):
    g1, g2 = st.columns(2)
    with g1:
        if not f.price_history.empty and intrinsic and f.price:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=f.price_history.index, y=f.price_history["Close"],
                                     name="Price", line=dict(color="#22c55e", width=2)))
            fig.add_hline(y=intrinsic, line_dash="dash", line_color="#3b82f6",
                          annotation_text=f"Intrinsic {fmt_money(intrinsic,f.currency)}")
            fig.add_hline(y=intrinsic * 0.75, line_dash="dot", line_color="#fbbf24",
                          annotation_text="Buy zone (-25%)")
            fig.update_layout(**dark_layout("Price vs Intrinsic"))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Price chart unavailable.")
    with g2:
        if not f.eps_history.empty:
            fig = go.Figure(go.Bar(x=year_index(f.eps_history), y=f.eps_history.values,
                                   marker_color="#22c55e"))
            fig.update_layout(**dark_layout("EPS history"))
            st.plotly_chart(fig, use_container_width=True)

    g3, g4 = st.columns(2)
    with g3:
        if not f.roe_history.empty:
            fig = go.Figure(go.Bar(x=year_index(f.roe_history), y=f.roe_history.values,
                marker_color=["#22c55e" if v>=15 else "#fbbf24" if v>=10 else "#ef4444"
                              for v in f.roe_history.values]))
            fig.add_hline(y=15, line_dash="dash", line_color="#22c55e",
                          annotation_text="Buffett 15%")
            fig.update_layout(**dark_layout("ROE % history"))
            st.plotly_chart(fig, use_container_width=True)
    with g4:
        if not f.de_history.empty:
            fig = go.Figure(go.Bar(x=year_index(f.de_history), y=f.de_history.values,
                marker_color=["#22c55e" if v<=0.5 else "#fbbf24" if v<=1.0 else "#ef4444"
                              for v in f.de_history.values]))
            fig.add_hline(y=0.5, line_dash="dash", line_color="#22c55e",
                          annotation_text="Safe 0.5")
            fig.update_layout(**dark_layout("Debt / Equity history"))
            st.plotly_chart(fig, use_container_width=True)

    g5, g6 = st.columns(2)
    with g5:
        if not f.fcf_history.empty:
            fig = go.Figure(go.Bar(x=year_index(f.fcf_history), y=f.fcf_history.values,
                marker_color=["#22c55e" if v>=0 else "#ef4444" for v in f.fcf_history.values]))
            fig.update_layout(**dark_layout("Free Cash Flow"))
            st.plotly_chart(fig, use_container_width=True)
    with g6:
        if not f.revenue_history.empty:
            fig = go.Figure(go.Bar(x=year_index(f.revenue_history), y=f.revenue_history.values,
                                   marker_color="#3b82f6"))
            fig.update_layout(**dark_layout("Revenue"))
            st.plotly_chart(fig, use_container_width=True)


def render_checklist(f: Fundamentals, mos, color):
    items = quality_checklist(f)
    passed = sum(1 for i in items if i["pass"])
    st.markdown(f"### AUTO-CHECKLIST — {passed}/{len(items)} PASS")

    rows = ['<div class="tranche-row head"><div>STATUS</div><div>CHECK</div><div>VALUE</div></div>']
    for i in items:
        icon = '<span class="chk-pass">✓ PASS</span>' if i["pass"] else '<span class="chk-fail">✗ FAIL</span>'
        rows.append(f"""<div class="tranche-row" style="grid-template-columns: 0.7fr 2fr 1fr">
          <div>{icon}</div><div>{i["check"]}</div><div>{i["value"]}</div></div>""")
    st.markdown("".join(rows), unsafe_allow_html=True)

    # MoS gauge
    if mos is not None:
        gauge_val = max(min(mos, 80), -80)
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=gauge_val, number={"suffix": "%"},
            title={"text": "Margin of Safety"},
            gauge={"axis": {"range": [-80, 80], "tickcolor": "#6b7280"},
                   "bar": {"color": color},
                   "bgcolor": "#0a0e0a", "borderwidth": 0,
                   "steps": [{"range": [-80, 0], "color": "#7f1d1d"},
                             {"range": [0, 25], "color": "#78350f"},
                             {"range": [25, 80], "color": "#14532d"}],
                   "threshold": {"line": {"color": "white", "width": 3}, "value": 25}}))
        fig.update_layout(**dark_layout("", height=260))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### QUALITATIVE — TICK WHAT YOU'VE VERIFIED")
    qa = [
        "Wide moat (brand, network, switching costs, scale)",
        "Honest, capital-disciplined management",
        "Held up through 2008 / 2020 recessions",
        "Customer base diversified (no >20% concentration)",
        "Not a commodity / cyclical business",
        "Within your circle of competence",
        "Position size sane (≤10% of portfolio)",
    ]
    for q in qa:
        st.checkbox(q, key=f"qa_{q}")


# ─── MAIN ───────────────────────────────────────────────────────────────────
st.markdown('<div class="tprompt" style="font-size:0.8rem; color:#22c55e; letter-spacing:0.2em;">BUFFETT_TERMINAL@v2 :: live_feed</div>',
            unsafe_allow_html=True)
st.markdown("# ◉ Stock Analyser")

if not run:
    st.markdown("""
    <div class="tcard" style="text-align:center; padding:2rem;">
      <div class="tlabel">READY</div>
      <div class="tvalue green">> awaiting ticker</div>
      <div class="tsub">Enter a US or ASX ticker in the sidebar and hit ANALYSE.</div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

ticker = normalize_ticker(ticker_input, market)
with st.spinner(f"> fetching {ticker}…"):
    try:
        f = fetch(ticker)
    except Exception as e:
        st.error(f"Could not load {ticker}: {e}")
        st.stop()

if f.price is None:
    st.error(f"No price data for {ticker}. Check the ticker symbol.")
    st.stop()

intrinsic = dcf_intrinsic_value(
    latest_fcf=f.fcf_latest,
    shares_outstanding=f.shares_outstanding,
    growth_rate=growth/100, discount_rate=discount/100,
    terminal_growth=terminal/100, net_cash=f.net_cash or 0,
)
mos = margin_of_safety(intrinsic, f.price)
score, notes = buffett_score(f)
verdict, color, reason = recommendation(mos, score)

render_header(f, mos, verdict, color, reason)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["◉ OVERVIEW", "▤ FUNDAMENTALS", "$ VALUATION",
                                         "▦ CHARTS", "✓ CHECKLIST"])
with tab1: render_overview(f, intrinsic, mos, verdict, color, score, t1, t2, t3, budget)
with tab2: render_fundamentals(f)
with tab3: render_valuation(f, intrinsic, growth, discount, terminal, mos)
with tab4: render_charts(f, intrinsic)
with tab5: render_checklist(f, mos, color)
