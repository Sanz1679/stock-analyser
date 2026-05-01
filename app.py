"""Clean, calm stock analyser. Light + dark mode, single ticker input, auto-DCF."""
from __future__ import annotations

import datetime as dt
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analyzer import (
    Fundamentals,
    auto_assumptions,
    dcf_intrinsic_value,
    dcf_sensitivity,
    fcf_yield,
    generate_insights,
    graham_number,
    margin_of_safety,
    quality_score,
    recommendation,
    reverse_dcf_growth,
    smart_fetch,
    ten_year_summary,
)

st.set_page_config(page_title="Stock Analyser", page_icon="●", layout="wide",
                   initial_sidebar_state="collapsed")

# ─── Theme state ────────────────────────────────────────────────────────────
if "theme" not in st.session_state:
    st.session_state.theme = "light"
THEME = st.session_state.theme

PALETTES = {
    "light": {
        "bg": "#f8fafc", "card": "#ffffff", "text": "#0f172a", "muted": "#64748b",
        "border": "#e2e8f0", "subtle": "#f1f5f9",
        "primary": "#2563eb", "positive": "#059669", "warn": "#d97706",
        "negative": "#dc2626", "neutral": "#94a3b8",
        "chart_bg": "#ffffff", "plotly_template": "plotly_white",
        "shadow": "0 1px 3px rgba(15,23,42,0.06), 0 1px 2px rgba(15,23,42,0.04)",
    },
    "dark": {
        "bg": "#0b1120", "card": "#111827", "text": "#e5e7eb", "muted": "#9ca3af",
        "border": "#1f2937", "subtle": "#0f172a",
        "primary": "#60a5fa", "positive": "#34d399", "warn": "#fbbf24",
        "negative": "#f87171", "neutral": "#6b7280",
        "chart_bg": "#111827", "plotly_template": "plotly_dark",
        "shadow": "0 1px 3px rgba(0,0,0,0.4), 0 1px 2px rgba(0,0,0,0.3)",
    },
}
P = PALETTES[THEME]

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"], .stApp, p, span, div, label, button, input, textarea, h1, h2, h3, h4, h5, h6 {{
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
}}
.stApp, .main {{ background: {P['bg']} !important; }}
.block-container {{ padding-top: 1.5rem !important; padding-bottom: 4rem !important; max-width: 1280px !important; }}

h1 {{ color: {P['text']} !important; font-weight: 600 !important; font-size: 1.5rem !important; margin: 0 !important; }}
h2 {{ color: {P['text']} !important; font-weight: 600 !important; font-size: 1.05rem !important; margin: 0.5rem 0 0.6rem !important; }}
h3 {{ color: {P['text']} !important; font-weight: 600 !important; font-size: 0.95rem !important; }}
p, label, span, div {{ color: {P['text']}; }}
.muted {{ color: {P['muted']} !important; }}

.card {{
    background: {P['card']}; border: 1px solid {P['border']}; border-radius: 10px;
    padding: 1rem 1.1rem; box-shadow: {P['shadow']}; margin-bottom: 0.75rem;
}}
.metric-group {{
    background: {P['card']}; border: 1px solid {P['border']}; border-radius: 10px;
    padding: 1rem 1.1rem; box-shadow: {P['shadow']}; height: 100%;
}}
.metric-group h4 {{
    margin: 0 0 0.7rem 0; color: {P['text']}; font-size: 0.78rem; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.06em;
}}
.metric-row {{ display: flex; justify-content: space-between; align-items: baseline;
              padding: 0.32rem 0; border-bottom: 1px solid {P['subtle']}; font-size: 0.85rem; }}
.metric-row:last-child {{ border-bottom: none; }}
.metric-row .lbl {{ color: {P['muted']}; }}
.metric-row .val {{ color: {P['text']}; font-weight: 500; }}

.verdict-pill {{
    display: inline-flex; align-items: center; gap: 0.4rem;
    padding: 0.32rem 0.85rem; border-radius: 999px; font-weight: 600;
    font-size: 0.82rem; letter-spacing: 0.04em;
}}
.v-positive {{ background: {'#dcfce7' if THEME=='light' else '#064e3b'}; color: {P['positive']}; }}
.v-warn {{ background: {'#fef3c7' if THEME=='light' else '#78350f'}; color: {P['warn']}; }}
.v-negative {{ background: {'#fee2e2' if THEME=='light' else '#7f1d1d'}; color: {P['negative']}; }}
.v-neutral {{ background: {P['subtle']}; color: {P['muted']}; }}

.day-change-pos {{ color: {P['positive']}; font-weight: 500; font-size: 0.9rem; }}
.day-change-neg {{ color: {P['negative']}; font-weight: 500; font-size: 0.9rem; }}

.ticker-bar {{
    display: flex; align-items: center; gap: 1rem; flex-wrap: wrap;
    padding: 0.4rem 0 1rem 0; border-bottom: 1px solid {P['border']}; margin-bottom: 1.2rem;
}}
.ticker-sym {{ font-size: 1.6rem; font-weight: 700; color: {P['text']}; }}
.ticker-name {{ color: {P['muted']}; font-size: 0.92rem; }}
.ticker-price {{ font-size: 1.4rem; font-weight: 600; color: {P['text']}; }}

.insight-card {{
    background: {P['card']}; border: 1px solid {P['border']}; border-radius: 10px;
    padding: 0.9rem 1.1rem; margin-bottom: 0.7rem; box-shadow: {P['shadow']};
}}
.insight-title {{ font-weight: 600; color: {P['text']}; font-size: 0.92rem; margin-bottom: 0.3rem; }}
.insight-text {{ color: {P['muted']}; font-size: 0.86rem; line-height: 1.5; }}
.insight-badge {{
    display: inline-block; padding: 0.18rem 0.55rem; border-radius: 5px;
    font-size: 0.66rem; font-weight: 600; letter-spacing: 0.06em;
    background: {P['subtle']}; color: {P['muted']}; text-transform: uppercase;
}}

.news-row {{
    padding: 0.55rem 0; border-bottom: 1px solid {P['subtle']};
    font-size: 0.86rem;
}}
.news-row:last-child {{ border-bottom: none; }}
.news-row a {{ color: {P['text']} !important; text-decoration: none; }}
.news-row a:hover {{ color: {P['primary']} !important; }}
.news-meta {{ color: {P['muted']}; font-size: 0.75rem; margin-top: 0.15rem; }}

section[data-testid="stSidebar"] {{ background: {P['card']} !important; border-right: 1px solid {P['border']}; }}

/* Inputs */
.stTextInput input {{
    background: {P['card']} !important; border: 1px solid {P['border']} !important;
    color: {P['text']} !important; border-radius: 8px !important; padding: 0.55rem 0.75rem !important;
}}
.stTextInput input:focus {{ border-color: {P['primary']} !important; box-shadow: 0 0 0 3px {P['primary']}22 !important; }}
.stButton button[kind="primary"] {{
    background: {P['primary']} !important; color: white !important;
    border: none !important; border-radius: 8px !important; font-weight: 500 !important;
}}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{ gap: 0.1rem; border-bottom: 1px solid {P['border']}; }}
.stTabs [data-baseweb="tab"] {{
    background: transparent !important; color: {P['muted']} !important;
    font-weight: 500 !important; font-size: 0.88rem !important;
    padding: 0.5rem 1rem !important; border-radius: 6px 6px 0 0 !important;
}}
.stTabs [aria-selected="true"] {{ color: {P['text']} !important; border-bottom: 2px solid {P['primary']} !important; }}

/* DataFrames */
[data-testid="stDataFrame"] {{ background: {P['card']}; border-radius: 8px; }}

/* Theme toggle button container alignment */
.theme-toggle {{ float: right; }}

hr {{ border-color: {P['border']} !important; }}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ─── helpers ────────────────────────────────────────────────────────────────
def fmt_money(x, currency="USD"):
    if x is None: return "—"
    sym = {"USD": "$", "AUD": "A$"}.get(currency, "")
    if abs(x) >= 1e12: return f"{sym}{x/1e12:.2f}T"
    if abs(x) >= 1e9:  return f"{sym}{x/1e9:.2f}B"
    if abs(x) >= 1e6:  return f"{sym}{x/1e6:.2f}M"
    return f"{sym}{x:,.2f}"

def fmt_pct(x, decimals=2):
    return f"{x:.{decimals}f}%" if x is not None else "—"

def fmt_num(x, decimals=2):
    return f"{x:.{decimals}f}" if x is not None else "—"

def fmt_date(x):
    if not x: return "—"
    try:
        if isinstance(x, (int, float)):
            return dt.datetime.fromtimestamp(x).strftime("%b %d, %Y")
        if isinstance(x, str):
            return x.split("T")[0] if "T" in x else x
        return str(x)
    except Exception:
        return str(x)

def metric_group(title: str, rows: list[tuple[str, str]]) -> str:
    body = "".join(f'<div class="metric-row"><span class="lbl">{l}</span><span class="val">{v}</span></div>'
                   for l, v in rows)
    return f'<div class="metric-group"><h4>{title}</h4>{body}</div>'

def chart_layout(title: str = "", height: int = 240):
    return dict(
        title=dict(text=title, font=dict(size=12, color=P['text']), x=0.02, y=0.95),
        height=height, template=P["plotly_template"],
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=P["chart_bg"],
        margin=dict(l=10, r=10, t=30, b=10),
        font=dict(family="Inter", size=10, color=P['muted']),
        xaxis=dict(gridcolor=P['border'], showline=False, zeroline=False),
        yaxis=dict(gridcolor=P['border'], showline=False, zeroline=False),
    )

def year_x(series: pd.Series):
    return [d.year if hasattr(d, "year") else str(d) for d in series.index]

def quarter_x(series: pd.Series):
    out = []
    for d in series.index:
        if hasattr(d, "year") and hasattr(d, "month"):
            q = (d.month - 1) // 3 + 1
            out.append(f"Q{q} {d.year}")
        else:
            out.append(str(d))
    return out


# ─── Header (theme toggle + title) ──────────────────────────────────────────
hdr_l, hdr_r = st.columns([6, 1])
with hdr_l:
    st.markdown("# ● Stock Analyser")
    st.markdown(f"<span class='muted'>Buffett-style fundamentals · live data · auto-DCF</span>",
                unsafe_allow_html=True)
with hdr_r:
    label = "🌙 Dark" if THEME == "light" else "☀️ Light"
    if st.button(label, use_container_width=True):
        st.session_state.theme = "dark" if THEME == "light" else "light"
        st.rerun()

st.markdown("")

# ─── Search ─────────────────────────────────────────────────────────────────
sc_l, sc_r = st.columns([5, 1])
with sc_l:
    query = st.text_input("Ticker or company name",
                          placeholder="e.g. AAPL, MSFT, KO, BHP, CBA",
                          label_visibility="collapsed")
with sc_r:
    go_btn = st.button("Analyse", type="primary", use_container_width=True)

if not query or not go_btn:
    st.markdown(f"""
    <div class='card' style='text-align:center; padding: 2.5rem;'>
      <div style='font-size: 2rem; margin-bottom: 0.5rem;'>●</div>
      <div style='font-size: 1.05rem; font-weight: 600; color: {P['text']};'>Type a ticker to begin</div>
      <div class='muted' style='margin-top: 0.5rem;'>
        Examples: <b>AAPL</b>, <b>MSFT</b> (US) · <b>BHP</b>, <b>CBA</b> (ASX, auto-detected)
      </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

with st.spinner("Loading…"):
    try:
        f = smart_fetch(query)
    except Exception as e:
        st.error(f"Could not load `{query}`: {e}")
        st.stop()

if f.price is None:
    st.error(f"No price data for `{query}`. Check the ticker symbol.")
    st.stop()

# ─── Compute everything once ────────────────────────────────────────────────
A = auto_assumptions(f)
intrinsic = dcf_intrinsic_value(
    f.fcf_latest, f.shares_outstanding,
    A["growth"], A["discount"], A["terminal"], A["years"], f.net_cash or 0,
)
mos = margin_of_safety(intrinsic, f.price)
score = quality_score(f)
verdict, vclass, reason = recommendation(mos, score)
graham = graham_number(f.eps_ttm, f.book_value_per_share)
graham_mos = ((graham - f.price) / graham * 100) if (graham and f.price) else None
rev_growth = reverse_dcf_growth(f.price, f.fcf_latest, f.shares_outstanding,
                                A["discount"], A["terminal"], A["years"], f.net_cash or 0)
fcfy = fcf_yield(f.market_cap, f.fcf_latest)
oey = fcf_yield(f.market_cap, f.owner_earnings_latest)
insights = generate_insights(f, intrinsic, mos, rev_growth)
ten_yr = ten_year_summary(f)

# Day change
change = None
if f.price and f.prev_close:
    change = (f.price - f.prev_close) / f.prev_close * 100
chg_html = ""
if change is not None:
    cls = "day-change-pos" if change >= 0 else "day-change-neg"
    sign = "+" if change >= 0 else ""
    diff = f.price - f.prev_close
    chg_html = f"<span class='{cls}'>{sign}{diff:.2f} ({sign}{change:.2f}%)</span>"

vmap = {"positive": "v-positive", "warn": "v-warn", "negative": "v-negative", "neutral": "v-neutral"}
vdot = {"positive": "●", "warn": "●", "negative": "●", "neutral": "●"}
mos_label = f" · {mos:+.0f}% MoS" if mos is not None else ""

st.markdown(f"""
<div class='ticker-bar'>
  <span class='ticker-sym'>{f.ticker}</span>
  <span class='ticker-name'>{f.name}{(' · ' + f.sector) if f.sector else ''}</span>
  <span style='margin-left:auto; display:flex; align-items:center; gap:0.9rem;'>
    <span class='ticker-price'>{fmt_money(f.price, f.currency)}</span>
    {chg_html}
    <span class='verdict-pill {vmap[vclass]}'>{vdot[vclass]} {verdict}{mos_label}</span>
  </span>
</div>
""", unsafe_allow_html=True)


# ─── Tabs ───────────────────────────────────────────────────────────────────
tab_overview, tab_financials, tab_valuation, tab_insights = st.tabs(
    ["Overview", "Financials", "Valuation", "Insights"]
)

# ════════ OVERVIEW ══════════
with tab_overview:
    # 5-column metric groups
    cols = st.columns(5)
    with cols[0]:
        st.markdown(metric_group("Valuation", [
            ("Market Cap", fmt_money(f.market_cap, f.currency)),
            ("P/E (TTM)", fmt_num(f.pe_ratio)),
            ("Forward P/E", fmt_num(f.forward_pe)),
            ("Price / Sales", fmt_num(f.price_to_sales)),
            ("Price / Book", fmt_num(f.price_to_book)),
            ("EV / EBITDA", fmt_num(f.ev_to_ebitda)),
        ]), unsafe_allow_html=True)
    with cols[1]:
        st.markdown(metric_group("Cash Flow", [
            ("FCF (latest)", fmt_money(f.fcf_latest, f.currency)),
            ("FCF Yield", fmt_pct(fcfy)),
            ("FCF Margin", fmt_pct(f.fcf_margin)),
            ("Owner Earnings", fmt_money(f.owner_earnings_latest, f.currency)),
            ("OE Yield", fmt_pct(oey)),
            ("SBC Impact", fmt_pct(f.sbc_impact)),
        ]), unsafe_allow_html=True)
    with cols[2]:
        st.markdown(metric_group("Margins & Growth", [
            ("Gross Margin", fmt_pct(f.gross_margin)),
            ("Operating Margin", fmt_pct(f.operating_margin)),
            ("Profit Margin", fmt_pct(f.profit_margin)),
            ("Revenue YoY", fmt_pct(f.revenue_growth_yoy)),
            ("Earnings YoY", fmt_pct(f.earnings_growth_yoy)),
            ("Revenue 5y CAGR", fmt_pct(f.revenue_growth)),
        ]), unsafe_allow_html=True)
    with cols[3]:
        st.markdown(metric_group("Balance", [
            ("Cash", fmt_money(f.total_cash, f.currency)),
            ("Debt", fmt_money(f.total_debt, f.currency)),
            ("Net Cash", fmt_money(f.net_cash, f.currency)),
            ("Debt / Equity", fmt_num(f.de_latest)),
            ("Current Ratio", fmt_num(f.current_ratio)),
            ("Interest Coverage", fmt_num(f.interest_coverage)),
        ]), unsafe_allow_html=True)
    with cols[4]:
        st.markdown(metric_group("Dividend", [
            ("Yield", fmt_pct(f.dividend_yield)),
            ("Payout Ratio", fmt_pct(f.payout_ratio)),
            ("Annual Rate", fmt_money(f.dividend_rate, f.currency)),
            ("Ex-Div Date", fmt_date(f.ex_dividend_date)),
            ("Buyback Yield", fmt_pct(f.buyback_yield)),
            ("Earnings", fmt_date(f.next_earnings)),
        ]), unsafe_allow_html=True)

    st.markdown("")
    # Price chart + news
    pc, nc = st.columns([2, 1])
    with pc:
        if not f.price_history.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=f.price_history.index, y=f.price_history["Close"],
                                     mode="lines",
                                     line=dict(color=P["primary"], width=2),
                                     fill="tozeroy",
                                     fillcolor=f"{P['primary']}1a",
                                     hovertemplate="%{y:.2f}<extra></extra>"))
            if intrinsic:
                fig.add_hline(y=intrinsic, line_dash="dot", line_color=P["positive"],
                              line_width=1, annotation_text=f"Intrinsic {fmt_money(intrinsic, f.currency)}",
                              annotation_position="top right",
                              annotation_font=dict(color=P["positive"], size=10))
            fig.update_layout(**chart_layout("Price · 5y", height=320))
            fig.update_yaxes(rangemode="tozero")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown('<div class="card">Price chart unavailable.</div>', unsafe_allow_html=True)
    with nc:
        st.markdown('<div class="card"><h4 style="margin:0 0 0.6rem 0;font-size:0.78rem;'
                    'text-transform:uppercase;letter-spacing:0.06em;">Recent News</h4>',
                    unsafe_allow_html=True)
        if f.news:
            for n in f.news[:5]:
                title = n.get("title", "")
                pub = n.get("publisher", "")
                url = n.get("url", "")
                t = n.get("time", "")
                if isinstance(t, (int, float)):
                    t = dt.datetime.fromtimestamp(t).strftime("%b %d")
                elif isinstance(t, str) and "T" in t:
                    t = t.split("T")[0]
                link = f'<a href="{url}" target="_blank">{title}</a>' if url else title
                st.markdown(f"""<div class="news-row">{link}
                    <div class="news-meta">{pub}{' · ' + str(t) if t else ''}</div></div>""",
                            unsafe_allow_html=True)
        else:
            st.markdown('<span class="muted">No recent headlines.</span>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


# ════════ FINANCIALS ══════════
with tab_financials:
    period = st.radio("Period", ["Annually", "Quarterly"], horizontal=True, label_visibility="collapsed")
    use_q = period == "Quarterly"

    series_map = {
        "Revenue": (f.q_revenue if use_q else f.revenue_history, "#f59e0b"),
        "EBITDA": (f.q_ebitda if use_q else f.ebitda_history, "#3b82f6"),
        "Net Income": (f.q_net_income if use_q else f.net_income_history, "#fb923c"),
        "Free Cash Flow": (f.q_fcf if use_q else f.fcf_history, "#f97316"),
        "EPS": (f.q_eps if use_q else f.eps_history, "#eab308"),
        "Operating Income": (f.q_operating_income if use_q else f.operating_income_history, "#10b981"),
        "Gross Profit": (f.q_gross_profit if use_q else f.gross_profit_history, "#14b8a6"),
    }

    # 4 columns × 2 rows = 8 charts (Price + 7 series)
    items = list(series_map.items())
    rows = [items[0:4], items[4:8]] if len(items) > 4 else [items]
    # Add price chart as first
    price_chart = ("Price", (f.price_history, P["primary"]))
    items = [price_chart] + items
    rows = [items[0:4], items[4:8]]

    for row in rows:
        cols = st.columns(4)
        for i, (name, (series, color)) in enumerate(row):
            with cols[i]:
                if name == "Price" and not series.empty:
                    fig = go.Figure(go.Scatter(x=series.index, y=series["Close"], mode="lines",
                                               line=dict(color=color, width=2),
                                               hovertemplate="%{y:.2f}<extra></extra>"))
                    fig.update_layout(**chart_layout(name, height=220))
                    st.plotly_chart(fig, use_container_width=True)
                elif hasattr(series, "empty") and not series.empty:
                    x = quarter_x(series) if use_q else year_x(series)
                    colors = [color if v >= 0 else P["negative"] for v in series.values]
                    fig = go.Figure(go.Bar(x=x, y=series.values, marker_color=colors,
                                           hovertemplate="%{y:,.2f}<extra></extra>"))
                    fig.update_layout(**chart_layout(name, height=220))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.markdown(f'<div class="card"><span class="muted">{name} unavailable</span></div>',
                                unsafe_allow_html=True)

    # 10-year summary table
    st.markdown("")
    st.markdown('<h2>10-Year Summary</h2>', unsafe_allow_html=True)
    if not ten_yr.empty:
        # Format large numbers
        def _fmt_cell(v, label):
            if pd.isna(v):
                return "—"
            if "%" in label:
                return f"{v:.1f}%"
            if "EPS" in label or "Equity" in label:
                return f"{v:.2f}"
            return fmt_money(v, f.currency)
        formatted = ten_yr.copy()
        for idx, lbl in enumerate(formatted.index):
            formatted.iloc[idx] = [_fmt_cell(v, lbl) for v in formatted.iloc[idx]]
        st.dataframe(formatted, use_container_width=True)
    else:
        st.markdown('<div class="card"><span class="muted">No multi-year data available.</span></div>',
                    unsafe_allow_html=True)


# ════════ VALUATION ══════════
with tab_valuation:
    # Verdict + reasoning card
    st.markdown(f"""
    <div class='card'>
      <div style='display:flex; align-items:center; gap:0.8rem; margin-bottom:0.5rem;'>
        <span class='verdict-pill {vmap[vclass]}'>{vdot[vclass]} {verdict}</span>
        <span class='muted' style='font-size:0.86rem;'>{reason}</span>
      </div>
      <div class='muted' style='font-size:0.82rem; margin-top:0.6rem;'>
        DCF assumptions auto-derived from this stock's history:
        growth <b>{A['growth']*100:.1f}%</b> (5y FCF CAGR, capped 3–12%) ·
        discount <b>{A['discount']*100:.0f}%</b> (Buffett's hurdle) ·
        terminal growth <b>{A['terminal']*100:.1f}%</b> (long-run GDP).
      </div>
    </div>
    """, unsafe_allow_html=True)

    # 3 valuation cards
    cv = st.columns(3)
    with cv[0]:
        mos_str = f"{mos:+.1f}% MoS" if mos is not None else "—"
        st.markdown(metric_group("DCF Intrinsic", [
            ("Per share", fmt_money(intrinsic, f.currency)),
            ("Margin of safety", mos_str),
            ("Quality score", f"{score}/3"),
        ]), unsafe_allow_html=True)
    with cv[1]:
        gmos = f"{graham_mos:+.1f}% MoS" if graham_mos is not None else "—"
        st.markdown(metric_group("Graham Number", [
            ("Per share", fmt_money(graham, f.currency)),
            ("Margin of safety", gmos),
            ("Cross-check", "Defensive investor"),
        ]), unsafe_allow_html=True)
    with cv[2]:
        rg_str = fmt_pct(rev_growth)
        rg_note = ("conservative" if rev_growth and rev_growth <= 8
                   else "moderate" if rev_growth and rev_growth <= 15 else "aggressive")
        st.markdown(metric_group("Reverse DCF", [
            ("Implied growth", rg_str),
            ("Read", rg_note if rev_growth else "—"),
            ("Time horizon", "10 years"),
        ]), unsafe_allow_html=True)

    # Sensitivity grid
    st.markdown("")
    st.markdown('<h2>DCF Sensitivity</h2>', unsafe_allow_html=True)
    if f.fcf_latest and f.shares_outstanding:
        sens = dcf_sensitivity(f.fcf_latest, f.shares_outstanding,
                               [0.04, 0.06, 0.08, 0.10, 0.12],
                               [0.08, 0.09, 0.10, 0.11, 0.12],
                               A["terminal"], A["years"], f.net_cash or 0)
        if sens is not None:
            st.markdown('<div class="muted" style="font-size:0.82rem; margin-bottom:0.4rem;">'
                        'Rows = discount rate · Columns = growth rate · Values = intrinsic per share'
                        '</div>', unsafe_allow_html=True)
            try:
                styled = sens.style.background_gradient(cmap="Greens", axis=None).format("{:,.2f}")
                st.dataframe(styled, use_container_width=True)
            except Exception:
                st.dataframe(sens.applymap(lambda v: f"{v:,.2f}" if v else "—"),
                             use_container_width=True)
    else:
        st.markdown('<div class="card"><span class="muted">Sensitivity grid unavailable '
                    '(no FCF or shares data).</span></div>', unsafe_allow_html=True)

    # ROE / D/E history
    st.markdown("")
    cl, cr = st.columns(2)
    with cl:
        if not f.roe_history.empty:
            colors = [P["positive"] if v >= 15 else P["warn"] if v >= 10 else P["negative"]
                      for v in f.roe_history.values]
            fig = go.Figure(go.Bar(x=year_x(f.roe_history), y=f.roe_history.values, marker_color=colors))
            fig.add_hline(y=15, line_dash="dot", line_color=P["positive"], line_width=1)
            fig.update_layout(**chart_layout("ROE % history", height=240))
            st.plotly_chart(fig, use_container_width=True)
    with cr:
        if not f.de_history.empty:
            colors = [P["positive"] if v <= 0.5 else P["warn"] if v <= 1 else P["negative"]
                      for v in f.de_history.values]
            fig = go.Figure(go.Bar(x=year_x(f.de_history), y=f.de_history.values, marker_color=colors))
            fig.add_hline(y=0.5, line_dash="dot", line_color=P["positive"], line_width=1)
            fig.update_layout(**chart_layout("Debt / Equity history", height=240))
            st.plotly_chart(fig, use_container_width=True)


# ════════ INSIGHTS ══════════
with tab_insights:
    ic1, ic2 = st.columns(2)
    with ic1:
        st.markdown(f'<h2 style="margin-bottom:0.4rem;">Competitive Advantages '
                    f'<span class="insight-badge">Auto-generated</span></h2>',
                    unsafe_allow_html=True)
        if insights["advantages"]:
            for item in insights["advantages"]:
                st.markdown(f"""<div class="insight-card">
                    <div class="insight-title">{item['title']}</div>
                    <div class="insight-text">{item['text']}</div>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown('<div class="card"><span class="muted">No standout strengths detected '
                        'from the available data.</span></div>', unsafe_allow_html=True)

    with ic2:
        st.markdown(f'<h2 style="margin-bottom:0.4rem;">Investment Risks '
                    f'<span class="insight-badge">Auto-generated</span></h2>',
                    unsafe_allow_html=True)
        if insights["risks"]:
            for item in insights["risks"]:
                st.markdown(f"""<div class="insight-card">
                    <div class="insight-title">{item['title']}</div>
                    <div class="insight-text">{item['text']}</div>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown('<div class="card"><span class="muted">No major red flags detected. '
                        'Always cross-check with qualitative judgement.</span></div>',
                        unsafe_allow_html=True)

    st.markdown("")
    st.markdown('<h2>Things to verify yourself</h2>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card">
      <div class="metric-row"><span class="lbl">Wide moat</span>
        <span class="val muted">brand, network effects, switching costs, scale</span></div>
      <div class="metric-row"><span class="lbl">Honest, capital-disciplined management</span>
        <span class="val muted">read shareholder letters, check insider activity</span></div>
      <div class="metric-row"><span class="lbl">Recession resilience</span>
        <span class="val muted">how did 2008 / 2020 earnings hold up?</span></div>
      <div class="metric-row"><span class="lbl">Customer concentration</span>
        <span class="val muted">no single customer >20% of revenue</span></div>
      <div class="metric-row"><span class="lbl">Within your circle of competence</span>
        <span class="val muted">if you can't explain it in a paragraph, skip it</span></div>
      <div class="metric-row"><span class="lbl">Position size</span>
        <span class="val muted">a great buy still shouldn't be >10% of your portfolio</span></div>
    </div>
    """, unsafe_allow_html=True)
