"""Exercise every code path with synthetic data (sandbox blocks yfinance)."""
import traceback
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import plotly.graph_objects as go
import numpy as np

from analyzer import (
    Fundamentals, auto_assumptions, dcf_intrinsic_value, dcf_sensitivity, fcf_yield,
    generate_insights, graham_number, margin_of_safety, quality_score,
    recommendation, reverse_dcf_growth, ten_year_summary,
)


def make_series(years, values):
    idx = pd.to_datetime([f"{y}-12-31" for y in years])
    return pd.Series(values, index=idx)


def make_quarterly(quarters, values):
    idx = pd.to_datetime([f"{y}-{m:02d}-30" for (y, m) in quarters])
    return pd.Series(values, index=idx)


def make_price_history():
    dates = pd.date_range("2020-01-01", "2024-12-31", freq="D")
    rng = np.random.default_rng(42)
    closes = 100 * np.cumprod(1 + rng.normal(0.0005, 0.015, len(dates)))
    return pd.DataFrame({
        "Open": closes * 0.99, "High": closes * 1.01,
        "Low": closes * 0.98, "Close": closes,
        "Volume": np.full(len(dates), 1_000_000, dtype=float),
    }, index=dates)


def quality_stock():
    """LLY-style: fast grower with lumpy FCF."""
    f = Fundamentals(ticker="QUAL", name="Quality Co", currency="USD", sector="Healthcare")
    f.price = 934.60
    f.prev_close = 851.21
    f.shares_outstanding = 900_000_000
    f.market_cap = 840_000_000_000
    f.enterprise_value = 850_000_000_000
    f.pe_ratio = 70.5
    f.forward_pe = 36.5
    f.eps_ttm = 13.2
    f.eps_forward = 25.6
    f.book_value_per_share = 18.4
    f.price_to_sales = 16.2
    f.price_to_book = 50.8
    f.ev_to_ebitda = 48.7
    f.dividend_yield = 0.66
    f.payout_ratio = 23.65
    f.dividend_rate = 6.0
    f.ex_dividend_date = 1734739200
    f.operating_margin = 45.62
    f.profit_margin = 36.15
    f.gross_margin = 81.5
    f.revenue_growth_yoy = 18.10
    f.earnings_growth_yoy = 23.58
    f.return_on_equity = 78.4
    f.return_on_assets = 17.2
    f.fifty_two_high = 972.53; f.fifty_two_low = 711.40
    f.fifty_day_avg = 880.0; f.two_hundred_day_avg = 850.0
    f.total_cash = 4_000_000_000
    f.total_debt = 25_000_000_000
    f.net_cash = -21_000_000_000
    f.next_earnings = "2025-10-29"
    f.news = [{"title": "Sample headline", "publisher": "Reuters",
               "url": "https://example.com", "time": 1714521600}]
    yrs = [2019, 2020, 2021, 2022, 2023]
    f.revenue_history = make_series(yrs, [22e9, 24e9, 28e9, 28e9, 34e9])
    f.net_income_history = make_series(yrs, [8e9, 6e9, 5e9, 6e9, 5e9])
    f.fcf_history = make_series(yrs, [4e9, 5e9, 1e9, -1e9, 2e9])  # lumpy capex
    f.fcf_latest = 2e9
    f.fcf_growth = 18.0
    f.fcf_margin = 5.9
    f.eps_history = make_series(yrs, [8.0, 6.5, 5.5, 6.5, 5.5])
    f.eps_growth = 7.0
    f.earnings_stability = 65.0
    f.gross_profit_history = make_series(yrs, [17e9, 19e9, 22e9, 22e9, 27e9])
    f.operating_income_history = make_series(yrs, [6.5e9, 6.0e9, 8.0e9, 9.0e9, 13.0e9])
    f.ebitda_history = make_series(yrs, [8e9, 7.5e9, 9.5e9, 10.5e9, 14.5e9])
    f.gross_margin_history = make_series(yrs, [77, 79, 78, 78, 79.5])
    f.operating_margin_history = make_series(yrs, [29.5, 25.0, 28.5, 32.1, 38.2])
    f.roe_history = make_series(yrs, [42, 38, 60, 70, 78])
    f.roe_avg = 57.6
    f.de_history = make_series(yrs, [1.4, 1.6, 1.5, 1.7, 1.7])
    f.de_latest = 1.7
    f.owner_earnings_history = make_series(yrs, [6e9, 7e9, 4e9, 2e9, 6e9])
    f.owner_earnings_latest = 6e9
    f.shares_history = make_series(yrs, [957e6, 950e6, 945e6, 905e6, 900e6])
    f.buyback_yield = 1.5
    f.interest_coverage = 12.5
    f.current_ratio = 1.3
    f.quick_ratio = 0.9
    f.sbc_history = make_series(yrs, [-300e6, -350e6, -400e6, -450e6, -500e6])
    f.sbc_impact = 25.0
    f.q_revenue = make_quarterly([(2023, 3), (2023, 6), (2023, 9), (2023, 12),
                                  (2024, 3), (2024, 6), (2024, 9), (2024, 12)],
                                 [7e9, 8e9, 9e9, 10e9, 9e9, 11e9, 12e9, 14e9])
    f.q_net_income = make_quarterly([(2024, 3), (2024, 6), (2024, 9), (2024, 12)],
                                    [2.3e9, 3.0e9, 3.0e9, 4.5e9])
    f.q_eps = make_quarterly([(2024, 3), (2024, 6), (2024, 9), (2024, 12)],
                             [2.5, 3.3, 3.3, 5.0])
    f.q_fcf = make_quarterly([(2024, 3), (2024, 6), (2024, 9), (2024, 12)],
                             [-500e6, 1e9, 800e6, 1.2e9])
    f.q_ebitda = make_quarterly([(2024, 3), (2024, 6)], [3e9, 4e9])
    f.q_operating_income = make_quarterly([(2024, 3), (2024, 6)], [3e9, 3.5e9])
    f.q_gross_profit = make_quarterly([(2024, 3), (2024, 6)], [7e9, 8.5e9])
    f.price_history = make_price_history()
    return f


def value_stock():
    """KO-style: stable, modest growth."""
    f = Fundamentals(ticker="VAL", name="Value Co", currency="USD", sector="Consumer Defensive")
    f.price = 65.0
    f.prev_close = 64.5
    f.shares_outstanding = 4_300_000_000
    f.market_cap = 280_000_000_000
    f.pe_ratio = 25.0
    f.eps_ttm = 2.6
    f.book_value_per_share = 6.5
    f.dividend_yield = 3.0
    f.payout_ratio = 75.0
    f.operating_margin = 28.0
    f.profit_margin = 22.0
    f.gross_margin = 60.0
    f.return_on_equity = 35.0
    f.fifty_two_high = 70; f.fifty_two_low = 55
    f.total_cash = 12e9; f.total_debt = 40e9; f.net_cash = -28e9
    yrs = [2019, 2020, 2021, 2022, 2023]
    f.revenue_history = make_series(yrs, [37e9, 33e9, 39e9, 43e9, 46e9])
    f.net_income_history = make_series(yrs, [9e9, 7.7e9, 9.8e9, 9.6e9, 10.7e9])
    f.fcf_history = make_series(yrs, [8.4e9, 8.7e9, 11.3e9, 9.5e9, 9.7e9])
    f.fcf_latest = 9.7e9; f.fcf_growth = 4.0; f.fcf_margin = 21.0
    f.eps_history = make_series(yrs, [2.0, 1.8, 2.3, 2.2, 2.5])
    f.eps_growth = 5.7; f.earnings_stability = 85.0
    f.roe_history = make_series(yrs, [38, 36, 42, 38, 35])
    f.roe_avg = 37.8
    f.de_history = make_series(yrs, [1.7, 1.9, 1.6, 1.7, 1.5])
    f.de_latest = 1.5
    f.owner_earnings_history = make_series(yrs, [9e9, 8.5e9, 11e9, 9.5e9, 10e9])
    f.owner_earnings_latest = 10e9
    f.gross_profit_history = make_series(yrs, [22e9, 20e9, 23e9, 26e9, 28e9])
    f.operating_income_history = make_series(yrs, [10e9, 9e9, 11e9, 12e9, 13e9])
    f.price_history = make_price_history()
    return f


def empty_stock():
    """Edge: missing most series."""
    f = Fundamentals(ticker="MIN", name="Minimal Co", currency="USD")
    f.price = 10.0
    f.prev_close = 10.0
    f.shares_outstanding = 1_000_000
    f.market_cap = 10_000_000
    return f


SAMPLES = [("QUAL (lumpy FCF)", quality_stock()),
           ("VAL (stable)", value_stock()),
           ("MIN (empty)", empty_stock())]

errors = []

def step(name, fn):
    try:
        out = fn()
        print(f"  OK  {name}")
        return out
    except Exception as e:
        tb = traceback.format_exc()
        print(f"  FAIL {name}: {type(e).__name__}: {e}")
        errors.append((name, tb))
        return None


for label, f in SAMPLES:
    print(f"\n=== {label} ===")
    A = step("auto_assumptions", lambda f=f: auto_assumptions(f))
    if not A: continue
    intrinsic = step("dcf_intrinsic_value", lambda f=f: dcf_intrinsic_value(
        A["fcf"], f.shares_outstanding, A["growth"], A["discount"],
        A["terminal"], A["years"], f.net_cash or 0))
    mos = step("margin_of_safety", lambda f=f: margin_of_safety(intrinsic, f.price))
    score = step("quality_score", lambda f=f: quality_score(f))
    step("recommendation", lambda f=f: recommendation(mos, score))
    step("graham_number", lambda f=f: graham_number(f.eps_ttm, f.book_value_per_share))
    rev_g = step("reverse_dcf_growth", lambda f=f: reverse_dcf_growth(
        f.price, A["fcf"], f.shares_outstanding, A["discount"],
        A["terminal"], A["years"], f.net_cash or 0))
    step("fcf_yield (mcap)", lambda f=f: fcf_yield(f.market_cap, f.fcf_latest))
    step("fcf_yield (oe)", lambda f=f: fcf_yield(f.market_cap, f.owner_earnings_latest))
    step("generate_insights", lambda f=f: generate_insights(f, intrinsic, mos, rev_g))
    ten = step("ten_year_summary", lambda f=f: ten_year_summary(f))
    step("dcf_sensitivity", lambda f=f: dcf_sensitivity(
        A["fcf"], f.shares_outstanding,
        [0.04, 0.06, 0.08, 0.10, 0.12], [0.08, 0.09, 0.10, 0.11, 0.12],
        A["terminal"], A["years"], f.net_cash or 0))

    # Mimic app's chart construction
    def build_charts():
        figs = []
        if not f.price_history.empty:
            fig = go.Figure(go.Scatter(
                x=f.price_history.index, y=f.price_history["Close"],
                mode="lines", line=dict(color="#2563eb", width=2),
                fill="tozeroy", fillcolor="rgba(37,99,235,0.1)"))
            if intrinsic:
                fig.add_hline(y=intrinsic, line_dash="dot", line_color="#059669")
            figs.append(("price", fig))
        for name, series in [("revenue", f.revenue_history), ("ni", f.net_income_history),
                             ("fcf", f.fcf_history), ("eps", f.eps_history),
                             ("oi", f.operating_income_history),
                             ("gp", f.gross_profit_history),
                             ("ebitda", f.ebitda_history),
                             ("roe", f.roe_history), ("de", f.de_history)]:
            if hasattr(series, "empty") and not series.empty:
                fig = go.Figure(go.Bar(x=[str(d) for d in series.index],
                                       y=series.values, marker_color="#3b82f6"))
                figs.append((name, fig))
        return figs
    step("build_charts (annual)", build_charts)

    def build_q_charts():
        figs = []
        for name, s in [("q_rev", f.q_revenue), ("q_ni", f.q_net_income),
                        ("q_eps", f.q_eps), ("q_fcf", f.q_fcf),
                        ("q_ebitda", f.q_ebitda), ("q_oi", f.q_operating_income),
                        ("q_gp", f.q_gross_profit)]:
            if hasattr(s, "empty") and not s.empty:
                fig = go.Figure(go.Bar(x=[str(d) for d in s.index],
                                       y=s.values, marker_color="#3b82f6"))
                figs.append((name, fig))
        return figs
    step("build_charts (quarterly)", build_q_charts)

    # Mimic 10y formatting (the new fix - build fresh string DataFrame)
    def format_ten_yr():
        if ten is None or ten.empty:
            return None
        def _fmt_cell(v, lbl):
            if pd.isna(v): return "—"
            if "%" in str(lbl): return f"{v:.1f}%"
            if "EPS" in str(lbl) or "Equity" in str(lbl): return f"{v:.2f}"
            return f"${v:,.2f}"
        return pd.DataFrame(
            {col: [_fmt_cell(ten.at[lbl, col], lbl) for lbl in ten.index]
             for col in ten.columns},
            index=ten.index,
        )
    step("format_ten_yr", format_ten_yr)

    # Mimic plotly heatmap sensitivity (replacing styled dataframe)
    def heatmap_sens():
        sens = dcf_sensitivity(A["fcf"], f.shares_outstanding,
                               [0.04, 0.06, 0.08, 0.10, 0.12],
                               [0.08, 0.09, 0.10, 0.11, 0.12],
                               A["terminal"], A["years"], f.net_cash or 0)
        if sens is None:
            return None
        text = [[f"{v:,.0f}" if v is not None else "—" for v in row] for row in sens.values]
        fig = go.Figure(data=go.Heatmap(
            z=sens.values, x=list(sens.columns), y=list(sens.index),
            colorscale=[[0, "#f1f5f9"], [1, "#059669"]],
            text=text, texttemplate="%{text}",
            hoverongaps=False))
        return fig
    step("heatmap_sens", heatmap_sens)


print("\n" + "="*60)
if errors:
    print(f"\nFAILURES: {len(errors)}")
    for name, tb in errors:
        print(f"\n--- {name} ---\n{tb}")
else:
    print("\nALL OK")
