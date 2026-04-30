"""Buffett-style stock analysis: fundamentals, DCF, margin of safety."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf


# yfinance row labels vary across versions/regions — try aliases.
ALIASES = {
    "net_income": ["Net Income", "Net Income Common Stockholders", "Net Income From Continuing Operation Net Minority Interest"],
    "equity": ["Stockholders Equity", "Total Stockholder Equity", "Common Stock Equity"],
    "total_debt": ["Total Debt"],
    "long_term_debt": ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"],
    "short_term_debt": ["Current Debt", "Short Long Term Debt"],
    "operating_cf": ["Operating Cash Flow", "Total Cash From Operating Activities", "Cash Flow From Continuing Operating Activities"],
    "capex": ["Capital Expenditure", "Capital Expenditures"],
    "free_cash_flow": ["Free Cash Flow"],
    "revenue": ["Total Revenue", "Operating Revenue"],
}


def _row(df: pd.DataFrame, key: str) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    for label in ALIASES.get(key, [key]):
        if label in df.index:
            return df.loc[label].dropna()
    return None


def normalize_ticker(ticker: str, market: str = "US") -> str:
    t = ticker.strip().upper()
    if market == "ASX" and not t.endswith(".AX"):
        return f"{t}.AX"
    return t


@dataclass
class Fundamentals:
    ticker: str
    name: str = ""
    currency: str = "USD"
    price: Optional[float] = None
    shares_outstanding: Optional[float] = None
    market_cap: Optional[float] = None

    roe_history: pd.Series = field(default_factory=pd.Series)
    de_history: pd.Series = field(default_factory=pd.Series)
    eps_history: pd.Series = field(default_factory=pd.Series)
    fcf_history: pd.Series = field(default_factory=pd.Series)
    revenue_history: pd.Series = field(default_factory=pd.Series)
    price_history: pd.DataFrame = field(default_factory=pd.DataFrame)

    roe_avg: Optional[float] = None
    de_latest: Optional[float] = None
    eps_growth: Optional[float] = None
    fcf_growth: Optional[float] = None
    fcf_latest: Optional[float] = None
    net_cash: Optional[float] = None


def _safe_get(info: dict, *keys, default=None):
    for k in keys:
        v = info.get(k)
        if v not in (None, "", 0):
            return v
    return default


def _cagr(series: pd.Series) -> Optional[float]:
    s = series.dropna().sort_index()
    if len(s) < 2:
        return None
    first, last = s.iloc[0], s.iloc[-1]
    years = max(len(s) - 1, 1)
    if first is None or first <= 0 or last is None or last <= 0:
        # Fall back to average year-on-year growth for negative/zero starts.
        pct = s.pct_change().dropna()
        if pct.empty:
            return None
        return float(pct.mean() * 100)
    return float(((last / first) ** (1 / years) - 1) * 100)


def fetch(ticker: str) -> Fundamentals:
    tk = yf.Ticker(ticker)
    info = {}
    try:
        info = tk.info or {}
    except Exception:
        info = {}

    fin = getattr(tk, "financials", pd.DataFrame())
    bal = getattr(tk, "balance_sheet", pd.DataFrame())
    cf = getattr(tk, "cashflow", pd.DataFrame())

    f = Fundamentals(ticker=ticker)
    f.name = _safe_get(info, "longName", "shortName", default=ticker) or ticker
    f.currency = _safe_get(info, "currency", default="USD") or "USD"
    f.price = _safe_get(info, "currentPrice", "regularMarketPrice", "previousClose")
    f.shares_outstanding = _safe_get(info, "sharesOutstanding", "impliedSharesOutstanding")
    f.market_cap = _safe_get(info, "marketCap")

    try:
        f.price_history = tk.history(period="5y", auto_adjust=True)
    except Exception:
        f.price_history = pd.DataFrame()

    ni = _row(fin, "net_income")
    eq = _row(bal, "equity")
    if ni is not None and eq is not None:
        common_idx = ni.index.intersection(eq.index)
        if len(common_idx) > 0:
            roe = (ni.loc[common_idx] / eq.loc[common_idx]) * 100
            f.roe_history = roe.sort_index()
            f.roe_avg = float(roe.mean())

    debt = _row(bal, "total_debt")
    if debt is None:
        ltd = _row(bal, "long_term_debt")
        std = _row(bal, "short_term_debt")
        if ltd is not None or std is not None:
            debt = (ltd if ltd is not None else 0) + (std if std is not None else 0)
            if hasattr(debt, "dropna"):
                debt = debt.dropna()
    if debt is not None and eq is not None:
        common_idx = debt.index.intersection(eq.index)
        if len(common_idx) > 0:
            de = (debt.loc[common_idx] / eq.loc[common_idx]).sort_index()
            f.de_history = de
            if not de.empty:
                f.de_latest = float(de.iloc[-1])

    if ni is not None and f.shares_outstanding:
        eps = ni / f.shares_outstanding
        f.eps_history = eps.sort_index()
        f.eps_growth = _cagr(f.eps_history)

    fcf = _row(cf, "free_cash_flow")
    if fcf is None:
        ocf = _row(cf, "operating_cf")
        capex = _row(cf, "capex")
        if ocf is not None:
            if capex is not None:
                common_idx = ocf.index.intersection(capex.index)
                fcf = (ocf.loc[common_idx] + capex.loc[common_idx]).dropna()
            else:
                fcf = ocf
    if fcf is not None and not fcf.empty:
        f.fcf_history = fcf.sort_index()
        f.fcf_latest = float(f.fcf_history.iloc[-1])
        f.fcf_growth = _cagr(f.fcf_history)

    rev = _row(fin, "revenue")
    if rev is not None:
        f.revenue_history = rev.sort_index()

    cash = _safe_get(info, "totalCash", default=0) or 0
    total_debt_info = _safe_get(info, "totalDebt", default=0) or 0
    f.net_cash = float(cash) - float(total_debt_info)

    return f


def dcf_intrinsic_value(
    latest_fcf: Optional[float],
    shares_outstanding: Optional[float],
    growth_rate: float = 0.08,
    discount_rate: float = 0.10,
    terminal_growth: float = 0.025,
    years: int = 10,
    net_cash: float = 0.0,
) -> Optional[float]:
    """Two-stage DCF — returns intrinsic value per share."""
    if not latest_fcf or latest_fcf <= 0 or not shares_outstanding or shares_outstanding <= 0:
        return None
    if discount_rate <= terminal_growth:
        return None

    pv_fcf = 0.0
    for year in range(1, years + 1):
        future = latest_fcf * ((1 + growth_rate) ** year)
        pv_fcf += future / ((1 + discount_rate) ** year)

    terminal_fcf = latest_fcf * ((1 + growth_rate) ** years) * (1 + terminal_growth)
    terminal_value = terminal_fcf / (discount_rate - terminal_growth)
    pv_terminal = terminal_value / ((1 + discount_rate) ** years)

    equity_value = pv_fcf + pv_terminal + (net_cash or 0)
    return equity_value / shares_outstanding


def margin_of_safety(intrinsic: Optional[float], price: Optional[float]) -> Optional[float]:
    if intrinsic is None or price is None or price <= 0:
        return None
    return (intrinsic - price) / intrinsic * 100


def buffett_score(f: Fundamentals) -> tuple[int, list[str]]:
    """Score 0-3 across ROE, debt, earnings growth — returns (score, notes)."""
    score = 0
    notes: list[str] = []

    if f.roe_avg is not None:
        if f.roe_avg >= 15:
            score += 1
            notes.append(f"ROE {f.roe_avg:.1f}% — strong (>=15%)")
        elif f.roe_avg >= 10:
            notes.append(f"ROE {f.roe_avg:.1f}% — moderate")
        else:
            notes.append(f"ROE {f.roe_avg:.1f}% — weak")
    else:
        notes.append("ROE — no data")

    if f.de_latest is not None:
        if f.de_latest <= 0.5:
            score += 1
            notes.append(f"Debt/Equity {f.de_latest:.2f} — low (<=0.5)")
        elif f.de_latest <= 1.0:
            notes.append(f"Debt/Equity {f.de_latest:.2f} — moderate")
        else:
            notes.append(f"Debt/Equity {f.de_latest:.2f} — high")
    else:
        notes.append("Debt/Equity — no data")

    if f.eps_growth is not None:
        if f.eps_growth >= 8:
            score += 1
            notes.append(f"EPS growth {f.eps_growth:.1f}%/yr — strong")
        elif f.eps_growth >= 3:
            notes.append(f"EPS growth {f.eps_growth:.1f}%/yr — moderate")
        else:
            notes.append(f"EPS growth {f.eps_growth:.1f}%/yr — weak")
    else:
        notes.append("EPS growth — no data")

    return score, notes


def recommendation(mos: Optional[float], score: int) -> tuple[str, str, str]:
    """Returns (verdict, color, reason)."""
    if mos is None:
        return "INSUFFICIENT DATA", "#888", "DCF inputs unavailable for this ticker."
    if mos >= 25 and score >= 2:
        return "BUY", "#16a34a", f"Trading {mos:.0f}% below intrinsic value with {score}/3 quality flags."
    if mos >= 0:
        return "HOLD", "#f59e0b", f"Margin of safety {mos:.0f}% — fair, not cheap."
    return "SELL / AVOID", "#dc2626", f"Trading {abs(mos):.0f}% above intrinsic value — overpriced."
