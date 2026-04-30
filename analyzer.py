"""Buffett-style stock analysis: fundamentals, DCF, margin of safety, tranches."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf


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
    "depreciation": ["Reconciled Depreciation", "Depreciation And Amortization", "Depreciation"],
    "operating_income": ["Operating Income", "Operating Revenue"],
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
    sector: str = ""
    price: Optional[float] = None
    prev_close: Optional[float] = None
    shares_outstanding: Optional[float] = None
    market_cap: Optional[float] = None
    pe_ratio: Optional[float] = None
    forward_pe: Optional[float] = None
    eps_ttm: Optional[float] = None
    book_value_per_share: Optional[float] = None
    dividend_yield: Optional[float] = None
    payout_ratio: Optional[float] = None
    operating_margin: Optional[float] = None
    profit_margin: Optional[float] = None
    fifty_two_high: Optional[float] = None
    fifty_two_low: Optional[float] = None
    fifty_day_avg: Optional[float] = None
    two_hundred_day_avg: Optional[float] = None

    roe_history: pd.Series = field(default_factory=pd.Series)
    de_history: pd.Series = field(default_factory=pd.Series)
    eps_history: pd.Series = field(default_factory=pd.Series)
    fcf_history: pd.Series = field(default_factory=pd.Series)
    revenue_history: pd.Series = field(default_factory=pd.Series)
    owner_earnings_history: pd.Series = field(default_factory=pd.Series)
    price_history: pd.DataFrame = field(default_factory=pd.DataFrame)

    roe_avg: Optional[float] = None
    de_latest: Optional[float] = None
    eps_growth: Optional[float] = None
    fcf_growth: Optional[float] = None
    revenue_growth: Optional[float] = None
    fcf_latest: Optional[float] = None
    owner_earnings_latest: Optional[float] = None
    earnings_stability: Optional[float] = None
    net_cash: Optional[float] = None


def _safe(info: dict, *keys, default=None):
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
        pct = s.pct_change().dropna()
        return float(pct.mean() * 100) if not pct.empty else None
    return float(((last / first) ** (1 / years) - 1) * 100)


def _stability(series: pd.Series) -> Optional[float]:
    """0-100 score; 100 = perfectly stable, 0 = chaotic."""
    s = series.dropna()
    if len(s) < 3 or s.mean() == 0:
        return None
    cv = abs(s.std() / s.mean())
    return float(max(0.0, min(100.0, 100 - cv * 100)))


def fetch(ticker: str) -> Fundamentals:
    tk = yf.Ticker(ticker)
    try:
        info = tk.info or {}
    except Exception:
        info = {}

    fin = getattr(tk, "financials", pd.DataFrame())
    bal = getattr(tk, "balance_sheet", pd.DataFrame())
    cf = getattr(tk, "cashflow", pd.DataFrame())

    f = Fundamentals(ticker=ticker)
    f.name = _safe(info, "longName", "shortName", default=ticker) or ticker
    f.currency = _safe(info, "currency", default="USD") or "USD"
    f.sector = _safe(info, "sector", default="") or ""
    f.price = _safe(info, "currentPrice", "regularMarketPrice", "previousClose")
    f.prev_close = _safe(info, "previousClose")
    f.shares_outstanding = _safe(info, "sharesOutstanding", "impliedSharesOutstanding")
    f.market_cap = _safe(info, "marketCap")
    f.pe_ratio = _safe(info, "trailingPE")
    f.forward_pe = _safe(info, "forwardPE")
    f.eps_ttm = _safe(info, "trailingEps")
    f.book_value_per_share = _safe(info, "bookValue")
    dy = _safe(info, "dividendYield")
    f.dividend_yield = float(dy) * 100 if dy and dy < 1 else (float(dy) if dy else None)
    pr = _safe(info, "payoutRatio")
    f.payout_ratio = float(pr) * 100 if pr else None
    om = _safe(info, "operatingMargins")
    f.operating_margin = float(om) * 100 if om else None
    pm = _safe(info, "profitMargins")
    f.profit_margin = float(pm) * 100 if pm else None
    f.fifty_two_high = _safe(info, "fiftyTwoWeekHigh")
    f.fifty_two_low = _safe(info, "fiftyTwoWeekLow")
    f.fifty_day_avg = _safe(info, "fiftyDayAverage")
    f.two_hundred_day_avg = _safe(info, "twoHundredDayAverage")

    try:
        f.price_history = tk.history(period="5y", auto_adjust=True)
    except Exception:
        f.price_history = pd.DataFrame()

    ni = _row(fin, "net_income")
    eq = _row(bal, "equity")
    if ni is not None and eq is not None:
        idx = ni.index.intersection(eq.index)
        if len(idx):
            roe = (ni.loc[idx] / eq.loc[idx]) * 100
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
        idx = debt.index.intersection(eq.index)
        if len(idx):
            de = (debt.loc[idx] / eq.loc[idx]).sort_index()
            f.de_history = de
            if not de.empty:
                f.de_latest = float(de.iloc[-1])

    if ni is not None and f.shares_outstanding:
        eps = ni / f.shares_outstanding
        f.eps_history = eps.sort_index()
        f.eps_growth = _cagr(f.eps_history)
        f.earnings_stability = _stability(f.eps_history)

    fcf = _row(cf, "free_cash_flow")
    if fcf is None:
        ocf = _row(cf, "operating_cf")
        capex = _row(cf, "capex")
        if ocf is not None:
            if capex is not None:
                idx = ocf.index.intersection(capex.index)
                fcf = (ocf.loc[idx] + capex.loc[idx]).dropna()
            else:
                fcf = ocf
    if fcf is not None and not fcf.empty:
        f.fcf_history = fcf.sort_index()
        f.fcf_latest = float(f.fcf_history.iloc[-1])
        f.fcf_growth = _cagr(f.fcf_history)

    # Owner earnings ~ Net Income + D&A − Capex (Buffett's preferred metric).
    da = _row(cf, "depreciation")
    capex = _row(cf, "capex")
    if ni is not None and capex is not None:
        idx = ni.index.intersection(capex.index)
        if da is not None:
            idx = idx.intersection(da.index)
            if len(idx):
                oe = ni.loc[idx] + da.loc[idx] + capex.loc[idx]
                f.owner_earnings_history = oe.sort_index()
                f.owner_earnings_latest = float(oe.sort_index().iloc[-1])
        elif len(idx):
            oe = ni.loc[idx] + capex.loc[idx]
            f.owner_earnings_history = oe.sort_index()
            f.owner_earnings_latest = float(oe.sort_index().iloc[-1])

    rev = _row(fin, "revenue")
    if rev is not None:
        f.revenue_history = rev.sort_index()
        f.revenue_growth = _cagr(f.revenue_history)

    cash = _safe(info, "totalCash", default=0) or 0
    total_debt_info = _safe(info, "totalDebt", default=0) or 0
    f.net_cash = float(cash) - float(total_debt_info)

    return f


# ─── Valuation ──────────────────────────────────────────────────────────────

def dcf_intrinsic_value(
    latest_fcf: Optional[float],
    shares_outstanding: Optional[float],
    growth_rate: float = 0.08,
    discount_rate: float = 0.10,
    terminal_growth: float = 0.025,
    years: int = 10,
    net_cash: float = 0.0,
) -> Optional[float]:
    if not latest_fcf or latest_fcf <= 0 or not shares_outstanding or shares_outstanding <= 0:
        return None
    if discount_rate <= terminal_growth:
        return None
    pv = 0.0
    for y in range(1, years + 1):
        pv += latest_fcf * ((1 + growth_rate) ** y) / ((1 + discount_rate) ** y)
    terminal = latest_fcf * ((1 + growth_rate) ** years) * (1 + terminal_growth) / (discount_rate - terminal_growth)
    pv += terminal / ((1 + discount_rate) ** years)
    return (pv + (net_cash or 0)) / shares_outstanding


def margin_of_safety(intrinsic: Optional[float], price: Optional[float]) -> Optional[float]:
    if intrinsic is None or price is None or price <= 0:
        return None
    return (intrinsic - price) / intrinsic * 100


def graham_number(eps: Optional[float], bvps: Optional[float]) -> Optional[float]:
    """Ben Graham's defensive-investor fair value ≈ √(22.5 × EPS × BVPS)."""
    if not eps or not bvps or eps <= 0 or bvps <= 0:
        return None
    return float((22.5 * eps * bvps) ** 0.5)


def reverse_dcf_growth(
    price: Optional[float],
    latest_fcf: Optional[float],
    shares_outstanding: Optional[float],
    discount_rate: float = 0.10,
    terminal_growth: float = 0.025,
    years: int = 10,
    net_cash: float = 0.0,
) -> Optional[float]:
    """Solve for the growth rate the market is pricing in. Returns % per year."""
    if not (price and latest_fcf and shares_outstanding) or latest_fcf <= 0 or shares_outstanding <= 0:
        return None
    lo, hi = -0.20, 0.50
    for _ in range(60):
        mid = (lo + hi) / 2
        v = dcf_intrinsic_value(latest_fcf, shares_outstanding, mid, discount_rate,
                                terminal_growth, years, net_cash)
        if v is None:
            return None
        if v < price:
            lo = mid
        else:
            hi = mid
    return mid * 100


def dcf_sensitivity(
    latest_fcf: Optional[float],
    shares_outstanding: Optional[float],
    growth_grid: list[float],
    discount_grid: list[float],
    terminal_growth: float = 0.025,
    years: int = 10,
    net_cash: float = 0.0,
) -> Optional[pd.DataFrame]:
    if not latest_fcf or not shares_outstanding:
        return None
    data = []
    for d in discount_grid:
        row = []
        for g in growth_grid:
            v = dcf_intrinsic_value(latest_fcf, shares_outstanding, g, d, terminal_growth, years, net_cash)
            row.append(v)
        data.append(row)
    return pd.DataFrame(
        data,
        index=[f"{d*100:.1f}%" for d in discount_grid],
        columns=[f"{g*100:.1f}%" for g in growth_grid],
    )


# ─── Position sizing ────────────────────────────────────────────────────────

def tranches(
    intrinsic: Optional[float],
    current_price: Optional[float],
    discounts: tuple[float, ...] = (0.15, 0.30, 0.45),
    allocations: tuple[float, ...] = (0.25, 0.35, 0.40),
    budget: float = 10000.0,
) -> list[dict]:
    if not intrinsic or intrinsic <= 0:
        return []
    out = []
    for d, a in zip(discounts, allocations):
        trigger = intrinsic * (1 - d)
        active = current_price is not None and current_price <= trigger
        distance = ((trigger / current_price) - 1) * 100 if current_price else None
        out.append({
            "discount_pct": d * 100,
            "trigger": trigger,
            "allocation_pct": a * 100,
            "dollars": budget * a,
            "active": active,
            "distance_pct": distance,
        })
    return out


def price_zone(price_history: pd.DataFrame, current: Optional[float],
               fifty_two_high: Optional[float] = None,
               fifty_two_low: Optional[float] = None) -> Optional[dict]:
    high, low = fifty_two_high, fifty_two_low
    if (high is None or low is None) and not price_history.empty:
        last = price_history.tail(252)
        col_high = "High" if "High" in last else "Close"
        col_low = "Low" if "Low" in last else "Close"
        high = float(last[col_high].max())
        low = float(last[col_low].min())
    if not high or not low or not current or high <= low:
        return None
    return {
        "high": high,
        "low": low,
        "current": current,
        "position_pct": (current - low) / (high - low) * 100,
    }


# ─── Scoring ────────────────────────────────────────────────────────────────

def buffett_score(f: Fundamentals) -> tuple[int, list[str]]:
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


def quality_checklist(f: Fundamentals) -> list[dict]:
    """Auto-graded 6-point quality screen."""
    items = []

    items.append({
        "check": "ROE >= 15% (avg)",
        "pass": f.roe_avg is not None and f.roe_avg >= 15,
        "value": f"{f.roe_avg:.1f}%" if f.roe_avg is not None else "—",
    })
    items.append({
        "check": "Debt/Equity <= 0.5",
        "pass": f.de_latest is not None and f.de_latest <= 0.5,
        "value": f"{f.de_latest:.2f}" if f.de_latest is not None else "—",
    })
    items.append({
        "check": "EPS growing >= 8%/yr",
        "pass": f.eps_growth is not None and f.eps_growth >= 8,
        "value": f"{f.eps_growth:.1f}%" if f.eps_growth is not None else "—",
    })
    items.append({
        "check": "FCF positive (latest)",
        "pass": f.fcf_latest is not None and f.fcf_latest > 0,
        "value": f"{f.fcf_latest/1e9:.2f}B" if f.fcf_latest is not None else "—",
    })
    items.append({
        "check": "Operating margin >= 15%",
        "pass": f.operating_margin is not None and f.operating_margin >= 15,
        "value": f"{f.operating_margin:.1f}%" if f.operating_margin is not None else "—",
    })
    items.append({
        "check": "Earnings stability >= 60",
        "pass": f.earnings_stability is not None and f.earnings_stability >= 60,
        "value": f"{f.earnings_stability:.0f}/100" if f.earnings_stability is not None else "—",
    })
    return items


def recommendation(mos: Optional[float], score: int) -> tuple[str, str, str]:
    if mos is None:
        return "INSUFFICIENT DATA", "#888888", "DCF inputs unavailable for this ticker."
    if mos >= 25 and score >= 2:
        return "BUY", "#22c55e", f"Trading {mos:.0f}% below intrinsic value with {score}/3 quality flags."
    if mos >= 0:
        return "HOLD", "#fbbf24", f"Margin of safety {mos:.0f}% — fair, not cheap."
    return "SELL / AVOID", "#ef4444", f"Trading {abs(mos):.0f}% above intrinsic value — overpriced."


def fcf_yield(market_cap: Optional[float], fcf: Optional[float]) -> Optional[float]:
    if not market_cap or not fcf or market_cap <= 0:
        return None
    return fcf / market_cap * 100
