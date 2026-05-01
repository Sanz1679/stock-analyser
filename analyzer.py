"""Buffett-style analysis: fundamentals, auto-DCF, insights, 10-year summary.

US-only. Live price + ratios + quarterly data come from yfinance; long-form
annual history (10+ years) comes from SEC EDGAR XBRL.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

import edgar


ALIASES = {
    "net_income": ["Net Income", "Net Income Common Stockholders",
                   "Net Income From Continuing Operation Net Minority Interest"],
    "equity": ["Stockholders Equity", "Total Stockholder Equity", "Common Stock Equity"],
    "total_debt": ["Total Debt"],
    "long_term_debt": ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"],
    "short_term_debt": ["Current Debt", "Short Long Term Debt"],
    "operating_cf": ["Operating Cash Flow", "Total Cash From Operating Activities",
                     "Cash Flow From Continuing Operating Activities"],
    "capex": ["Capital Expenditure", "Capital Expenditures"],
    "free_cash_flow": ["Free Cash Flow"],
    "revenue": ["Total Revenue", "Operating Revenue"],
    "gross_profit": ["Gross Profit"],
    "operating_income": ["Operating Income", "Total Operating Income As Reported"],
    "ebitda": ["EBITDA", "Normalized EBITDA"],
    "depreciation": ["Reconciled Depreciation", "Depreciation And Amortization", "Depreciation"],
    "interest_expense": ["Interest Expense", "Interest Expense Non Operating"],
    "current_assets": ["Current Assets", "Total Current Assets"],
    "current_liabilities": ["Current Liabilities", "Total Current Liabilities"],
    "inventory": ["Inventory"],
    "total_assets": ["Total Assets"],
    "shares_basic": ["Basic Average Shares", "Diluted Average Shares"],
    "sbc": ["Stock Based Compensation"],
}


def _row(df: pd.DataFrame, key: str) -> Optional[pd.Series]:
    if df is None or df.empty:
        return None
    for label in ALIASES.get(key, [key]):
        if label in df.index:
            return df.loc[label].dropna()
    return None


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
    s = series.dropna()
    if len(s) < 3 or s.mean() == 0:
        return None
    cv = abs(s.std() / s.mean())
    return float(max(0.0, min(100.0, 100 - cv * 100)))


# ─── Ticker resolution ──────────────────────────────────────────────────────

def resolve_ticker(query: str) -> str:
    """Resolve user input to a US ticker symbol."""
    q = (query or "").strip()
    if not q:
        return ""
    if len(q) <= 6 and " " not in q:
        return q.upper()
    try:
        s = yf.Search(q, max_results=1)
        quotes = getattr(s, "quotes", None) or []
        if quotes:
            sym = quotes[0].get("symbol", q).upper()
            # Reject anything with an exchange suffix — this is a US-only tool.
            if "." not in sym:
                return sym
    except Exception:
        pass
    return q.upper()


def smart_fetch(query: str) -> "Fundamentals":
    """Resolve and fetch a US ticker. Returns Fundamentals (price may be None on failure)."""
    return fetch(resolve_ticker(query))


# ─── Data class ─────────────────────────────────────────────────────────────

@dataclass
class Fundamentals:
    ticker: str
    name: str = ""
    currency: str = "USD"  # always USD — this is a US-only analyser
    sector: str = ""
    industry: str = ""
    price: Optional[float] = None
    prev_close: Optional[float] = None
    shares_outstanding: Optional[float] = None
    market_cap: Optional[float] = None
    enterprise_value: Optional[float] = None
    pe_ratio: Optional[float] = None
    forward_pe: Optional[float] = None
    eps_ttm: Optional[float] = None
    eps_forward: Optional[float] = None
    book_value_per_share: Optional[float] = None
    price_to_sales: Optional[float] = None
    price_to_book: Optional[float] = None
    ev_to_ebitda: Optional[float] = None
    peg_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    payout_ratio: Optional[float] = None
    dividend_rate: Optional[float] = None
    ex_dividend_date: Optional[str] = None
    operating_margin: Optional[float] = None
    profit_margin: Optional[float] = None
    gross_margin: Optional[float] = None
    revenue_growth_yoy: Optional[float] = None
    earnings_growth_yoy: Optional[float] = None
    return_on_equity: Optional[float] = None
    return_on_assets: Optional[float] = None
    fifty_two_high: Optional[float] = None
    fifty_two_low: Optional[float] = None
    fifty_day_avg: Optional[float] = None
    two_hundred_day_avg: Optional[float] = None
    total_cash: Optional[float] = None
    total_debt: Optional[float] = None
    net_cash: Optional[float] = None
    next_earnings: Optional[str] = None
    news: list = field(default_factory=list)

    # Annual histories
    roe_history: pd.Series = field(default_factory=pd.Series)
    de_history: pd.Series = field(default_factory=pd.Series)
    eps_history: pd.Series = field(default_factory=pd.Series)
    fcf_history: pd.Series = field(default_factory=pd.Series)
    revenue_history: pd.Series = field(default_factory=pd.Series)
    gross_profit_history: pd.Series = field(default_factory=pd.Series)
    operating_income_history: pd.Series = field(default_factory=pd.Series)
    net_income_history: pd.Series = field(default_factory=pd.Series)
    ebitda_history: pd.Series = field(default_factory=pd.Series)
    owner_earnings_history: pd.Series = field(default_factory=pd.Series)
    operating_margin_history: pd.Series = field(default_factory=pd.Series)
    gross_margin_history: pd.Series = field(default_factory=pd.Series)
    sbc_history: pd.Series = field(default_factory=pd.Series)
    shares_history: pd.Series = field(default_factory=pd.Series)
    price_history: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Quarterly histories
    q_revenue: pd.Series = field(default_factory=pd.Series)
    q_net_income: pd.Series = field(default_factory=pd.Series)
    q_ebitda: pd.Series = field(default_factory=pd.Series)
    q_fcf: pd.Series = field(default_factory=pd.Series)
    q_eps: pd.Series = field(default_factory=pd.Series)
    q_operating_income: pd.Series = field(default_factory=pd.Series)
    q_gross_profit: pd.Series = field(default_factory=pd.Series)

    # Aggregates
    roe_avg: Optional[float] = None
    de_latest: Optional[float] = None
    eps_growth: Optional[float] = None
    fcf_growth: Optional[float] = None
    revenue_growth: Optional[float] = None
    fcf_latest: Optional[float] = None
    fcf_margin: Optional[float] = None
    owner_earnings_latest: Optional[float] = None
    earnings_stability: Optional[float] = None
    interest_coverage: Optional[float] = None
    current_ratio: Optional[float] = None
    quick_ratio: Optional[float] = None
    sbc_impact: Optional[float] = None
    buyback_yield: Optional[float] = None

    # Data source diagnostics
    edgar_years: Optional[int] = None     # # of distinct fiscal years pulled from EDGAR
    edgar_status: str = "not attempted"   # "ok" | "not attempted" | error message


# ─── Fetch ──────────────────────────────────────────────────────────────────

def fetch(ticker: str) -> Fundamentals:
    tk = yf.Ticker(ticker)
    try:
        info = tk.info or {}
    except Exception:
        info = {}

    fin = getattr(tk, "financials", pd.DataFrame())
    bal = getattr(tk, "balance_sheet", pd.DataFrame())
    cf = getattr(tk, "cashflow", pd.DataFrame())
    qfin = getattr(tk, "quarterly_financials", pd.DataFrame())
    qbal = getattr(tk, "quarterly_balance_sheet", pd.DataFrame())
    qcf = getattr(tk, "quarterly_cashflow", pd.DataFrame())

    f = Fundamentals(ticker=ticker)
    f.name = _safe(info, "longName", "shortName", default=ticker) or ticker
    f.currency = "USD"
    f.sector = _safe(info, "sector", default="") or ""
    f.industry = _safe(info, "industry", default="") or ""
    f.price = _safe(info, "currentPrice", "regularMarketPrice", "previousClose")
    f.prev_close = _safe(info, "previousClose")
    f.shares_outstanding = _safe(info, "sharesOutstanding", "impliedSharesOutstanding")
    f.market_cap = _safe(info, "marketCap")
    f.enterprise_value = _safe(info, "enterpriseValue")
    f.pe_ratio = _safe(info, "trailingPE")
    f.forward_pe = _safe(info, "forwardPE")
    f.eps_ttm = _safe(info, "trailingEps")
    f.eps_forward = _safe(info, "forwardEps")
    f.book_value_per_share = _safe(info, "bookValue")
    f.price_to_sales = _safe(info, "priceToSalesTrailing12Months")
    f.price_to_book = _safe(info, "priceToBook")
    f.ev_to_ebitda = _safe(info, "enterpriseToEbitda")
    f.peg_ratio = _safe(info, "pegRatio")
    f.fifty_two_high = _safe(info, "fiftyTwoWeekHigh")
    f.fifty_two_low = _safe(info, "fiftyTwoWeekLow")
    f.fifty_day_avg = _safe(info, "fiftyDayAverage")
    f.two_hundred_day_avg = _safe(info, "twoHundredDayAverage")

    dy = _safe(info, "dividendYield")
    f.dividend_yield = float(dy) * 100 if dy and dy < 1 else (float(dy) if dy else None)
    pr = _safe(info, "payoutRatio")
    f.payout_ratio = float(pr) * 100 if pr else None
    f.dividend_rate = _safe(info, "dividendRate")
    f.ex_dividend_date = _safe(info, "exDividendDate")

    om = _safe(info, "operatingMargins")
    f.operating_margin = float(om) * 100 if om else None
    pm = _safe(info, "profitMargins")
    f.profit_margin = float(pm) * 100 if pm else None
    gm = _safe(info, "grossMargins")
    f.gross_margin = float(gm) * 100 if gm else None

    rg = _safe(info, "revenueGrowth")
    f.revenue_growth_yoy = float(rg) * 100 if rg else None
    eg = _safe(info, "earningsGrowth", "earningsQuarterlyGrowth")
    f.earnings_growth_yoy = float(eg) * 100 if eg else None

    roe = _safe(info, "returnOnEquity")
    f.return_on_equity = float(roe) * 100 if roe else None
    roa = _safe(info, "returnOnAssets")
    f.return_on_assets = float(roa) * 100 if roa else None

    f.total_cash = _safe(info, "totalCash", default=0) or 0
    f.total_debt = _safe(info, "totalDebt", default=0) or 0
    f.net_cash = float(f.total_cash) - float(f.total_debt)

    cr = _safe(info, "currentRatio")
    f.current_ratio = float(cr) if cr else None
    qr = _safe(info, "quickRatio")
    f.quick_ratio = float(qr) if qr else None

    # Earnings date
    try:
        cal = tk.calendar
        if isinstance(cal, dict):
            ed = cal.get("Earnings Date")
            if isinstance(ed, list) and ed:
                f.next_earnings = str(ed[0])
            elif ed:
                f.next_earnings = str(ed)
    except Exception:
        pass

    # News
    try:
        raw_news = (tk.news or [])[:6]
        f.news = []
        for n in raw_news:
            content = n.get("content") or n
            f.news.append({
                "title": content.get("title", ""),
                "publisher": (content.get("provider") or {}).get("displayName") or n.get("publisher", ""),
                "url": (content.get("canonicalUrl") or {}).get("url") or n.get("link", ""),
                "time": content.get("pubDate") or n.get("providerPublishTime", ""),
            })
    except Exception:
        f.news = []

    # Price history
    try:
        f.price_history = tk.history(period="5y", auto_adjust=True)
    except Exception:
        f.price_history = pd.DataFrame()

    # Annual series
    ni = _row(fin, "net_income")
    eq = _row(bal, "equity")
    if ni is not None:
        f.net_income_history = ni.sort_index()
    if ni is not None and eq is not None:
        idx = ni.index.intersection(eq.index)
        if len(idx):
            roe_series = (ni.loc[idx] / eq.loc[idx]) * 100
            f.roe_history = roe_series.sort_index()
            f.roe_avg = float(roe_series.mean())

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

    da = _row(cf, "depreciation")
    capex_series = _row(cf, "capex")
    if ni is not None and capex_series is not None:
        idx = ni.index.intersection(capex_series.index)
        if da is not None:
            idx = idx.intersection(da.index)
            if len(idx):
                oe = ni.loc[idx] + da.loc[idx] + capex_series.loc[idx]
                f.owner_earnings_history = oe.sort_index()
                f.owner_earnings_latest = float(oe.sort_index().iloc[-1])
        elif len(idx):
            oe = ni.loc[idx] + capex_series.loc[idx]
            f.owner_earnings_history = oe.sort_index()
            f.owner_earnings_latest = float(oe.sort_index().iloc[-1])

    rev = _row(fin, "revenue")
    if rev is not None:
        f.revenue_history = rev.sort_index()
        f.revenue_growth = _cagr(f.revenue_history)
        if f.fcf_latest and not f.revenue_history.empty:
            latest_rev = f.revenue_history.iloc[-1]
            if latest_rev and latest_rev > 0:
                f.fcf_margin = f.fcf_latest / latest_rev * 100

    gp = _row(fin, "gross_profit")
    if gp is not None:
        f.gross_profit_history = gp.sort_index()
        if rev is not None:
            idx = gp.index.intersection(rev.index)
            if len(idx):
                f.gross_margin_history = (gp.loc[idx] / rev.loc[idx] * 100).sort_index()

    oi = _row(fin, "operating_income")
    if oi is not None:
        f.operating_income_history = oi.sort_index()
        if rev is not None:
            idx = oi.index.intersection(rev.index)
            if len(idx):
                f.operating_margin_history = (oi.loc[idx] / rev.loc[idx] * 100).sort_index()

    eb = _row(fin, "ebitda")
    if eb is not None:
        f.ebitda_history = eb.sort_index()

    sbc = _row(cf, "sbc")
    if sbc is not None:
        f.sbc_history = sbc.sort_index()
        if f.fcf_latest and not sbc.empty:
            latest_sbc = float(sbc.sort_index().iloc[-1])
            if f.fcf_latest > 0:
                f.sbc_impact = -latest_sbc / f.fcf_latest * 100

    sh = _row(fin, "shares_basic")
    if sh is not None:
        f.shares_history = sh.sort_index()
        if len(sh) >= 2:
            first, last = sh.iloc[0], sh.iloc[-1]
            if first and first > 0:
                f.buyback_yield = (first - last) / first * 100 / max(len(sh) - 1, 1)

    # Interest coverage
    int_exp = _row(fin, "interest_expense")
    if oi is not None and int_exp is not None:
        latest_oi = oi.sort_index().iloc[-1] if not oi.empty else None
        latest_ie = abs(int_exp.sort_index().iloc[-1]) if not int_exp.empty else None
        if latest_oi and latest_ie and latest_ie > 0:
            f.interest_coverage = float(latest_oi / latest_ie)

    # Quarterly
    q_rev = _row(qfin, "revenue")
    if q_rev is not None:
        f.q_revenue = q_rev.sort_index()
    q_ni = _row(qfin, "net_income")
    if q_ni is not None:
        f.q_net_income = q_ni.sort_index()
        if f.shares_outstanding:
            f.q_eps = (q_ni / f.shares_outstanding).sort_index()
    q_oi = _row(qfin, "operating_income")
    if q_oi is not None:
        f.q_operating_income = q_oi.sort_index()
    q_gp = _row(qfin, "gross_profit")
    if q_gp is not None:
        f.q_gross_profit = q_gp.sort_index()
    q_eb = _row(qfin, "ebitda")
    if q_eb is not None:
        f.q_ebitda = q_eb.sort_index()
    q_fcf = _row(qcf, "free_cash_flow")
    if q_fcf is None:
        q_ocf = _row(qcf, "operating_cf")
        q_capex = _row(qcf, "capex")
        if q_ocf is not None and q_capex is not None:
            idx = q_ocf.index.intersection(q_capex.index)
            q_fcf = (q_ocf.loc[idx] + q_capex.loc[idx]).dropna()
    if q_fcf is not None:
        f.q_fcf = q_fcf.sort_index()

    # Overlay 10+ year SEC EDGAR data where available — replaces yfinance
    # short histories with the full XBRL record for US filers.
    _apply_edgar_history(f)

    return f


def _apply_edgar_history(f: Fundamentals) -> None:
    """Pull 10-year XBRL annual data from SEC EDGAR and replace short yfinance series.

    Silent no-op if the ticker is not in EDGAR (foreign filer, ETF, fund) or the
    request fails — we keep whatever yfinance gave us. Diagnostic info is stored
    on the Fundamentals so the UI can show whether EDGAR succeeded.
    """
    try:
        hist = edgar.fetch_history(f.ticker)
    except Exception as exc:
        f.edgar_status = f"exception: {exc}"
        return
    if not hist:
        f.edgar_status = edgar.last_error() or "no data"
        return
    f.edgar_status = "ok"
    # Pick the longest series as the "years available" indicator.
    f.edgar_years = max(len(s) for s in hist.values())

    rev = hist.get("revenue")
    ni = hist.get("net_income")
    gp = hist.get("gross_profit")
    oi = hist.get("operating_income")
    eq = hist.get("equity")
    ocf = hist.get("operating_cf")
    capex = hist.get("capex")
    eps_d = hist.get("eps_diluted") or hist.get("eps_basic")
    da = hist.get("depreciation")
    sbc = hist.get("sbc")
    shares = hist.get("shares_diluted") or hist.get("shares_outstanding")
    ltd = hist.get("long_term_debt")
    std = hist.get("short_term_debt")
    int_exp = hist.get("interest_expense")

    if rev is not None:
        f.revenue_history = rev.sort_index()
        f.revenue_growth = _cagr(f.revenue_history)
    if ni is not None:
        f.net_income_history = ni.sort_index()
    if gp is not None:
        f.gross_profit_history = gp.sort_index()
        if rev is not None:
            idx = gp.index.intersection(rev.index)
            if len(idx):
                f.gross_margin_history = (gp.loc[idx] / rev.loc[idx] * 100).sort_index()
    if oi is not None:
        f.operating_income_history = oi.sort_index()
        if rev is not None:
            idx = oi.index.intersection(rev.index)
            if len(idx):
                f.operating_margin_history = (oi.loc[idx] / rev.loc[idx] * 100).sort_index()

    # FCF = operating cash flow - capex (capex reported positive in EDGAR cash-flow)
    if ocf is not None and capex is not None:
        idx = ocf.index.intersection(capex.index)
        if len(idx):
            fcf = (ocf.loc[idx] - capex.loc[idx]).dropna().sort_index()
            f.fcf_history = fcf
            if not fcf.empty:
                f.fcf_latest = float(fcf.iloc[-1])
                f.fcf_growth = _cagr(fcf)
            if rev is not None and not fcf.empty:
                latest_rev = rev.sort_index().iloc[-1]
                if latest_rev and latest_rev > 0:
                    f.fcf_margin = float(fcf.iloc[-1]) / float(latest_rev) * 100

    # EBITDA = Operating Income + D&A
    if oi is not None and da is not None:
        idx = oi.index.intersection(da.index)
        if len(idx):
            f.ebitda_history = (oi.loc[idx] + da.loc[idx]).sort_index()

    # ROE = Net Income / Equity
    if ni is not None and eq is not None:
        idx = ni.index.intersection(eq.index)
        idx = [i for i in idx if eq.loc[i] and eq.loc[i] != 0]
        if idx:
            roe = (ni.loc[idx] / eq.loc[idx] * 100).sort_index()
            f.roe_history = roe
            f.roe_avg = float(roe.tail(5).mean())

    # Debt / Equity
    debt = None
    if ltd is not None and std is not None:
        idx = ltd.index.intersection(std.index)
        if len(idx):
            debt = (ltd.loc[idx] + std.loc[idx]).sort_index()
    elif ltd is not None:
        debt = ltd.sort_index()
    elif std is not None:
        debt = std.sort_index()
    if debt is not None and eq is not None:
        idx = debt.index.intersection(eq.index)
        idx = [i for i in idx if eq.loc[i] and eq.loc[i] != 0]
        if idx:
            de = (debt.loc[idx] / eq.loc[idx]).sort_index()
            f.de_history = de
            if not de.empty:
                f.de_latest = float(de.iloc[-1])

    # EPS series (per-share, already)
    if eps_d is not None:
        f.eps_history = eps_d.sort_index()
        f.eps_growth = _cagr(f.eps_history)
        f.earnings_stability = _stability(f.eps_history)

    # Owner earnings = Net Income + D&A - CapEx
    if ni is not None and capex is not None:
        idx = ni.index.intersection(capex.index)
        if da is not None:
            idx2 = idx.intersection(da.index) if hasattr(idx, "intersection") else idx
            if len(idx2):
                oe = (ni.loc[idx2] + da.loc[idx2] - capex.loc[idx2]).sort_index()
                f.owner_earnings_history = oe
                f.owner_earnings_latest = float(oe.iloc[-1])
        elif len(idx):
            oe = (ni.loc[idx] - capex.loc[idx]).sort_index()
            f.owner_earnings_history = oe
            f.owner_earnings_latest = float(oe.iloc[-1])

    # SBC
    if sbc is not None:
        f.sbc_history = sbc.sort_index()
        if f.fcf_latest and f.fcf_latest > 0:
            latest_sbc = float(f.sbc_history.iloc[-1])
            f.sbc_impact = latest_sbc / f.fcf_latest * 100

    # Shares + buyback yield
    if shares is not None:
        f.shares_history = shares.sort_index()
        if len(shares) >= 2:
            first, last = shares.iloc[0], shares.iloc[-1]
            if first and first > 0:
                f.buyback_yield = (first - last) / first * 100 / max(len(shares) - 1, 1)

    # Interest coverage (latest year)
    if oi is not None and int_exp is not None and not oi.empty and not int_exp.empty:
        latest_oi = float(oi.sort_index().iloc[-1])
        latest_ie = abs(float(int_exp.sort_index().iloc[-1]))
        if latest_ie > 0:
            f.interest_coverage = latest_oi / latest_ie


# ─── Auto-DCF ───────────────────────────────────────────────────────────────

def auto_assumptions(f: Fundamentals) -> dict:
    """Realistic DCF inputs derived from history.

    - Normalised FCF: 3-year mean (smooths out lumpy capex years).
    - Growth: historical FCF CAGR clipped to 3-18%.
    - Discount: 10% (long-run equity hurdle).
    - Terminal growth: 2.5% (long-run GDP).
    """
    if not f.fcf_history.empty and len(f.fcf_history) >= 3:
        normalized_fcf = float(f.fcf_history.tail(3).mean())
    else:
        normalized_fcf = f.fcf_latest

    growth = f.fcf_growth if f.fcf_growth is not None else f.eps_growth
    if growth is None:
        growth = 6.0
    growth = float(max(3.0, min(18.0, growth)))
    return {
        "growth": growth / 100,
        "discount": 0.10,
        "terminal": 0.025,
        "years": 10,
        "fcf": normalized_fcf,
    }


def dcf_intrinsic_value(latest_fcf, shares_outstanding, growth_rate=0.08,
                        discount_rate=0.10, terminal_growth=0.025,
                        years=10, net_cash=0.0) -> Optional[float]:
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


def margin_of_safety(intrinsic, price) -> Optional[float]:
    if intrinsic is None or price is None or price <= 0:
        return None
    return (intrinsic - price) / intrinsic * 100


def graham_number(eps, bvps) -> Optional[float]:
    if not eps or not bvps or eps <= 0 or bvps <= 0:
        return None
    return float((22.5 * eps * bvps) ** 0.5)


def reverse_dcf_growth(price, latest_fcf, shares_outstanding,
                       discount_rate=0.10, terminal_growth=0.025,
                       years=10, net_cash=0.0) -> Optional[float]:
    if not (price and latest_fcf and shares_outstanding) or latest_fcf <= 0 or shares_outstanding <= 0:
        return None
    lo, hi = -0.20, 0.50
    mid = 0.0
    for _ in range(60):
        mid = (lo + hi) / 2
        v = dcf_intrinsic_value(latest_fcf, shares_outstanding, mid,
                                discount_rate, terminal_growth, years, net_cash)
        if v is None:
            return None
        if v < price:
            lo = mid
        else:
            hi = mid
    return mid * 100


def dcf_sensitivity(latest_fcf, shares_outstanding, growth_grid, discount_grid,
                    terminal_growth=0.025, years=10, net_cash=0.0) -> Optional[pd.DataFrame]:
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


def fcf_yield(market_cap, fcf) -> Optional[float]:
    if not market_cap or not fcf or market_cap <= 0:
        return None
    return fcf / market_cap * 100


# ─── Verdict ────────────────────────────────────────────────────────────────

def quality_score(f: Fundamentals) -> int:
    score = 0
    if f.roe_avg is not None and f.roe_avg >= 15:
        score += 1
    if f.de_latest is not None and f.de_latest <= 0.5:
        score += 1
    if f.eps_growth is not None and f.eps_growth >= 8:
        score += 1
    return score


def recommendation(mos, score) -> tuple[str, str, str]:
    """Quality-adjusted verdict.

    Strong businesses (score >=2) get a softer 'OVERVALUED — wait for pullback'
    instead of a screaming AVOID; only weak businesses trading above intrinsic
    get AVOID.
    """
    if mos is None:
        return "NO DATA", "neutral", "Not enough financial data to value this stock."

    if score >= 2:  # quality business
        if mos >= 25:
            return "BUY", "positive", (
                f"{mos:.0f}% margin of safety on a quality business "
                f"({score}/3 Buffett flags)."
            )
        if mos >= -15:
            return "HOLD", "warn", (
                f"Quality business near fair value ({mos:+.0f}% vs DCF). "
                f"Reasonable to hold, modest premium acceptable."
            )
        return "OVERVALUED", "negative", (
            f"Strong business, but priced ~{abs(mos):.0f}% above DCF intrinsic. "
            f"Consider waiting for a pullback or confirm growth thesis exceeds "
            f"the model's assumptions."
        )

    # weaker quality
    if mos >= 40:
        return "DEEP VALUE", "warn", (
            f"Cheap on DCF ({mos:.0f}% MoS) but only {score}/3 quality flags — "
            f"verify the growth thesis and check for value-trap signals."
        )
    if mos >= 0:
        return "HOLD", "warn", (
            f"Fair value ({mos:.0f}% MoS) but quality concerns — "
            f"only {score}/3 Buffett flags."
        )
    return "AVOID", "negative", (
        f"~{abs(mos):.0f}% above DCF intrinsic with weak quality scores "
        f"({score}/3) — risk of permanent capital loss."
    )


# ─── Heuristic insights ─────────────────────────────────────────────────────

def generate_insights(f: Fundamentals, intrinsic, mos, reverse_growth) -> dict:
    """Return {advantages: [...], risks: [...]} grounded in the data."""
    adv: list[dict] = []
    rsk: list[dict] = []

    # ─── Advantages ─────────
    if f.roe_avg is not None and f.roe_avg >= 15:
        adv.append({"title": "Strong returns on capital",
                    "text": f"5-year average ROE of {f.roe_avg:.1f}% indicates the business "
                            f"compounds shareholder capital efficiently — Buffett's "
                            f"15% threshold is comfortably cleared."})
    if f.de_latest is not None and f.de_latest <= 0.5 and (f.net_cash or 0) > 0:
        adv.append({"title": "Fortress balance sheet",
                    "text": f"Debt-to-equity of {f.de_latest:.2f} and net cash position "
                            f"of {_money(f.net_cash, f.currency)} provides resilience "
                            f"through downturns and optionality for buybacks or M&A."})
    if f.fcf_growth is not None and f.fcf_growth >= 8 and (f.fcf_latest or 0) > 0:
        adv.append({"title": "Compounding cash generation",
                    "text": f"Free cash flow has grown {f.fcf_growth:.1f}% per year, "
                            f"with the latest year producing {_money(f.fcf_latest, f.currency)} — "
                            f"a sign of durable economics, not just accounting profits."})
    if f.operating_margin is not None and f.operating_margin >= 20:
        adv.append({"title": "Pricing power",
                    "text": f"Operating margins of {f.operating_margin:.1f}% are well above "
                            f"the typical 10–15% range, suggesting customers value the product "
                            f"enough to absorb premium pricing."})
    if f.earnings_stability is not None and f.earnings_stability >= 70:
        adv.append({"title": "Predictable earnings",
                    "text": f"Earnings stability score of {f.earnings_stability:.0f}/100 — "
                            f"low year-to-year volatility makes future cash flows easier to model."})
    if f.gross_margin is not None and f.gross_margin >= 50:
        adv.append({"title": "High gross margins",
                    "text": f"Gross margin of {f.gross_margin:.1f}% leaves ample room for "
                            f"reinvestment in growth or returning capital to shareholders."})
    if (f.dividend_yield is not None and f.dividend_yield > 0
            and (f.payout_ratio or 0) < 60):
        adv.append({"title": "Sustainable dividend",
                    "text": f"Yield of {f.dividend_yield:.2f}% with a payout ratio of "
                            f"{f.payout_ratio:.0f}% — well-covered by earnings, room to grow."})
    if f.buyback_yield is not None and f.buyback_yield > 1:
        adv.append({"title": "Active buybacks",
                    "text": f"Share count has been declining ~{f.buyback_yield:.1f}% per year, "
                            f"boosting per-share metrics independent of business growth."})
    if mos is not None and mos >= 25:
        adv.append({"title": "Trading below intrinsic value",
                    "text": f"Current price implies a {mos:.0f}% margin of safety against "
                            f"a conservative DCF — limited downside, asymmetric upside."})

    # ─── Risks ─────────
    if f.de_latest is not None and f.de_latest > 1.0:
        rsk.append({"title": "Elevated leverage",
                    "text": f"Debt-to-equity of {f.de_latest:.2f} amplifies earnings sensitivity "
                            f"to rates and downturns. A bad year could pressure the dividend or capex."})
    if f.roe_avg is not None and f.roe_avg < 10:
        rsk.append({"title": "Weak returns on capital",
                    "text": f"Average ROE of {f.roe_avg:.1f}% is below the 15% threshold — "
                            f"the business may be reinvesting capital at unattractive rates."})
    if f.fcf_latest is not None and f.fcf_latest < 0:
        rsk.append({"title": "Negative free cash flow",
                    "text": f"Latest FCF of {_money(f.fcf_latest, f.currency)} means the business "
                            f"is consuming cash, not producing it. Acceptable in growth phases, "
                            f"risky in mature businesses."})
    if f.eps_growth is not None and f.eps_growth < 0:
        rsk.append({"title": "Earnings decline",
                    "text": f"EPS has shrunk {abs(f.eps_growth):.1f}% per year over the period — "
                            f"a structural headwind unless reversal catalysts are visible."})
    if f.pe_ratio is not None and f.pe_ratio > 30:
        rsk.append({"title": "Premium valuation",
                    "text": f"P/E of {f.pe_ratio:.1f} prices in strong growth. "
                            f"Any disappointment can trigger sharp multiple compression."})
    if reverse_growth is not None and reverse_growth > 15:
        rsk.append({"title": "High market expectations",
                    "text": f"Today's price implies the market expects {reverse_growth:.1f}% "
                            f"FCF growth per year for a decade. That bar gets harder as the "
                            f"company scales."})
    if f.payout_ratio is not None and f.payout_ratio > 80:
        rsk.append({"title": "Dividend at risk",
                    "text": f"Payout ratio of {f.payout_ratio:.0f}% leaves little buffer — "
                            f"a single weak year could force a cut."})
    if f.earnings_stability is not None and f.earnings_stability < 40:
        rsk.append({"title": "Earnings volatility",
                    "text": f"Stability score of {f.earnings_stability:.0f}/100 — large swings "
                            f"in earnings make valuation harder and increase risk of a "
                            f"poorly-timed entry."})
    if f.sbc_impact is not None and abs(f.sbc_impact) > 15:
        rsk.append({"title": "Stock-based compensation drag",
                    "text": f"SBC consumes ~{abs(f.sbc_impact):.0f}% of free cash flow — "
                            f"reported FCF overstates the cash actually available to shareholders."})
    if mos is not None and mos < -25:
        rsk.append({"title": "Trading well above intrinsic value",
                    "text": f"Current price is {abs(mos):.0f}% above a conservative DCF estimate — "
                            f"better to wait for a pullback or confirm growth assumptions."})

    return {"advantages": adv[:5], "risks": rsk[:5]}


def _money(x, currency="USD"):
    if x is None:
        return "—"
    if abs(x) >= 1e12: return f"${x/1e12:.2f}T"
    if abs(x) >= 1e9:  return f"${x/1e9:.2f}B"
    if abs(x) >= 1e6:  return f"${x/1e6:.2f}M"
    return f"${x:,.2f}"


# ─── 10-year summary ────────────────────────────────────────────────────────

def ten_year_summary(f: Fundamentals) -> pd.DataFrame:
    """One row per metric, columns = years (latest 10)."""
    rows: dict[str, pd.Series] = {}
    if not f.revenue_history.empty: rows["Revenue"] = f.revenue_history
    if not f.gross_profit_history.empty: rows["Gross Profit"] = f.gross_profit_history
    if not f.operating_income_history.empty: rows["Operating Income"] = f.operating_income_history
    if not f.net_income_history.empty: rows["Net Income"] = f.net_income_history
    if not f.fcf_history.empty: rows["Free Cash Flow"] = f.fcf_history
    if not f.eps_history.empty: rows["EPS"] = f.eps_history
    if not f.gross_margin_history.empty: rows["Gross Margin %"] = f.gross_margin_history
    if not f.operating_margin_history.empty: rows["Op Margin %"] = f.operating_margin_history
    if not f.roe_history.empty: rows["ROE %"] = f.roe_history
    if not f.de_history.empty: rows["Debt / Equity"] = f.de_history

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).T
    # Format columns as years
    df.columns = [c.year if hasattr(c, "year") else str(c) for c in df.columns]
    # Keep at most 12 most-recent columns (a decade plus a couple)
    if df.shape[1] > 12:
        df = df.iloc[:, -12:]
    return df
