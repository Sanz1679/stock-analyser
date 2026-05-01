"""SEC EDGAR XBRL fetcher — 10+ year annual fundamentals for US listed companies.

Uses the public SEC API (no key required). Spec: https://www.sec.gov/edgar/sec-api-documentation
"""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Optional

import pandas as pd


USER_AGENT = "StockAnalyser-OSS sanz1679@users.noreply.github.com"
TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

_cik_cache: dict[str, str] = {}
_facts_cache: dict[str, dict] = {}
_tickers_payload: Optional[dict] = None
_last_error: Optional[str] = None


def last_error() -> Optional[str]:
    """Return last error from EDGAR (None if last call succeeded)."""
    return _last_error


# Map our internal field names to candidate us-gaap concepts (most-preferred first).
CONCEPTS: dict[str, list[str]] = {
    "revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
    ],
    "gross_profit": ["GrossProfit"],
    "operating_income": ["OperatingIncomeLoss"],
    "net_income": [
        "NetIncomeLoss",
        "ProfitLoss",
    ],
    "eps_diluted": ["EarningsPerShareDiluted"],
    "eps_basic": ["EarningsPerShareBasic"],
    "operating_cf": [
        "NetCashProvidedByUsedInOperatingActivities",
        "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
    ],
    "capex": [
        "PaymentsToAcquirePropertyPlantAndEquipment",
        "PaymentsToAcquireProductiveAssets",
    ],
    "assets": ["Assets"],
    "current_assets": ["AssetsCurrent"],
    "current_liabilities": ["LiabilitiesCurrent"],
    "equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
    "long_term_debt": ["LongTermDebtNoncurrent", "LongTermDebt"],
    "short_term_debt": ["LongTermDebtCurrent", "DebtCurrent", "ShortTermBorrowings"],
    "shares_diluted": ["WeightedAverageNumberOfDilutedSharesOutstanding"],
    "shares_outstanding": [
        "CommonStockSharesOutstanding",
        "EntityCommonStockSharesOutstanding",
    ],
    "depreciation": [
        "DepreciationDepletionAndAmortization",
        "DepreciationAmortizationAndAccretionNet",
        "DepreciationAndAmortization",
        "Depreciation",
    ],
    "sbc": [
        "ShareBasedCompensation",
        "AllocatedShareBasedCompensationExpense",
    ],
    "interest_expense": [
        "InterestExpense",
        "InterestExpenseDebt",
    ],
    "cash": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
    ],
}


def _http_get_json(url: str, timeout: float = 15.0) -> Optional[dict]:
    global _last_error
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT,
                                               "Accept": "application/json",
                                               "Accept-Encoding": "identity",
                                               "Host": urllib.parse.urlparse(url).netloc})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        _last_error = f"HTTP {e.code} from {url}"
        return None
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
        _last_error = f"network error: {e}"
        return None
    except json.JSONDecodeError as e:
        _last_error = f"bad JSON: {e}"
        return None


def get_cik(ticker: str) -> Optional[str]:
    """Resolve ticker -> 10-digit zero-padded CIK string."""
    global _tickers_payload
    t = (ticker or "").upper().strip()
    if not t:
        return None
    if t in _cik_cache:
        return _cik_cache[t]
    if _tickers_payload is None:
        _tickers_payload = _http_get_json(TICKERS_URL) or {}
    for entry in _tickers_payload.values():
        if str(entry.get("ticker", "")).upper() == t:
            cik = str(entry.get("cik_str", "")).zfill(10)
            _cik_cache[t] = cik
            return cik
    return None


def get_company_facts(cik: str) -> Optional[dict]:
    if cik in _facts_cache:
        return _facts_cache[cik]
    data = _http_get_json(FACTS_URL.format(cik=cik))
    if data:
        _facts_cache[cik] = data
    return data


def _pick_unit(units: dict) -> Optional[str]:
    for candidate in ("USD", "USD/shares", "shares", "pure"):
        if candidate in units and units[candidate]:
            return candidate
    return next(iter(units), None) if units else None


def _annual_series(facts: dict, concept_aliases: list[str]) -> Optional[pd.Series]:
    """Build an annual fiscal-year-end series, merging across concept aliases.

    Many filers switch concept names mid-history (e.g. ``Revenues`` →
    ``RevenueFromContractWithCustomerExcludingAssessedTax`` after ASC 606 in 2018).
    To get a continuous 10-year history we walk the alias list in priority order
    and let later aliases fill in fiscal years not yet covered by earlier ones.

    - Filters to 10-K filings with fiscal period FY.
    - Within a single concept, the latest-filed value wins (handles 10-K/A).
    """
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    rows: dict[int, dict] = {}
    for concept in concept_aliases:
        node = us_gaap.get(concept)
        if not node:
            continue
        units = node.get("units", {})
        unit_key = _pick_unit(units)
        if not unit_key:
            continue
        for it in units[unit_key]:
            if it.get("fp") != "FY":
                continue
            if not str(it.get("form", "")).startswith("10-K"):
                continue
            fy = it.get("fy")
            val = it.get("val")
            end = it.get("end")
            filed = it.get("filed", "")
            if fy is None or val is None or not end:
                continue
            existing = rows.get(fy)
            if existing is None:
                rows[fy] = {"end": end, "val": float(val), "filed": filed,
                            "concept": concept}
            elif existing["concept"] == concept and filed > existing["filed"]:
                rows[fy].update({"end": end, "val": float(val), "filed": filed})
            # If existing came from a higher-priority concept, leave it.
    if not rows:
        return None
    ordered = sorted(rows.items(), key=lambda kv: kv[0])
    return pd.Series(
        [r[1]["val"] for r in ordered],
        index=pd.DatetimeIndex([r[1]["end"] for r in ordered]),
    )


def fetch_history(ticker: str) -> dict[str, pd.Series]:
    """Return dict of 10+ year annual series. Empty dict on failure."""
    global _last_error
    _last_error = None
    cik = get_cik(ticker)
    if not cik:
        if not _last_error:
            _last_error = f"ticker {ticker!r} not found in SEC tickers list"
        return {}
    facts = get_company_facts(cik)
    if not facts:
        return {}
    out: dict[str, pd.Series] = {}
    for key, aliases in CONCEPTS.items():
        s = _annual_series(facts, aliases)
        if s is not None and not s.empty:
            out[key] = s
    if not out:
        _last_error = f"CIK {cik} has no recognised us-gaap concepts"
    return out
