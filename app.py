"""
CORPORATE CREDIT RISK PLATFORM — Streamlit app
Credit risk analysis on live SEC EDGAR + market data.
Ratings and exposures illustrative; methodology is the deliverable.
"""
import time
from datetime import date

import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
from scipy.stats import norm
from scipy.optimize import fsolve
from scipy.interpolate import PchipInterpolator
import yfinance as yf

# ──────────────────────────────────────────────────────────────────────
#  CONFIG
# ──────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Corporate Credit Risk Platform", page_icon="📊", layout="wide")

SEC_USER_AGENT = st.secrets.get("SEC_USER_AGENT", "Corporate Credit Risk Platform contact@example.com")

LGD = 0.45
RISK_FREE = 0.045
TRADING_DAYS = 252

PALETTE = {"safe": "#2e8b57", "warning": "#d4a017", "distress": "#c0392b",
           "accent": "#2c5f8a", "muted": "#8a8a8a"}
GREEN_B = ("#2e8b57", 0.18)
YELLOW_B = ("#d4a017", 0.15)
RED_B = ("#c0392b", 0.18)

plt.rcParams.update({"figure.dpi": 110, "font.size": 10, "axes.titleweight": "bold",
                     "figure.facecolor": "white", "axes.facecolor": "white"})

RATING_PD = {
    "AAA": 0.0001, "AA+": 0.0002, "AA": 0.0003, "AA-": 0.0004,
    "A+": 0.0006, "A": 0.0008, "A-": 0.0012,
    "BBB+": 0.0018, "BBB": 0.0028, "BBB-": 0.0045,
    "BB+": 0.0080, "BB": 0.0140, "BB-": 0.0240,
    "B+": 0.0400, "B": 0.0600, "B-": 0.0900, "CCC": 0.1400,
}
BUCKETS = ["AAA", "AA", "A", "BBB", "BB", "B"]

# ticker: (sector, rating, EAD $M, tenor yrs) — 5 names per sector
UNIVERSE = {
    "MSFT": ("Technology", "AAA", 2000, 5),
    "AAPL": ("Technology", "AA+", 2000, 5),
    "GOOGL": ("Technology", "AA+", 1800, 5),
    "ORCL": ("Technology", "BBB", 1400, 4),
    "IBM":  ("Technology", "A-", 1300, 4),
    "NVDA": ("Semiconductors", "A+", 2000, 3),
    "MU":   ("Semiconductors", "BBB", 1500, 4),
    "INTC": ("Semiconductors", "BBB", 1500, 4),
    "MRVL": ("Semiconductors", "BBB-", 1000, 4),
    "SMCI": ("Semiconductors", "BB-", 500, 3),
    "WMT":  ("Consumer & Retail", "AA", 1800, 4),
    "COST": ("Consumer & Retail", "AA-", 1500, 4),
    "KO":   ("Consumer & Retail", "A+", 1200, 4),
    "TGT":  ("Consumer & Retail", "A", 1000, 4),
    "NKE":  ("Consumer & Retail", "AA-", 1000, 4),
    "JNJ":  ("Healthcare", "AAA", 1500, 5),
    "UNH":  ("Healthcare", "A+", 1500, 5),
    "MRK":  ("Healthcare", "A+", 1200, 5),
    "PFE":  ("Healthcare", "A", 1200, 5),
    "ABBV": ("Healthcare", "A-", 1100, 5),
    "CAT":  ("Industrials & Autos", "A", 1400, 5),
    "HON":  ("Industrials & Autos", "A", 1300, 5),
    "DE":   ("Industrials & Autos", "A", 1300, 5),
    "F":    ("Industrials & Autos", "BBB-", 1200, 5),
    "GM":   ("Industrials & Autos", "BBB", 1200, 5),
    "DAL":  ("Travel & Airlines", "BBB-", 1000, 5),
    "RCL":  ("Travel & Airlines", "BBB-", 800, 6),
    "CCL":  ("Travel & Airlines", "BBB-", 700, 6),
    "UAL":  ("Travel & Airlines", "BB", 700, 5),
    "AAL":  ("Travel & Airlines", "B+", 600, 5),
    "DIS":  ("Media & Telecom", "A-", 1300, 5),
    "T":    ("Media & Telecom", "BBB+", 1400, 5),
    "VZ":   ("Media & Telecom", "BBB+", 1400, 5),
    "WBD":  ("Media & Telecom", "BB+", 900, 5),
    "PARA": ("Media & Telecom", "BB+", 700, 5),
}

TAGS = {
    "ta": ["Assets"], "tl": ["Liabilities"],
    "ca": ["AssetsCurrent"], "cl": ["LiabilitiesCurrent"],
    "re": ["RetainedEarningsAccumulatedDeficit"],
    "ebit": ["OperatingIncomeLoss",
             "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest"],
    "eq": ["StockholdersEquity",
           "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "ni": ["NetIncomeLoss"],
    "cfo": ["NetCashProvidedByUsedInOperatingActivities",
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "ltd": ["LongTermDebtNoncurrent", "LongTermDebt", "LongTermDebtAndCapitalLeaseObligations"],
    "rev": ["RevenueFromContractWithCustomerExcludingAssessedTax",
            "RevenueFromContractWithCustomerIncludingAssessedTax",
            "Revenues", "SalesRevenueNet", "SalesRevenueGoodsNet"],
    "cogs": ["CostOfGoodsAndServicesSold", "CostOfRevenue", "CostOfGoodsSold"],
    "intx": ["InterestExpense", "InterestExpenseDebt", "InterestAndDebtExpense"],
    "ar": ["AccountsReceivableNetCurrent"],
    "ppe": ["PropertyPlantAndEquipmentNet"],
    "dep": ["DepreciationDepletionAndAmortization", "Depreciation"],
    "sga": ["SellingGeneralAndAdministrativeExpense"],
    "sh": ["CommonStockSharesOutstanding", "CommonStockSharesIssued",
           "WeightedAverageNumberOfDilutedSharesOutstanding"],
    "gp": ["GrossProfit"],
}

# ──────────────────────────────────────────────────────────────────────
#  DATA LAYER (cached)
# ──────────────────────────────────────────────────────────────────────
HEADERS = {"User-Agent": SEC_USER_AGENT, "Accept-Encoding": "gzip, deflate"}


@st.cache_data(ttl=86400, show_spinner=False)
def sec_ticker_map():
    r = requests.get("https://www.sec.gov/files/company_tickers.json",
                     headers=HEADERS, timeout=60)
    d = r.json()
    return {row["ticker"].upper(): int(row["cik_str"]) for row in d.values()}


@st.cache_data(ttl=86400, show_spinner=False)
def sec_facts(ticker: str):
    cik = sec_ticker_map().get(ticker.upper())
    if cik is None:
        return None
    for attempt in range(3):
        try:
            r = requests.get(
                f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json",
                headers=HEADERS, timeout=60)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
        except requests.RequestException:
            pass
        time.sleep(1.0 * (attempt + 1))
    return None


@st.cache_data(ttl=86400, show_spinner=False)
def price_history(ticker: str, period: str = "5y") -> pd.DataFrame:
    try:
        h = yf.Ticker(ticker).history(period=period)
        return h if not h.empty else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=86400, show_spinner=False)
def market_snapshot(ticker: str) -> dict:
    out = {"market_cap": np.nan, "shares": np.nan}
    try:
        t = yf.Ticker(ticker)
        fi = t.fast_info
        out["market_cap"] = float(fi.get("market_cap") or np.nan)
        out["shares"] = float(fi.get("shares") or np.nan)
    except Exception:
        pass
    return out


def _series(facts, tags, unit="USD", form="10-K", annual_only=False):
    out = {}
    if not facts:
        return out
    for tag in tags:
        try:
            obs = facts["facts"]["us-gaap"][tag]["units"][unit]
        except (KeyError, TypeError):
            continue
        for o in obs:
            if not o.get("form", "").startswith(form):
                continue
            end = o.get("end")
            filed = o.get("filed", "0")
            if not end:
                continue
            if annual_only and o.get("start"):
                try:
                    dd = (date.fromisoformat(end) - date.fromisoformat(o["start"])).days
                    if not (300 <= dd <= 400):
                        continue
                except Exception:
                    pass
            yr = int(end[:4])
            if yr not in out or filed > out[yr][1]:
                out[yr] = (o.get("val"), filed)
    return {y: v[0] for y, v in out.items()}


def _total_liabilities(facts):
    tl = _series(facts, TAGS["tl"])
    if tl:
        return tl
    ta = _series(facts, TAGS["ta"])
    eq = _series(facts, TAGS["eq"])
    return {y: ta[y] - eq[y] for y in ta if y in eq}


def _smooth(x, y, n=300):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 3:
        return x, y
    xs = np.linspace(x.min(), x.max(), n)
    return xs, PchipInterpolator(x, y)(xs)


# ──────────────────────────────────────────────────────────────────────
#  MODELS
# ──────────────────────────────────────────────────────────────────────
def default_point(ticker):
    f = sec_facts(ticker)
    if not f:
        return np.nan
    cl = _series(f, TAGS["cl"])
    ltd = _series(f, TAGS["ltd"])
    common = set(cl) & set(ltd)
    if common:
        y = max(common)
        return cl[y] + 0.5 * ltd[y]
    tl = _total_liabilities(f)
    return 0.5 * tl[max(tl)] if tl else np.nan


@st.cache_data(ttl=86400, show_spinner=False)
def score_company(ticker):
    """Latest Altman Z'', Merton DD/PD, Piotroski F, Beneish M for one name."""
    f = sec_facts(ticker)
    out = {"altman_z": np.nan, "merton_dd": np.nan, "merton_PD": np.nan,
           "piotroski_f": np.nan, "beneish_m": np.nan}
    if not f:
        return out
    ta = _series(f, TAGS["ta"]); tl = _total_liabilities(f)
    ca = _series(f, TAGS["ca"]); cl = _series(f, TAGS["cl"])
    re = _series(f, TAGS["re"]); ebit = _series(f, TAGS["ebit"], annual_only=True)
    eq = _series(f, TAGS["eq"])
    yrs = sorted(set(ta) & set(tl) & set(ca) & set(cl) & set(re) & set(ebit))
    if yrs:
        y = yrs[-1]
        try:
            x4 = 1.05 * (eq[y] / tl[y]) if (y in eq and tl[y]) else 0
            out["altman_z"] = (6.56 * (ca[y] - cl[y]) / ta[y] + 3.26 * re[y] / ta[y]
                               + 6.72 * ebit[y] / ta[y] + x4)
        except Exception:
            pass
    snap = market_snapshot(ticker)
    hist = price_history(ticker, "1y")
    D = default_point(ticker)
    E = snap["market_cap"]
    if not hist.empty and not np.isnan(E) and not np.isnan(D) and D > 0 and E > 0:
        lr = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
        sE = float(lr.std() * np.sqrt(TRADING_DAYS))
        if sE > 0:
            def eqs(p):
                V, sV = p
                if V <= 0 or sV <= 0:
                    return [1e6, 1e6]
                d1 = (np.log(V / D) + (RISK_FREE + 0.5 * sV ** 2)) / sV
                d2 = d1 - sV
                return [V * norm.cdf(d1) - D * np.exp(-RISK_FREE) * norm.cdf(d2) - E,
                        V * norm.cdf(d1) * sV / E - sE]
            try:
                V, sV = fsolve(eqs, [E + D, sE * E / (E + D)])
                if V > 0 and sV > 0:
                    dd = (np.log(V / D) + (RISK_FREE - 0.5 * sV ** 2)) / sV
                    out["merton_dd"] = float(dd)
                    out["merton_PD"] = float(norm.cdf(-dd))
            except Exception:
                pass
    ni = _series(f, TAGS["ni"], annual_only=True); cfo = _series(f, TAGS["cfo"], annual_only=True)
    ltd = _series(f, TAGS["ltd"]); rev = _series(f, TAGS["rev"], annual_only=True)
    cogs = _series(f, TAGS["cogs"], annual_only=True); gp0 = _series(f, TAGS["gp"], annual_only=True)
    sh = _series(f, TAGS["sh"], unit="shares")
    pyrs = sorted(set(ni) & set(ta) & set(cfo) & set(ca) & set(cl))
    if len(pyrs) >= 2:
        y, p = pyrs[-1], pyrs[-1] - 1
        if p in pyrs:
            try:
                s_ = 0
                s_ += ni[y] / ta[y] > 0
                s_ += cfo[y] > 0
                s_ += (ni[y] / ta[y]) > (ni[p] / ta[p])
                s_ += cfo[y] > ni[y]
                s_ += (ltd.get(y, 0) / ta[y]) < (ltd.get(p, 0) / ta[p])
                s_ += (ca[y] / cl[y]) > (ca[p] / cl[p])
                s_ += (sh.get(y, 1) <= sh.get(p, 1)) if (y in sh and p in sh) else 1
                def gp(yy):
                    if yy in gp0:
                        return gp0[yy]
                    if yy in rev and yy in cogs:
                        return rev[yy] - cogs[yy]
                    return None
                gpy, gpp = gp(y), gp(p)
                if y in rev and p in rev and gpy is not None and gpp is not None:
                    s_ += (gpy / rev[y]) > (gpp / rev[p])
                    s_ += (rev[y] / ta[y]) > (rev[p] / ta[p])
                out["piotroski_f"] = int(s_)
            except Exception:
                pass
    ar = _series(f, TAGS["ar"]); ppe = _series(f, TAGS["ppe"])
    dep = _series(f, TAGS["dep"], annual_only=True); sga = _series(f, TAGS["sga"], annual_only=True)
    byrs = sorted(set(ar) & set(rev) & set(cogs) & set(ca) & set(ppe) & set(ta) & set(tl))
    if len(byrs) >= 2:
        y = byrs[-1]
        p = y - 1 if y - 1 in byrs else byrs[-2]
        try:
            gm_y = (rev[y] - cogs[y]) / rev[y]
            gm_p = (rev[p] - cogs[p]) / rev[p]
            if gm_y > 0 and gm_p > 0:
                DSRI = (ar[y] / rev[y]) / (ar[p] / rev[p])
                GMI = gm_p / gm_y
                AQI = (1 - (ca[y] + ppe[y]) / ta[y]) / (1 - (ca[p] + ppe[p]) / ta[p])
                SGI = rev[y] / rev[p]
                DEPI = ((dep.get(p, 0) / (dep.get(p, 0) + ppe[p]))
                        / (dep.get(y, 0) / (dep.get(y, 0) + ppe[y]))) if dep.get(y) else 1
                SGAI = ((sga.get(y, 0) / rev[y]) / (sga.get(p, 0) / rev[p])) if sga.get(p) else 1
                LVGI = (tl[y] / ta[y]) / (tl[p] / ta[p])
                TATA = (ni.get(y, 0) - cfo.get(y, 0)) / ta[y]
                out["beneish_m"] = (-4.84 + 0.92 * DSRI + 0.528 * GMI + 0.404 * AQI
                                    + 0.892 * SGI + 0.115 * DEPI - 0.172 * SGAI
                                    + 4.679 * TATA - 0.327 * LVGI)
        except Exception:
            pass
    return out


@st.cache_data(ttl=86400, show_spinner=True)
def build_master():
    rows = {}
    for tk, (sector, rating, ead, tenor) in UNIVERSE.items():
        s = score_company(tk)
        rows[tk] = {"sector": sector, "rating": rating, "EAD": ead, "tenor": tenor,
                    "rating_PD": RATING_PD[rating], **s}
    m = pd.DataFrame(rows).T
    for c in ["EAD", "tenor", "rating_PD", "altman_z", "merton_dd", "merton_PD",
              "piotroski_f", "beneish_m"]:
        m[c] = pd.to_numeric(m[c], errors="coerce")
    m["stage"] = m["rating"].apply(
        lambda r: 3 if r in ["B+", "B", "B-", "CCC"] else 2 if r.startswith("BB") else 1)
    m["ECL_12m"] = m["rating_PD"] * LGD * m["EAD"]
    m["PD_lifetime"] = 1 - (1 - m["rating_PD"]) ** m["tenor"]
    m["ECL_lifetime"] = m["PD_lifetime"] * LGD * m["EAD"]
    m["provision"] = np.where(m["stage"] == 1, m["ECL_12m"], m["ECL_lifetime"])
    m["altman_zone"] = m["altman_z"].apply(
        lambda z: "n/a" if pd.isna(z) else "Safe" if z > 2.6 else "Grey" if z > 1.1 else "Distress")
    return m


# ──────────────────────────────────────────────────────────────────────
#  OUTLOOK ENGINES
# ──────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=86400, show_spinner=False)
def accounting_engine(tk):
    f = sec_facts(tk)
    if not f:
        return None
    ta = _series(f, TAGS["ta"]); tl = _total_liabilities(f)
    ca = _series(f, TAGS["ca"]); cl = _series(f, TAGS["cl"])
    ebit = _series(f, TAGS["ebit"], annual_only=True)
    ni = _series(f, TAGS["ni"], annual_only=True); cfo = _series(f, TAGS["cfo"], annual_only=True)
    rev = _series(f, TAGS["rev"], annual_only=True); cogs = _series(f, TAGS["cogs"], annual_only=True)
    intx = _series(f, TAGS["intx"], annual_only=True)
    yrs = sorted(set(ta) & set(rev) & set(ni))
    if len(yrs) < 3:
        return None
    y, p = yrs[-1], yrs[-2]
    out = {"fy": y}
    out["rev_growth_1y"] = (rev[y] - rev[p]) / abs(rev[p]) if rev.get(p) else np.nan
    def gm(yy):
        if yy in rev and yy in cogs and rev[yy]:
            return (rev[yy] - cogs[yy]) / rev[yy]
        return np.nan
    out["gross_margin"] = gm(y); gmp = gm(p)
    out["margin_improving"] = None if (np.isnan(out["gross_margin"]) or np.isnan(gmp)) \
        else out["gross_margin"] > gmp
    out["roa"] = ni[y] / ta[y]
    roap = ni[p] / ta[p] if (p in ni and p in ta) else np.nan
    out["roa_improving"] = None if np.isnan(roap) else out["roa"] > roap
    out["cfo_positive"] = cfo.get(y, 0) > 0
    out["leverage"] = tl[y] / ta[y] if y in tl else np.nan
    levp = tl[p] / ta[p] if p in tl else np.nan
    out["deleveraging"] = None if (np.isnan(out["leverage"]) or np.isnan(levp)) \
        else out["leverage"] < levp
    out["int_coverage"] = ebit[y] / intx[y] if (y in ebit and intx.get(y, 0) and intx.get(y, 0) > 0) else np.nan
    sc = score_company(tk)
    out["altman_z"] = sc["altman_z"]
    return out


@st.cache_data(ttl=86400, show_spinner=False)
def market_engine(tk):
    h = price_history(tk, "1y")
    if h.empty:
        return None
    px = h["Close"]
    out = {}
    out["ret_1y"] = float(px.iloc[-1] / px.iloc[0] - 1)
    out["ret_3m"] = float(px.iloc[-1] / px.iloc[-63] - 1) if len(px) > 63 else np.nan
    out["vol_ann"] = float(np.log(px / px.shift(1)).dropna().std() * np.sqrt(252))
    try:
        out["pe"] = yf.Ticker(tk).info.get("trailingPE", np.nan)
    except Exception:
        out["pe"] = np.nan
    spy = price_history("SPY", "1y")
    if not spy.empty:
        s = spy["Close"]
        out["rel_perf_1y"] = out["ret_1y"] - float(s.iloc[-1] / s.iloc[0] - 1)
    else:
        out["rel_perf_1y"] = np.nan
    return out


def credit_engine(tk, acct):
    rating = UNIVERSE.get(tk, (None, "NR"))[1]
    out = {"rating": rating}
    if acct is None:
        return out
    z = acct.get("altman_z", np.nan); lev = acct.get("leverage", np.nan)
    cov = acct.get("int_coverage", np.nan); roa = acct.get("roa", np.nan)
    score = 0
    if not np.isnan(z):
        score += 2 if z > 6 else 1 if z > 2.6 else -1 if z > 1.1 else -2
    if not np.isnan(lev):
        score += 1 if lev < 0.5 else 0 if lev < 0.7 else -1
    if not np.isnan(cov):
        score += 1 if cov > 8 else 0 if cov > 3 else -1
    if not np.isnan(roa):
        score += 1 if roa > 0.08 else 0 if roa > 0 else -1
    implied = "AAA" if score >= 5 else "AA" if score >= 4 else "A" if score >= 2 else \
              "BBB" if score >= 0 else "BB" if score >= -2 else "B"
    out["implied_bucket"] = implied
    actual = rating.rstrip("+-")
    out["rating_gap"] = (BUCKETS.index(implied) - BUCKETS.index(actual)) \
        if actual in BUCKETS else np.nan
    return out


def forensic_check(tk, acct, mkt=None, cred=None):
    flags = []
    sc = score_company(tk)
    m = sc.get("beneish_m", np.nan)
    if not np.isnan(m) and m > -1.78:
        flags.append(f"Beneish M {m:.2f} above -1.78 (possible manipulation)")
    if cred is not None and mkt is not None:
        gap = cred.get("rating_gap", np.nan)
        rel = mkt.get("rel_perf_1y", np.nan)
        vol = mkt.get("vol_ann", np.nan)
        if not np.isnan(gap) and gap <= -2:
            sev_rel = (not np.isnan(rel)) and rel < -0.30
            sev_vol = (not np.isnan(vol)) and vol > 0.70
            if sev_rel or sev_vol:
                msg = "Fundamentals imply " + str(cred["implied_bucket"]) + " vs rated " + str(cred["rating"])
                msg += " but market severely punishing"
                if not np.isnan(rel):
                    msg += " (rel " + f"{rel*100:+.0f}" + "%"
                if not np.isnan(vol):
                    msg += ", vol " + f"{vol*100:.0f}" + "%)"
                msg += " - gap likely reflects DISTRUST, not agency lag"
                flags.append(msg)
    return (len(flags) > 0, "; ".join(flags))


def verdict(acct, mkt, cred, tk=None):
    reasons = []
    pos = neg = total = 0

    def vote(cond, why_pos, why_neg, weight=1):
        nonlocal pos, neg, total
        if cond is None:
            return
        total += weight
        if cond:
            pos += weight
            reasons.append(f"[+] {why_pos}")
        else:
            neg += weight
            reasons.append(f"[-] {why_neg}")

    if acct:
        vote(acct.get("margin_improving"), "Gross margin improving", "Gross margin compressing")
        vote(acct.get("roa_improving"), "Profitability (ROA) improving", "Profitability declining")
        vote(acct.get("cfo_positive"), "Positive operating cash flow", "Negative operating cash flow")
        vote(acct.get("deleveraging"), "Deleveraging (debt/assets falling)", "Leverage rising")
        g = acct.get("rev_growth_1y", np.nan)
        if not np.isnan(g):
            vote(g > 0.05, f"Revenue growing ({g*100:.0f}% y/y)",
                 f"Revenue flat/declining ({g*100:.0f}% y/y)")
        z = acct.get("altman_z", np.nan)
        if not np.isnan(z):
            vote(z > 2.6, f"Altman Z in safe zone ({z:.1f})", f"Altman Z weak ({z:.1f})")
    if cred and not np.isnan(cred.get("rating_gap", np.nan)):
        vote(cred["rating_gap"] < 0,
             f"Fundamentals resemble a higher rating ({cred['implied_bucket']} vs {cred['rating']})",
             f"Fundamentals resemble same/lower rating ({cred['implied_bucket']} vs {cred['rating']})",
             weight=2)
    if mkt and not np.isnan(mkt.get("rel_perf_1y", np.nan)):
        vote(mkt["rel_perf_1y"] < 0.10,
             "Market hasn't fully repriced (room for convergence)",
             "Market already repriced (thesis largely in the price)")

    conviction = int(100 * pos / total) if total else 50
    flagged, why = forensic_check(tk, acct, mkt, cred)
    if flagged:
        reasons.append("[!] FORENSIC VETO: " + why)
        conviction = min(conviction, 50)

    call = "OVERPERFORM" if conviction >= 65 else "UNDERPERFORM" if conviction <= 35 else "NEUTRAL"
    return call, conviction, reasons


def outlook_report_text(tk):
    """Colab-style text report. Returns (text, call, conviction) or (None, None, None)."""
    acct = accounting_engine(tk)
    mkt = market_engine(tk)
    cred = credit_engine(tk, acct)
    if acct is None:
        return None, None, None
    L = []
    L.append("=" * 66)
    L.append(f"  COMPANY OUTLOOK REPORT — {tk}   ({date.today()})")
    L.append("=" * 66)
    L.append("")
    L.append(f"ACCOUNTING (FY{acct['fy']})")
    L.append(f"  Revenue growth (1y):  {acct['rev_growth_1y']*100:6.1f}%")
    mi = acct["margin_improving"]
    L.append(f"  Gross margin:         {acct['gross_margin']*100:6.1f}%  "
             f"({'improving' if mi else 'compressing' if mi is not None else 'n/a'})")
    ri = acct["roa_improving"]
    L.append(f"  ROA:                  {acct['roa']*100:6.1f}%  "
             f"({'improving' if ri else 'declining' if ri is not None else 'n/a'})")
    dl = acct["deleveraging"]
    L.append(f"  Leverage (TL/TA):     {acct['leverage']:6.2f}  "
             f"({'deleveraging' if dl else 'rising' if dl is not None else 'n/a'})")
    ic = acct["int_coverage"]
    if not np.isnan(ic):
        L.append(f"  Interest coverage:    {ic:6.1f}x")
    else:
        L.append("  Interest coverage:       n/a")
    if not np.isnan(acct["altman_z"]):
        L.append(f"  Altman Z'':           {acct['altman_z']:6.2f}")
    else:
        L.append("  Altman Z'':              n/a")
    if mkt:
        L.append("")
        L.append("MARKET")
        if not np.isnan(mkt.get("rel_perf_1y", np.nan)):
            L.append(f"  1y return:            {mkt['ret_1y']*100:6.1f}%   (vs S&P: {mkt['rel_perf_1y']*100:+.1f}%)")
        else:
            L.append(f"  1y return:            {mkt['ret_1y']*100:6.1f}%")
        if not np.isnan(mkt.get("ret_3m", np.nan)):
            L.append(f"  3m return:            {mkt['ret_3m']*100:6.1f}%")
        L.append(f"  Volatility (ann.):    {mkt['vol_ann']*100:6.1f}%")
        pe = mkt.get("pe", np.nan)
        try:
            pe = float(pe)
        except (TypeError, ValueError):
            pe = np.nan
        if not np.isnan(pe):
            L.append(f"  Trailing P/E:         {pe:6.1f}")
    L.append("")
    L.append("CREDIT")
    L.append(f"  Agency rating:        {cred['rating']}")
    L.append(f"  Fundamentals imply:   {cred.get('implied_bucket','n/a')} (heuristic, reported figures only)")
    gap = cred.get("rating_gap", np.nan)
    if not np.isnan(gap):
        d = "stronger than rated" if gap < 0 else "weaker than rated" if gap > 0 else "in line with rating"
        L.append(f"  Read:                 fundamentals {d}")
    call, conv, reasons = verdict(acct, mkt, cred, tk)
    L.append("")
    L.append("-" * 66)
    L.append(f"VERDICT: {call}   -   conviction {conv}/100")
    L.append("")
    for r in reasons:
        L.append(f"  {r}")
    L.append("")
    L.append("Thesis breaks if: revenue growth stalls, leverage reverses, or")
    L.append("margin improvement fails to hold through the next two filings.")
    L.append("=" * 66)
    return "\n".join(L), call, conv


# ──────────────────────────────────────────────────────────────────────
#  CHARTS
# ──────────────────────────────────────────────────────────────────────
def altman_chart(tk):
    f = sec_facts(tk)
    if not f:
        return None
    ta = _series(f, TAGS["ta"]); tl = _total_liabilities(f)
    ca = _series(f, TAGS["ca"]); cl = _series(f, TAGS["cl"])
    re = _series(f, TAGS["re"]); ebit = _series(f, TAGS["ebit"], annual_only=True)
    eq = _series(f, TAGS["eq"])
    years = sorted(set(ta) & set(tl) & set(ca) & set(cl) & set(re) & set(ebit))
    if len(years) < 2:
        return None
    rows = []
    for y in years:
        if not ta[y]:
            continue
        x4 = 1.05 * (eq[y] / tl[y]) if (y in eq and tl[y]) else 0
        rows.append({"year": y,
                     "X1: Working Capital / Assets": 6.56 * (ca[y] - cl[y]) / ta[y],
                     "X2: Retained Earnings / Assets": 3.26 * re[y] / ta[y],
                     "X3: EBIT / Assets": 6.72 * ebit[y] / ta[y],
                     "X4: Book Equity / Liabilities": x4})
    ts = pd.DataFrame(rows).set_index("year")
    ts["Z"] = ts.sum(axis=1)
    cols = list(ts.columns[:-1])
    ccol = [PALETTE["accent"], PALETTE["safe"], "#e08a3c", "#8e5aa8"]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 8),
                                 gridspec_kw={"height_ratios": [2, 1], "hspace": 0.45})
    fig.suptitle(f"{tk} — Altman Z'' Deep Dive", fontsize=14, fontweight="bold", y=0.96)
    bottom = np.zeros(len(ts))
    for c, col in zip(cols, ccol):
        a1.bar(ts.index.astype(str), ts[c], bottom=bottom, label=c, color=col, width=0.72)
        bottom += ts[c].values
    a1.axhline(2.6, ls="--", color=PALETTE["safe"], lw=1.1)
    a1.axhline(1.1, ls="--", color=PALETTE["distress"], lw=1.1)
    a1.set_ylabel("Weighted contribution")
    a1.legend(loc="upper left", fontsize=7)
    a1.set_title("Component breakdown", fontweight="bold", fontsize=10)
    a1.tick_params(axis="x", rotation=45, labelsize=7)
    a1.spines[["top", "right"]].set_visible(False)
    zmin = min(ts["Z"].min(), 0) - 1
    zmax = ts["Z"].max() + 1
    a2.axhspan(2.6, zmax, color=GREEN_B[0], alpha=GREEN_B[1])
    a2.axhspan(1.1, 2.6, color=YELLOW_B[0], alpha=YELLOW_B[1])
    a2.axhspan(zmin, 1.1, color=RED_B[0], alpha=RED_B[1])
    xs, ys = _smooth(ts.index, ts["Z"].values)
    a2.plot(xs, ys, color=PALETTE["accent"], lw=2.2)
    a2.text(ts.index[0], 2.7, " SAFE", fontsize=7, color="#2e8b57", va="bottom", fontweight="bold")
    a2.text(ts.index[0], 1.2, " GREY ZONE", fontsize=7, color="#a07908", va="bottom", fontweight="bold")
    a2.text(ts.index[0], zmin + 0.2, " DISTRESS", fontsize=7, color="#c0392b", va="bottom", fontweight="bold")
    a2.set_ylim(zmin, zmax)
    a2.set_xticks(list(ts.index))
    a2.set_xticklabels(ts.index, rotation=45, fontsize=7)
    a2.set_ylabel("Z'' Score")
    a2.set_title("Total Z'' over time", fontweight="bold", fontsize=10)
    a2.spines[["top", "right"]].set_visible(False)
    return fig


def piotroski_chart(tk):
    f = sec_facts(tk)
    if not f:
        return None
    ni = _series(f, TAGS["ni"], annual_only=True); ta = _series(f, TAGS["ta"])
    cfo = _series(f, TAGS["cfo"], annual_only=True); ltd = _series(f, TAGS["ltd"])
    ca = _series(f, TAGS["ca"]); cl = _series(f, TAGS["cl"])
    rev = _series(f, TAGS["rev"], annual_only=True); cogs = _series(f, TAGS["cogs"], annual_only=True)
    gp0 = _series(f, TAGS["gp"], annual_only=True); sh = _series(f, TAGS["sh"], unit="shares")

    def gp(y):
        if y in gp0:
            return gp0[y]
        if y in rev and y in cogs:
            return rev[y] - cogs[y]
        return None

    years = sorted(set(ni) & set(ta) & set(cfo) & set(ca) & set(cl))
    rows = []
    for y in years:
        p = y - 1
        if p not in years:
            continue
        try:
            roa = ni[y] / ta[y]; roa_p = ni[p] / ta[p]
            ltd_y = ltd.get(y, 0.0); ltd_p = ltd.get(p, 0.0)
            rec = {"ROA>0": ni[y] / ta[y] > 0, "CFO>0": cfo[y] > 0, "dROA>0": roa > roa_p,
                   "CFO>NI": cfo[y] > ni[y],
                   "dLeverage<0": (ltd_y / ta[y]) < (ltd_p / ta[p]),
                   "dCurrent>0": (ca[y] / cl[y]) > (ca[p] / cl[p]),
                   "No dilution": (sh.get(y, 1) <= sh.get(p, 1)) if (y in sh and p in sh) else True}
            gpy, gpp = gp(y), gp(p)
            if y in rev and p in rev and gpy is not None and gpp is not None:
                rec["dMargin>0"] = (gpy / rev[y]) > (gpp / rev[p])
                rec["dTurnover>0"] = (rev[y] / ta[y]) > (rev[p] / ta[p])
            else:
                rec["dMargin>0"] = None
                rec["dTurnover>0"] = None
            rows.append({"year": y, **rec, "F": sum(1 for v in rec.values() if v is True)})
        except (TypeError, ZeroDivisionError):
            continue
    if len(rows) < 2:
        return None
    ts = pd.DataFrame(rows).set_index("year")
    sc = ["ROA>0", "CFO>0", "dROA>0", "CFO>NI", "dLeverage<0", "dCurrent>0",
          "No dilution", "dMargin>0", "dTurnover>0"]
    grid = np.full((len(sc), len(ts)), 0.5)
    for j, (yr, row) in enumerate(ts.iterrows()):
        for i, s in enumerate(sc):
            v = row[s]
            grid[i, j] = 1 if v is True else (0 if v is False else 0.5)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 8),
                                 gridspec_kw={"height_ratios": [2, 1], "hspace": 0.55})
    fig.suptitle(f"{tk} — Piotroski F-Score Deep Dive", fontsize=14, fontweight="bold", y=0.96)
    cmap = mcolors.ListedColormap(["#d9534f", "#cccccc", "#2e7d5b"])
    a1.imshow(grid, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    a1.set_yticks(range(len(sc)))
    a1.set_yticklabels(sc, fontsize=7)
    a1.set_xticks(range(len(ts)))
    a1.set_xticklabels(ts.index, rotation=45, fontsize=7)
    a1.set_title("Signal breakdown (green = pass / red = fail / grey = n/a)",
                 fontweight="bold", fontsize=10)
    for i in range(len(sc)):
        for j in range(len(ts)):
            if grid[i, j] == 0.5:
                a1.text(j, i, "n/a", ha="center", va="center", fontsize=5, color="#555")
    a2.axhspan(7, 9.4, color=GREEN_B[0], alpha=GREEN_B[1])
    a2.axhspan(3, 7, color=YELLOW_B[0], alpha=YELLOW_B[1])
    a2.axhspan(-0.4, 3, color=RED_B[0], alpha=RED_B[1])
    xs, ys = _smooth(ts.index, ts["F"].values)
    a2.plot(xs, np.clip(ys, 0, 9), color=PALETTE["accent"], lw=2.2)
    a2.text(ts.index[0], 7.1, " STRONG", fontsize=7, color="#2e8b57", va="bottom", fontweight="bold")
    a2.text(ts.index[0], 3.1, " MODERATE", fontsize=7, color="#a07908", va="bottom", fontweight="bold")
    a2.text(ts.index[0], -0.2, " WEAK", fontsize=7, color="#c0392b", va="bottom", fontweight="bold")
    a2.set_ylim(-0.4, 9.4)
    a2.set_xticks(list(ts.index))
    a2.set_xticklabels(ts.index, rotation=45, fontsize=7)
    a2.set_ylabel("F-Score (0-9)")
    a2.set_title("Total F-Score over time", fontweight="bold", fontsize=10)
    a2.spines[["top", "right"]].set_visible(False)
    return fig


def merton_quarterly_chart(tk):
    hist = price_history(tk, "5y")
    if hist.empty or len(hist) < 300:
        return None
    D = default_point(tk)
    snap = market_snapshot(tk)
    shares = snap["shares"]
    if np.isnan(D) or D <= 0 or np.isnan(shares):
        return None
    px = hist["Close"]
    rows = []
    step = 63
    for i in range(step, len(px), step):
        window = px.iloc[i - step:i]
        E = float(px.iloc[i]) * shares
        sE = float(np.log(window / window.shift(1)).dropna().std() * np.sqrt(TRADING_DAYS))
        if sE <= 0:
            continue
        def eqs(pp):
            V, sV = pp
            if V <= 0 or sV <= 0:
                return [1e6, 1e6]
            d1 = (np.log(V / D) + (RISK_FREE + 0.5 * sV ** 2)) / sV
            d2 = d1 - sV
            return [V * norm.cdf(d1) - D * np.exp(-RISK_FREE) * norm.cdf(d2) - E,
                    V * norm.cdf(d1) * sV / E - sE]
        try:
            V, sV = fsolve(eqs, [E + D, sE * E / (E + D)])
            if V > 0 and sV > 0:
                dd = (np.log(V / D) + (RISK_FREE - 0.5 * sV ** 2)) / sV
                rows.append({"date": px.index[i], "PD": float(norm.cdf(-dd)) * 100})
        except Exception:
            continue
    if len(rows) < 3:
        return None
    ts = pd.DataFrame(rows).set_index("date")
    fig, ax = plt.subplots(figsize=(11, 4.8))
    fig.suptitle(f"{tk} — Merton Default Probability by Year",
                 fontsize=13, fontweight="bold", y=0.97)
    pmax = max(ts["PD"].max() * 1.15, 3)
    ax.axhspan(0, 0.5, color=GREEN_B[0], alpha=GREEN_B[1])
    ax.axhspan(0.5, 2, color=YELLOW_B[0], alpha=YELLOW_B[1])
    ax.axhspan(2, pmax, color=RED_B[0], alpha=RED_B[1])
    xnum = mdates.date2num(ts.index.to_pydatetime())
    xs, ys = _smooth(xnum, ts["PD"].values)
    ax.plot(mdates.num2date(xs), np.clip(ys, 0, None), color=PALETTE["accent"], lw=2.2)
    ax.text(ts.index[0], 0.05, " LOW", fontsize=7, color="#2e8b57", va="bottom", fontweight="bold")
    ax.text(ts.index[0], 0.55, " MODERATE", fontsize=7, color="#a07908", va="bottom", fontweight="bold")
    ax.text(ts.index[0], 2.1, " ELEVATED", fontsize=7, color="#c0392b", va="bottom", fontweight="bold")
    ax.set_ylim(0, pmax)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_ylabel("Default probability (%)")
    ax.set_title("Rolling market-based PD (quarterly, smoothed)", fontsize=9, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def merton_daily_chart(tk):
    hist = price_history(tk, "max")
    snap = market_snapshot(tk)
    D = default_point(tk)
    shares = snap["shares"]
    if hist.empty or np.isnan(shares) or np.isnan(D) or D <= 0:
        return None
    px = hist["Close"]
    E = px * shares
    ret = np.log(px / px.shift(1))
    sE = ret.rolling(90).std() * np.sqrt(TRADING_DAYS)
    V = E + D
    sV = sE * E / V
    DD = (np.log(V / D) + (RISK_FREE - 0.5 * sV ** 2)) / sV
    PD = pd.Series(norm.cdf(-DD) * 100, index=px.index).dropna().clip(lower=1e-4)
    if PD.empty:
        return None
    fig, ax = plt.subplots(figsize=(11, 4.8))
    fig.suptitle(f"{tk} — Daily Market-Implied Default Risk (full history)",
                 fontsize=13, fontweight="bold", y=0.97)
    pmax = max(PD.max() * 1.3, 3)
    ax.axhspan(1e-4, 0.5, color=GREEN_B[0], alpha=GREEN_B[1])
    ax.axhspan(0.5, 2, color=YELLOW_B[0], alpha=YELLOW_B[1])
    ax.axhspan(2, pmax, color=RED_B[0], alpha=RED_B[1])
    ax.plot(PD.index, PD.values, color=PALETTE["accent"], lw=0.9)
    ax.fill_between(PD.index, 1e-4, PD.values, color=PALETTE["accent"], alpha=0.08)
    ax.set_yscale("symlog", linthresh=0.01)
    ax.set_ylim(1e-4, pmax)
    ax.set_yticks([0, 0.01, 0.1, 0.5, 2, 10, 30])
    ax.set_yticklabels(["0", "0.01%", "0.1%", "0.5%", "2%", "10%", "30%"])
    ax.text(PD.index[0], 0.02, " LOW", fontsize=7, color="#2e8b57", va="bottom", fontweight="bold")
    ax.text(PD.index[0], 0.6, " MODERATE", fontsize=7, color="#a07908", va="bottom", fontweight="bold")
    ax.text(PD.index[0], 2.4, " ELEVATED", fontsize=7, color="#c0392b", va="bottom", fontweight="bold")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_ylabel("Default probability (log)")
    ax.set_title("Merton approximation / rolling 90-day volatility / latest-year debt",
                 fontsize=8.5, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def sector_map(master):
    d = master.copy()
    d["z_risk"] = d["altman_z"].apply(lambda z: np.nan if pd.isna(z) else max(0, min(1, (6 - z) / 6)))
    d["dd_risk"] = d["merton_dd"].apply(lambda x: np.nan if pd.isna(x) else max(0, min(1, (8 - x) / 8)))
    d["rt_risk"] = d["rating_PD"] / 0.14
    d["risk"] = d[["z_risk", "dd_risk", "rt_risk"]].mean(axis=1)
    sv = d.groupby("sector").agg(risk=("risk", "mean"), EAD=("EAD", "sum"),
                                 names=("risk", "size")).sort_values("risk")
    fig, ax = plt.subplots(figsize=(10, 0.55 * len(sv) + 1.5))
    rc = lambda v: PALETTE["safe"] if v < 0.33 else PALETTE["warning"] if v < 0.6 \
        else PALETTE["distress"]
    y = np.arange(len(sv))
    ax.axvspan(0, 0.33, color=GREEN_B[0], alpha=GREEN_B[1])
    ax.axvspan(0.33, 0.6, color=YELLOW_B[0], alpha=YELLOW_B[1])
    ax.axvspan(0.6, 1.0, color=RED_B[0], alpha=RED_B[1])
    ax.barh(y, sv["risk"], color=[rc(v) for v in sv["risk"]], height=0.55, zorder=3)
    for i, (s, r) in enumerate(sv.iterrows()):
        ax.text(r["risk"] + 0.015, i,
                f"{r['risk']*100:.0f}  (${r['EAD']:,.0f}M / {int(r['names'])} names)",
                va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(sv.index, fontsize=9, fontweight="bold")
    ax.set_xlim(0, 1.02)
    ax.set_xticks([0, 0.33, 0.6, 1.0])
    ax.set_xticklabels(["0", "33", "60", "100"])
    ax.set_xlabel("Average risk score (0 = safest / 100 = riskiest)")
    ax.set_title("Sector Risk Map", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.invert_yaxis()
    return fig


def stage_chart(master):
    sp = master.groupby("stage")["provision"].sum()
    se = master.groupby("stage")["EAD"].sum()
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(sp))
    w = 0.35
    ax.bar(x - w / 2, se / se.sum() * 100, w, color=PALETTE["muted"], label="% exposure")
    ax.bar(x + w / 2, sp / sp.sum() * 100, w, color=PALETTE["distress"], label="% provision")
    for i, (e, p) in enumerate(zip(se / se.sum() * 100, sp / sp.sum() * 100)):
        ax.text(i - w / 2, e, f"{e:.0f}%", ha="center", va="bottom", fontsize=8)
        ax.text(i + w / 2, p, f"{p:.0f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Stage {int(s)}" for s in sp.index])
    ax.set_title("Exposure vs Provision by Stage", fontweight="bold")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def ecl_chart(row):
    GREEN, AMBER, RED = "#2e7d5b", "#c99512", "#b23a3a"
    stg = int(row["stage"])
    scol = {1: GREEN, 2: AMBER, 3: RED}[stg]
    vals = [row["ECL_12m"], row["ECL_lifetime"], row["provision"]]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    b = ax.bar(["12-Month\nLoss", "Lifetime\nLoss", "Booked\nProvision"], vals,
               color=["#d8dee5", "#d8dee5", scol])
    for bar, v in zip(b, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"${v:.1f}M",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("$M")
    ax.set_title("Expected Credit Loss", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────
#  ADDITIONAL CHART — Beneish single-year index view
# ──────────────────────────────────────────────────────────────────────
def beneish_chart(tk):
    f = sec_facts(tk)
    if not f:
        return None
    ar = _series(f, TAGS["ar"]); rev = _series(f, TAGS["rev"], annual_only=True)
    cogs = _series(f, TAGS["cogs"], annual_only=True)
    ca = _series(f, TAGS["ca"]); ppe = _series(f, TAGS["ppe"]); ta = _series(f, TAGS["ta"])
    dep = _series(f, TAGS["dep"], annual_only=True); sga = _series(f, TAGS["sga"], annual_only=True)
    ni = _series(f, TAGS["ni"], annual_only=True); cfo = _series(f, TAGS["cfo"], annual_only=True)
    tl = _total_liabilities(f)
    years = sorted(set(ar) & set(rev) & set(cogs) & set(ca) & set(ppe) & set(ta) & set(tl))
    if len(years) < 2:
        return None
    y = years[-1]
    p = y - 1 if y - 1 in years else years[-2]
    gm_y = (rev[y] - cogs[y]) / rev[y] if rev[y] else 0
    gm_p = (rev[p] - cogs[p]) / rev[p] if rev[p] else 0
    if gm_y <= 0 or gm_p <= 0:
        return None
    try:
        DSRI = (ar[y] / rev[y]) / (ar[p] / rev[p]); GMI = gm_p / gm_y
        AQI = (1 - (ca[y] + ppe[y]) / ta[y]) / (1 - (ca[p] + ppe[p]) / ta[p]); SGI = rev[y] / rev[p]
        DEPI = ((dep.get(p, 0) / (dep.get(p, 0) + ppe[p]))
                / (dep.get(y, 0) / (dep.get(y, 0) + ppe[y]))) if dep.get(y) else 1
        SGAI = ((sga.get(y, 0) / rev[y]) / (sga.get(p, 0) / rev[p])) if sga.get(p) else 1
        LVGI = (tl[y] / ta[y]) / (tl[p] / ta[p]); TATA = (ni.get(y, 0) - cfo.get(y, 0)) / ta[y]
        M = (-4.84 + 0.92 * DSRI + 0.528 * GMI + 0.404 * AQI + 0.892 * SGI + 0.115 * DEPI
             - 0.172 * SGAI + 4.679 * TATA - 0.327 * LVGI)
    except (TypeError, ZeroDivisionError):
        return None
    idx = ["DSRI", "GMI", "AQI", "SGI", "DEPI", "SGAI", "TATA", "LVGI"]
    vals = [DSRI, GMI, AQI, SGI, DEPI, SGAI, TATA, LVGI]
    baseline = [1, 1, 1, 1, 1, 1, 0, 1]
    flag = "Possible manipulator" if M > -1.78 else "No red flags"
    fig, ax = plt.subplots(figsize=(11, 4.6))
    fig.suptitle(f"{tk} — Beneish M-Score (FY{y})", fontsize=13, fontweight="bold", y=0.97)
    ax.bar(idx, vals, color=PALETTE["accent"], width=0.6, label="Company", zorder=2)
    ax.plot(idx, baseline, "o--", color=PALETTE["muted"], label="Benign baseline", zorder=3)
    ax.axhline(1, color=PALETTE["muted"], lw=0.6, alpha=0.5)
    ax.set_title(f"8 indices vs benign baseline   |   M = {M:.2f}   |   {flag}",
                 fontsize=10, fontweight="bold")
    ax.set_ylabel("Index value")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    return fig


# ──────────────────────────────────────────────────────────────────────
#  UI — COMPANY WORKSTATION
# ──────────────────────────────────────────────────────────────────────
st.sidebar.title("Corporate Credit Risk Platform")
st.sidebar.caption("Internal credit analysis system - live SEC EDGAR + market data")
page = st.sidebar.radio("Mode", ["Company Workstation", "Portfolio Dashboard"])
st.sidebar.markdown("---")
st.sidebar.caption("Ratings & exposures illustrative - methodology is the deliverable.")

ZONE_ICON = {"Safe": ":green[● Safe]", "Grey": ":orange[● Grey zone]",
             "Distress": ":red[● Distress]", "n/a": "○ n/a"}


def clean_ticker(raw):
    """Validate free-text ticker: uppercase, SEC-listed, sane characters."""
    t = (raw or "").strip().upper().replace(".", "-")
    if not t or len(t) > 8 or not all(c.isalnum() or c == "-" for c in t):
        return None
    return t if t in sec_ticker_map() else None


def scores_line(tk):
    sc = score_company(tk)
    z = sc["altman_z"]
    zone = "n/a" if pd.isna(z) else "Safe" if z > 2.6 else "Grey" if z > 1.1 else "Distress"
    parts = []
    if pd.notna(z):
        parts.append(f"Altman Z'' `{z:.2f}` {ZONE_ICON[zone]}")
    if pd.notna(sc["piotroski_f"]):
        parts.append(f"Piotroski `{int(sc['piotroski_f'])}/9`")
    if pd.notna(sc["beneish_m"]):
        parts.append(f"Beneish `{sc['beneish_m']:.2f}`")
    if pd.notna(sc["merton_PD"]):
        parts.append(f"Merton PD `{sc['merton_PD']*100:.3f}%`")
    return "**Latest scores:**  " + "  ·  ".join(parts) if parts else \
        "**Latest scores:** insufficient filing data"


if page == "Company Workstation":
    top1, top2 = st.columns([2, 3])
    with top1:
        raw = st.text_input("Company lookup - enter any US-listed ticker",
                            value="CCL", max_chars=8,
                            help="Any ticker on SEC EDGAR. The 35-name modeled book "
                                 "adds exposure, staging and ECL on top.")
    tk = clean_ticker(raw)
    if tk is None:
        st.error(f"'{raw}' not found on SEC EDGAR - check the ticker "
                 "(use dashes for share classes, e.g. BRK-B).")
        st.stop()
    in_book = tk in UNIVERSE
    with top2:
        st.write("")
        with st.spinner("Pulling filings..."):
            st.markdown(scores_line(tk))

    if in_book:
        master = build_master()
        row = master.loc[tk]
        sector = UNIVERSE[tk][0]
        st.markdown(f"## {tk}  ·  {sector}")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Rating", row["rating"])
        c2.metric("Stage", f"{int(row['stage'])}", help="IFRS 9 stage, rating-implied")
        c3.metric("Exposure", f"${row['EAD']:,.0f}M")
        c4.metric("Tenor", f"{int(row['tenor'])}y")
        c5.metric("Rating PD (1y)", f"{row['rating_PD']*100:.2f}%")
        c6.metric("Provision", f"${row['provision']:.1f}M")
    else:
        st.markdown(f"## {tk}")
        st.info("Not in the modeled portfolio - filing and market analytics available; "
                "exposure, staging and ECL require a booked position.")
        row = None
    st.markdown("---")

    modules = ["Overview", "Altman Z''", "Piotroski F", "Beneish M",
               "Market Risk (Merton)", "Outlook Report", "SEC Filings"]
    if in_book:
        modules.insert(5, "ECL & Provisioning")
    module = st.radio("Analysis modules", modules, horizontal=True,
                      label_visibility="collapsed")

    if module == "Overview":
        c1, c2 = st.columns([1, 1])
        sc = score_company(tk)
        with c1:
            st.subheader("Credit position")
            if in_book:
                st.write(f"- **Sector:** {UNIVERSE[tk][0]}")
                st.write(f"- **Rating (assigned proxy):** {row['rating']} - "
                         f"1y PD {row['rating_PD']*100:.2f}%")
                st.write(f"- **IFRS 9 stage (rating-implied):** Stage {int(row['stage'])}")
                st.write(f"- **Exposure:** ${row['EAD']:,.0f}M over {int(row['tenor'])} years")
                st.write(f"- **Booked provision:** ${row['provision']:.1f}M "
                         f"({row['provision']/row['EAD']*100:.2f}% coverage)")
            else:
                st.write("- **Status:** not in modeled book (no assigned rating/exposure)")
                st.write("- All model analytics below computed live from this "
                         "company's SEC filings and market data")
            st.subheader("Model snapshot")
            zval = sc["altman_z"]
            zone = "n/a" if pd.isna(zval) else "Safe" if zval > 2.6 else \
                "Grey" if zval > 1.1 else "Distress"
            snap = pd.DataFrame({
                "Model": ["Altman Z''", "Piotroski F", "Beneish M", "Merton DD", "Merton PD"],
                "Value": [f"{zval:.2f}" if pd.notna(zval) else "n/a",
                          f"{int(sc['piotroski_f'])}/9" if pd.notna(sc['piotroski_f']) else "n/a",
                          f"{sc['beneish_m']:.2f}" if pd.notna(sc['beneish_m']) else "n/a",
                          f"{sc['merton_dd']:.2f}" if pd.notna(sc['merton_dd']) else "n/a",
                          f"{sc['merton_PD']*100:.3f}%" if pd.notna(sc['merton_PD']) else "n/a"],
                "Read": [zone,
                         "Strong" if pd.notna(sc['piotroski_f']) and sc['piotroski_f'] >= 7
                         else "Moderate" if pd.notna(sc['piotroski_f']) and sc['piotroski_f'] >= 4
                         else "Weak" if pd.notna(sc['piotroski_f']) else "n/a",
                         "Flag" if pd.notna(sc['beneish_m']) and sc['beneish_m'] > -1.78
                         else "Clean" if pd.notna(sc['beneish_m']) else "n/a",
                         "-", "-"]})
            st.dataframe(snap, hide_index=True, use_container_width=True)
            st.caption("Open a module above for the full chart on any of these.")
        with c2:
            if in_book:
                st.subheader("Expected credit loss")
                st.pyplot(ecl_chart(row), use_container_width=True)
            else:
                st.subheader("Market risk preview")
                with st.spinner("Computing daily Merton series..."):
                    fig = merton_daily_chart(tk)
                if fig:
                    st.pyplot(fig, use_container_width=True)
                else:
                    st.info("Daily Merton series unavailable for this name.")

    elif module == "Altman Z''":
        st.caption("Accounting-based distress score. Components stacked by year; "
                   "trajectory against safe / grey / distress zones.")
        with st.spinner("Building from SEC filings..."):
            fig = altman_chart(tk)
        if fig:
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("Insufficient filing data.")
    elif module == "Piotroski F":
        st.caption("Nine binary fundamental signals per year; total score against "
                   "strong / moderate / weak zones.")
        with st.spinner("Building from SEC filings..."):
            fig = piotroski_chart(tk)
        if fig:
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("Insufficient filing data.")
    elif module == "Beneish M":
        st.caption("Earnings-manipulation screen: eight forensic indices vs a benign baseline. "
                   "M above -1.78 = possible manipulator.")
        with st.spinner("Building from SEC filings..."):
            fig = beneish_chart(tk)
        if fig:
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("Not meaningful for this name (missing fields or negative gross margin).")
    elif module == "Market Risk (Merton)":
        st.caption("Structural model: equity as a call option on assets. "
                   "Quarterly solved series plus daily approximation over full history.")
        with st.spinner("Solving quarterly Merton model..."):
            fig = merton_quarterly_chart(tk)
        if fig:
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("Quarterly series unavailable.")
        with st.spinner("Computing daily approximation..."):
            fig = merton_daily_chart(tk)
        if fig:
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("Daily series unavailable.")
    elif module == "ECL & Provisioning":
        c1, c2 = st.columns([1, 1])
        with c1:
            st.subheader("IFRS 9 mechanics")
            st.write(f"- **Stage {int(row['stage'])}** (rating-implied): "
                     + ("12-month ECL basis" if row["stage"] == 1 else "lifetime ECL basis"))
            st.write(f"- 12-month ECL = PD x LGD x EAD = "
                     f"{row['rating_PD']*100:.2f}% x {int(LGD*100)}% x ${row['EAD']:,.0f}M "
                     f"= **${row['ECL_12m']:.1f}M**")
            st.write(f"- Lifetime PD over {int(row['tenor'])}y = "
                     f"{row['PD_lifetime']*100:.2f}% -> lifetime ECL **${row['ECL_lifetime']:.1f}M**")
            st.write(f"- **Booked provision: ${row['provision']:.1f}M** "
                     f"({row['provision']/row['EAD']*100:.2f}% coverage)")
            st.caption("Real IFRS 9 staging uses lender-private data; staging here is "
                       "rating-implied and labeled as such.")
        with c2:
            st.pyplot(ecl_chart(row), use_container_width=True)

    elif module == "Outlook Report":
        with st.spinner("Running three engines + verdict..."):
            text, call, conv = outlook_report_text(tk)
        if text is None:
            st.info("Insufficient filing data for this name.")
        else:
            color = {"OVERPERFORM": "green", "NEUTRAL": "orange", "UNDERPERFORM": "red"}[call]
            st.markdown(f"### Verdict: :{color}[{call}] - conviction {conv}/100")
            if not in_book:
                st.caption("Note: no assigned rating for this name, so the rating-gap "
                           "vote and distrust veto are inactive - verdict rests on "
                           "accounting and market signals only.")
            st.code(text, language=None)

    else:  # SEC Filings
        cik = sec_ticker_map().get(tk, "")
        base = ("https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany"
                f"&CIK={cik}&dateb=&owner=include&count=20&type=")
        st.subheader(f"{tk} - primary sources on SEC EDGAR")
        st.markdown(f"- [10-K annual reports]({base}10-K) - full audited statements + MD&A")
        st.markdown(f"- [10-Q quarterly reports]({base}10-Q) - interim statements")
        st.markdown(f"- [8-K current reports]({base}8-K) - material events as they happen")
        st.markdown(f"- [DEF 14A proxy statements]({base}DEF+14A) - governance & compensation")
        st.markdown(f"- [All filings](https://www.sec.gov/cgi-bin/browse-edgar?"
                    f"action=getcompany&CIK={cik}&dateb=&owner=include&count=40)")
        st.caption("Every model input in this system traces back to these filings "
                   "plus market prices - same sources a credit analyst reads.")

else:  # Portfolio Dashboard
    st.header("Portfolio Dashboard")
    st.caption(f"{len(UNIVERSE)} companies / 5 per sector - first load builds the master "
               "table from EDGAR (cached 24h), so give it a minute")
    master = build_master()
    cikmap = sec_ticker_map()
    view = master[["sector", "rating", "altman_z", "altman_zone", "piotroski_f",
                   "beneish_m", "merton_dd", "merton_PD", "rating_PD", "EAD",
                   "stage", "provision"]].copy()
    view["merton_PD"] = (view["merton_PD"] * 100).round(3)
    view["rating_PD"] = (view["rating_PD"] * 100).round(2)
    view["10-K filings"] = [
        ("https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany"
         f"&CIK={cikmap.get(t, '')}&type=10-K&dateb=&owner=include&count=10")
        for t in view.index]
    st.dataframe(view.round(2), use_container_width=True,
                 column_config={"10-K filings": st.column_config.LinkColumn(
                     "10-K filings", display_text="View 10-Ks")})
    st.pyplot(sector_map(master), use_container_width=True)
    st.pyplot(stage_chart(master), use_container_width=True)
