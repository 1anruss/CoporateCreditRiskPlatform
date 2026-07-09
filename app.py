"""
NOVA CREDIT PLATFORM — Streamlit app
Corporate credit risk analysis on live SEC EDGAR + market data.
Ratings and exposures illustrative; methodology is the deliverable.
"""
import time
import json
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
st.set_page_config(page_title="Nova Credit Platform", page_icon="📊", layout="wide")

SEC_USER_AGENT = st.secrets.get("SEC_USER_AGENT", "Nova Credit Platform contact@example.com")

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

# ticker: (sector, rating, EAD $M, tenor yrs)
UNIVERSE = {
    "MSFT": ("Software", "AAA", 2000, 5),
    "JNJ":  ("Healthcare", "AAA", 1500, 5),
    "AAPL": ("Consumer Tech", "AA+", 2000, 5),
    "WMT":  ("Retail", "AA", 1800, 4),
    "KO":   ("Consumer", "A+", 1200, 4),
    "NVDA": ("Semiconductors", "A+", 2000, 3),
    "MU":   ("Semiconductors", "BBB", 1500, 4),
    "INTC": ("Semiconductors", "BBB", 1500, 4),
    "MRVL": ("Semiconductors", "BBB-", 1000, 4),
    "F":    ("Autos", "BBB-", 1200, 5),
    "DAL":  ("Airlines", "BBB-", 1000, 5),
    "WBD":  ("Media", "BB+", 900, 5),
    "WDC":  ("Hardware", "BB+", 800, 4),
    "CCL":  ("Travel", "BBB-", 700, 6),
    "X":    ("Materials", "BB", 600, 5),
    "SMCI": ("Hardware", "BB-", 500, 3),
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
    out = {"market_cap": np.nan, "shares": np.nan, "pe": np.nan}
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
    if cl and ltd:
        y = max(set(cl) & set(ltd)) if set(cl) & set(ltd) else None
        if y:
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
    # Merton
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
    # Piotroski (latest year)
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
    # Beneish (latest year)
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
#  OUTLOOK ENGINES (verdict + forensic veto)
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
    fig.suptitle(f"{tk} — Daily Market-Implied Default Risk",
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
    ax.set_title("Merton approximation · rolling 90-day volatility · latest-year debt",
                 fontsize=8.5, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    return fig


def gauge_row(master_row):
    z = master_row["altman_z"]; dd = master_row["merton_dd"]; mpd = master_row["merton_PD"]
    zr = np.nan if pd.isna(z) else max(0, min(1, (6 - z) / 6))
    ddr = np.nan if pd.isna(dd) else max(0, min(1, (8 - dd) / 8))
    rtr = min(1, master_row["rating_PD"] / 0.14)

    def rcol(v):
        return RED_B[0] if v > 0.6 else YELLOW_B[0] if v > 0.3 else GREEN_B[0]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    specs = [(zr, "Altman Z'' (Accounting)", f"{z:.1f}" if pd.notna(z) else "n/a",
              master_row["altman_zone"]),
             (ddr, "Merton (Market)", f"DD {dd:.1f}" if pd.notna(dd) else "n/a",
              f"PD {mpd*100:.2f}%" if pd.notna(mpd) else ""),
             (rtr, "Agency Rating", master_row["rating"],
              f"PD {master_row['rating_PD']*100:.2f}%")]
    for ax, (v, title, big, small) in zip(axes, specs):
        ax.axis("off")
        th = np.linspace(np.pi, 0, 100)
        ax.plot(np.cos(th), np.sin(th), color="#dcdcdc", lw=9, solid_capstyle="round")
        if pd.notna(v):
            n = max(1, int(v * 100))
            ax.plot(np.cos(th[:n]), np.sin(th[:n]), color=rcol(v), lw=9, solid_capstyle="round")
        ax.text(0, 0.28, big, ha="center", fontsize=13, fontweight="bold", color="#1a1a1a")
        ax.text(0, -0.05, small, ha="center", fontsize=9, color="#5a6b7b")
        ax.text(0, -0.45, title, ha="center", fontsize=9, fontweight="bold", color="#1b2a41")
        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-0.6, 1.15)
    return fig


def quadrant_chart(master):
    q = master.dropna(subset=["altman_z", "merton_dd"])
    fig, ax = plt.subplots(figsize=(9, 7))
    for tk, row in q.iterrows():
        zs, ds = row["altman_z"] >= 2.6, row["merton_dd"] >= 6
        c = PALETTE["safe"] if (zs and ds) else PALETTE["distress"] if (not zs and not ds) \
            else PALETTE["warning"]
        ax.scatter(row["altman_z"], row["merton_dd"], s=110, color=c, edgecolor="white", zorder=3)
        ax.annotate(tk, (row["altman_z"], row["merton_dd"]), fontsize=8,
                    xytext=(5, 4), textcoords="offset points")
    ax.axvline(2.6, ls="--", color=PALETTE["muted"], lw=1)
    ax.axhline(6, ls="--", color=PALETTE["muted"], lw=1)
    ax.set_xlabel("Altman Z (accounting)")
    ax.set_ylabel("Merton DD (market)")
    ax.set_title("Accounting vs Market Risk", fontweight="bold")
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
    fig, ax = plt.subplots(figsize=(10, 0.5 * len(sv) + 1.5))
    rc = lambda v: PALETTE["safe"] if v < 0.33 else PALETTE["warning"] if v < 0.6 \
        else PALETTE["distress"]
    y = np.arange(len(sv))
    ax.axvspan(0, 0.33, color=GREEN_B[0], alpha=GREEN_B[1])
    ax.axvspan(0.33, 0.6, color=YELLOW_B[0], alpha=YELLOW_B[1])
    ax.axvspan(0.6, 1.0, color=RED_B[0], alpha=RED_B[1])
    ax.barh(y, sv["risk"], color=[rc(v) for v in sv["risk"]], height=0.55, zorder=3)
    for i, (s, r) in enumerate(sv.iterrows()):
        ax.text(r["risk"] + 0.015, i,
                f"{r['risk']*100:.0f}  (${r['EAD']:,.0f}M · {int(r['names'])})",
                va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels(sv.index, fontsize=9, fontweight="bold")
    ax.set_xlim(0, 1.02)
    ax.set_xticks([0, 0.33, 0.6, 1.0])
    ax.set_xticklabels(["0", "33", "60", "100"])
    ax.set_xlabel("Average risk score")
    ax.set_title("Sector Risk Map", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.invert_yaxis()
    return fig


# ──────────────────────────────────────────────────────────────────────
#  UI
# ──────────────────────────────────────────────────────────────────────
st.sidebar.title("Nova Credit Platform")
st.sidebar.caption("Corporate credit risk on live SEC EDGAR + market data")
page = st.sidebar.radio("View", ["Company Deep Dive", "Platform Dashboard", "Outlook Board"])
st.sidebar.markdown("---")
st.sidebar.caption("Ratings & exposures illustrative — methodology is the deliverable. "
                   "Built by Ian Casamano · Nova Holdings LLC")

master = build_master()

if page == "Company Deep Dive":
    tk = st.selectbox("Company", sorted(UNIVERSE.keys()),
                      index=sorted(UNIVERSE.keys()).index("CCL"))
    row = master.loc[tk]
    st.header(f"{tk} — Credit Profile")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rating", row["rating"])
    c2.metric("Stage (rating-implied)", f"Stage {int(row['stage'])}")
    c3.metric("Exposure", f"${row['EAD']:,.0f}M")
    c4.metric("Provision", f"${row['provision']:.1f}M")
    st.pyplot(gauge_row(row), use_container_width=True)
    with st.spinner("Building Altman deep dive from SEC filings..."):
        fig = altman_chart(tk)
    if fig:
        st.pyplot(fig, use_container_width=True)
    else:
        st.info("Altman time-series unavailable for this name.")
    with st.spinner("Computing daily market-implied default risk..."):
        fig = merton_daily_chart(tk)
    if fig:
        st.pyplot(fig, use_container_width=True)
    else:
        st.info("Daily Merton series unavailable for this name.")

elif page == "Platform Dashboard":
    st.header("Platform Dashboard — 16-name universe")
    view = master[["sector", "rating", "altman_z", "altman_zone", "piotroski_f",
                   "beneish_m", "merton_dd", "merton_PD", "rating_PD", "EAD",
                   "stage", "provision"]].copy()
    view["merton_PD"] = (view["merton_PD"] * 100).round(3)
    view["rating_PD"] = (view["rating_PD"] * 100).round(2)
    st.dataframe(view.round(2), use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(quadrant_chart(master), use_container_width=True)
    with c2:
        st.pyplot(sector_map(master), use_container_width=True)
        sp = master.groupby("stage")["provision"].sum()
        se = master.groupby("stage")["EAD"].sum()
        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(sp)); w = 0.35
        ax.bar(x - w/2, se / se.sum() * 100, w, color=PALETTE["muted"], label="% exposure")
        ax.bar(x + w/2, sp / sp.sum() * 100, w, color=PALETTE["distress"], label="% provision")
        ax.set_xticks(x)
        ax.set_xticklabels([f"Stage {int(s)}" for s in sp.index])
        ax.set_title("Exposure vs Provision by Stage", fontweight="bold")
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig, use_container_width=True)

else:  # Outlook Board
    st.header("Outlook Board — rule-based calls with forensic veto")
    st.caption("Rules decide the call; conviction (0-100) sizes the evidence. "
               "A forensic flag hard-caps conviction at 50 — no high-conviction calls "
               "on financials the market distrusts.")
    picks = st.multiselect("Companies", sorted(UNIVERSE.keys()),
                           default=["WDC", "SMCI", "MU", "INTC", "CCL", "F"])
    results = []
    prog = st.progress(0.0)
    for i, tk in enumerate(picks):
        acct = accounting_engine(tk)
        mkt = market_engine(tk)
        cred = credit_engine(tk, acct)
        if acct is None:
            continue
        call, conv, reasons = verdict(acct, mkt, cred, tk)
        results.append({"ticker": tk, "call": call, "conviction": conv,
                        "veto": any(r.startswith("[!]") for r in reasons),
                        "reasons": reasons})
        prog.progress((i + 1) / max(len(picks), 1))
    prog.empty()
    if results:
        CALL_COLOR = {"OVERPERFORM": PALETTE["safe"], "NEUTRAL": PALETTE["warning"],
                      "UNDERPERFORM": PALETTE["distress"]}
        board = pd.DataFrame(results).set_index("ticker").sort_values("conviction")
        fig, ax = plt.subplots(figsize=(10, 0.6 * len(board) + 1.5))
        y = np.arange(len(board))
        ax.barh(y, board["conviction"], color=[CALL_COLOR[c] for c in board["call"]], height=0.6)
        for i, (tk, r) in enumerate(board.iterrows()):
            tag = f"{r['call']}  {r['conviction']}" + ("  ⚑ veto" if r["veto"] else "")
            ax.text(r["conviction"] + 2, i, tag, va="center", fontsize=9, fontweight="bold")
        ax.axvline(35, ls="--", color=PALETTE["muted"], lw=0.8)
        ax.axvline(65, ls="--", color=PALETTE["muted"], lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(board.index, fontsize=10, fontweight="bold")
        ax.set_xlim(0, 118)
        ax.set_xlabel("Conviction (0-100)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.invert_yaxis()
        st.pyplot(fig, use_container_width=True)
        for r in results:
            with st.expander(f"{r['ticker']} — {r['call']} ({r['conviction']}/100)"):
                for line in r["reasons"]:
                    st.write(line)
