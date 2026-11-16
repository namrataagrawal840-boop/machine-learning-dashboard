"""Descriptive / diagnostic statistics computed from the filtered data (no modelling)."""
from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd
from scipy import stats as ss

MIN_N_DEFAULT = 30


def share(mask: pd.Series | None) -> tuple[float, int, int]:
    """(percentage, numerator, denominator) for a boolean series with NA excluded."""
    if mask is None:
        return (math.nan, 0, 0)
    m = mask.dropna()
    n = int(len(m))
    k = int(m.astype(bool).sum()) if n else 0
    return ((100.0 * k / n) if n else math.nan, k, n)


def fmt_pct(p: float, k: int | None = None, n: int | None = None, digits: int = 1) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "n/a"
    s = f"{p:.{digits}f}%"
    if k is not None and n is not None:
        s += f" ({k:,}/{n:,})"
    return s


def fmt_inr(x: float | None) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"₹{x:,.0f}"


def fmt_num(x, digits: int = 2) -> str:
    try:
        if x is None or math.isnan(float(x)):
            return "n/a"
    except (TypeError, ValueError):
        return str(x)
    return f"{float(x):,.{digits}f}"


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return (100 * (c - h), 100 * (c + h))


def describe_numeric(s: pd.Series) -> dict:
    v = pd.to_numeric(s, errors="coerce").dropna()
    if v.empty:
        return {"n": 0}
    q1, med, q3 = v.quantile([0.25, 0.5, 0.75])
    return {"n": int(len(v)), "mean": float(v.mean()), "median": float(med), "q1": float(q1), "q3": float(q3),
            "min": float(v.min()), "max": float(v.max()), "std": float(v.std(ddof=1)) if len(v) > 1 else math.nan,
            "zero_share": float((v == 0).mean() * 100)}


def iqr_outliers(s: pd.Series, k: float = 1.5) -> tuple[pd.Series, float, float]:
    v = pd.to_numeric(s, errors="coerce")
    q1, q3 = v.quantile(0.25), v.quantile(0.75)
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    return ((v < lo) | (v > hi)) & v.notna(), float(lo), float(hi)


def crosstab_share(df: pd.DataFrame, row: str, col: str, normalize: str = "index") -> tuple[pd.DataFrame, pd.DataFrame]:
    """(percentage table, count table) with NA rows dropped."""
    d = df[[row, col]].dropna()
    if d.empty:
        return pd.DataFrame(), pd.DataFrame()
    counts = pd.crosstab(d[row], d[col], dropna=True)
    counts = counts.loc[(counts.sum(axis=1) > 0), (counts.sum(axis=0) > 0)]
    if counts.empty:
        return counts, counts
    if normalize == "index":
        pct = counts.div(counts.sum(axis=1), axis=0) * 100
    else:
        pct = counts.div(counts.sum(axis=0), axis=1) * 100
    return pct, counts


def multi_share_by(df: pd.DataFrame, group: str, ms_cols: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """% of answering respondents in each group selecting each option. Returns (pct table, group n)."""
    sub = df[[group] + ms_cols].dropna(subset=[group])
    answered = sub[ms_cols].notna().any(axis=1)
    sub = sub[answered]
    if sub.empty:
        return pd.DataFrame(), pd.Series(dtype=int)
    g = sub.groupby(group, observed=True)
    n = g.size()
    pct = g[ms_cols].apply(lambda x: x.fillna(False).astype(float).mean() * 100)
    return pct, n


def cramers_v(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    """Bias-corrected Cramér's V (Bergsma 2013), chi-square p-value, n."""
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    n = len(d)
    if n < 2:
        return (math.nan, math.nan, n)
    t = pd.crosstab(d["x"], d["y"])
    if t.shape[0] < 2 or t.shape[1] < 2:
        return (math.nan, math.nan, n)
    chi2, p, _, _ = ss.chi2_contingency(t, correction=False)
    phi2 = chi2 / n
    r, k = t.shape
    phi2c = max(0.0, phi2 - (k - 1) * (r - 1) / max(n - 1, 1))
    rc = r - (r - 1) ** 2 / max(n - 1, 1)
    kc = k - (k - 1) ** 2 / max(n - 1, 1)
    den = min(kc - 1, rc - 1)
    return (math.sqrt(phi2c / den) if den > 0 else math.nan, float(p), n)


def corr_matrix(df: pd.DataFrame, cols: list[str], method: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Pairwise-complete correlations: (r, p-value, n)."""
    cols = [c for c in cols if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().sum() > 2]
    r = pd.DataFrame(np.nan, index=cols, columns=cols)
    p = r.copy()
    n = pd.DataFrame(0, index=cols, columns=cols)
    for a, b in itertools.combinations_with_replacement(cols, 2):
        d = df[[a, b]].apply(pd.to_numeric, errors="coerce").dropna() if a != b else \
            df[[a]].apply(pd.to_numeric, errors="coerce").dropna()
        nn = len(d)
        n.loc[a, b] = n.loc[b, a] = nn
        if a == b:
            r.loc[a, b], p.loc[a, b] = 1.0, 0.0
            continue
        if nn < 3 or d[a].nunique() < 2 or d[b].nunique() < 2:
            continue
        res = ss.pearsonr(d[a], d[b]) if method == "pearson" else ss.spearmanr(d[a], d[b])
        r.loc[a, b] = r.loc[b, a] = float(res[0])
        p.loc[a, b] = p.loc[b, a] = float(res[1])
    return r, p, n


def cramers_matrix(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cols = [c for c in cols if c in df.columns and df[c].nunique(dropna=True) >= 2]
    v = pd.DataFrame(np.nan, index=cols, columns=cols)
    p = v.copy()
    n = pd.DataFrame(0, index=cols, columns=cols)
    for a, b in itertools.combinations_with_replacement(cols, 2):
        if a == b:
            v.loc[a, b], n.loc[a, b] = 1.0, int(df[a].notna().sum())
            continue
        vv, pp, nn = cramers_v(df[a], df[b])
        v.loc[a, b] = v.loc[b, a] = vv
        p.loc[a, b] = p.loc[b, a] = pp
        n.loc[a, b] = n.loc[b, a] = nn
    return v, p, n


def strength_word(r: float, kind: str = "corr") -> str:
    if r is None or math.isnan(r):
        return "undetermined"
    a = abs(r)
    if kind == "v":
        return "negligible" if a < 0.1 else "weak" if a < 0.2 else "moderate" if a < 0.4 else "relatively strong"
    return "negligible" if a < 0.1 else "weak" if a < 0.3 else "moderate" if a < 0.5 else "strong"


def top_pair(r: pd.DataFrame) -> tuple[str, str, float] | None:
    best = None
    for a, b in itertools.combinations(r.columns, 2):
        v = r.loc[a, b]
        if pd.notna(v) and (best is None or abs(v) > abs(best[2])):
            best = (a, b, float(v))
    return best


def group_median(df: pd.DataFrame, group: str, value: str, min_n: int) -> pd.DataFrame:
    d = df[[group, value]].dropna()
    if d.empty:
        return pd.DataFrame(columns=[group, "n", "median", "mean", "q1", "q3", "reliable"])
    g = d.groupby(group, observed=True)[value]
    out = pd.DataFrame({"n": g.size(), "median": g.median(), "mean": g.mean(),
                        "q1": g.quantile(0.25), "q3": g.quantile(0.75)}).reset_index()
    out["reliable"] = out["n"] >= min_n
    return out


def group_share(df: pd.DataFrame, group: str, flag: str, min_n: int) -> pd.DataFrame:
    d = df[[group, flag]].dropna()
    if d.empty:
        return pd.DataFrame(columns=[group, "n", "k", "pct", "reliable"])
    g = d.groupby(group, observed=True)[flag]
    out = pd.DataFrame({"n": g.size(), "k": g.apply(lambda s: int(s.astype(bool).sum()))}).reset_index()
    out["pct"] = 100 * out["k"] / out["n"]
    out["reliable"] = out["n"] >= min_n
    return out
