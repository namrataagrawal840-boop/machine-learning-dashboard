"""Microzone-level market calculations shared by the Geographic, Partner and Strategy pages."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from dashboard.data.filters import FILTER_PREFIX
from dashboard.data.transforms import norm_text

COMPONENTS = [
    ("demand", "Demand", "High stated booking intent (% of respondents scoring 8–10)"),
    ("unmet", "Unmet need", "Unmet storage need (% — working definition)"),
    ("acceptance", "Customer acceptance", "Accept partner-based venues (% of Q21 answers)"),
    ("readiness", "Partner readiness", "Assessed partners willing to store non-customers (% of assessed in zone)"),
    ("feasibility", "Operational feasibility",
     "Mean of: willing partners open across the service window, with secure space, accepting digital handover (%)"),
]
DEFAULT_WEIGHTS = {"demand": 25, "unmet": 25, "acceptance": 20, "readiness": 15, "feasibility": 15}


def _key(v) -> str:
    return norm_text(v) if pd.notna(v) else ""


def filter_market(ctx, df: pd.DataFrame | None) -> pd.DataFrame:
    """Apply the global city / microzone filter to partner, competitor or geo records."""
    if df is None:
        return pd.DataFrame()
    out = df
    for fkey, col, alt in (("city", "city", None), ("microzone", "microzone", "locality")):
        sel = st.session_state.get(FILTER_PREFIX + fkey) or []
        c = col if col in out.columns else (alt if alt and alt in out.columns else None)
        if sel and c:
            wanted = {_key(s) for s in sel}
            out = out[out[c].map(_key).isin(wanted)]
    return out


def partner_frames(ctx) -> list[tuple[str, pd.DataFrame, str]]:
    frames = []
    if ctx.partner_audit is not None:
        frames.append(("Partner audit", ctx.partner_audit, ctx.r("partner_audit").provenance))
    if ctx.partner_survey is not None:
        frames.append(("Partner survey", ctx.partner_survey, ctx.r("partner_survey").provenance))
    return frames


def primary_partner_frame(ctx) -> tuple[str, pd.DataFrame | None]:
    """Audit is preferred (field-assessed); otherwise the partner survey."""
    if ctx.partner_audit is not None:
        return "Partner audit", ctx.partner_audit
    if ctx.partner_survey is not None:
        return "Partner survey", ctx.partner_survey
    return "", None


def zone_column(df: pd.DataFrame) -> str | None:
    for c in ("microzone__label", "microzone", "locality__label", "locality"):
        if c in df.columns and df[c].notna().any():
            return c
    return None


def _pct(s: pd.Series) -> float:
    s = s.dropna()
    return float(s.astype(bool).mean() * 100) if len(s) else np.nan


def consumer_zone_table(cf: pd.DataFrame) -> pd.DataFrame:
    zc = zone_column(cf)
    cc = "city__label" if "city__label" in cf else None
    if zc is None or cc is None:
        return pd.DataFrame()
    d = cf.dropna(subset=[cc, zc])
    rows = []
    for (city, zone), g in d.groupby([cc, zc], observed=True):
        row = {"city": city, "microzone": zone, "respondents": len(g)}
        for col, name in (("long_gap", "long_gap_pct"), ("unmet_need", "unmet_pct"), ("high_intent", "high_intent_pct"),
                          ("accepts_partner", "acceptance_pct")):
            row[name] = _pct(g[col]) if col in g else np.nan
            row[name.replace("_pct", "_n")] = int(g[col].notna().sum()) if col in g else 0
        row["median_storage_wtp"] = float(g["storage_wtp_inr__value"].median()) if "storage_wtp_inr__value" in g else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def partner_zone_table(pf: pd.DataFrame | None, window: tuple[float, float]) -> pd.DataFrame:
    if pf is None or pf.empty or "city" not in pf or zone_column(pf) is None:
        return pd.DataFrame()
    zc = zone_column(pf)
    rows = []
    for (city, zone), g in pf.dropna(subset=["city", zc]).groupby(["city", zc]):
        willing = g[g["is_willing"].fillna(False).astype(bool)] if "is_willing" in g else g.iloc[0:0]
        row = {"city_key": _key(city), "zone_key": _key(zone), "partners_assessed": len(g),
               "partners_willing": len(willing) if "is_willing" in g else np.nan,
               "verified_partners": int((g["is_verified"] & g.get("is_willing", True)).fillna(False).sum())
               if "evidence_status" in g else np.nan,
               "readiness_pct": _pct(g["is_willing"]) if "is_willing" in g else np.nan,
               "capacity_willing": float(willing["storage_capacity__value"].sum(min_count=1))
               if "storage_capacity__value" in g and len(willing) else np.nan}
        feas = []
        if {"opening_time__hour", "closing_time__hour"} <= set(g.columns) and len(willing):
            w = willing.dropna(subset=["opening_time__hour", "closing_time__hour"])
            if len(w):
                ok = (w["opening_time__hour"] <= window[0]) & (w["closing_time__hour"] >= window[1])
                row["hours_compatible_pct"] = float(ok.mean() * 100)
                feas.append(row["hours_compatible_pct"])
        if "secure_storage_available__rank" in g and len(willing):
            s = willing["secure_storage_available__rank"].dropna()
            if len(s):
                row["secure_space_pct"] = float((s >= 2).mean() * 100)
                feas.append(row["secure_space_pct"])
        if "digital_handover_acceptance__rank" in g and len(willing):
            s = willing["digital_handover_acceptance__rank"].dropna()
            if len(s):
                row["digital_handover_pct"] = float((s >= 2).mean() * 100)
                feas.append(row["digital_handover_pct"])
        row["feasibility_pct"] = float(np.mean(feas)) if feas else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def competitor_zone_table(comp: pd.DataFrame | None) -> pd.DataFrame:
    if comp is None or comp.empty or "city" not in comp or zone_column(comp) is None:
        return pd.DataFrame()
    zc = zone_column(comp)
    g = comp.dropna(subset=["city", zc]).groupby([comp["city"].map(_key), comp[zc].map(_key)])
    out = pd.DataFrame({"competitor_records": g.size(),
                        "competitors_verified": g["evidence__status"].apply(lambda s: int((s == "Verified").sum()))})
    out.index.names = ["city_key", "zone_key"]
    return out.reset_index()


def geo_zone_table(geo: pd.DataFrame | None) -> pd.DataFrame:
    if geo is None or geo.empty or "nearest_storage_option" not in geo or "city" not in geo or zone_column(geo) is None:
        return pd.DataFrame()
    zc = zone_column(geo)
    d = geo.dropna(subset=["nearest_storage_option"]).copy()
    d["city_key"], d["zone_key"] = d["city"].map(_key), d[zc].map(_key)
    if "nearest_storage_distance_m__value" in d:
        d = d.sort_values("nearest_storage_distance_m__value")
    d = d.drop_duplicates(["city_key", "zone_key"])
    out = d[["city_key", "zone_key", "nearest_storage_option"]].copy()
    if "nearest_storage_distance_m__value" in d:
        out["nearest_storage_distance_m"] = d["nearest_storage_distance_m__value"]
        out["distance_type"] = d["distance_type"] if "distance_type" in d else "not stated"
    out["nearest_storage_evidence"] = d["evidence__status"]
    return out


def zone_table(ctx, window: tuple[float, float] = (11.0, 17.0)) -> tuple[pd.DataFrame, dict]:
    """Combined microzone table + availability flags."""
    info = {"partner_source": "", "has_partner": False, "has_competitor": ctx.competitors is not None,
            "has_geo": ctx.geo is not None}
    cz = consumer_zone_table(ctx.cf) if ctx.cf is not None else pd.DataFrame()
    if cz.empty:
        return cz, info
    cz["city_key"], cz["zone_key"] = cz["city"].map(_key), cz["microzone"].map(_key)
    src, pf = primary_partner_frame(ctx)
    pz = partner_zone_table(filter_market(ctx, pf) if pf is not None else None, window)
    if not pz.empty:
        info["has_partner"], info["partner_source"] = True, src
        cz = cz.merge(pz, on=["city_key", "zone_key"], how="left")
    comp = competitor_zone_table(filter_market(ctx, ctx.competitors) if ctx.competitors is not None else None)
    if not comp.empty:
        cz = cz.merge(comp, on=["city_key", "zone_key"], how="left")
        cz[["competitor_records", "competitors_verified"]] = cz[["competitor_records", "competitors_verified"]].fillna(0)
    elif ctx.competitors is not None:
        cz["competitor_records"] = 0
        cz["competitors_verified"] = 0
    gz = geo_zone_table(ctx.geo)
    if not gz.empty:
        cz = cz.merge(gz, on=["city_key", "zone_key"], how="left")
    return cz, info


def score(zt: pd.DataFrame, weights: dict, scaling: str, min_n: int, allow_partial: bool) -> pd.DataFrame:
    z = zt.copy()
    src = {"demand": "high_intent_pct", "unmet": "unmet_pct", "acceptance": "acceptance_pct",
           "readiness": "readiness_pct", "feasibility": "feasibility_pct"}
    reliable = z["respondents"] >= min_n
    for comp, col in src.items():
        v = z[col].astype(float) if col in z else pd.Series(np.nan, index=z.index)
        if scaling == "relative":
            ref = v[reliable].dropna()
            lo, hi = (ref.min(), ref.max()) if len(ref) else (np.nan, np.nan)
            v = (v - lo) / (hi - lo) * 100 if pd.notna(lo) and hi > lo else v * np.nan
            v = v.clip(0, 100)
        z[f"score_{comp}"] = v
    sc = [f"score_{c}" for c in src]
    w = np.array([weights.get(c, 0) for c in src], dtype=float)
    vals = z[sc].to_numpy(dtype=float)
    have = ~np.isnan(vals)
    complete = have[:, w > 0].all(axis=1) if (w > 0).any() else np.zeros(len(z), bool)
    wsum_avail = (have * w).sum(axis=1)
    weighted = np.nansum(np.nan_to_num(vals) * w, axis=1)
    combined = np.where(complete, weighted / max(w.sum(), 1e-9), np.nan)
    if allow_partial:
        combined = np.where(~complete & (wsum_avail > 0), weighted / np.where(wsum_avail > 0, wsum_avail, 1), combined)
    z["opportunity_score"] = combined
    z["score_basis"] = np.where(complete, "all components", np.where(allow_partial & (wsum_avail > 0),
                                                                      "partial (re-weighted)", "incomplete"))
    z["weakest_component"] = [
        (min(((c, r[f"score_{c}"]) for c in src if pd.notna(r[f"score_{c}"]) and weights.get(c, 0) > 0),
             key=lambda t: t[1], default=(None, None))[0] or "—") for _, r in z.iterrows()]
    z["evidence_status"] = [evidence_label(r, min_n) for _, r in z.iterrows()]
    return z


def evidence_label(r: pd.Series, min_n: int) -> str:
    if r["respondents"] < min_n:
        return f"Insufficient consumer sample (n < {min_n})"
    if "competitor_records" not in r.index:
        return "Competitor evidence missing"
    if pd.isna(r.get("partners_assessed", np.nan)):
        return "Partner evidence missing"
    if r.get("competitors_verified", 0) == 0 and r.get("competitor_records", 0) > 0:
        return "Competitor records not yet verified"
    return "Comparable — needs field verification"
