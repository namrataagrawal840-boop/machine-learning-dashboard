"""Dashboard-level parsing of survey answers into analysable columns.

These are presentation transforms on an in-memory copy — not data cleaning.
Unrecognised answers are kept and reported, "Not applicable" stays distinct
from missing, and missing values are never turned into zero.

Added columns (only when the source column exists):
  <field>__label   ordered categorical (scale levels, then special answers)
  <field>__rank    ordinal position (NaN for special / missing / unrecognised)
  <field>__value   approximate quantity (band midpoint or the supplied number)
  <field>__status  valid | not_applicable | special | missing | unrecognised | out_of_range
  <field>::<option>  boolean, one per option of a multi-select question
Derived working definitions: high_intent, long_gap, has_problem, search_difficulty,
access_barrier, hotel_sufficient, unmet_need, accepts_partner.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import streamlit as st

from dashboard.config.schema import (ACCESS_BARRIERS, DIFFICULT_SEARCH, MULTI_SELECT, NOMINAL_FIELDS, NUMERIC_FIELDS, ORDINALS,
                                     PARTNER_VENUES, SCORE_SCALES)
from dashboard.data.mapping import find_encoded_columns, normalise_name, to_bool

MS_SEP = "::"
_SPLIT = re.compile(r"\s*[;|]\s*")


def norm_text(v: object) -> str:
    s = str(v).strip().lower()
    s = re.sub(r"[–—−]", "-", s)
    s = re.sub(r"\s+", " ", s)
    return s.rstrip(".;: ").strip()


def _to_number(v: object) -> float:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    s = re.sub(r"(₹|rs\.?|inr|,|\s)", "", str(v).strip().lower())
    try:
        return float(s)
    except ValueError:
        return np.nan


# ------------------------------------------------------------------ ordinal
def parse_ordinal(series: pd.Series, field: str) -> pd.DataFrame:
    spec = ORDINALS[field]
    levels = [lv[0] for lv in spec["levels"]]
    specials = [sp[0] for sp in spec.get("special", [])]
    bins = spec.get("numeric_bins")
    lookup: dict = {}
    for raw in series.dropna().unique():
        t = norm_text(raw)
        res = None
        if t == "" or t in {"nan", "none_given"}:
            res = (None, np.nan, np.nan, "missing")
        for lbl, pats in spec.get("special", []):
            if res is None and any(re.search(p, t) for p in pats):
                res = (lbl, np.nan, np.nan, "not_applicable" if "applicable" in lbl.lower() else "special")
        if res is None:
            for i, (lbl, pats, val) in enumerate(spec["levels"]):
                if any(re.search(p, t) for p in pats):
                    res = (lbl, float(i), val, "valid")
                    break
        if res is None and bins:
            x = _to_number(raw)
            if not np.isnan(x):
                for lo, hi, lbl in bins:
                    if lo <= x < hi:
                        res = (lbl, float(levels.index(lbl)), x, "valid")
                        break
                if res is None:
                    res = (None, np.nan, x, "out_of_range")
        if res is None:
            res = (str(raw).strip(), np.nan, np.nan, "unrecognised")
        lookup[raw] = res
    mapped = series.map(lambda v: lookup.get(v, (None, np.nan, np.nan, "missing")) if pd.notna(v)
                        else (None, np.nan, np.nan, "missing"))
    out = pd.DataFrame(mapped.tolist(), index=series.index, columns=["label", "rank", "value", "status"])
    unrec = sorted({x for x in out.loc[out["status"] == "unrecognised", "label"].dropna()})
    cats = levels + specials + unrec
    out["label"] = pd.Categorical(out["label"], categories=cats, ordered=True)
    return out


def parse_score(series: pd.Series, lo: int, hi: int) -> pd.DataFrame:
    def first_num(v):
        if pd.isna(v):
            return np.nan
        if isinstance(v, (int, float, np.integer, np.floating)):
            return float(v)
        m = re.match(r"\s*(-?\d+(\.\d+)?)", str(v))
        return float(m.group(1)) if m else np.nan
    val = series.map(first_num).astype(float)
    status = np.where(series.isna(), "missing",
                      np.where(val.isna(), "unrecognised",
                               np.where((val < lo) | (val > hi), "out_of_range", "valid")))
    valid = status == "valid"
    v = val.where(valid)
    lbl = v.map(lambda x: f"{int(x)}" if pd.notna(x) and float(x).is_integer() else (f"{x:g}" if pd.notna(x) else None))
    cats = [str(i) for i in range(lo, hi + 1)]
    extra = sorted({x for x in lbl.dropna().unique() if x not in cats})
    return pd.DataFrame({"label": pd.Categorical(lbl, categories=cats + extra, ordered=True),
                         "rank": v, "value": v, "status": status}, index=series.index)


def parse_numeric(series: pd.Series) -> pd.DataFrame:
    val = series.map(_to_number).astype(float)
    status = np.where(series.isna(), "missing", np.where(val.isna(), "unrecognised",
                                                         np.where(val < 0, "out_of_range", "valid")))
    return pd.DataFrame({"value": val.where(status == "valid"), "status": status}, index=series.index)


# ------------------------------------------------------------------ multi-select
def canonical_option(field: str, text: str) -> str:
    t = norm_text(text)
    for lbl, pats in MULTI_SELECT.get(field, []):
        if any(re.search(p, t) for p in pats):
            return lbl
    return str(text).strip().rstrip(".").strip() or "(blank)"


def parse_multiselect(df: pd.DataFrame, field: str, alias_group: str = "consumer") -> tuple[pd.DataFrame, str]:
    """Return (boolean option matrix, source description). Empty matrix when unavailable.

    Rows with no answer at all are NaN in every option column (not False)."""
    if field in df.columns and df[field].notna().any() and not _looks_binary(df[field]):
        answered = df[field].notna() & (df[field].astype(str).str.strip() != "")
        tokens = df[field].where(answered).map(
            lambda v: [] if pd.isna(v) else [canonical_option(field, t) for t in _SPLIT.split(str(v)) if t.strip()])
        options: list[str] = []
        for lst in tokens:
            for o in lst:
                if o not in options:
                    options.append(o)
        order = [lbl for lbl, _ in MULTI_SELECT.get(field, []) if lbl in options] + \
                [o for o in options if o not in {lbl for lbl, _ in MULTI_SELECT.get(field, [])}]
        mat = pd.DataFrame({o: tokens.map(lambda lst, o=o: o in lst) for o in order}, index=df.index)
        mat = mat.astype("boolean")
        if len(order):
            mat.loc[~answered.values] = pd.NA
        return mat, "semicolon-separated answers"
    enc = find_encoded_columns(df, field, alias_group)
    if enc:
        mat = pd.DataFrame(index=df.index)
        for c in enc:
            suffix = re.sub(rf"^({re.escape(field)}|q\d+)_+", "", c)
            lbl = canonical_option(field, suffix.replace("_", " "))
            b = to_bool(df[c])
            mat[lbl] = (mat[lbl] | b) if lbl in mat.columns else b
        any_answer = mat.notna().any(axis=1)
        mat = mat.where(any_answer, other=pd.NA)
        return mat.astype("boolean"), f"{len(enc)} binary-encoded columns"
    return pd.DataFrame(index=df.index), ""


def _looks_binary(s: pd.Series) -> bool:
    vals = {str(v).strip().lower() for v in s.dropna().unique()[:20]}
    return bool(vals) and vals <= {"0", "1", "0.0", "1.0", "true", "false"}


def ms_columns(df: pd.DataFrame, field: str) -> list[str]:
    return [c for c in df.columns if c.startswith(field + MS_SEP)]


def ms_option(col: str) -> str:
    return col.split(MS_SEP, 1)[1]


# ------------------------------------------------------------------ consumer
CONSUMER_ORDINALS = ("checkout_gap_hours", "luggage_management_hours", "bag_count", "problem_frequency", "trips_12m",
                     "group_size", "age_group", "storage_search_difficulty", "maximum_detour", "past_spend_inr",
                     "pilot_interest")
CONSUMER_MULTI = ("existing_storage_methods", "existing_barriers", "acceptable_venue_types", "security_requirements",
                  "preferred_features")


@st.cache_data(show_spinner=False, max_entries=8)
def prepare_consumer(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    d = df.copy()
    meta: dict = {"ordinal": [], "score": [], "numeric": [], "multi": {}, "parse_issues": {}, "definitions": {},
                  "numeric_source": set()}
    new_cols: dict[str, pd.Series] = {}
    for f in CONSUMER_ORDINALS:
        if f in d.columns:
            p = parse_ordinal(d[f], f)
            for c in p.columns:
                new_cols[f"{f}__{c}"] = p[c]
            meta["ordinal"].append(f)
            if pd.api.types.is_numeric_dtype(d[f]):
                meta["numeric_source"].add(f)
    for f, (lo, hi) in SCORE_SCALES.items():
        if f in d.columns:
            p = parse_score(d[f], lo, hi)
            for c in p.columns:
                new_cols[f"{f}__{c}"] = p[c]
            meta["score"].append(f)
    for f in NUMERIC_FIELDS:
        if f in d.columns:
            p = parse_numeric(d[f])
            new_cols[f"{f}__value"] = p["value"]
            new_cols[f"{f}__status"] = p["status"]
            meta["numeric"].append(f)
    for f in NOMINAL_FIELDS:
        if f in d.columns:
            new_cols[f"{f}__label"] = d[f].map(lambda v: str(v).strip().rstrip(".").strip() if pd.notna(v) and
                                               str(v).strip() != "" else None)
    for f in CONSUMER_MULTI:
        mat, src = parse_multiselect(d, f)
        if not mat.empty and mat.shape[1]:
            for c in mat.columns:
                new_cols[f"{f}{MS_SEP}{c}"] = mat[c]
            new_cols[f"{f}__n_selected"] = mat.fillna(False).astype(int).sum(axis=1).where(mat.notna().any(axis=1))
            meta["multi"][f] = src
    d = pd.concat([d, pd.DataFrame(new_cols, index=d.index)], axis=1)

    for f in list(meta["ordinal"]) + list(meta["score"]) + list(meta["numeric"]):
        st_col = d[f"{f}__status"]
        bad = st_col.isin(["unrecognised", "out_of_range"])
        if bad.any():
            meta["parse_issues"][f] = {
                "count": int(bad.sum()),
                "examples": [str(x) for x in d.loc[bad, f].dropna().astype(str).unique()[:5]],
            }

    _derive(d, meta)
    return d, meta


def _derive(d: pd.DataFrame, meta: dict) -> None:
    defs = meta["definitions"]
    # High stated intent
    if "booking_intent_score__value" in d:
        v = d["booking_intent_score__value"]
        d["high_intent"] = (v >= 8).astype("boolean").where(v.notna())
        defs["high_intent"] = "High_Intent = 1 when booking_intent_score (Q23) is 8–10; 0 when 0–7. Stated intent only."
    elif "high_intent" in d:
        d["high_intent"] = to_bool(d["high_intent"])
        defs["high_intent"] = "Taken from the supplied high_intent column."

    # Long checkout gap
    if "checkout_gap_hours__rank" in d:
        lbl = d["checkout_gap_hours__label"].astype(object)
        long_ = d["checkout_gap_hours__rank"] >= 3  # rank 3 == "3–4 h"
        d["long_gap"] = long_.astype("boolean").where(d["checkout_gap_hours__status"] == "valid")
        defs["long_gap"] = ("Checkout-to-departure gap of 3 hours or more (Q11 ≥ '3–4 hours'). "
                            "'Not applicable' and missing answers are excluded from the denominator.")
        del lbl

    # Experiences the problem: long gap AND had to manage luggage personally for ≥ 1 h
    if "long_gap" in d:
        hp = d["long_gap"].copy()
        if "luggage_management_hours__rank" in d:
            managed = (d["luggage_management_hours__rank"] >= 2).astype("boolean").where(
                d["luggage_management_hours__status"] == "valid")
            hp = (d["long_gap"] & managed)
            defs["has_problem"] = ("Gap ≥ 3 h (Q11) AND personally managed luggage for ≥ 1 h after checkout (Q12).")
        else:
            defs["has_problem"] = "Gap ≥ 3 h (Q11). Q12 unavailable, so luggage-management time is not checked."
        d["has_problem"] = hp.astype("boolean")

    # Search difficulty
    if "storage_search_difficulty__label" in d:
        lbl = d["storage_search_difficulty__label"].astype(object)
        known = d["storage_search_difficulty__status"].isin(["valid", "special"])
        d["search_difficulty"] = lbl.isin(DIFFICULT_SEARCH).astype("boolean").where(known & (lbl != "Did not search"))
        defs["search_difficulty"] = ("Q18 answer is 'Difficult', 'Very difficult' or 'Unaware of any facility'. "
                                     "'Did not search' is excluded from the denominator.")

    # Access barriers / hotel sufficiency from Q19
    ms_bar = ms_columns(d, "existing_barriers")
    if ms_bar:
        access = [c for c in ms_bar if ms_option(c) in ACCESS_BARRIERS]
        answered = d[ms_bar].notna().any(axis=1)
        d["access_barrier"] = (d[access].fillna(False).any(axis=1) if access else pd.Series(False, index=d.index)
                               ).astype("boolean").where(answered)
        hs = [c for c in ms_bar if ms_option(c) == "Free hotel storage sufficient"]
        d["hotel_sufficient"] = (d[hs[0]].fillna(False) if hs else pd.Series(False, index=d.index)
                                 ).astype("boolean").where(answered)

    # Unmet need
    if "has_problem" in d and ("search_difficulty" in d or "access_barrier" in d):
        sd = d["search_difficulty"] if "search_difficulty" in d else pd.Series(pd.NA, index=d.index, dtype="boolean")
        ab = d["access_barrier"] if "access_barrier" in d else pd.Series(pd.NA, index=d.index, dtype="boolean")
        lacking = (sd.fillna(False) | ab.fillna(False)).astype("boolean").where(sd.notna() | ab.notna())
        unmet = d["has_problem"] & lacking
        if "hotel_sufficient" in d:
            unmet = unmet & ~d["hotel_sufficient"].fillna(False)
        d["unmet_need"] = unmet.astype("boolean")
        defs["unmet_need"] = ("Experiences the problem (above) AND reports a storage-access gap — Q18 difficult/"
                              "very difficult/unaware OR Q19 cites no facility nearby, too far, hard to locate/book "
                              "or inconvenient hours — AND did NOT say free hotel storage was sufficient.")

    # Accepts partner venues (Q21)
    ms_ven = ms_columns(d, "acceptable_venue_types")
    if ms_ven:
        partner_cols = [c for c in ms_ven if ms_option(c) in PARTNER_VENUES]
        answered = d[ms_ven].notna().any(axis=1)
        d["accepts_partner"] = (d[partner_cols].fillna(False).any(axis=1) if partner_cols
                                else pd.Series(False, index=d.index)).astype("boolean").where(answered)
        defs["accepts_partner"] = ("Q21 includes at least one partner-based venue: verified hotels, restaurants/cafés, "
                                   "retail stores, or any registered business meeting security requirements.")


# ------------------------------------------------------------------ partner / market
def parse_time(v: object) -> float:
    """Return hour of day as float (e.g. 21.5) or NaN."""
    if pd.isna(v):
        return np.nan
    if isinstance(v, (int, float, np.integer, np.floating)):
        x = float(v)
        return x if 0 <= x <= 24 else np.nan
    t = str(v).strip().lower().replace(".", ":")
    if t in {"24 hours", "24h", "24/7", "24x7", "open 24 hours"}:
        return np.nan
    m = re.match(r"^(\d{1,2})(?::(\d{2}))?\s*(am|pm)?$", t)
    if not m:
        return np.nan
    h, mnt, ap = int(m.group(1)), int(m.group(2) or 0), m.group(3)
    if ap == "pm" and h < 12:
        h += 12
    if ap == "am" and h == 12:
        h = 0
    return h + mnt / 60 if 0 <= h <= 24 else np.nan


def evidence_status(s: pd.Series | None, index) -> pd.Series:
    """Map supplied evidence/verification values to the five allowed statuses.

    Records without a recognisable status are 'Needs verification' — never assumed verified."""
    if s is None:
        return pd.Series("Needs verification", index=index)

    def m(v):
        if pd.isna(v):
            return "Needs verification"
        t = norm_text(v)
        if t in {"true", "1", "yes", "y"} or t.startswith("verified"):
            return "Verified"
        if "company" in t:
            return "Company-reported"
        if "third" in t:
            return "Third-party reported"
        if "inactive" in t or "unavailable" in t or "closed" in t:
            return "Inactive / unavailable"
        return "Needs verification"
    return s.map(m)


@st.cache_data(show_spinner=False, max_entries=8)
def prepare_partner(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    d = df.copy()
    meta: dict = {"ordinal": [], "multi": {}}
    new_cols: dict[str, pd.Series] = {}
    for f in ("partner_willingness", "digital_handover_acceptance", "pickup_delivery_acceptance", "pilot_participation",
              "secure_storage_available"):
        if f in d.columns:
            p = parse_ordinal(d[f], f)
            for c in p.columns:
                new_cols[f"{f}__{c}"] = p[c]
            meta["ordinal"].append(f)
    for f in ("security_features", "partner_barriers"):
        mat, src = parse_multiselect(d, f, "partner")
        if not mat.empty and mat.shape[1]:
            for c in mat.columns:
                new_cols[f"{f}{MS_SEP}{c}"] = mat[c]
            meta["multi"][f] = src
    if "storage_capacity" in d:
        cap = d["storage_capacity"].map(_to_number).astype(float)
        new_cols["storage_capacity__value"] = cap.where(cap >= 0)
    if "expected_payment_inr" in d:
        pay = d["expected_payment_inr"].map(_to_number).astype(float)
        new_cols["expected_payment_inr__value"] = pay.where(pay >= 0)
    for c in ("opening_time", "closing_time"):
        if c in d:
            new_cols[f"{c}__hour"] = d[c].map(parse_time).astype(float)
    if "cctv_available" in d:
        new_cols["cctv_available__bool"] = to_bool(d["cctv_available"])
    for c in ("latitude", "longitude"):
        if c in d:
            new_cols[f"{c}__value"] = pd.to_numeric(d[c], errors="coerce")
    d = pd.concat([d, pd.DataFrame(new_cols, index=d.index)], axis=1)
    d["evidence__status"] = evidence_status(d.get("evidence_status"), d.index)
    d["is_verified"] = d["evidence__status"] == "Verified"
    if "partner_willingness__rank" in d:
        d["is_willing"] = (d["partner_willingness__rank"] >= 2).astype("boolean").where(
            d["partner_willingness__status"] == "valid")
        meta["willing_definition"] = ("Willing = P8 'Definitely yes' or 'Yes, subject to security procedures'. "
                                      "'Maybe, depending on compensation' is counted as not yet willing.")
    if "cctv_available__bool" not in d and f"security_features{MS_SEP}CCTV" in d:
        d["cctv_available__bool"] = d[f"security_features{MS_SEP}CCTV"]
    return d, meta


def valid_coords(d: pd.DataFrame) -> pd.Series:
    if "latitude" not in d or "longitude" not in d:
        return pd.Series(False, index=d.index)
    lat = pd.to_numeric(d["latitude"], errors="coerce")
    lon = pd.to_numeric(d["longitude"], errors="coerce")
    return lat.between(-90, 90) & lon.between(-180, 180) & ~((lat == 0) & (lon == 0))


COMPETITOR_CATEGORIES = [
    ("Official airport / railway storage", [r"official", r"airport", r"railway", r"cloak", r"irctc", r"station"]),
    ("Smart locker operator", [r"locker"]),
    ("Aggregator network", [r"aggregat", r"marketplace", r"network", r"platform"]),
    ("Prospective business partner", [r"prospect", r"partner"]),
    ("Other short-term storage", [r"storage", r"left luggage", r"baggage"]),
]


@st.cache_data(show_spinner=False, max_entries=8)
def prepare_competitors(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["evidence__status"] = evidence_status(d.get("evidence_status"), d.index)

    def cat(row):
        for col in ("competitor_category", "service_type"):
            v = row.get(col)
            if pd.notna(v):
                t = norm_text(v)
                for lbl, pats in COMPETITOR_CATEGORIES:
                    if any(re.search(p, t) for p in pats):
                        return lbl
        return "Unclassified"
    d["category__label"] = d.apply(cat, axis=1) if len(d) else pd.Series(dtype=object)
    for c in ("hourly_price", "daily_price"):
        if c in d:
            d[f"{c}__value"] = d[c].map(_to_number).astype(float)
    for c in ("pickup_available", "delivery_available"):
        if c in d:
            d[f"{c}__bool"] = to_bool(d[c])
    d["has_coords"] = valid_coords(d)
    return d


@st.cache_data(show_spinner=False, max_entries=8)
def prepare_geo(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["evidence__status"] = evidence_status(d.get("evidence_status"), d.index)
    d["has_coords"] = valid_coords(d)
    if "nearest_storage_distance_m" in d:
        d["nearest_storage_distance_m__value"] = pd.to_numeric(d["nearest_storage_distance_m"], errors="coerce")
    return d


def slug(s: str) -> str:
    return normalise_name(s)
