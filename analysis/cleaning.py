"""Cleaning & transformation of the raw consumer survey export.

Every decision is recorded in data_cleaning_log.csv so the dashboard's Data Quality page can show it.
Principles: keep legitimate extremes (flag, don't delete); keep zero WTP; keep 'Not applicable' distinct from
missing; never impute stated answers.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from analysis import synthetic as syn

CANON = {
    "travel_mode": syn.MODE, "accommodation_type": syn.ACCOM, "travel_purpose": syn.PURPOSE,
    "city": list(syn.MICROZONES),
}


def _norm(v) -> str:
    return re.sub(r"\s+", " ", str(v).strip().lower()).rstrip(".")


def _money(v):
    if pd.isna(v):
        return np.nan
    s = re.sub(r"(₹|rs\.?|inr|,|\s)", "", str(v).lower())
    try:
        return float(s)
    except ValueError:
        return np.nan


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    log = []
    d = raw.copy()

    def note(step, column, action, rows, reason):
        log.append({"step": len(log) + 1, "column": column, "action": action, "rows_affected": int(rows),
                    "reason": reason})

    n0 = len(d)
    d = d.drop_duplicates()
    d = d.drop_duplicates(subset="respondent_id", keep="first")
    note(1, "all", "Removed exact duplicate submissions", n0 - len(d),
         "Repeated submissions would double-count respondents.")

    mask = d["travel_status"].astype(str).str.startswith("Neither")
    note(2, "travel_status", "Removed screened-out respondents (Q1 = Neither)", mask.sum(),
         "Outside the target population; the survey should have ended at Q1.")
    d = d[~mask]

    for col, options in CANON.items():
        lookup = {_norm(o): o for o in options}
        before = d[col].copy()
        d[col] = d[col].map(lambda v: lookup.get(_norm(v), v) if pd.notna(v) else v)
        note(3, col, "Standardised case, spacing and punctuation to the questionnaire wording",
             (before.astype(str) != d[col].astype(str)).sum(), "Same answer written differently.")

    for col in ("storage_wtp_inr", "pickup_delivery_wtp_inr"):
        before = d[col].copy()
        d[col] = d[col].map(_money)
        changed = before.notna() & (before.astype(str) != d[col].astype(str)) & d[col].notna()
        note(4, col, "Converted text amounts (e.g. '₹200', '200 rs') to numbers", changed.sum(),
             "Numeric analysis needs a consistent number in rupees.")
        neg = d[col] < 0
        note(5, col, "Set negative amounts to missing", neg.sum(), "A negative WTP is impossible; not imputed.")
        d.loc[neg, col] = np.nan

    s = pd.to_numeric(d["booking_intent_score"], errors="coerce")
    bad = s.notna() & ~s.between(0, 10)
    note(6, "booking_intent_score", "Set scores outside 0–10 to missing", bad.sum(), "Outside the Q23 scale.")
    d["booking_intent_score"] = s.where(~bad)

    incomplete = d["booking_intent_score"].isna() & d["first_service_preference"].isna() & d["pilot_interest"].isna()
    note(7, "Q23–Q30", "Removed responses abandoned before Section D", incomplete.sum(),
         "No outcome answers at all; they cannot contribute to demand or pricing analysis.")
    d = d[~incomplete]

    na = (d["past_trip_experience"] == "No — planning only").sum()
    note(8, "Q11–Q19", "Kept planning-only respondents; past-trip answers left as 'Not applicable' / blank",
         na, "Distinguishes 'not applicable' from missing; flagged by past_trip_experience.")

    for col in ("storage_wtp_inr", "pickup_delivery_wtp_inr"):
        v = d[col]
        q1, q3 = v.quantile([.25, .75])
        hi = q3 + 3 * (q3 - q1)
        flag = v > hi
        d[f"{col.replace('_inr', '')}_extreme_flag"] = flag.astype(int)
        note(9, col, f"Flagged (not removed) values above Q3 + 3×IQR (₹{hi:,.0f})", flag.sum(),
             "Legitimate extreme answers are kept; the flag lets analyses test sensitivity.")
        note(10, col, "Kept zero WTP as a valid answer", (v == 0).sum(), "₹0 means 'would not pay', not missing.")

    d["high_intent"] = np.where(d["booking_intent_score"].isna(), np.nan, (d["booking_intent_score"] >= 8).astype(float))
    note(11, "high_intent", "Derived High_Intent = 1 if Q23 is 8–10, else 0", d["high_intent"].notna().sum(),
         "Classification target (stated intent, not conversion).")
    d["data_source"] = "synthetic"
    note(12, "data_source", "Labelled every record as synthetic", len(d), "Provenance must travel with the data.")
    d = d.sort_values("respondent_id").reset_index(drop=True)
    return d, pd.DataFrame(log)
