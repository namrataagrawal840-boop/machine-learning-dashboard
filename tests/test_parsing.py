"""Unit tests for answer parsing and statistics."""
from __future__ import annotations

import numpy as np
import pandas as pd

from dashboard.components import stats as S
from dashboard.data.mapping import map_columns
from dashboard.data.transforms import parse_multiselect, parse_ordinal, parse_score, parse_time, prepare_consumer
from dashboard.views.p08_association import parse_items


def test_checkout_gap_bands_and_not_applicable():
    s = pd.Series(["No gap.", "Less than 1 hour.", "3–4 hours.", "More than 8 hours.", "Not applicable.", None, "weird"])
    p = parse_ordinal(s, "checkout_gap_hours")
    assert list(p["status"]) == ["valid", "valid", "valid", "valid", "not_applicable", "missing", "unrecognised"]
    assert p.loc[2, "label"] == "3–4 h" and p.loc[2, "rank"] == 3
    assert np.isnan(p.loc[4, "rank"])  # not applicable is never zero


def test_numeric_bins():
    p = parse_ordinal(pd.Series([0, 0.5, 1, 3, 8, 8.5]), "checkout_gap_hours")
    assert list(p["label"].astype(str)) == ["No gap", "< 1 h", "1–2 h", "3–4 h", "7–8 h", "> 8 h"]


def test_detour_does_not_confuse_15_and_5_minutes():
    p = parse_ordinal(pd.Series(["Up to a 15-minute walk.", "Up to a 5-minute walk."]), "maximum_detour")
    assert list(p["label"].astype(str)) == ["≤ 15-min walk", "≤ 5-min walk"]


def test_score_range():
    p = parse_score(pd.Series([0, 10, 11, "8 — likely", None]), 0, 10)
    assert list(p["status"]) == ["valid", "valid", "out_of_range", "valid", "missing"]


def test_multiselect_semicolon_keeps_unanswered_as_missing():
    df = pd.DataFrame({"security_requirements": ["CCTV surveillance.; Tamper-evident luggage seals.", None,
                                                 "Secure, individual luggage compartments."]})
    m, src = parse_multiselect(df, "security_requirements")
    assert "CCTV" in m.columns and "Individual compartments" in m.columns
    assert m.loc[0, "CCTV"] and pd.isna(m.loc[1, "CCTV"])


def test_alias_mapping():
    df = pd.DataFrame({"Q23": [1], "Respondent ID": ["a"], "q15_hotel": [1], "q15_airport": [0]})
    out, ren, miss = map_columns(df, "consumer", ("booking_intent_score", "respondent_id"))
    assert "booking_intent_score" in out and "respondent_id" in out and "q15_hotel" in out


def test_wtp_zero_kept_and_missing_not_zero():
    df = pd.DataFrame({"storage_wtp_inr": [0, "₹150", None, -5], "booking_intent_score": [8, 2, None, 9]})
    d, meta = prepare_consumer(df)
    assert d["storage_wtp_inr__value"].tolist()[:2] == [0.0, 150.0]
    assert np.isnan(d["storage_wtp_inr__value"].iloc[2]) and np.isnan(d["storage_wtp_inr__value"].iloc[3])
    assert d["high_intent"].tolist()[:2] == [True, False] and pd.isna(d["high_intent"].iloc[2])


def test_time_parsing():
    assert parse_time("9 AM") == 9 and parse_time("21:30") == 21.5 and parse_time("11 PM") == 23
    assert np.isnan(parse_time("late"))


def test_cramers_v_bounds():
    x = pd.Series(list("aabbccaabbcc") * 5)
    v, p, n = S.cramers_v(x, x)
    assert 0.9 < v <= 1.0 and n == 60


def test_rule_item_parsing():
    assert parse_items("frozenset({'CCTV', 'Secure, individual compartments'})") == ("CCTV", "Secure, individual compartments")
    assert parse_items("a; b") == ("a", "b")
