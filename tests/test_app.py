"""End-to-end tests: every page renders without exceptions across data scenarios.

Run with:  python -m pytest -q tests
Uses throwaway fixtures written to a temporary folder (see tests/fixtures.py).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from tests import fixtures as fx

APP = str(Path(__file__).resolve().parents[1] / "streamlit_app.py")


def run_all_pages(data_dir: Path, state: dict | None = None, notices: bool = True) -> dict[str, list[str]]:
    """Return {page: [problems]} — uncaught exceptions, plus error notices when notices=True."""
    os.environ["LUGGAGE_DATA_DIR"] = str(data_dir)
    at = AppTest.from_file(APP, default_timeout=240)
    for k, v in (state or {}).items():
        at.session_state[k] = v
    at.run()
    problems: dict[str, list[str]] = {}
    for h, meta in at._registered_pages.items():
        at._page_hash = h
        at.run()
        errs = [str(e.value) for e in at.exception]
        if notices:
            errs += [m.value for m in at.markdown if "lc-notice error" in m.value and "leakage" not in m.value.lower()]
        if errs:
            problems[meta["page_name"]] = errs
    return problems


@pytest.fixture()
def full(tmp_path):
    d = fx.write_all(tmp_path / "data")
    fx.competitors().to_csv(d / "competitor_locations.csv", index=False)
    fx.geo().to_csv(d / "geographic_market_audit.csv", index=False)
    fx.partners().to_csv(d / "partner_audit.csv", index=False)
    fx.booking_funnel().to_csv(d / "booking_funnel.csv", index=False)
    return d


def test_no_data(tmp_path):
    (tmp_path / "data").mkdir()
    assert run_all_pages(tmp_path / "data") == {}


def test_full_data(full):
    from dashboard.views.p03_diagnostics import DIAG
    assert run_all_pages(full, {"diag_sel": [f"{q} · {t}" for q, t in DIAG]}) == {}


def test_consumer_only(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    fx.consumer().to_csv(d / "consumer_survey_clean.csv", index=False)
    assert run_all_pages(d) == {}


def test_missing_columns_and_aliases(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    c = fx.consumer().drop(columns=["luggage_management_hours", "existing_barriers", "storage_wtp_inr",
                                    "acceptable_venue_types", "city"])
    c = c.rename(columns={"checkout_gap_hours": "Q11", "booking_intent_score": "Booking Intent"})
    c.to_csv(d / "consumer_survey_clean.csv", index=False)
    assert run_all_pages(d) == {}


def test_binary_encoded_and_numeric(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    c = fx.consumer()
    rng = np.random.default_rng(1)
    for opt in ["cctv", "tamper_evident_seals", "photos"]:
        c[f"security_requirements__{opt}"] = rng.integers(0, 2, len(c))
    c = c.drop(columns=["security_requirements"])
    c["checkout_gap_hours"] = rng.choice([0, 0.5, 2, 3, 4.5, 6, 9, np.nan], len(c))
    c["bag_count"] = rng.integers(1, 6, len(c))
    c.loc[0, "booking_intent_score"] = 14  # out of range
    c.loc[1, "travel_mode"] = "train"  # case variant
    c.to_csv(d / "consumer_survey_clean.csv", index=False)
    assert run_all_pages(d) == {}


def test_empty_and_malformed_files(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    (d / "consumer_survey_clean.csv").write_text("")
    (d / "association_rules.csv").write_text("foo,bar\n1,2\n")
    (d / "classification_predictions.csv").write_text("model,actual,predicted\nA,yes,no\nA,maybe,1\n")
    assert run_all_pages(d, notices=False) == {}  # no crashes
    problems = run_all_pages(d)
    assert "Association rules could not be used" in " ".join(problems.get("8 · Association Mining", []))


def test_tight_filters_and_small_samples(full):
    state = {"flt_city": ["Goa"], "flt_microzone": ["Panaji"], "flt_trips_12m": ["7+"], "set_min_n": 200}
    assert run_all_pages(full, state) == {}


def test_filter_to_nothing(full):
    state = {"flt_city": ["Goa"], "flt_microzone": ["Bandra"]}  # stale microzone is dropped, Goa remains
    assert run_all_pages(full, state) == {}


def test_source_files_unchanged(full):
    before = {p.name: p.read_bytes() for p in full.iterdir()}
    run_all_pages(full)
    after = {p.name: p.read_bytes() for p in full.iterdir()}
    assert before == after


def test_repository_data_renders_cleanly():
    """The committed synthetic datasets and model outputs render on every page with no errors or warnings of
    unrecognised answers."""
    repo_data = Path(APP).parent / "data"
    if not (repo_data / "consumer_survey_clean.csv").exists():
        pytest.skip("no committed data")
    from dashboard.views.p03_diagnostics import DIAG
    assert run_all_pages(repo_data, {"diag_sel": [f"{q} · {t}" for q, t in DIAG]}) == {}
    from dashboard.data.transforms import prepare_consumer
    _, meta = prepare_consumer(pd.read_csv(repo_data / "consumer_survey_clean.csv"))
    assert meta["parse_issues"] == {}
