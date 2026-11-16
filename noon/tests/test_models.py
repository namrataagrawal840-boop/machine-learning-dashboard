"""Checks on the analysis outputs: no leakage, realistic (not perfect) scores, and decisions that respect the rules."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RES, DATA, SIM = ROOT / "data" / "results", ROOT / "data", ROOT / "data" / "sim"
S = json.loads((RES / "summary.json").read_text())


def _r(name, folder=RES):
    return pd.read_csv(folder / f"{name}.csv", keep_default_na=False, na_values=[""])


def test_no_hidden_columns_published():
    for f in ["orders", "returns", "tickets"]:
        assert not [c for c in _r(f, DATA).columns if c.startswith("_")], f
    for f in ["p1_orders", "p2_returns", "p3_tickets"]:
        cols = _r(f, SIM).columns
        assert "engagement" not in cols and "true_effect" not in cols, f


def test_two_sizes_flag_exists_on_orders_not_only_returns():
    o = _r("orders", DATA)
    r = _r("returns", DATA)
    assert o["multi_size_basket"].sum() > r["ordered_two_sizes"].sum()  # flag lives on orders; not every two-size basket is returned
    conf = r["ordered_two_sizes"].sum() / o["multi_size_basket"].sum()
    assert 0.4 < conf < 0.95


def test_scores_realistic_not_perfect():
    for f in ["p1_cls_metrics", "p2_cls_metrics", "p3_churn_metrics", "cust_churn_metrics"]:
        auc = _r(f)["roc_auc"].max()
        assert 0.6 < auc < 0.9, (f, auc)
    assert 0.7 < S["p3"]["topic_accuracy"] < 0.95


def test_green_tier_respects_fraud_cap_on_unseen_returns():
    p2 = S["p2"]
    assert p2["green_loss_share_all"] <= 0.006
    tiers = _r("p2_tiers").set_index("tier")
    assert tiers.loc["Green", "actual_fail_rate"] < tiers.loc["Amber", "actual_fail_rate"] < tiers.loc["Red", "actual_fail_rate"]


def test_model_chosen_by_cross_validation():
    sel = _r("p2_model_selection")
    assert sel["chosen"].sum() == 1
    assert sel.loc[sel["chosen"], "cv_auc_mean"].iloc[0] == sel["cv_auc_mean"].max()


def test_time_split_cv_uses_only_future_months():
    t = _r("val_p1_time_cv")
    assert set(t["fold"]) == {"2026-06", "2026-07", "2026-08", "2026-09", "2026-10"}


def test_sim_response_models_have_sensible_signs():
    co = _r("sim_response_models").set_index(["model", "feature"])["coef"]
    assert co[("p1_rep", "late")] < 0 and co[("p1_rep", "update sent")] > 0
    assert co[("p2_rep", "QC failed")] < 0 and co[("p2_rep", "instant refund")] > 0
    assert co[("p3_churn", "extra contacts")] > 0 and co[("p3_churn", "CSAT")] < 0


def test_simulator_is_fixed_and_reproducible():
    from noonbi import sim
    d0 = _r("p2_returns", SIM)
    a = sim.run_p2(d0, sim.P2_DEFAULTS)
    b = sim.run_p2(d0, sim.P2_DEFAULTS)
    assert (a["s_rep"] == b["s_rep"]).all()
    lo, hi = a.attrs["bands"]["Bought again within 90 days"]
    assert lo <= hi


def test_live_scoring_uses_observable_features_only():
    from live import engine
    m = engine.live_models()["p2_fail"]
    assert not [f for f in m["features"] if "returner" in f or f.startswith("_")]
    low = engine.score_return({"prior_returns_12m": 0, "account_age_months": 48, "days_since_delivery": 10, "item_value_aed": 100,
                               "photos_uploaded": 1})
    high = engine.score_return({"prior_returns_12m": 6, "account_age_months": 2, "days_since_delivery": 1, "item_value_aed": 900,
                                "photos_uploaded": 0, "return_reason = Changed my mind": 1})
    assert low < high
