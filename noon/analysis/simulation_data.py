"""Records for the solution simulator ("run the simulation" with fixed numbers).

For each problem we take records the models never trained on (the test months / test split) and store
everything needed to replay them twice: once under today's process and once with the solution switched on.

The same random number (`u`) decides each customer's outcome in both runs ("common random numbers"),
so the only thing that differs between the two runs is the process. Customer responses come from models
estimated on the recorded data (analysis/outcome_models.py); hidden generator variables are NOT stored here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from analysis.models import RS, _design

SIM_SEED = 777


def problem1_records(orders: pd.DataFrame, te: pd.DataFrame, customers: pd.DataFrame, cust: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SIM_SEED)
    o = orders.sort_values("order_date").copy()
    o["prior_orders"] = o.groupby("customer_id").cumcount()
    # uplift model trained on the Apr–Jun randomised test, scored on the unseen Aug–Oct orders
    tst = o[o["recovery_test_group"] != "Not in test"].copy()
    feats = ["membership", "emirate", "category"]
    nums = ["tenure_months", "order_value_aed", "days_late", "prior_orders"]
    X = _design(tst, feats, nums)
    mt = RandomForestClassifier(300, min_samples_leaf=25, random_state=RS, n_jobs=4)
    mc = RandomForestClassifier(300, min_samples_leaf=25, random_state=RS, n_jobs=4)
    tr = tst["recovery_test_group"] == "Treatment"
    mt.fit(X[tr], tst.loc[tr, "repurchase_60d"])
    mc.fit(X[~tr], tst.loc[~tr, "repurchase_60d"])
    s = te.merge(o[["order_id", "prior_orders", "membership", "tenure_months"]], on="order_id", how="left", suffixes=("_x", ""))
    Xs = _design(s, feats, nums).reindex(columns=X.columns, fill_value=0)
    s["pred_uplift"] = mt.predict_proba(Xs)[:, 1] - mc.predict_proba(Xs)[:, 1]
    s["clv"] = s["customer_id"].map(cust.set_index("customer_id")["clv_next12m_aed"])
    s["u"] = rng.random(len(s))
    keep = ["order_id", "order_date", "customer_id", "emirate", "zone", "category", "item_size", "seller_id", "courier",
            "fulfilment", "sale_event", "address_quality", "order_value_aed", "promised_days", "actual_hours", "actual_days",
            "late", "days_late", "proactive_update_sent", "failed_first_attempt", "unavailable_gps_verified", "p_late",
            "risk_tier", "p90_promise_days", "pred_uplift", "membership", "tenure_months", "prior_orders", "clv", "u"]
    return s[keep].sort_values(["order_date", "order_id"]).reset_index(drop=True)


def problem2_records(returns: pd.DataFrame, scores: pd.DataFrame, customers: pd.DataFrame, cust: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SIM_SEED + 1)
    r = returns.merge(scores[["return_id", "p_fail"]], on="return_id", how="inner")
    r = r[r["instant_refund_test_group"] != "Instant refund"].copy()  # keep today's real refund times
    r["clv"] = r["customer_id"].map(cust.set_index("customer_id")["clv_next12m_aed"])
    r["u"] = rng.random(len(r))
    r["u_ticket"] = rng.random(len(r))
    keep = ["return_id", "order_date", "customer_id", "emirate", "category", "item_size", "return_reason", "item_value_aed",
            "prior_returns_12m", "account_age_months", "photos_uploaded", "refund_method", "fulfilment", "refund_days",
            "qc_passed", "return_cost_aed", "p_fail", "membership", "clv", "u", "u_ticket"]
    return r[keep].sort_values(["order_date", "return_id"]).reset_index(drop=True)


def problem3_records(tickets: pd.DataFrame, pred: pd.DataFrame, customers: pd.DataFrame, cust: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SIM_SEED + 2)
    t = tickets.merge(pred, on="ticket_id", how="inner").copy()
    good_fix = {"Delivery delay": "Expedite", "Failed delivery": "Re-attempt", "Refund delay": "Wallet refund now",
                "Refund rejected": "Re-inspect", "Warranty / seller": "Replace item", "Wrong / damaged item": "Replace item",
                "Payment / billing": "Reverse charge"}
    t["best_fix"] = t["topic"].map(good_fix)
    t["used_best_fix"] = (t["fix_applied"] == t["best_fix"]).astype(int)
    t["routed_correctly"] = (t["predicted_topic"] == t["topic"]).astype(int)
    t["clv"] = t["customer_id"].map(cust.set_index("customer_id")["clv_next12m_aed"])
    t["csat_missing"] = t["csat"].isna().astype(int)
    t["csat"] = t["csat"].fillna(t["csat"].median())
    for k in ("u", "u_reopen", "u_recontact"):
        t[k] = rng.random(len(t))
    keep = ["ticket_id", "created_date", "customer_id", "emirate", "channel", "topic", "predicted_topic", "routed_correctly",
            "text", "angry", "contacts", "resolution_hours", "reopened", "template_reply", "csat", "fix_applied", "best_fix",
            "used_best_fix", "recontact_7d", "order_value_aed", "tenure_months", "membership", "csat_missing", "clv",
            "churn_90d", "u", "u_reopen", "u_recontact"]
    return t[keep].sort_values(["created_date", "ticket_id"]).reset_index(drop=True)
