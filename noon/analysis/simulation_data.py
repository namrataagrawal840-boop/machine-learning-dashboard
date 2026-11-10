"""Records for the solution simulator ("run the simulation" with fixed numbers).

For each problem we take records the models never trained on (the test months / test split) and store
everything needed to replay them twice: once under today's process and once with the solution switched on.

The same random number (`u`) decides each customer's outcome in both runs ("common random numbers"),
so the only thing that differs between the two runs is the process. The behaviour formulas are the same
ones used to generate the synthetic data (analysis/synthetic.py), so the simulator is consistent with it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from analysis.models import RS, _design

SIM_SEED = 777


def _persuade_effect(orders: pd.DataFrame, eng: pd.Series) -> pd.Series:
    """True (hidden) effect of a recovery credit, identical to the generator's rule."""
    e = orders["customer_id"].map(eng).to_numpy()
    first_late = (~orders.duplicated(["customer_id", "late"]) & (orders["late"] == 1)).to_numpy()
    pers = (np.abs(e - 0.1) < 0.75) & (orders["membership"].eq("Non-member").to_numpy() | first_late)
    return pd.Series(np.select([pers, e > 0.85, e < -1.0], [0.15, 0.005, 0.0], 0.02), index=orders.index)


def problem1_records(orders: pd.DataFrame, te: pd.DataFrame, customers: pd.DataFrame, cust: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SIM_SEED)
    eng = customers.set_index("customer_id")["_engagement"]
    o = orders.sort_values("order_date").copy()
    o["true_effect"] = _persuade_effect(o, eng)
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
    s = te.merge(o[["order_id", "true_effect", "prior_orders"]], on="order_id", how="left")
    Xs = _design(s, feats, nums).reindex(columns=X.columns, fill_value=0)
    s["pred_uplift"] = mt.predict_proba(Xs)[:, 1] - mc.predict_proba(Xs)[:, 1]
    s["engagement"] = s["customer_id"].map(eng)
    s["clv"] = s["customer_id"].map(cust.set_index("customer_id")["clv_next12m_aed"])
    s["u"] = rng.random(len(s))
    keep = ["order_id", "order_date", "customer_id", "emirate", "zone", "category", "item_size", "seller_id", "courier",
            "fulfilment", "sale_event", "address_quality", "order_value_aed", "promised_days", "actual_hours", "actual_days",
            "late", "days_late", "proactive_update_sent", "failed_first_attempt", "unavailable_gps_verified", "p_late",
            "risk_tier", "p90_promise_days", "pred_uplift", "true_effect", "engagement", "clv", "u"]
    return s[keep].sort_values(["order_date", "order_id"]).reset_index(drop=True)


def problem2_records(returns: pd.DataFrame, scores: pd.DataFrame, customers: pd.DataFrame, cust: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SIM_SEED + 1)
    eng = customers.set_index("customer_id")["_engagement"]
    r = returns.merge(scores[["return_id", "p_fail"]], on="return_id", how="inner")
    r = r[r["instant_refund_test_group"] != "Instant refund"].copy()  # keep today's real refund times
    r["engagement"] = r["customer_id"].map(eng)
    r["clv"] = r["customer_id"].map(cust.set_index("customer_id")["clv_next12m_aed"])
    r["u"] = rng.random(len(r))
    r["u_ticket"] = rng.random(len(r))
    keep = ["return_id", "order_date", "customer_id", "emirate", "category", "item_size", "return_reason", "item_value_aed",
            "prior_returns_12m", "account_age_months", "photos_uploaded", "refund_method", "fulfilment", "refund_days",
            "qc_passed", "return_cost_aed", "p_fail", "engagement", "clv", "u", "u_ticket"]
    return r[keep].sort_values(["order_date", "return_id"]).reset_index(drop=True)


def problem3_records(tickets: pd.DataFrame, pred: pd.DataFrame, customers: pd.DataFrame, cust: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SIM_SEED + 2)
    eng = customers.set_index("customer_id")["_engagement"]
    t = tickets.merge(pred, on="ticket_id", how="inner").copy()
    good_fix = {"Delivery delay": "Expedite", "Failed delivery": "Re-attempt", "Refund delay": "Wallet refund now",
                "Refund rejected": "Re-inspect", "Warranty / seller": "Replace item", "Wrong / damaged item": "Replace item",
                "Payment / billing": "Reverse charge"}
    t["best_fix"] = t["topic"].map(good_fix)
    t["used_best_fix"] = (t["fix_applied"] == t["best_fix"]).astype(int)
    t["routed_correctly"] = (t["predicted_topic"] == t["topic"]).astype(int)
    t["engagement"] = t["customer_id"].map(eng)
    t["clv"] = t["customer_id"].map(cust.set_index("customer_id")["clv_next12m_aed"])
    t["csat"] = t["csat"].fillna(t["csat"].median())
    for k in ("u", "u_reopen", "u_recontact"):
        t[k] = rng.random(len(t))
    keep = ["ticket_id", "created_date", "customer_id", "emirate", "channel", "topic", "predicted_topic", "routed_correctly",
            "text", "angry", "contacts", "resolution_hours", "reopened", "template_reply", "csat", "fix_applied", "best_fix",
            "used_best_fix", "recontact_7d", "order_value_aed", "tenure_months", "membership", "engagement", "clv",
            "churn_90d", "u", "u_reopen", "u_recontact"]
    return t[keep].sort_values(["created_date", "ticket_id"]).reset_index(drop=True)
