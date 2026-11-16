"""Customer-response models for the solution simulator, ESTIMATED FROM THE DATA (not copied from the generator).

The simulator has to answer "if this order had not been late / this refund had come in 1 day, would the customer
have come back?". The generator's hidden formulas know the answer, but a real analyst never sees them, so the
simulator must not use them either. Instead we fit small logistic models on the recorded data only:

  p1_rep     bought again within 60 days   ~ late, days late, update sent, credit, value, tenure, member, prior orders
  p1_ticket  delivery complaint on order   ~ late, late×update, late×days late, failed attempt, failed×GPS proof
  p2_rep     bought again within 90 days   ~ QC failed, refund days, instant refund, value, member, prior returns
  p2_ticket  refund-chase ticket on return ~ QC failed, slow (>8 days), days over 8
  p3_churn   churned within 90 days        ~ extra contacts, days to resolve, reopened, CSAT, CSAT missing, recontact…
  p3_csat    CSAT score (linear)           ~ days to resolve, reopened, extra contacts, template reply

The credit and instant-refund effects are identified by the randomised tests; the others are observational
(associations that may include confounding) — the dashboard says so. 100 bootstrap refits give uncertainty bands.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

from noonbi.sim import FEATURES, features

B = 100
RS = 42


def _fit(X, y, logistic=True):
    if logistic:
        m = LogisticRegression(C=100.0, max_iter=5000).fit(X, y)
        return np.r_[m.intercept_[0], m.coef_[0]]
    m = LinearRegression().fit(X, y)
    return np.r_[m.intercept_, m.coef_]


def _model(kind, cols: dict, y, logistic=True) -> dict:
    X = features(kind, cols)
    rng = np.random.default_rng(RS)
    point = _fit(X, y, logistic)
    boot = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(B):
            i = rng.integers(0, len(y), len(y))
            boot.append(_fit(X[i], y[i], logistic))
    return {"features": FEATURES[kind], "logistic": logistic, "n": int(len(y)), "point": point.round(6).tolist(),
            "boot": np.array(boot).round(6).tolist(), "base_rate": float(np.mean(y))}


def build(orders: pd.DataFrame, returns: pd.DataFrame, tickets: pd.DataFrame) -> dict:
    o = orders.sort_values("order_date").copy()
    o["prior_orders"] = o.groupby("customer_id").cumcount()
    tk_orders = tickets[tickets["topic"].isin(["Delivery delay", "Failed delivery"])]["order_id"].unique()
    oc = {"late": o["late"], "days_late": o["days_late"], "update": o["proactive_update_sent"], "credit": (o["credit_aed"] > 0).astype(int),
          "value": o["order_value_aed"], "tenure": o["tenure_months"], "membership": o["membership"], "prior_orders": o["prior_orders"],
          "failed": o["failed_first_attempt"], "gps": o["unavailable_gps_verified"]}
    out = {"p1_rep": _model("p1_rep", oc, o["repurchase_60d"].to_numpy()),
           "p1_ticket": _model("p1_ticket", oc, o["order_id"].isin(tk_orders).astype(int).to_numpy())}

    r = returns.copy()
    fail = (r["qc_passed"] == 0).astype(int)
    rk = tickets[tickets["topic"].isin(["Refund delay", "Refund rejected"])]["order_id"].unique()
    rc = {"fail": fail, "refund_days": r["refund_days"].fillna(0), "instant": (r["instant_refund_test_group"] == "Instant refund").astype(int),
          "value": r["item_value_aed"], "membership": r["membership"], "prior_returns": r["prior_returns_12m"]}
    out["p2_rep"] = _model("p2_rep", rc, r["repurchase_90d"].to_numpy())
    out["p2_ticket"] = _model("p2_ticket", rc, r["order_id"].isin(rk).astype(int).to_numpy())

    t = tickets.copy()
    tc = {"contacts": t["contacts"], "res_h": t["resolution_hours"], "reopened": t["reopened"],
          "csat": t["csat"].fillna(t["csat"].median()), "csat_missing": t["csat"].isna().astype(int), "recontact": t["recontact_7d"],
          "rejected": (t["topic"] == "Refund rejected").astype(int), "tenure": t["tenure_months"], "membership": t["membership"],
          "template": t["template_reply"]}
    out["p3_churn"] = _model("p3_churn", tc, t["churn_90d"].to_numpy())
    ans = t["csat"].notna().to_numpy()
    out["p3_csat"] = _model("p3_csat", {k: pd.Series(v).to_numpy()[ans] for k, v in tc.items()}, t.loc[ans, "csat"].to_numpy(), logistic=False)
    return out


def coef_table(m: dict) -> pd.DataFrame:
    rows = []
    for name, mod in m.items():
        b = np.array(mod["boot"])
        for j, f in enumerate(["(intercept)"] + mod["features"]):
            rows.append({"model": name, "feature": f, "coef": mod["point"][j], "low": float(np.percentile(b[:, j], 10)),
                         "high": float(np.percentile(b[:, j], 90)), "n": mod["n"], "type": "logistic" if mod["logistic"] else "linear"})
    return pd.DataFrame(rows)


def save(m: dict, path: Path) -> None:
    path.write_text(json.dumps(m), encoding="utf-8")
