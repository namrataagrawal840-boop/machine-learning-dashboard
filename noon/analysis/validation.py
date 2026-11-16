"""Model checking: the tests a careful analyst runs before trusting a model.

* Cross-validation (5 folds) instead of one lucky split; time-ordered folds for delivery (train on the past only).
* Small hyper-parameter search, chosen on cross-validated AUC of the TRAINING data only.
* Calibration: does a score of 0.10 really mean a 10% chance? (Brier score + reliability table)
* Cost-based cut-offs: thresholds chosen from business costs and the fraud cap, not 0.5 by habit.
* Qini curve for the uplift model; bootstrap confidence intervals for AUC and the A/B tests.
* Fairness: are new-account customers or one emirate pushed into the strict tier more often for the same outcome?
* Rolling-origin back-test for the demand forecasts (several forecast start dates, not one).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RS = 42


def _boost(**kw):
    from analysis.models import _boost_clf
    m = _boost_clf()
    m.set_params(**kw)
    return m


def _candidates():
    from analysis.models import BOOST_NAME
    c = {}
    for C in (0.05, 0.3, 1.0):
        c[f"Logistic regression (C={C})"] = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
    for leaf in (10, 30):
        c[f"Random forest (leaf={leaf})"] = RandomForestClassifier(300, min_samples_leaf=leaf, max_features=0.4,
                                                                   n_jobs=4, random_state=RS)
    for depth in (2, 3):
        c[f"{BOOST_NAME} (depth={depth})"] = _boost(max_depth=depth, n_estimators=250, learning_rate=0.05)
    return c


def cv_auc(model, X, y, folds=5):
    skf = StratifiedKFold(folds, shuffle=True, random_state=RS)
    aucs, oof = [], np.zeros(len(y))
    X = np.asarray(X, dtype=float)
    for tr, te in skf.split(X, y):
        m = clone(model).fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], oof[te]))
    return float(np.mean(aucs)), float(np.std(aucs)), oof


def select_and_calibrate(X_tr, y_tr, label):
    """Pick the best candidate by 5-fold CV AUC on the training data, then calibrate it (Platt / sigmoid)."""
    rows, best = [], None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, mdl in _candidates().items():
            mean, sd, oof = cv_auc(mdl, X_tr, y_tr)
            rows.append({"problem": label, "model": name, "cv_auc_mean": mean, "cv_auc_sd": sd})
            if best is None or mean > best[1]:
                best = (name, mean, mdl, oof)
        name, _, mdl, oof_raw = best
        Xa = np.asarray(X_tr, dtype=float)
        raw = clone(mdl).fit(Xa, y_tr)
        # out-of-fold Platt-calibrated scores; keep calibration only if it lowers the out-of-fold Brier score
        skf = StratifiedKFold(5, shuffle=True, random_state=RS + 1)
        oof_cal = np.zeros(len(y_tr))
        for tr, te in skf.split(Xa, y_tr):
            oof_cal[te] = CalibratedClassifierCV(clone(mdl), method="sigmoid", cv=3).fit(Xa[tr], y_tr[tr]).predict_proba(Xa[te])[:, 1]
        b_raw, b_cal = brier_score_loss(y_tr, oof_raw), brier_score_loss(y_tr, oof_cal)
        use_cal = b_cal < b_raw - 1e-4
        final = CalibratedClassifierCV(clone(mdl), method="sigmoid", cv=5).fit(Xa, y_tr) if use_cal else raw
        oof = oof_cal if use_cal else oof_raw
    table = pd.DataFrame(rows).sort_values("cv_auc_mean", ascending=False).reset_index(drop=True)
    table["chosen"] = table["model"] == name

    class _Wrap:
        def __init__(self, m):
            self.m = m

        def predict_proba(self, X):
            return self.m.predict_proba(np.asarray(X, dtype=float))

    cal_any = CalibratedClassifierCV(clone(mdl), method="sigmoid", cv=5).fit(Xa, y_tr)
    return {"table": table, "name": name, "model": _Wrap(final), "oof": oof, "calibrated": bool(use_cal),
            "raw_test": lambda X: raw.predict_proba(np.asarray(X, dtype=float))[:, 1],
            "cal_test": lambda X: cal_any.predict_proba(np.asarray(X, dtype=float))[:, 1],
            "brier_oof_raw": float(b_raw), "brier_oof_cal": float(b_cal)}


def calibration_table(y, scores: dict, label, bins=10):
    rows = []
    for name, p in scores.items():
        q = pd.qcut(p, bins, labels=False, duplicates="drop")
        d = pd.DataFrame({"p": p, "y": y, "bin": q}).groupby("bin").agg(mean_pred=("p", "mean"), observed=("y", "mean"),
                                                                        n=("y", "size")).reset_index()
        d["version"], d["problem"], d["brier"] = name, label, float(brier_score_loss(y, p))
        rows.append(d)
    return pd.concat(rows, ignore_index=True)


def cutoff_curve(p, y, value, cuts=None):
    """For each Green cut-off: how many returns get an instant refund, and how much money is paid out early on items
    that then fail inspection — as a share of Green value and of ALL refund value (the fraud cap is on the latter)."""
    cuts = np.round(np.arange(0.01, 0.301, 0.005), 3) if cuts is None else cuts
    total = max(value.sum(), 1)
    rows = []
    for c in cuts:
        g = p < c
        lost = value[g & (y == 1)].sum()
        rows.append({"cutoff": c, "green_share": float(g.mean()),
                     "value_at_risk": float(lost / max(value[g].sum(), 1)),
                     "loss_share_all_refunds": float(lost / total),
                     "fail_rate_in_green": float(y[g].mean()) if g.any() else np.nan})
    return pd.DataFrame(rows)


def green_cutoff(oof, y, value, cap=0.005):
    cur = cutoff_curve(oof, y, value)
    ok = cur[cur["loss_share_all_refunds"] <= cap]
    return float(ok["cutoff"].max()) if len(ok) else 0.01


def fairness_table(te: pd.DataFrame) -> pd.DataFrame:
    """For returns that PASSED inspection (honest customers), how often was each group still held back (not Green)?"""
    te = te.assign(account_band=np.where(te["account_age_months"] < 6, "New account (<6 months)", "Established account"))
    rows = []
    for col in ("account_band", "emirate"):
        for g, d in te.groupby(col):
            honest = d[d["qc_failed"] == 0]
            rows.append({"group_type": "Account age" if col == "account_band" else "Emirate", "group": g, "returns": len(d),
                         "green_share": (d["tier"] == "Green").mean(), "red_share": (d["tier"] == "Red").mean(),
                         "honest_held_back": (honest["tier"] != "Green").mean(),
                         "honest_sent_red": (honest["tier"] == "Red").mean(), "actual_fail_rate": d["qc_failed"].mean()})
    return pd.DataFrame(rows)


def bootstrap_auc(y, p, n=400):
    rng = np.random.default_rng(RS)
    vals = []
    idx = np.arange(len(y))
    for _ in range(n):
        b = rng.choice(idx, len(idx))
        if y[b].min() != y[b].max():
            vals.append(roc_auc_score(y[b], p[b]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def bootstrap_ab(control, treat, n=2000):
    rng = np.random.default_rng(RS)
    d = [rng.choice(treat, len(treat)).mean() - rng.choice(control, len(control)).mean() for _ in range(n)]
    return float(np.percentile(d, 2.5) * 100), float(np.percentile(d, 97.5) * 100)


def export_logit(X, y, name) -> dict:
    """A small, explainable logistic model the live platform can apply with plain numpy (no sklearn needed)."""
    X = pd.DataFrame(X).astype(float)
    mu, sd = X.mean(), X.std().replace(0, 1)
    m = LogisticRegression(C=0.3, max_iter=3000).fit((X - mu) / sd, y)
    return {"name": name, "features": list(X.columns), "mean": mu.round(6).tolist(), "sd": sd.round(6).tolist(),
            "coef": m.coef_[0].round(6).tolist(), "intercept": float(m.intercept_[0])}


# ----------------------------------------------------------------------------------- whole-project checks
def p1_time_cv(orders: pd.DataFrame) -> pd.DataFrame:
    """Expanding-window CV by month: train on all months before M, test on month M (Jun–Oct 2026)."""
    from analysis.models import BOOST_NAME, P1_CATS, P1_NUMS, _boost_clf, _design
    o = orders.copy()
    o["m"] = o["order_date"].dt.to_period("M")
    rows = []
    for mth in pd.period_range("2026-06", "2026-10", freq="M"):
        tr, te = o[o["m"] < mth].copy(), o[o["m"] == mth].copy()
        st = tr.groupby("seller_id").agg(seller_avg_handover_h=("seller_handover_hours", "mean"), seller_late_rate=("late", "mean"))
        for d in (tr, te):
            d["seller_avg_handover_h"] = d["seller_id"].map(st["seller_avg_handover_h"]).fillna(st["seller_avg_handover_h"].mean())
            d["seller_late_rate"] = d["seller_id"].map(st["seller_late_rate"]).fillna(st["seller_late_rate"].mean())
        Xa, Xb = _design(tr, P1_CATS, P1_NUMS), _design(te, P1_CATS, P1_NUMS)
        Xb = Xb.reindex(columns=Xa.columns, fill_value=0)
        for name, mdl in [("Logistic regression", make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))),
                          (BOOST_NAME, _boost_clf())]:
            pr = mdl.fit(Xa, tr["late"]).predict_proba(Xb)[:, 1]
            rows.append({"problem": "Late delivery", "fold": str(mth), "model": name, "auc": roc_auc_score(te["late"], pr),
                         "test_orders": len(te), "late_rate": te["late"].mean()})
    return pd.DataFrame(rows)


def stratified_cv(X, y, label, models: dict) -> pd.DataFrame:
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, mdl in models.items():
            mean, sd, _ = cv_auc(mdl, X, np.asarray(y))
            rows.append({"problem": label, "model": name, "cv_auc_mean": mean, "cv_auc_sd": sd})
    return pd.DataFrame(rows)


def p1_cost_threshold(p, y, cost_fp=8.0, cost_fn=35.0) -> pd.DataFrame:
    """Flagging an order costs ~AED 8 (priority courier / message); missing a late order costs ~AED 35 in lost repeat profit."""
    rows = []
    for t in np.round(np.arange(0.05, 0.96, 0.05), 2):
        f = p >= t
        rows.append({"threshold": t, "flagged_share": f.mean(), "missed_late": int(((~f) & (y == 1)).sum()),
                     "false_alarms": int((f & (y == 0)).sum()),
                     "cost_aed": float(cost_fp * (f & (y == 0)).sum() + cost_fn * ((~f) & (y == 1)).sum()
                                       + cost_fp * 0.5 * (f & (y == 1)).sum())})
    d = pd.DataFrame(rows)
    d["best"] = d["cost_aed"] == d["cost_aed"].min()
    return d


def qini(treated, outcome, uplift) -> tuple[pd.DataFrame, float]:
    """Qini curve: extra buyers gained if we target the top x% by predicted uplift, vs. random targeting."""
    order = np.argsort(-uplift)
    t, yv = np.asarray(treated)[order], np.asarray(outcome)[order]
    n = len(t)
    nt, nc = np.cumsum(t), np.cumsum(1 - t)
    yt, yc = np.cumsum(yv * t), np.cumsum(yv * (1 - t))
    q = yt - yc * np.divide(nt, nc, out=np.zeros(n), where=nc > 0)
    frac = np.arange(1, n + 1) / n
    step = max(1, n // 100)
    d = pd.DataFrame({"share_targeted": frac[::step], "qini": q[::step], "random": (q[-1] * frac)[::step]})
    auuc = float(np.trapezoid(q - q[-1] * frac, frac) / n)
    return d, auuc


def forecast_backtest(daily: pd.DataFrame) -> pd.DataFrame:
    """Rolling-origin test: forecast 28 days ahead from four different start dates, SARIMA vs regression vs naive."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from analysis.models import demand_features
    d = daily.groupby("date", as_index=False)["orders"].sum()
    X = demand_features(d["date"])
    ex = X[[c for c in X.columns if c.startswith(("sin", "cos", "ev_"))]].astype(float)
    y = np.log(d["orders"].to_numpy())
    rows = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for start in ["2026-05-01", "2026-06-15", "2026-08-01", "2026-09-15"]:
            tr = (d["date"] < start).to_numpy()
            te = ((d["date"] >= start) & (d["date"] < pd.Timestamp(start) + pd.Timedelta(days=28))).to_numpy()
            act = d.loc[te, "orders"].to_numpy()
            reg = LinearRegression().fit(X[tr], y[tr])
            pr = np.exp(reg.predict(X[te]))
            keep = ex.columns[ex[tr].std() > 0]  # an input that never varied in training cannot be learned
            fit = SARIMAX(y[tr], exog=ex.loc[tr, keep].to_numpy(), order=(1, 1, 1),
                          seasonal_order=(1, 0, 1, 7)).fit(disp=False, maxiter=150)
            ps = np.exp(fit.get_forecast(int(te.sum()), exog=ex.loc[te, keep].to_numpy()).predicted_mean)
            naive = np.tile(d.loc[tr, "orders"].to_numpy()[-7:], 5)[: te.sum()]
            for name, pp in [("Seasonal ARIMA", ps), ("Time-series regression", pr), ("Repeat last week", naive)]:
                rows.append({"origin": start, "model": name, "mape": float(np.mean(np.abs(act - pp) / act))})
    return pd.DataFrame(rows)
