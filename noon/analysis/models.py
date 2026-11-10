"""Step 3/4 — the algorithms behind the three solutions. Outputs are tidy tables for the dashboard.

Every model is trained on one part of the data and judged on a part it has never seen.
XGBoost is used when installed; otherwise sklearn's gradient boosting is used and labelled so.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error, precision_score,
                             r2_score, recall_score, roc_auc_score, roc_curve, silhouette_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except Exception:  # pragma: no cover
    from sklearn.ensemble import GradientBoostingClassifier as XGBClassifier  # type: ignore
    from sklearn.ensemble import GradientBoostingRegressor as XGBRegressor  # type: ignore
    HAS_XGB = False

RS = 42
BOOST_NAME = "XGBoost" if HAS_XGB else "Gradient boosting (sklearn)"


def _boost_clf():
    if HAS_XGB:
        return XGBClassifier(n_estimators=350, max_depth=5, learning_rate=0.06, subsample=0.9, colsample_bytree=0.8,
                             eval_metric="logloss", random_state=RS, n_jobs=4)
    return XGBClassifier(random_state=RS)


def _boost_reg(**kw):
    if HAS_XGB:
        return XGBRegressor(n_estimators=400, max_depth=5, learning_rate=0.06, subsample=0.9, colsample_bytree=0.8,
                            random_state=RS, n_jobs=4, **kw)
    return XGBRegressor(random_state=RS)


def _design(df: pd.DataFrame, cats: list[str], nums: list[str]) -> pd.DataFrame:
    X = pd.get_dummies(df[cats].astype(str), prefix=cats, prefix_sep=" = ", dtype=float)
    return pd.concat([X, df[nums].astype(float)], axis=1)


def _clf_suite(X_tr, X_te, y_tr, y_te, models: dict, threshold: float = 0.5, label: str = ""):
    rows, rocs, preds = [], [], {}
    for name, m in models.items():
        if name == "Logistic regression":
            sc = StandardScaler().fit(X_tr)
            m.fit(sc.transform(X_tr), y_tr)
            p = m.predict_proba(sc.transform(X_te))[:, 1]
        else:
            m.fit(X_tr, y_tr)
            p = m.predict_proba(X_te)[:, 1]
        yhat = (p >= threshold).astype(int)
        rows.append({"problem": label, "model": name, "accuracy": accuracy_score(y_te, yhat),
                     "precision": precision_score(y_te, yhat, zero_division=0), "recall": recall_score(y_te, yhat),
                     "f1": f1_score(y_te, yhat), "roc_auc": roc_auc_score(y_te, p), "n_test": len(y_te),
                     "positive_rate": float(np.mean(y_te))})
        fpr, tpr, _ = roc_curve(y_te, p)
        step = max(1, len(fpr) // 120)
        rocs.append(pd.DataFrame({"problem": label, "model": name, "fpr": fpr[::step], "tpr": tpr[::step]}))
        preds[name] = p
    return pd.DataFrame(rows), pd.concat(rocs, ignore_index=True), preds


def _importance(model, cols, label, top=15):
    imp = getattr(model, "feature_importances_", None)
    if imp is None:
        return pd.DataFrame()
    return (pd.DataFrame({"problem": label, "feature": cols, "importance": imp})
            .sort_values("importance", ascending=False).head(top))


def _confusion(y, p, thr, label, model):
    tn, fp, fn, tp = confusion_matrix(y, (p >= thr).astype(int), labels=[0, 1]).ravel()
    return pd.DataFrame([{"problem": label, "model": model, "true_neg": tn, "false_pos": fp, "false_neg": fn, "true_pos": tp}])


def _silhouette_range(X, ks, label):
    out = []
    for k in ks:
        km = KMeans(k, n_init=10, random_state=RS).fit(X)
        s = silhouette_score(X, km.labels_, sample_size=min(4000, len(X)), random_state=RS)
        out.append({"problem": label, "k": k, "silhouette": s, "inertia": km.inertia_})
    return pd.DataFrame(out)


def _apriori_rules(items: pd.DataFrame, consequent: str, min_support=0.01, min_conf=0.05, label=""):
    from mlxtend.frequent_patterns import apriori, association_rules
    fi = apriori(items.astype(bool), min_support=min_support, use_colnames=True, max_len=4)
    rules = association_rules(fi, metric="confidence", min_threshold=min_conf)
    rules = rules[rules["consequents"].apply(lambda s: s == frozenset([consequent]))]
    rules = rules[rules["lift"] > 1].copy()
    rules["if"] = rules["antecedents"].apply(lambda s: " + ".join(sorted(s)))
    rules["then"] = consequent
    rules["n_items"] = rules["antecedents"].apply(len)
    rules["problem"] = label
    # drop redundant rules: a longer rule is kept only if it is clearly stronger than every shorter rule inside it
    rules = rules.sort_values("n_items")
    kept = []
    for row in rules.itertuples():
        subs = [k for k in kept if k[0] < row.antecedents]
        if any(row.confidence <= k[1] + 0.02 for k in subs):
            continue
        if "Yellow Friday" in row.antecedents and "Sale day" in row.antecedents:
            continue
        kept.append((row.antecedents, row.confidence, row.Index))
    rules = rules.loc[[k[2] for k in kept]]
    out = rules[["problem", "if", "then", "n_items", "support", "confidence", "lift"]]
    return out.sort_values(["lift", "support"], ascending=False).reset_index(drop=True)


def _two_prop(a_succ, a_n, b_succ, b_n):
    pa, pb = a_succ / a_n, b_succ / b_n
    diff = pb - pa
    se = np.sqrt(pa * (1 - pa) / a_n + pb * (1 - pb) / b_n)
    pool = (a_succ + b_succ) / (a_n + b_n)
    z = diff / np.sqrt(pool * (1 - pool) * (1 / a_n + 1 / b_n))
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return {"control_rate": pa, "treatment_rate": pb, "uplift_pts": diff * 100,
            "ci_low_pts": (diff - 1.96 * se) * 100, "ci_high_pts": (diff + 1.96 * se) * 100, "z": z, "p_value": p,
            "control_n": a_n, "treatment_n": b_n}


# ===================================================================== Problem 1 — delivery
P1_CATS = ["emirate", "zone_type", "category", "item_size", "fulfilment", "courier", "time_slot", "address_quality",
           "weekday", "sale_event", "membership"]
P1_NUMS = ["km_from_hub", "order_value_aed", "seller_avg_handover_h", "seller_late_rate"]


def problem1(orders: pd.DataFrame, returns: pd.DataFrame) -> dict:
    out = {}
    o = orders.copy()
    split = pd.Timestamp("2026-08-01")
    tr, te = o[o["order_date"] < split].copy(), o[o["order_date"] >= split].copy()
    seller_stats = tr.groupby("seller_id").agg(seller_avg_handover_h=("seller_handover_hours", "mean"),
                                               seller_late_rate=("late", "mean"))
    glob = seller_stats.mean()
    for d in (tr, te):
        d["seller_avg_handover_h"] = d["seller_id"].map(seller_stats["seller_avg_handover_h"]).fillna(glob.iloc[0])
        d["seller_late_rate"] = d["seller_id"].map(seller_stats["seller_late_rate"]).fillna(glob.iloc[1])
    X_tr, X_te = _design(tr, P1_CATS, P1_NUMS), _design(te, P1_CATS, P1_NUMS)
    X_te = X_te.reindex(columns=X_tr.columns, fill_value=0)
    y_tr, y_te = tr["late"].to_numpy(), te["late"].to_numpy()
    boost = _boost_clf()
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=5, n_jobs=4, random_state=RS)
    models = {"Logistic regression": LogisticRegression(max_iter=2000), "Random forest": rf, BOOST_NAME: boost}
    m, roc, preds = _clf_suite(X_tr, X_te, y_tr, y_te, models, 0.4, "Late delivery")
    out["p1_cls_metrics"], out["p1_roc"] = m, roc
    out["p1_importance"] = _importance(boost, list(X_tr.columns), "Late delivery")
    p = preds[BOOST_NAME]
    out["p1_confusion"] = _confusion(y_te, p, 0.4, "Late delivery", BOOST_NAME)
    te["p_late"] = p
    te["risk_tier"] = pd.cut(p, [-0.01, 0.25, 0.55, 1.01], labels=["Low risk", "Medium risk", "High risk"]).astype(str)
    tiers = te.groupby("risk_tier").agg(orders=("order_id", "size"), actual_late_rate=("late", "mean"),
                                        avg_p_late=("p_late", "mean")).reindex(["Low risk", "Medium risk", "High risk"]).reset_index()
    tiers["share_of_orders"] = tiers["orders"] / tiers["orders"].sum()
    tiers["share_of_all_late"] = (tiers["orders"] * tiers["actual_late_rate"]) / (te["late"].sum())
    tiers["action"] = ["Keep normal promise", "Adjusted date at checkout + tracking updates",
                       "Priority courier + early apology with a new date"]
    out["p1_risk_tiers"] = tiers

    # ---- delivery-time regression: mean (MAE) and P90 promise
    reg = _boost_reg()
    reg.fit(X_tr, tr["actual_hours"])
    pred_h = reg.predict(X_te)
    q90 = HistGradientBoostingRegressor(loss="quantile", quantile=0.9, max_iter=400, learning_rate=0.06, random_state=RS)
    q90.fit(X_tr, tr["actual_hours"])
    p90_h = q90.predict(X_te)
    te["pred_hours"], te["p90_hours"] = pred_h, p90_h
    te["p90_promise_days"] = np.maximum(1, np.ceil(p90_h / 24)).astype(int)
    out["p1_reg_metrics"] = pd.DataFrame([{
        "model": f"{BOOST_NAME} regressor (average time)", "mae_hours": mean_absolute_error(te["actual_hours"], pred_h),
        "mae_days": mean_absolute_error(te["actual_hours"], pred_h) / 24, "r2": r2_score(te["actual_hours"], pred_h)},
        {"model": "Quantile gradient boosting (P90 time)", "mae_hours": np.nan, "mae_days": np.nan,
         "r2": np.nan, "coverage": float(np.mean(te["actual_hours"] <= p90_h))}])
    cmp = []
    for name, col in [("Today: fixed promise", "promised_days"), ("P90 model promise", "p90_promise_days")]:
        kept = (te["actual_days"] <= te[col]).mean()
        cmp.append({"promise": name, "kept_rate": kept, "late_rate": 1 - kept, "avg_promised_days": te[col].mean(),
                    "orders": len(te)})
    out["p1_promise_compare"] = pd.DataFrame(cmp)
    by_em = te.groupby("emirate").apply(lambda d: pd.Series({
        "orders": len(d), "today_kept": (d["actual_days"] <= d["promised_days"]).mean(),
        "p90_kept": (d["actual_days"] <= d["p90_promise_days"]).mean(),
        "today_days": d["promised_days"].mean(), "p90_days": d["p90_promise_days"].mean()}), include_groups=False).reset_index()
    out["p1_promise_by_emirate"] = by_em
    out["p1_pred_sample"] = te.sample(min(1500, len(te)), random_state=RS)[
        ["order_id", "emirate", "category", "item_size", "actual_hours", "pred_hours", "p90_hours", "late", "p_late"]]

    # ---- weak-address model
    acat = ["address_quality", "zone_type", "time_slot", "emirate"]
    Xa = _design(o, acat, ["km_from_hub"])
    Xa_tr, Xa_te, ya_tr, ya_te = train_test_split(Xa, o["failed_first_attempt"], test_size=0.25, random_state=RS,
                                                  stratify=o["failed_first_attempt"])
    sc = StandardScaler().fit(Xa_tr)
    lr = LogisticRegression(max_iter=2000).fit(sc.transform(Xa_tr), ya_tr)
    pa = lr.predict_proba(sc.transform(Xa_te))[:, 1]
    out["p1_address_metrics"] = pd.DataFrame([{"model": "Logistic regression", "roc_auc": roc_auc_score(ya_te, pa),
                                               "base_rate": float(ya_te.mean())}])
    top = pd.Series(pa).rank(pct=True) >= 0.8
    out["p1_address_capture"] = pd.DataFrame([{
        "ask_for_pin_share": 0.2, "failed_caught": float(ya_te.to_numpy()[top.to_numpy()].sum() / ya_te.sum())}])
    out["p1_address_by_type"] = (o.groupby(["address_quality", "zone_type"])["failed_first_attempt"].agg(["mean", "size"])
                                 .reset_index().rename(columns={"mean": "failed_rate", "size": "orders"}))
    f = o[o["failed_first_attempt"] == 1]
    out["p1_unavailable"] = pd.DataFrame([{"failed_attempts": len(f), "gps_verified": int(f["unavailable_gps_verified"].sum()),
                                           "no_gps_proof": int((f["unavailable_gps_verified"] == 0).sum())}])

    # ---- K-Means on sellers
    rr = returns.groupby("seller_id").size()
    s = o.groupby("seller_id").agg(orders=("order_id", "size"), avg_handover_h=("seller_handover_hours", "mean"),
                                   late_rate=("late", "mean"), avg_days_late=("days_late", "mean"))
    s["return_rate"] = (rr.reindex(s.index).fillna(0) / s["orders"])
    s = s[s["orders"] >= 20]
    feats = ["avg_handover_h", "late_rate", "return_rate"]
    Xs = StandardScaler().fit_transform(s[feats])
    out["p1_seller_k"] = _silhouette_range(Xs, range(2, 7), "Sellers")
    km = KMeans(3, n_init=20, random_state=RS).fit(Xs)
    s["cluster"] = km.labels_
    order_ = s.groupby("cluster")["late_rate"].mean().sort_values().index
    names = dict(zip(order_, ["Good", "Average", "Poor"]))
    s["seller_tier"] = s["cluster"].map(names)
    s["express_badge"] = np.where(s["seller_tier"] == "Poor", "Removed until improved", "Kept")
    out["p1_sellers"] = s.reset_index()
    out["p1_seller_profile"] = (s.groupby("seller_tier")[["orders", "avg_handover_h", "late_rate", "return_rate"]]
                                .agg({"orders": "sum", "avg_handover_h": "mean", "late_rate": "mean", "return_rate": "mean"})
                                .join(s.groupby("seller_tier").size().rename("sellers")).reindex(["Good", "Average", "Poor"]).reset_index())

    # ---- K-Means on zones
    z = o[o["zone"] != "Unknown"].groupby(["zone", "emirate", "zone_type"]).agg(
        orders=("order_id", "size"), late_rate=("late", "mean"), avg_days_late=("days_late", "mean"),
        avg_hours=("actual_hours", "mean"), km_from_hub=("km_from_hub", "first"),
        failed_rate=("failed_first_attempt", "mean")).reset_index()
    from analysis.synthetic import ZONE_DF
    z = z.merge(ZONE_DF[["zone", "lat", "lon"]], on="zone", how="left")
    zf = ["late_rate", "avg_days_late", "failed_rate", "orders"]
    Xz = StandardScaler().fit_transform(z[zf])
    out["p1_zone_k"] = _silhouette_range(Xz, range(2, 6), "Zones")
    kmz = KMeans(3, n_init=20, random_state=RS).fit(Xz)
    z["cluster"] = kmz.labels_
    zo = z.groupby("cluster")["late_rate"].mean().sort_values().index
    z["zone_group"] = z["cluster"].map(dict(zip(zo, ["Reliable", "Watch", "Chronic problem"])))
    out["p1_zones"] = z

    # ---- Apriori: conditions before a delay
    poor = set(s.index[s["seller_tier"] == "Poor"])
    items = pd.DataFrame({
        "Sale day": o["on_sale"] == 1, "Yellow Friday": o["sale_event"] == "Yellow Friday",
        "Bulky item": o["item_size"] == "Bulky", "Evening order": o["evening"] == 1,
        "Partial address": o["address_quality"] == "Partial address", "Seller ships directly": o["fulfilment"] == "Ships directly",
        "Poor seller": o["seller_id"].isin(poor), "3PL courier": o["courier"] != "In-house fleet",
        "Villa / remote zone": o["zone_type"].isin(["Villa community", "Remote"]), "Late": o["late"] == 1})
    out["p1_rules"] = _apriori_rules(items, "Late", 0.01, 0.3, "Delivery")
    return out, te


def uplift_and_ab(orders: pd.DataFrame) -> dict:
    out = {}
    t = orders[orders["recovery_test_group"] != "Not in test"].copy()
    prior = orders.groupby("customer_id").cumcount()
    t["prior_orders"] = prior.loc[t.index]
    t["treated"] = (t["recovery_test_group"] == "Treatment").astype(int)
    feats = _design(t, ["membership", "emirate", "category"], ["tenure_months", "order_value_aed", "days_late", "prior_orders"])
    rng = np.random.default_rng(RS)
    fold = rng.integers(0, 2, len(t))
    uplift = np.zeros(len(t))
    for k in (0, 1):  # two-fold: train on one half, score the other (no customer scores itself)
        trn, tst = fold != k, fold == k
        mt = RandomForestClassifier(300, min_samples_leaf=25, random_state=RS, n_jobs=4)
        mc = RandomForestClassifier(300, min_samples_leaf=25, random_state=RS, n_jobs=4)
        a = trn & (t["treated"].to_numpy() == 1)
        b = trn & (t["treated"].to_numpy() == 0)
        mt.fit(feats[a], t["repurchase_60d"][a])
        mc.fit(feats[b], t["repurchase_60d"][b])
        uplift[tst] = mt.predict_proba(feats[tst])[:, 1] - mc.predict_proba(feats[tst])[:, 1]
    t["pred_uplift"] = uplift
    t["decile"] = pd.qcut(-t["pred_uplift"], 5, labels=[f"Top {i*20}%" if i == 1 else f"{(i-1)*20}–{i*20}%" for i in range(1, 6)])
    dec = t.groupby("decile", observed=True).apply(lambda d: pd.Series({
        "customers": len(d), "treated_rate": d.loc[d.treated == 1, "repurchase_60d"].mean(),
        "control_rate": d.loc[d.treated == 0, "repurchase_60d"].mean(), "avg_pred_uplift": d["pred_uplift"].mean()}),
        include_groups=False).reset_index()
    dec["observed_uplift_pts"] = (dec["treated_rate"] - dec["control_rate"]) * 100
    out["p1_uplift_groups"] = dec
    ab = _two_prop(int(t.loc[t.treated == 0, "repurchase_60d"].sum()), int((t.treated == 0).sum()),
                   int(t.loc[t.treated == 1, "repurchase_60d"].sum()), int((t.treated == 1).sum()))
    ab.update({"test": "Solution 1: early message + credit for likely-late orders", "metric": "Bought again within 60 days"})
    return out, ab


def demand_forecast(daily: pd.DataFrame) -> dict:
    d = daily.groupby("date", as_index=False)["orders"].sum()
    from analysis.synthetic import EVENTS
    future = pd.DataFrame({"date": pd.date_range(d["date"].max() + pd.Timedelta(days=1), periods=45)})
    allx = pd.concat([d, future], ignore_index=True)
    t = (allx["date"] - allx["date"].min()).dt.days.to_numpy()
    X = pd.DataFrame({"trend": t})
    for k in range(1, 4):
        X[f"sin{k}"] = np.sin(2 * np.pi * k * allx["date"].dt.dayofyear / 365.25)
        X[f"cos{k}"] = np.cos(2 * np.pi * k * allx["date"].dt.dayofyear / 365.25)
    for dw in range(1, 7):
        X[f"dow{dw}"] = (allx["date"].dt.dayofweek == dw).astype(int)
    ev = pd.Series(0, index=allx.index)
    names = {}
    for name, a, b, _ in EVENTS:
        names[name] = [(pd.Timestamp(a), pd.Timestamp(b)), (pd.Timestamp(a) - pd.DateOffset(years=1), pd.Timestamp(b) - pd.DateOffset(years=1)),
                       (pd.Timestamp(a) + pd.DateOffset(years=1), pd.Timestamp(b) + pd.DateOffset(years=1))]
    names["Ramadan"] = [(pd.Timestamp("2026-02-18"), pd.Timestamp("2026-03-19")), (pd.Timestamp("2025-03-01"), pd.Timestamp("2025-03-29"))]
    names["Eid al-Fitr"] = [(pd.Timestamp("2026-03-20"), pd.Timestamp("2026-03-23")), (pd.Timestamp("2025-03-30"), pd.Timestamp("2025-04-02"))]
    for name, spans in names.items():
        col = np.zeros(len(allx))
        for a, b in spans:
            col[(allx["date"] >= a) & (allx["date"] <= b)] = 1
        X[f"ev_{name}"] = col
    y = np.log(allx["orders"])
    cut = pd.Timestamp("2026-08-01")
    trn = allx["date"] < cut
    tst = (allx["date"] >= cut) & allx["orders"].notna()
    lr = LinearRegression().fit(X[trn], y[trn])
    pred = np.exp(lr.predict(X))
    res = allx.assign(fitted=pred.round(), part=np.where(trn, "Train", np.where(tst, "Test", "Forecast")))
    mape = float(np.mean(np.abs(res.loc[tst, "orders"] - res.loc[tst, "fitted"]) / res.loc[tst, "orders"]))
    lr2 = LinearRegression().fit(X[allx["orders"].notna()], y[allx["orders"].notna()])
    res.loc[res["part"] == "Forecast", "fitted"] = np.exp(lr2.predict(X[res["part"] == "Forecast"])).round()
    resid = np.exp(y[trn]) - pred[trn]
    res["low"] = res["fitted"] * (1 - 1.28 * np.std(resid / pred[trn]))
    res["high"] = res["fitted"] * (1 + 1.28 * np.std(resid / pred[trn]))
    ev_eff = pd.DataFrame({"event": [c[3:] for c in X.columns if c.startswith("ev_")],
                           "uplift_pct": [(np.exp(lr2.coef_[i]) - 1) * 100 for i, c in enumerate(X.columns) if c.startswith("ev_")]})
    return {"p1_demand": res, "p1_demand_events": ev_eff}, {"mape_test": mape}


# ===================================================================== Problem 2 — refunds
P2_CATS = ["category", "return_reason", "refund_method", "fulfilment", "membership", "item_size"]
P2_NUMS = ["prior_returns_12m", "account_age_months", "days_since_delivery", "item_value_aed", "photos_uploaded",
           "seller_quality_flag", "ordered_two_sizes"]


def problem2(returns: pd.DataFrame, orders: pd.DataFrame, cust: pd.DataFrame) -> dict:
    out, info = {}, {}
    r = returns.copy()
    r["qc_failed"] = 1 - r["qc_passed"]
    X = _design(r, P2_CATS, P2_NUMS)
    X_tr, X_te, y_tr, y_te, i_tr, i_te = train_test_split(X, r["qc_failed"], r.index, test_size=0.3, random_state=RS,
                                                          stratify=r["qc_failed"])
    rf = RandomForestClassifier(n_estimators=400, min_samples_leaf=4, class_weight="balanced_subsample", n_jobs=4, random_state=RS)
    models = {"Logistic regression": LogisticRegression(max_iter=3000, class_weight="balanced"), "Random forest": rf,
              BOOST_NAME: _boost_clf()}
    m, roc, preds = _clf_suite(X_tr, X_te, y_tr.to_numpy(), y_te.to_numpy(), models, 0.5, "Return fails QC")
    out["p2_cls_metrics"], out["p2_roc"] = m, roc
    out["p2_importance"] = _importance(rf, list(X.columns), "Return fails QC")
    p = preds["Random forest"]
    out["p2_confusion"] = _confusion(y_te.to_numpy(), p, 0.5, "Return fails QC", "Random forest")
    te = r.loc[i_te].copy()
    te["p_fail"] = p
    te["tier"] = np.select([p < 0.15, p < 0.5], ["Green", "Amber"], "Red")
    tiers = te.groupby("tier").agg(returns=("return_id", "size"), actual_fail_rate=("qc_failed", "mean"),
                                   avg_value=("item_value_aed", "mean"),
                                   value=("item_value_aed", "sum")).reindex(["Green", "Amber", "Red"]).reset_index()
    tiers["share"] = tiers["returns"] / tiers["returns"].sum()
    loss = te.loc[(te["tier"] == "Green") & (te["qc_failed"] == 1), "item_value_aed"].sum()
    tiers["refund_timing"] = ["When courier collects the item", "When warehouse scans it (≤24h)", "After full quality check"]
    out["p2_tiers"] = tiers
    out["p2_test_scores"] = te[["return_id", "p_fail", "qc_failed", "item_value_aed", "category", "return_reason"]].round(4)
    info["green_fail_value_share"] = float(loss / te.loc[te["tier"] == "Green", "item_value_aed"].sum())
    info["green_share"] = float((te["tier"] == "Green").mean())
    info["refund_days_today_median"] = float(r["refund_days"].median())

    # ---- K-Means on RFM + return behaviour
    c = cust.copy()
    c["return_rate"] = c["returns_12m"] / c["orders_12m"]
    feats = pd.DataFrame({"recency": c["recency_days"], "frequency": np.log1p(c["orders_12m"]),
                          "monetary": np.log1p(c["spend_12m_aed"]), "return_rate": c["return_rate"].clip(0, 2),
                          "rejected": c["rejected_refunds"].clip(0, 3)})
    Xc = StandardScaler().fit_transform(feats)
    out["p2_cluster_k"] = _silhouette_range(Xc, range(3, 8), "Customers (RFM + returns)")
    km = KMeans(5, n_init=20, random_state=RS).fit(Xc)
    c["cluster"] = km.labels_
    prof = c.groupby("cluster").agg(customers=("customer_id", "size"), recency_days=("recency_days", "mean"),
                                    orders_12m=("orders_12m", "mean"), spend_12m_aed=("spend_12m_aed", "mean"),
                                    return_rate=("return_rate", "mean"), rejected_refunds=("rejected_refunds", "mean"),
                                    clv=("clv_next12m_aed", "mean"), churn=("churned_next90d", "mean")).reset_index()

    def _name(row):
        if row.rejected_refunds > 0.5:
            return "Disputed / high-risk returners"
        if row.return_rate > 0.35:
            return "Frequent returners (size swaps)"
        if row.orders_12m >= prof["orders_12m"].quantile(0.75) and row.recency_days < 60:
            return "Loyal low-returners"
        if row.recency_days > 150:
            return "Lapsing customers"
        return "Occasional shoppers"
    prof["segment"] = prof.apply(_name, axis=1)
    dup = prof["segment"].duplicated(keep=False)
    prof.loc[dup, "segment"] = prof.loc[dup, "segment"] + " " + (prof.loc[dup].groupby("segment").cumcount() + 1).astype(str)
    c["segment"] = c["cluster"].map(prof.set_index("cluster")["segment"])
    out["p2_segments"] = prof
    out["p2_segment_sample"] = c.sample(min(2500, len(c)), random_state=RS)[
        ["customer_id", "segment", "recency_days", "orders_12m", "spend_12m_aed", "return_rate", "clv_next12m_aed"]]

    # ---- Apriori: what goes with a return
    o = orders.copy()
    o["returned"] = o["order_id"].isin(r["order_id"]).astype(int)
    two = r.set_index("order_id")["ordered_two_sizes"]
    poor = set(r.loc[r["seller_quality_flag"] == 1, "seller_id"])
    items = pd.DataFrame({
        "Fashion": o["category"] == "Fashion", "Electronics": o["category"] == "Electronics",
        "Ordered two sizes": o["order_id"].map(two).fillna(0).astype(bool),
        "Poor-quality seller": o["seller_id"].isin(poor), "Late delivery": o["late"] == 1,
        "Sale day": o["on_sale"] == 1, "Seller ships directly": o["fulfilment"] == "Ships directly",
        "Order value AED 500+": o["order_value_aed"] >= 500, "Returned": o["returned"] == 1})
    out["p2_rules"] = _apriori_rules(items, "Returned", 0.004, 0.05, "Returns")

    # ---- refund-time regression (what makes a refund slow)
    ok = r[r["refund_days"].notna() & (r["instant_refund_test_group"] != "Instant refund")].copy()
    Xr = _design(ok, ["refund_method", "fulfilment", "category", "emirate"], ["item_value_aed"])
    Xr = Xr.drop(columns=[c for c in ["refund_method = noon wallet", "fulfilment = Fulfilled by noon",
                                      "category = Grocery", "emirate = Dubai"] if c in Xr.columns])
    a, b, ya, yb = train_test_split(Xr, ok["refund_days"], test_size=0.3, random_state=RS)
    lin = LinearRegression().fit(a, ya)
    out["p2_refund_reg"] = pd.DataFrame([{"model": "Linear regression", "mae_days": mean_absolute_error(yb, lin.predict(b)),
                                          "r2": r2_score(yb, lin.predict(b)), "intercept": lin.intercept_}])
    coefs = pd.DataFrame({"factor": Xr.columns, "extra_days": lin.coef_})
    out["p2_refund_coefs"] = coefs.reindex(coefs["extra_days"].abs().sort_values(ascending=False).index).head(10)

    # ---- keep-it partial refund (cost of taking an item back)
    Xk = _design(r, ["item_size", "category"], ["item_value_aed"])
    ka, kb, kya, kyb = train_test_split(Xk, r["return_cost_aed"], test_size=0.3, random_state=RS)
    kl = LinearRegression().fit(ka, kya)
    r["pred_return_cost"] = kl.predict(Xk)
    r["keep_it_offer"] = (0.8 * r["pred_return_cost"]).clip(lower=0)
    r["keep_it_sensible"] = (r["pred_return_cost"] > 0.4 * r["item_value_aed"]) & r["return_reason"].isin(
        ["Damaged in delivery", "Not as described", "Defective / not working"])
    keep = r.groupby("category").agg(returns=("return_id", "size"), avg_value=("item_value_aed", "mean"),
                                     avg_cost=("pred_return_cost", "mean"), avg_offer=("keep_it_offer", "mean"),
                                     keep_it_share=("keep_it_sensible", "mean")).reset_index()
    keep["cost_share_of_value"] = keep["avg_cost"] / keep["avg_value"]
    out["p2_keep_it"] = keep
    info["keep_it_mae"] = float(mean_absolute_error(kyb, kl.predict(kb)))
    info["keep_it_r2"] = float(r2_score(kyb, kl.predict(kb)))

    # ---- A/B: instant vs standard refund for low-risk customers
    t = r[r["instant_refund_test_group"] != "Not in test"]
    ab = _two_prop(int(t.loc[t["instant_refund_test_group"] == "Standard refund", "repurchase_90d"].sum()),
                   int((t["instant_refund_test_group"] == "Standard refund").sum()),
                   int(t.loc[t["instant_refund_test_group"] == "Instant refund", "repurchase_90d"].sum()),
                   int((t["instant_refund_test_group"] == "Instant refund").sum()))
    inst = t[t["instant_refund_test_group"] == "Instant refund"]
    ab.update({"test": "Solution 2: instant refund for low-risk customers", "metric": "Bought again within 90 days",
               "guardrail_fail_rate": float(inst["qc_failed"].mean() if "qc_failed" in inst else 1 - inst["qc_passed"].mean()),
               "guardrail_value_share": float(inst.loc[inst["qc_passed"] == 0, "item_value_aed"].sum() / inst["item_value_aed"].sum())})
    return out, info, ab


# ===================================================================== Problem 3 — service
def problem3(tickets: pd.DataFrame) -> dict:
    out, info = {}, {}
    t = tickets[tickets["has_text"] == 1].copy()
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, stop_words="english", sublinear_tf=True)
    Xt = vec.fit_transform(t["text"])
    a, b, ya, yb, ia, ib = train_test_split(Xt, t["topic"], t.index, test_size=0.25, random_state=RS, stratify=t["topic"])
    clf = LogisticRegression(max_iter=3000, C=4)
    clf.fit(a, ya)
    pred = clf.predict(b)
    out["p3_test_pred"] = pd.DataFrame({"ticket_id": t.loc[ib, "ticket_id"].to_numpy(), "predicted_topic": pred,
                                        "p_angry_dummy": 0}).drop(columns="p_angry_dummy")
    labels = sorted(t["topic"].unique())
    cm = confusion_matrix(yb, pred, labels=labels)
    out["p3_topic_confusion"] = pd.DataFrame(cm, index=labels, columns=labels).reset_index().rename(columns={"index": "actual"})
    info["topic_accuracy"] = float(accuracy_score(yb, pred))
    info["topic_macro_f1"] = float(f1_score(yb, pred, average="macro"))
    names = np.array(vec.get_feature_names_out())
    tw = []
    for i, lab in enumerate(clf.classes_):
        top = np.argsort(clf.coef_[i])[::-1][:8]
        tw.append({"topic": lab, "top_words": ", ".join(names[top])})
    out["p3_topic_words"] = pd.DataFrame(tw)
    ang = LogisticRegression(max_iter=3000, class_weight="balanced").fit(a, t.loc[ia, "angry"])
    info["anger_auc"] = float(roc_auc_score(t.loc[ib, "angry"], ang.predict_proba(b)[:, 1]))
    samp = t.loc[ib].head(12).copy()
    samp["predicted_topic"] = clf.predict(vec.transform(samp["text"]))
    samp["p_angry"] = ang.predict_proba(vec.transform(samp["text"]))[:, 1].round(2)
    out["p3_ticket_examples"] = samp[["ticket_id", "text", "topic", "predicted_topic", "angry", "p_angry"]]

    # ---- K-Means themes on review / ticket text (no labels used)
    from sklearn.decomposition import TruncatedSVD
    svd = TruncatedSVD(60, random_state=RS).fit(Xt)
    Z = svd.transform(Xt)
    Z = Z / np.linalg.norm(Z, axis=1, keepdims=True).clip(1e-9)
    out["p3_theme_k"] = _silhouette_range(Z, range(4, 11), "Ticket themes")
    k = 7  # silhouette keeps rising slowly with k; 7 is where groups are still easy to name and act on
    km = KMeans(k, n_init=20, random_state=RS).fit(Z)
    t["theme_cluster"] = km.labels_
    centers = svd.inverse_transform(km.cluster_centers_)
    themes = []
    for ci in range(k):
        top = names[np.argsort(centers[ci])[::-1][:6]]
        sub = t[t["theme_cluster"] == ci]
        dom = sub["topic"].value_counts()
        themes.append({"theme": f"Theme {ci + 1}", "top_terms": ", ".join(top), "tickets": len(sub),
                       "share": len(sub) / len(t), "main_actual_topic": dom.index[0], "purity": dom.iloc[0] / len(sub),
                       "angry_share": sub["angry"].mean(), "churn_90d": sub["churn_90d"].mean()})
    out["p3_themes"] = pd.DataFrame(themes).sort_values("tickets", ascending=False)
    info["theme_k"] = k
    monthly = t.assign(month=t["created_date"].dt.to_period("M").astype(str)).groupby(["month", "topic"]).size().rename("tickets").reset_index()
    out["p3_topic_month"] = monthly

    # ---- resolution time regression
    tk = tickets.copy()
    Xr = _design(tk, ["topic", "channel"], ["contacts", "template_reply"])
    Xr = Xr.drop(columns=[c for c in ["topic = Delivery delay", "channel = Chat"] if c in Xr.columns])
    ra, rb, rya, ryb = train_test_split(Xr, tk["resolution_hours"], test_size=0.3, random_state=RS)
    lin = LinearRegression().fit(ra, rya)
    out["p3_resolution_coefs"] = pd.DataFrame({"factor": Xr.columns, "extra_hours": lin.coef_}).sort_values("extra_hours", ascending=False)
    info["resolution_mae_h"] = float(mean_absolute_error(ryb, lin.predict(rb)))
    info["resolution_r2"] = float(r2_score(ryb, lin.predict(rb)))
    info["resolution_intercept"] = float(lin.intercept_)

    # ---- churn after a complaint (classification)
    tk["csat_score"] = tk["csat"].fillna(tk["csat"].median())
    Xc = _design(tk, ["topic", "channel", "membership"], ["contacts", "resolution_hours", "reopened", "csat_score",
                                                          "template_reply", "tenure_months", "order_value_aed", "angry"])
    ca, cb, cya, cyb = train_test_split(Xc, tk["churn_90d"], test_size=0.3, random_state=RS, stratify=tk["churn_90d"])
    boost = _boost_clf()
    models = {"Logistic regression": LogisticRegression(max_iter=3000, class_weight="balanced"),
              "Random forest": RandomForestClassifier(300, min_samples_leaf=5, class_weight="balanced_subsample", random_state=RS, n_jobs=4),
              BOOST_NAME: boost}
    m, roc, preds = _clf_suite(ca, cb, cya.to_numpy(), cyb.to_numpy(), models, 0.3, "Churn after complaint")
    out["p3_churn_metrics"], out["p3_churn_roc"] = m, roc
    out["p3_churn_importance"] = _importance(boost, list(Xc.columns), "Churn after complaint")

    # ---- best fix per issue (decision tree + simple table)
    fx = tickets.groupby(["topic", "fix_applied"]).agg(tickets=("ticket_id", "size"), recontact_7d=("recontact_7d", "mean"),
                                                       csat=("csat", "mean")).reset_index()
    best = fx.loc[fx.groupby("topic")["recontact_7d"].idxmin()].rename(columns={"fix_applied": "suggested_fix"})
    out["p3_fix_table"], out["p3_best_fix"] = fx, best
    Xd = _design(tickets, ["topic", "fix_applied"], ["contacts"])
    dt = DecisionTreeClassifier(max_depth=4, min_samples_leaf=60, random_state=RS).fit(Xd, tickets["recontact_7d"])
    info["fix_tree_rules"] = export_text(dt, feature_names=list(Xd.columns), max_depth=4)

    # ---- market response: what each extra contact / day costs in churn
    mr = LinearRegression().fit(np.column_stack([tickets["contacts"], tickets["resolution_hours"] / 24]), tickets["churn_90d"])
    info["churn_per_contact_pts"] = float(mr.coef_[0] * 100)
    info["churn_per_day_pts"] = float(mr.coef_[1] * 100)
    by_c = tickets.assign(contacts_band=tickets["contacts"].clip(1, 5).astype(int).astype(str).replace({"5": "5+"})).groupby(
        "contacts_band").agg(tickets=("ticket_id", "size"), churn_90d=("churn_90d", "mean"), csat=("csat", "mean")).reset_index()
    out["p3_churn_by_contacts"] = by_c
    tickets = tickets.assign(res_band=pd.cut(tickets["resolution_hours"], [0, 24, 48, 96, 168, 1e9],
                                             labels=["<1 day", "1–2 days", "2–4 days", "4–7 days", "7+ days"]).astype(str))
    out["p3_churn_by_resolution"] = tickets.groupby("res_band").agg(tickets=("ticket_id", "size"), churn_90d=("churn_90d", "mean"),
                                                                    csat=("csat", "mean")).reset_index()
    return out, info


def event_chain_rules(orders, returns, tickets, cust) -> pd.DataFrame:
    c = cust.set_index("customer_id")
    no_update = orders[(orders["late"] == 1) & (orders["proactive_update_sent"] == 0)].groupby("customer_id").size()
    slow = returns[returns["refund_days"] > 8].groupby("customer_id").size()
    items = pd.DataFrame(index=c.index)
    items["Late delivery"] = c["late_orders"] > 0
    items["No update when late"] = no_update.reindex(c.index).fillna(0) > 0
    items["Failed delivery"] = c["failed_deliveries"] > 0
    items["Slow refund (8+ days)"] = slow.reindex(c.index).fillna(0) > 0
    items["Refund rejected"] = c["rejected_refunds"] > 0
    items["Complaint"] = c["tickets_12m"] > 0
    items["Repeat contact"] = c["repeat_contacts"] > 0
    items["Churned"] = c["churned_next90d"] == 1
    a = _apriori_rules(items, "Churned", 0.01, 0.05, "Event chain")
    b = _apriori_rules(items.drop(columns=["Churned", "Repeat contact"]), "Complaint", 0.01, 0.05, "Event chain")
    return pd.concat([b.head(8), a.head(12)], ignore_index=True)


# ===================================================================== customer foundations (Blogs 2–5)
def customer_models(cust: pd.DataFrame) -> dict:
    out, info = {}, {}
    c = cust.copy()
    nums = ["orders_12m", "spend_12m_aed", "recency_days", "tenure_months", "late_orders", "failed_deliveries",
            "returns_12m", "rejected_refunds", "tickets_12m", "repeat_contacts"]
    X = _design(c, ["membership", "emirate", "age_band"], nums)
    a, b, ya, yb, ia, ib = train_test_split(X, c["clv_next12m_aed"], c.index, test_size=0.3, random_state=RS)
    lin = LinearRegression().fit(a, ya)
    bo = _boost_reg().fit(a, ya)
    rows = []
    for nm, mdl in [("Linear regression", lin), (f"{BOOST_NAME} regressor", bo)]:
        p = mdl.predict(b)
        rows.append({"model": nm, "mae_aed": mean_absolute_error(yb, p), "r2": r2_score(yb, p)})
    out["cust_clv_metrics"] = pd.DataFrame(rows)
    c["clv_pred"] = bo.predict(X)
    c["clv_tier"] = pd.qcut(c["clv_pred"], [0, .5, .8, 1], labels=["Low", "Mid", "High"]).astype(str)
    out["cust_clv_tiers"] = c.groupby("clv_tier").agg(customers=("customer_id", "size"), avg_pred_clv=("clv_pred", "mean"),
                                                      avg_actual_clv=("clv_next12m_aed", "mean"),
                                                      churn=("churned_next90d", "mean")).reindex(["Low", "Mid", "High"]).reset_index()
    out["cust_clv_sample"] = pd.DataFrame({"actual": yb, "predicted": bo.predict(b)}).sample(min(1500, len(yb)), random_state=RS)
    # churn (Blog 4)
    ca, cb, cya, cyb = train_test_split(X, c["churned_next90d"], test_size=0.3, random_state=RS, stratify=c["churned_next90d"])
    boost = _boost_clf()
    m, roc, _ = _clf_suite(ca, cb, cya.to_numpy(), cyb.to_numpy(),
                           {"Logistic regression": LogisticRegression(max_iter=3000), BOOST_NAME: boost}, 0.35, "Customer churn")
    out["cust_churn_metrics"], out["cust_churn_roc"] = m, roc
    out["cust_churn_importance"] = _importance(boost, list(X.columns), "Customer churn", 12)
    # next purchase day (Blog 5) — classes
    act = c[c["next_purchase_days"].notna()].copy()
    act["npd_class"] = pd.cut(act["next_purchase_days"], [0, 14, 45, 999], labels=["0–14 days", "15–45 days", "46+ days"]).astype(str)
    Xn = X.loc[act.index]
    na, nb, nya, nyb = train_test_split(Xn, act["npd_class"], test_size=0.3, random_state=RS, stratify=act["npd_class"])
    rf = RandomForestClassifier(300, min_samples_leaf=5, random_state=RS, n_jobs=4).fit(na, nya)
    pn = rf.predict(nb)
    lab = ["0–14 days", "15–45 days", "46+ days"]
    out["cust_npd_confusion"] = pd.DataFrame(confusion_matrix(nyb, pn, labels=lab), index=lab, columns=lab).reset_index().rename(columns={"index": "actual"})
    info["npd_accuracy"] = float(accuracy_score(nyb, pn))
    info["npd_baseline"] = float(nyb.value_counts(normalize=True).max())
    return out, info
