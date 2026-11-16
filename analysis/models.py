"""Group-stage analytics on the cleaned (synthetic) consumer survey.

Produces the model-output files the dashboard reads:
  classification_metrics.csv / classification_predictions.csv   (High_Intent, stated intent)
  cluster_assignments.csv / cluster_profiles.csv                 (K-means on pre-intent behaviour)
  association_rules.csv                                          (Apriori on multi-select answers)
  regression_metrics.csv / regression_predictions.csv / regression_coefficients.csv  (Linear, Ridge, Lasso)

Leakage rule: Q23 (the target) and Q24–Q30 are never used as classification predictors.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV, LinearRegression, LogisticRegression, RidgeCV
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score,
                             silhouette_score)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from analysis import synthetic as syn

SEED = 42


def _idx(options, v):
    """Position of an answer in its option list (matching ignores case/punctuation)."""
    if pd.isna(v):
        return np.nan
    t = str(v).strip().lower().rstrip(".")
    for i, o in enumerate(options):
        if o.lower().rstrip(".") == t:
            return float(i)
    return np.nan


def _short(s: str) -> str:
    return re.sub(r"[^0-9a-z]+", "_", s.lower().rstrip(".")).strip("_")[:40]


def _multi(d: pd.DataFrame, col: str, options: list[str], prefix: str) -> pd.DataFrame:
    out = {}
    split = d[col].fillna("").astype(str).str.split(r"\s*;\s*")
    for o in options:
        out[f"{prefix}__{_short(o)}"] = split.map(lambda lst, o=o: float(o in lst))
    m = pd.DataFrame(out, index=d.index)
    m[d[col].isna().values] = np.nan
    return m


def pre_intent_features(d: pd.DataFrame) -> pd.DataFrame:
    """Predictors from Q2–Q22 only."""
    X = pd.DataFrame(index=d.index)
    X["planning_only"] = (d["past_trip_experience"] != "Yes").astype(float)
    X["international_resident"] = (d["residence_country"] != "India").astype(float)
    age = d["age_group"].map(lambda v: _idx(syn.AGE[:5], v))
    X["age_rank"] = age
    X["trips_rank"] = d["trips_12m"].map(lambda v: _idx(syn.TRIPS, v))
    X["group_rank"] = d["group_size"].map(lambda v: _idx(syn.GROUP, v))
    X["checkout_gap_rank"] = d["checkout_gap_hours"].map(lambda v: _idx(syn.GAP, v))
    X["luggage_mgmt_rank"] = d["luggage_management_hours"].map(lambda v: _idx(syn.MGMT, v))
    X["bags_rank"] = d["bag_count"].map(lambda v: _idx(syn.BAGS, v))
    X["gap_frequency_rank"] = d["problem_frequency"].map(lambda v: _idx(syn.FREQ, v))
    X["inconvenience"] = pd.to_numeric(d["inconvenience_score"], errors="coerce")
    sd = d["storage_search_difficulty"].map(lambda v: _idx(syn.SEARCH, v))
    X["search_difficulty_rank"] = sd.where(sd <= 4)
    X["unaware_of_storage"] = (sd == 5).astype(float).where(sd.notna())
    X["did_not_search"] = (sd == 6).astype(float).where(sd.notna())
    det = d["maximum_detour"].map(lambda v: _idx(syn.DETOUR, v))
    X["max_detour_rank"] = det.where(det <= 4)
    X["pickup_only"] = (det == 5).astype(float).where(det.notna())
    for col, opts in (("city", list(syn.MICROZONES)), ("travel_purpose", syn.PURPOSE), ("travel_mode", syn.MODE),
                      ("accommodation_type", syn.ACCOM)):
        for o in opts[1:]:  # first category is the reference level
            X[f"{col}__{_short(o)}"] = (d[col] == o).astype(float)
    X = pd.concat([X, _multi(d, "existing_storage_methods", syn.METHODS, "method"),
                   _multi(d, "existing_barriers", syn.BARRIERS, "barrier"),
                   _multi(d, "acceptable_venue_types", syn.VENUES, "venue"),
                   _multi(d, "security_requirements", syn.SECURITY, "security")], axis=1)
    return X


# ---------------------------------------------------------------- classification
def run_classification(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = d[d["high_intent"].notna()]
    X, y = pre_intent_features(data), data["high_intent"].astype(int)
    ids = data["respondent_id"]
    Xtr, Xte, ytr, yte, idtr, idte = train_test_split(X, y, ids, test_size=0.25, stratify=y, random_state=SEED)
    models = {
        "Logistic Regression": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                             LogisticRegression(max_iter=2000, C=0.5)),
        "KNN": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), KNeighborsClassifier(n_neighbors=25)),
        "Decision Tree": make_pipeline(SimpleImputer(strategy="median"),
                                       DecisionTreeClassifier(max_depth=5, min_samples_leaf=20, random_state=SEED)),
        "Random Forest": make_pipeline(SimpleImputer(strategy="median"),
                                       RandomForestClassifier(n_estimators=400, min_samples_leaf=5, random_state=SEED)),
        "Gradient Boosting": make_pipeline(SimpleImputer(strategy="median"),
                                           GradientBoostingClassifier(random_state=SEED)),
    }
    feats = "; ".join(X.columns)
    mrows, prows = [], []
    for name, m in models.items():
        m.fit(Xtr, ytr)
        for split, XX, yy, ii in (("train", Xtr, ytr, idtr), ("test", Xte, yte, idte)):
            p = m.predict_proba(XX)[:, 1]
            yh = (p >= 0.5).astype(int)
            mrows.append({"model": name, "target": "high_intent (stated, Q23 8–10)", "split": split,
                          "accuracy": accuracy_score(yy, yh), "precision": precision_score(yy, yh, zero_division=0),
                          "recall": recall_score(yy, yh), "f1": f1_score(yy, yh), "roc_auc": roc_auc_score(yy, p),
                          "pr_auc": average_precision_score(yy, p), "n": len(yy), "positive_rate": float(yy.mean()),
                          "features": feats})
            if split == "test":
                prows.append(pd.DataFrame({"respondent_id": ii.values, "model": name, "target": "high_intent",
                                           "split": split, "actual": yy.values, "predicted": yh,
                                           "probability": p.round(4)}))
    return pd.DataFrame(mrows).round(4), pd.concat(prows, ignore_index=True)


# ---------------------------------------------------------------- clustering
CLUSTER_FEATURES = ["trips_rank", "group_rank", "checkout_gap_rank", "luggage_mgmt_rank", "bags_rank",
                    "gap_frequency_rank", "inconvenience", "search_difficulty_rank", "max_detour_rank", "pickup_only",
                    "method__carried_the_luggage_with_me", "method__asked_my_hotel_accommodation_to_hold_it",
                    "international_resident"]


def run_clustering(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """K-means on pre-intent behaviour of respondents with past-trip experience (planning-only excluded)."""
    data = d[d["past_trip_experience"] == "Yes"]
    X = pre_intent_features(data)[CLUSTER_FEATURES]
    Z = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()).fit_transform(X)
    scores = {}
    for k in range(3, 7):
        lab = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit_predict(Z)
        scores[k] = silhouette_score(Z, lab, sample_size=min(1500, len(Z)), random_state=SEED)
    k = max(scores, key=scores.get)
    km = KMeans(n_clusters=k, n_init=50, random_state=SEED).fit(Z)
    lab = km.labels_
    assign = pd.DataFrame({"respondent_id": data["respondent_id"].values, "cluster": lab})
    prof = X.assign(cluster=lab).groupby("cluster").mean().round(3)
    prof.insert(0, "size", pd.Series(lab).value_counts().sort_index().values)
    prof.insert(1, "share_pct", (prof["size"] / prof["size"].sum() * 100).round(1))
    prof["k_selected"] = k
    prof["silhouette_k"] = round(scores[k], 4)
    prof["silhouette_by_k"] = "; ".join(f"k={kk}: {v:.3f}" for kk, v in scores.items())
    prof["features_used"] = "; ".join(CLUSTER_FEATURES)
    prof["note"] = "Means of standardisation inputs (ranks / 0–1 flags); planning-only respondents not clustered."
    return assign, prof.reset_index()


# ---------------------------------------------------------------- association rules
def run_association(d: pd.DataFrame) -> pd.DataFrame:
    from dashboard.data.transforms import canonical_option

    tx = []
    fields = [("existing_storage_methods", "Current"), ("acceptable_venue_types", "Venue"),
              ("security_requirements", "Needs"), ("preferred_features", "Wants")]
    for _, r in d.iterrows():
        items = set()
        for f, pre in fields:
            if pd.notna(r[f]):
                for t in str(r[f]).split(";"):
                    if t.strip():
                        items.add(f"{pre}: {canonical_option(f, t)}")
        if pd.notna(r["first_service_preference"]):
            fs = str(r["first_service_preference"]).lower()
            lab = ("Locker" if "locker" in fs else "Partner drop-off" if "drop my" in fs else "Pickup→store"
                   if "stored at a partner" in fs else "Store→deliver" if "stored and then" in fs else
                   "Pickup→deliver" if "direct pickup" in fs else "None")
            items.add(f"First: {lab}")
        tx.append(items)
    all_items = sorted(set().union(*tx))
    onehot = pd.DataFrame([[i in t for i in all_items] for t in tx], columns=all_items)
    freq = apriori(onehot, min_support=0.05, use_colnames=True, max_len=3)
    rules = association_rules(freq, metric="confidence", min_threshold=0.35)
    rules = rules[rules["lift"] > 1.0].sort_values("lift", ascending=False)
    out = pd.DataFrame({
        "antecedents": rules["antecedents"].map(lambda s: "; ".join(sorted(s))),
        "consequents": rules["consequents"].map(lambda s: "; ".join(sorted(s))),
        "support": rules["support"].round(4), "confidence": rules["confidence"].round(4),
        "lift": rules["lift"].round(4), "leverage": rules["leverage"].round(4),
        "count": (rules["support"] * len(onehot)).round().astype(int),
    })
    return out.head(400).reset_index(drop=True)


# ---------------------------------------------------------------- regression
def run_regression(d: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    mrows, prows, crows = [], [], []
    for target in ("storage_wtp_inr", "pickup_delivery_wtp_inr"):
        data = d[d[target].notna()]
        X, y = pre_intent_features(data), data[target].astype(float)
        Xtr, Xte, ytr, yte, idtr, idte = train_test_split(X, y, data["respondent_id"], test_size=0.25,
                                                          random_state=SEED)
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": RidgeCV(alphas=np.logspace(-2, 3, 30)),
            "Lasso Regression": LassoCV(alphas=np.logspace(-1, 2.5, 30), max_iter=20000, random_state=SEED),
        }
        for name, est in models.items():
            m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), est).fit(Xtr, ytr)
            for split, XX, yy, ii in (("train", Xtr, ytr, idtr), ("test", Xte, yte, idte)):
                pr = m.predict(XX)
                mrows.append({"model": name, "target": target, "split": split, "mae": mean_absolute_error(yy, pr),
                              "rmse": float(np.sqrt(mean_squared_error(yy, pr))), "r2": r2_score(yy, pr), "n": len(yy),
                              "alpha": getattr(m[-1], "alpha_", np.nan)})
                if split == "test":
                    prows.append(pd.DataFrame({"respondent_id": ii.values, "model": name, "target": target,
                                               "split": split, "actual": yy.values, "predicted": pr.round(2)}))
            crows.append(pd.DataFrame({"model": name, "target": target, "feature": X.columns,
                                       "coefficient": np.round(m[-1].coef_, 4),
                                       "scale": "₹ per 1 standard deviation of the feature"}))
    return pd.DataFrame(mrows).round(4), pd.concat(prows, ignore_index=True), pd.concat(crows, ignore_index=True)
