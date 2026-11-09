"""Compute every number and figure used in the written report from the files in data/.

Usage:  python -m report.build_assets      -> report/assets/*.png and report/assets/content.json
Then:   node report/build_docx.js           -> report/Luggage_Marketplace_Analytics_Report.docx
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as ss  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dashboard.components import stats as S  # noqa: E402
from dashboard.data.transforms import ms_columns, ms_option, prepare_consumer, prepare_partner  # noqa: E402

DATA, OUT = ROOT / "data", ROOT / "report" / "assets"
OUT.mkdir(parents=True, exist_ok=True)
BLUE, GREEN, LAV, PEACH, GOLD, TEAL, ROSE, SLATE = "#668DBA", "#78A48D", "#9C88C9", "#E3A27A", "#D9B85C", "#7FB3C8", "#C98B9F", "#8E9AAB"
TEXT, MUTED = "#263548", "#6B788A"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": "#C9D3DF", "axes.labelcolor": TEXT,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.titlesize": 10, "axes.titlecolor": TEXT, "figure.dpi": 160})

raw = pd.read_csv(DATA / "consumer_survey_raw_synthetic.csv")
clean = pd.read_csv(DATA / "consumer_survey_clean.csv")
log = pd.read_csv(DATA / "data_cleaning_log.csv")
d, meta = prepare_consumer(clean)
d = d.copy()
ca = pd.read_csv(DATA / "cluster_assignments.csv")
d = d.merge(ca.assign(cluster="Cluster " + ca["cluster"].astype(str)), on="respondent_id", how="left")
partners, _ = prepare_partner(pd.read_csv(DATA / "partner_survey.csv"))
comp = pd.read_csv(DATA / "competitor_locations.csv")
geo = pd.read_csv(DATA / "geographic_market_audit.csv")
C: dict = {}


def pct(flag):
    p, k, n = S.share(d[flag])
    return {"pct": round(p, 1), "k": k, "n": n}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def barh(ax, labels, vals, colour, fmt="{:.1f}%"):
    y = np.arange(len(labels))
    ax.barh(y, vals, color=colour)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    for i, v in enumerate(vals):
        ax.text(v + max(vals) * 0.01, i, fmt.format(v), va="center", fontsize=8, color=TEXT)
    ax.set_xlim(0, max(vals) * 1.18)


def dist(col, order=None):
    s = d[col].dropna().astype(str)
    vc = s.value_counts()
    if order is not None:
        vc = vc.reindex([o for o in order if o in vc.index])
    return vc, len(s)


# ---------------------------------------------------------------- headline numbers
C["n_raw"], C["n_clean"], C["n_cols"] = len(raw), len(clean), clean.shape[1]
C["dup_raw"] = int(raw.duplicated().sum())
C["by_city"] = d["city"].value_counts().to_dict()
for f in ("long_gap", "has_problem", "search_difficulty", "access_barrier", "unmet_need", "accepts_partner", "high_intent"):
    C[f] = pct(f)
for f in ("storage_wtp_inr__value", "pickup_delivery_wtp_inr__value"):
    C[f] = {k: (round(v, 1) if isinstance(v, float) else v) for k, v in S.describe_numeric(d[f]).items()}
C["planning_only"] = int((clean["past_trip_experience"] != "Yes").sum())
C["log"] = log[["step", "column", "action", "rows_affected", "reason"]].to_dict("records")
miss = raw.isna().mean().mul(100).sort_values(ascending=False)
C["raw_missing_top"] = [{"column": c, "pct": round(v, 1)} for c, v in miss.head(6).items()]
C["raw_cells_missing"] = int(raw.isna().sum().sum())
C["clean_cells_missing"] = int(clean.isna().sum().sum())

# ---------------------------------------------------------------- figures: descriptive
fig, axes = plt.subplots(1, 2, figsize=(8, 2.8))
vc, n = dist("checkout_gap_hours__label", list(d["checkout_gap_hours__label"].cat.categories))
axes[0].bar(range(len(vc)), vc / n * 100, color=[LAV if "h" in i or "gap" in i else SLATE for i in vc.index])
axes[0].set_xticks(range(len(vc)), vc.index, rotation=35, ha="right")
axes[0].set_ylabel("% of respondents")
axes[0].set_title("Checkout-to-departure gap (Q11)")
vc2, n2 = dist("luggage_management_hours__label", list(d["luggage_management_hours__label"].cat.categories))
axes[1].bar(range(len(vc2)), vc2 / n2 * 100, color=BLUE)
axes[1].set_xticks(range(len(vc2)), vc2.index, rotation=35, ha="right")
axes[1].set_title("Time personally managing luggage (Q12)")
save(fig, "fig_gap")
C["gap_dist"] = {k: round(v / n * 100, 1) for k, v in vc.items()}

steps = [("long_gap", "Gap ≥ 3 h"), ("has_problem", "Gap ≥ 3 h & managed ≥ 1 h"), ("search_difficulty", "Storage hard to find / unaware"),
         ("access_barrier", "Cites an access barrier"), ("unmet_need", "Unmet storage need")]
fig, ax = plt.subplots(figsize=(7, 2.5))
barh(ax, [s[1] for s in steps], [C[s[0]]["pct"] for s in steps], LAV)
ax.set_xlabel("% of respondents with a valid answer")
ax.set_title("Problem prevalence cascade")
save(fig, "fig_cascade")


def ms(field):
    cols = ms_columns(d, field)
    ans = d[cols].notna().any(axis=1)
    sub = d.loc[ans, cols].fillna(False).astype(bool)
    s = (sub.mean() * 100).sort_values(ascending=False)
    return [(ms_option(c), round(v, 1)) for c, v in s.items()], int(ans.sum())


C["methods"], C["methods_n"] = ms("existing_storage_methods")
C["security"], C["security_n"] = ms("security_requirements")
C["venues"], C["venues_n"] = ms("acceptable_venue_types")
C["barriers"], C["barriers_n"] = ms("existing_barriers")
C["features"], C["features_n"] = ms("preferred_features")
fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.2))
m = C["methods"][:8]
barh(axes[0], [x[0] for x in m], [x[1] for x in m], BLUE)
axes[0].set_title(f"Current methods (Q15, n={C['methods_n']:,})")
s_ = C["security"][:8]
barh(axes[1], [x[0] for x in s_], [x[1] for x in s_], GREEN)
axes[1].set_title(f"Essential trust features (Q22, n={C['security_n']:,})")
save(fig, "fig_methods_security")

fig, axes = plt.subplots(1, 3, figsize=(9, 2.7))
vc, n = dist("inconvenience_score__label", [str(i) for i in range(1, 6)])
axes[0].bar(vc.index, vc / n * 100, color=PEACH)
axes[0].set_title("Inconvenience (Q17, 1–5)")
axes[0].set_ylabel("% of respondents")
vc, n = dist("booking_intent_score__label", [str(i) for i in range(11)])
axes[1].bar(vc.index, vc / n * 100, color=[GOLD if int(i) >= 8 else "#E8DDB5" for i in vc.index])
axes[1].set_title("Booking intent (Q23, 0–10)")
w = d["storage_wtp_inr__value"].dropna()
axes[2].hist(w[w <= w.quantile(0.99)], bins=30, color=BLUE)
axes[2].axvline(w.median(), ls=":", color=TEXT)
axes[2].set_title("Storage WTP ₹ (≤ 99th pct)")
save(fig, "fig_inconv_intent_wtp")
C["inconv_45"] = round(float((d["inconvenience_score__value"].dropna() >= 4).mean() * 100), 1)

fs = d["first_service_preference__label"].dropna()
fsv = (fs.value_counts(normalize=True) * 100).round(1)
C["first_service"] = list(fsv.items())
C["storage_first"] = round(float(fs.str.lower().str.contains("drop my luggage|locker").mean() * 100), 1)
C["pickup_first"] = round(float(fs.str.lower().str.contains("pick|deliver").mean() * 100), 1)
C["barrier_top"] = list((d["main_nonuse_barrier__label"].value_counts(normalize=True) * 100).round(1).head(5).items())
C["pricing_top"] = list((d["preferred_pricing_model__label"].value_counts(normalize=True) * 100).round(1).head(3).items())

# ---------------------------------------------------------------- diagnostics
def rho(a, b):
    x = d[[a, b]].dropna()
    r, p = ss.spearmanr(x[a], x[b])
    return {"rho": round(float(r), 3), "p": float(p), "n": len(x)}


v = d[(d["checkout_gap_hours__status"] == "valid") & (d["luggage_management_hours__status"] == "valid")]
C["d1"] = rho("checkout_gap_hours__rank", "luggage_management_hours__rank")
ct = pd.crosstab(v["checkout_gap_hours__label"], v["luggage_management_hours__label"], normalize="index") * 100
ct = ct.loc[(ct.sum(axis=1) > 0), (ct.sum(axis=0) > 0)]
fig, ax = plt.subplots(figsize=(6.4, 3))
im = ax.imshow(ct.values, cmap="Blues", vmin=0, vmax=100, aspect="auto")
ax.set_xticks(range(ct.shape[1]), ct.columns, rotation=30, ha="right")
ax.set_yticks(range(ct.shape[0]), ct.index)
for i in range(ct.shape[0]):
    for j in range(ct.shape[1]):
        ax.text(j, i, f"{ct.values[i, j]:.0f}", ha="center", va="center", fontsize=7,
                color="white" if ct.values[i, j] > 55 else TEXT)
ax.set_xlabel("Time managing luggage (Q12)")
ax.set_ylabel("Checkout gap (Q11)")
ax.set_title("Row % — gap vs luggage-management time")
save(fig, "fig_d1")

hold = [c for c in ms_columns(d, "existing_storage_methods") if ms_option(c).startswith("Hotel")][0]
acc = d.groupby("accommodation_type__label").agg(n=("respondent_id", "size"),
                                                 held=(hold, lambda s: s.dropna().astype(bool).mean() * 100),
                                                 difficulty=("search_difficulty", lambda s: s.dropna().astype(bool).mean() * 100),
                                                 unmet=("unmet_need", lambda s: s.dropna().astype(bool).mean() * 100))
acc = acc[acc["n"] >= 30].sort_values("held")
C["d4"] = acc.round(1).reset_index().to_dict("records")
fig, ax = plt.subplots(figsize=(7, 2.8))
yy = np.arange(len(acc))
ax.barh(yy - 0.25, acc["held"], 0.25, color=GREEN, label="Accommodation held luggage %")
ax.barh(yy, acc["difficulty"], 0.25, color=PEACH, label="Storage hard to find / unaware %")
ax.barh(yy + 0.25, acc["unmet"], 0.25, color=LAV, label="Unmet need %")
ax.set_yticks(yy, [f"{i} (n={int(n)})" for i, n in zip(acc.index, acc["n"])])
ax.invert_yaxis()
ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.4, -0.12), ncol=3)
ax.set_title("Accommodation type vs storage availability")
save(fig, "fig_d4")

C["d7"] = rho("inconvenience_score__value", "booking_intent_score__value")
hi = d.groupby("inconvenience_score__value")["high_intent"].apply(lambda s: s.dropna().astype(bool).mean() * 100)
C["d7_hi"] = {int(k): round(v, 1) for k, v in hi.items()}
fig, ax = plt.subplots(figsize=(6, 2.6))
data_ = [d.loc[d["inconvenience_score__value"] == i, "booking_intent_score__value"].dropna() for i in range(1, 6)]
ax.boxplot(data_, tick_labels=[f"{i}\n(n={len(x)})" for i, x in zip(range(1, 6), data_)], patch_artist=True,
           boxprops=dict(facecolor="#DCE6F2", color=BLUE), medianprops=dict(color=TEXT))
ax2 = ax.twinx()
ax2.plot(range(1, 6), [hi.get(i, np.nan) for i in range(1, 6)], color=GOLD, marker="o")
ax2.set_ylabel("High-intent %", color=GOLD)
ax.set_xlabel("Inconvenience score (Q17)")
ax.set_ylabel("Booking intent (Q23)")
ax.set_title("Inconvenience vs likelihood of booking")
save(fig, "fig_d7")

C["d6"] = rho("bag_count__rank", "booking_intent_score__value")
C["d6_hi"] = d.groupby("bag_count__label", observed=True)["high_intent"].apply(
    lambda s: round(s.dropna().astype(bool).mean() * 100, 1)).to_dict()
sub = d[d["checkout_gap_hours__status"] == "valid"]
vv, pp, nn = S.cramers_v(sub["travel_mode__label"], sub["checkout_gap_hours__label"])
C["d5"] = {"v": round(vv, 3), "p": pp, "n": nn,
           "long_by_mode": sub.groupby("travel_mode__label")["long_gap"].apply(lambda s: round(s.dropna().astype(bool).mean() * 100, 1)).to_dict()}
C["d10"] = rho("trips_12m__rank", "storage_wtp_inr__value")
C["d10_med"] = d.groupby("trips_12m__label", observed=True)["storage_wtp_inr__value"].median().round(0).to_dict()
C["wtp_city"] = d.groupby("city")["storage_wtp_inr__value"].median().round(0).to_dict()
C["wtp_purpose"] = d.groupby("travel_purpose__label")["storage_wtp_inr__value"].agg(["median", "size"]).round(0).reset_index().to_dict("records")

# ---------------------------------------------------------------- correlation
sp_vars = {"trips_12m__rank": "Trips", "group_size__rank": "Group", "checkout_gap_hours__rank": "Gap",
           "luggage_management_hours__rank": "Luggage time", "bag_count__rank": "Bags", "problem_frequency__rank": "Gap freq.",
           "inconvenience_score__value": "Inconvenience", "storage_search_difficulty__rank": "Search diff.",
           "booking_intent_score__value": "Intent", "storage_wtp_inr__value": "Storage WTP",
           "pickup_delivery_wtp_inr__value": "Pickup WTP"}
r, p, n = S.corr_matrix(d, list(sp_vars), "spearman")
r = r.rename(index=sp_vars, columns=sp_vars)
fig, ax = plt.subplots(figsize=(6.6, 5))
ax.imshow(r.values, cmap="RdBu", vmin=-1, vmax=1)
ax.set_xticks(range(len(r)), r.columns, rotation=45, ha="right")
ax.set_yticks(range(len(r)), r.index)
for i in range(len(r)):
    for j in range(len(r)):
        ax.text(j, i, f"{r.values[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                color="white" if abs(r.values[i, j]) > 0.6 else TEXT)
ax.set_title("Spearman rank correlation (ordinal variables)")
save(fig, "fig_spearman")
pairs = [(a, b, float(r.loc[a, b])) for i, a in enumerate(r.index) for b in r.columns[i + 1:]]
pairs.sort(key=lambda t: abs(t[2]), reverse=True)
C["spearman_top"] = [(a, b, round(v, 2)) for a, b, v in pairs[:6]]

nom = {"city__label": "City", "travel_purpose__label": "Purpose", "travel_mode__label": "Transport",
       "accommodation_type__label": "Accommodation", "first_service_preference__label": "First service",
       "main_nonuse_barrier__label": "Non-use barrier", "high_intent": "High intent", "age_group__label": "Age"}
v_, p_, n_ = S.cramers_matrix(d, list(nom))
v_ = v_.rename(index=nom, columns=nom)
fig, ax = plt.subplots(figsize=(5.6, 4.4))
ax.imshow(v_.values, cmap="Blues", vmin=0, vmax=0.6)
ax.set_xticks(range(len(v_)), v_.columns, rotation=45, ha="right")
ax.set_yticks(range(len(v_)), v_.index)
for i in range(len(v_)):
    for j in range(len(v_)):
        ax.text(j, i, f"{v_.values[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                color="white" if v_.values[i, j] > 0.45 else TEXT)
ax.set_title("Cramér's V (categorical variables)")
save(fig, "fig_cramers")
pairs = [(a, b, float(v_.loc[a, b])) for i, a in enumerate(v_.index) for b in v_.columns[i + 1:]]
pairs.sort(key=lambda t: t[2], reverse=True)
C["cramers_top"] = [(a, b, round(x, 2)) for a, b, x in pairs[:5]]
pw = d[["storage_wtp_inr__value", "pickup_delivery_wtp_inr__value"]].dropna()
pr_ = ss.pearsonr(pw.iloc[:, 0], pw.iloc[:, 1])
C["pearson_wtp"] = {"r": round(float(pr_[0]), 3), "n": len(pw)}

# ---------------------------------------------------------------- partners, market, models
C["partners"] = {"n": len(partners), "willing": pct_p if (pct_p := S.share(partners["is_willing"])) else None,
                 "capacity_willing": int(partners.loc[partners["is_willing"].fillna(False).astype(bool), "storage_capacity__value"].sum()),
                 "by_type": partners.assign(t=partners["establishment_type"].str.rstrip(".")).groupby("t")["is_willing"]
                 .apply(lambda s: round(s.dropna().astype(bool).mean() * 100, 1)).to_dict(),
                 "digital": S.share(partners["digital_handover_acceptance__rank"].dropna() >= 2),
                 "courier": S.share(partners["pickup_delivery_acceptance__rank"].dropna() >= 2)}
C["competitors"] = comp[["competitor_name", "city", "competitor_category", "price_note", "evidence_status", "source_url"]] \
    .fillna("").to_dict("records")
C["comp_summary"] = comp.groupby(["city", "competitor_category"]).size().unstack(fill_value=0).to_dict("index")
C["n_nodes"] = len(geo)

cm = pd.read_csv(DATA / "classification_metrics.csv")
C["cls"] = cm[cm["split"] == "test"][["model", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]].round(3).to_dict("records")
prof = pd.read_csv(DATA / "cluster_profiles.csv")
C["clusters"] = {"k": int(prof["k_selected"].iloc[0]), "silhouette": prof["silhouette_by_k"].iloc[0],
                 "sizes": prof[["cluster", "size", "share_pct"]].to_dict("records")}
ov = d.dropna(subset=["cluster"]).groupby("cluster").agg(
    n=("respondent_id", "size"), high_intent=("high_intent", lambda s: s.dropna().astype(bool).mean() * 100),
    unmet=("unmet_need", lambda s: s.dropna().astype(bool).mean() * 100),
    gap=("checkout_gap_hours__rank", "mean"), inconv=("inconvenience_score__value", "mean"),
    bags=("bag_count__rank", "mean"), trips=("trips_12m__rank", "mean"),
    wtp=("storage_wtp_inr__value", "median")).round(2)
C["cluster_overlay"] = ov.reset_index().to_dict("records")
rules = pd.read_csv(DATA / "association_rules.csv")
C["rules_top"] = rules.sort_values(["lift", "support"], ascending=False).head(6)[
    ["antecedents", "consequents", "support", "confidence", "lift"]].round(3).to_dict("records")
C["rules_n"] = len(rules)
rm = pd.read_csv(DATA / "regression_metrics.csv")
C["reg"] = rm[rm["split"] == "test"][["model", "target", "mae", "rmse", "r2"]].round(3).to_dict("records")
coef = pd.read_csv(DATA / "regression_coefficients.csv")
rc = coef[(coef["model"] == "Ridge Regression") & (coef["target"] == "storage_wtp_inr")]
rc = rc.reindex(rc["coefficient"].abs().sort_values(ascending=False).index).head(6)
C["reg_coef"] = rc[["feature", "coefficient"]].round(1).to_dict("records")

# survey-intent funnel
cur = pd.Series(True, index=d.index)
fun = [("All respondents", len(d))]
for col, nm in (("long_gap", "Checkout gap ≥ 3 h"), ("unmet_need", "Unmet storage need"),
                ("accepts_partner", "Accepts partner venues"), ("high_intent", "High stated intent (8–10)")):
    cur &= d[col].fillna(False).astype(bool)
    fun.append((nm, int(cur.sum())))
cur &= d["pilot_interest__rank"] >= 2
fun.append(("Pilot: 'Yes, definitely'", int(cur.sum())))
C["funnel"] = fun
fig, ax = plt.subplots(figsize=(6.4, 2.5))
barh(ax, [f[0] for f in fun], [f[1] for f in fun], LAV, fmt="{:,.0f}")
ax.set_title("Survey-intent funnel (not sales)")
save(fig, "fig_funnel")

# segment by city and city table
city_tbl = d.groupby("city").agg(n=("respondent_id", "size"),
                                 long_gap=("long_gap", lambda s: s.dropna().astype(bool).mean() * 100),
                                 unmet=("unmet_need", lambda s: s.dropna().astype(bool).mean() * 100),
                                 high_intent=("high_intent", lambda s: s.dropna().astype(bool).mean() * 100),
                                 wtp=("storage_wtp_inr__value", "median")).round(1)
C["city_tbl"] = city_tbl.reset_index().to_dict("records")

(OUT / "content.json").write_text(json.dumps(C, indent=1, default=lambda o: o if not isinstance(o, (np.integer, np.floating)) else o.item(), ensure_ascii=False), encoding="utf-8")
print("assets written:", sorted(p.name for p in OUT.glob("*.png")))
