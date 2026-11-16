"""Model Quality & Fairness — the checks an analyst runs before trusting any number on the other pages."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from noonbi import cards, charts, theme
from noonbi import data as D
from noonbi.views import common as C

REALISM = [
    ("Hidden causes the models never see", "Customer loyalty and 'life events', driver skill, daily depot load, inspector strictness, "
     "how much each customer complains", "No model can reach a perfect score; part of every outcome is genuinely unpredictable"),
    ("Measurement noise", "Seller handover time recorded with ±30% error, 4% of address labels wrong, 0.5% of QC results mis-keyed, "
     "6% of tickets tagged with the wrong topic by agents", "Models learn from imperfect labels, as in real logs"),
    ("Change over time (drift)", "From Aug 2026 a 3PL courier got faster and 40 new sellers with no history joined",
     "Train-on-past / test-on-future is harder than a random split"),
    ("Missing for a reason", "15% of returns have no photo flag (old app version); unhappy customers skip the CSAT survey more often",
     "Missing values are kept as their own signal instead of being filled in silently"),
    ("Overlapping behaviour", "Abusers, size-swappers and serial returners share habits; only some abusers use new accounts",
     "No single rule separates good and bad returns"),
    ("Outcomes from a separate future process", "Churn, CLV and next purchase are read off a simulated next year driven by hidden "
     "buying rates and life events, not written as a formula of the features", "Customer models cannot simply reverse-engineer a formula"),
    ("Messy text", "Typos, Arabizi ('yalla', 'mafi update'), one-line 'any update??' messages, 30% of tickets mention two issues",
     "Text classification drops from near-perfect to a realistic level"),
    ("A real-looking demand series", "Event sizes differ by year, a new Abu Dhabi hub (level shift), a platform outage, correlated noise",
     "Forecast models must be told about known one-offs, as a real forecaster would"),
]


def render(f: D.Filters) -> None:
    cards.page_header("Model Quality & Fairness",
                      "How the data was made realistic, and the checks run before any model result was used: cross-validation, "
                      "calibration, cost-based thresholds, uplift checks, bootstrap intervals, fairness and forecast back-tests.",
                      "Validation", [("Models", "derived"), ("Data", "synthetic")])
    S = D.summary()
    cv = D.table("val_cv_summary")
    best = cv.sort_values("cv_auc_mean", ascending=False).groupby("problem").head(1).set_index("problem")
    cards.kpi_grid([
        cards.kpi("Late delivery · CV ROC-AUC", f"{best.loc['Late delivery', 'cv_auc_mean']:.2f}", "5 monthly folds, train on past only", "derived", theme.PEACH),
        cards.kpi("Refund QC fail · CV ROC-AUC", f"{best.loc['Return fails QC', 'cv_auc_mean']:.2f}", "5 folds, best of 7 settings", "derived", theme.LAVENDER),
        cards.kpi("Churn after complaint · CV ROC-AUC", f"{best.loc['Churn after complaint', 'cv_auc_mean']:.2f}", "5 stratified folds", "derived", theme.BLUE),
        cards.kpi("Ticket topic accuracy", C.pct(S.get("p3", {}).get("topic_accuracy", np.nan), 0), "Messy text, 6% wrong agent tags", "derived", theme.GREEN),
    ])
    tabs = st.tabs(["1 · How the data was made realistic", "2 · Cross-validation", "3 · Thresholds from costs", "4 · Fairness",
                    "5 · Tests & forecasts", "6 · Simulator response models"])
    with tabs[0]:
        _realism()
    with tabs[1]:
        _cv(cv)
    with tabs[2]:
        _costs()
    with tabs[3]:
        _fairness()
    with tabs[4]:
        _tests()
    with tabs[5]:
        _response()


def _realism() -> None:
    with cards.card("The data does not hand the models their answers", "What makes this synthetic data behave like real logs?"):
        st.markdown(cards.html_table([{"Realism added": f"<b>{cards.esc(a)}</b>", "What it means in the data": cards.esc(b),
                                       "Effect on the models": cards.esc(c)} for a, b, c in REALISM]), unsafe_allow_html=True)
        cards.explanation("Rules generate the data, but the models see only what noon would record: no hidden traits, no true labels.",
                          "Scores dropped from 'too good to be true' (AUC 0.86–0.97) to realistic levels (0.65–0.80), and one model "
                          "(next-purchase timing) now does no better than a simple guess — reported honestly rather than hidden.",
                          "The solutions still hold up because they use models for ranking and timing, not for perfect yes/no answers.",
                          "It is still synthetic: the patterns are plausible, not noon's real numbers.")


def _cv(cv: pd.DataFrame) -> None:
    c1, c2 = st.columns([1, 1.1])
    with c1, cards.card("Cross-validated ROC-AUC (mean ± sd over 5 folds)"):
        charts.table(cv, "val_cv", "cross_validation.csv", column_config={
            "cv_auc_mean": st.column_config.ProgressColumn("CV ROC-AUC", min_value=0.5, max_value=1, format="%.3f"),
            "cv_auc_sd": st.column_config.NumberColumn("± sd", format="%.3f")})
    with c2, cards.card("Late-delivery model, month by month", "Train on all earlier months, test on the next"):
        t = D.table("val_p1_time_cv")
        fig = go.Figure()
        for i, (mname, d) in enumerate(t.groupby("model", sort=False)):
            fig.add_scatter(x=d["fold"], y=d["auc"], mode="lines+markers", name=mname, line=dict(width=2.6, color=theme.CATEGORICAL[i]))
        fig.update_layout(xaxis_title="Test month", yaxis_title="ROC-AUC", yaxis_range=[0.5, 1], legend=dict(orientation="h", y=-0.25))
        charts.show(fig, "val_p1_time", t, height=330)
    cards.explanation("Each model is trained and tested five times on different slices. For delivery the folds follow time, so the "
                      "model never learns from the future.",
                      "The ± sd shows how much a score moves between slices; models whose scores differ by less than that are tied. "
                      "The August dip is real drift: a courier contract changed and new sellers arrived.",
                      "Prefer the simpler model when scores are tied (easier to explain to customers and auditors), and retrain monthly.",
                      "Five folds on one year of synthetic data; real data would need a longer history.")


def _costs() -> None:
    d = D.table("val_p1_cost")
    b = d[d["best"]].iloc[0]
    with cards.card("Late-delivery flag: which threshold costs least?", "Cost of false alarms vs missed late orders"):
        fig = go.Figure()
        fig.add_scatter(x=d["threshold"], y=d["cost_aed"], mode="lines+markers", line=dict(color=theme.PEACH, width=3), name="Total cost")
        fig.add_vline(x=b["threshold"], line=dict(color=theme.GREEN, dash="dash"), annotation_text=f"lowest cost at {b['threshold']:.2f}")
        fig.update_layout(xaxis_title="Flag the order if predicted late risk ≥", yaxis_title="Cost on test orders (AED)")
        charts.show(fig, "val_cost", d, height=320)
        cards.explanation("Assumed costs: a flag (priority courier, message) ≈ AED 8; a missed late order ≈ AED 35 of lost repeat profit.",
                          f"The cheapest threshold is {b['threshold']:.2f}, flagging {C.pct(b['flagged_share'], 0)} of orders. The curve is flat "
                          "near the bottom, so anything from about 0.15 to 0.30 costs nearly the same.",
                          "Thresholds come from business costs, not the default 0.5. The Green refund cut-off (page 6) is set the same way, "
                          "from the 0.5% fraud budget.",
                          "Cost figures are assumptions; finance should replace them, and the chart updates.")


def _fairness() -> None:
    fa = D.table("p2_fairness")
    with cards.card("Are honest customers held back more in some groups?", "Refund tiers for returns that PASSED inspection"):
        c1, c2 = st.columns([1.1, 1])
        with c1:
            g = fa.copy()
            g["label"] = g["group_type"] + ": " + g["group"]
            g = g.sort_values(["group_type", "honest_sent_red"])
            fig = charts.bar(g["label"], g["honest_sent_red"] * 100, g["returns"], g["returns"], horizontal=True, colour=theme.RED,
                             axis_title="% of honest returns sent to Red (inspect first)", hover_label="Sent to Red %")
            charts.show(fig, "val_fair", g, height=420)
        with c2:
            charts.table(fa, "val_fair_t", "fairness.csv", column_config={
                c: st.column_config.ProgressColumn(c.replace("_", " ").capitalize(), min_value=0, max_value=1, format="%.2f")
                for c in ["green_share", "red_share", "honest_held_back", "honest_sent_red", "actual_fail_rate"]})
        new = fa[fa["group"].str.startswith("New")].iloc[0]
        old = fa[fa["group"].str.startswith("Established")].iloc[0]
        cards.explanation("For honest customers only (item passed QC): how many were not refunded instantly, and how many were "
                          "sent to the full-inspection Red tier.",
                          f"Emirates are treated alike. New accounts are not: {C.pct(new['honest_sent_red'], 0)} of honest new-account returns go to Red "
                          f"vs {C.pct(old['honest_sent_red'], 0)} for established accounts, and none reach Green. Account age is a real "
                          f"risk signal (new accounts fail {C.pct(new['actual_fail_rate'], 0)} of the time) but it also penalises genuine new customers.",
                          "Red never means 'refuse', only 'inspect first'. Add a fast lane for new accounts that upload photos, review this table "
                          "monthly, and never use nationality, gender or age band as inputs (they are not used).",
                          "Synthetic data; with real data this check must be repeated by the risk team.")


def _tests() -> None:
    ab, bt = D.table("val_ab_bootstrap"), D.table("val_forecast_backtest")
    ab0 = D.table("ab_tests")
    c1, c2 = st.columns(2)
    with c1, cards.card("A/B tests: z-test vs bootstrap", "Do two different methods agree?"):
        rows = []
        for i, r in ab.iterrows():
            z = ab0.iloc[i]
            rows.append({"test": r["test"], "uplift (pts)": f"{r['uplift_pts']:+.1f}",
                         "z-test 95% CI": f"{z['ci_low_pts']:.1f} to {z['ci_high_pts']:.1f}",
                         "bootstrap 95% CI": f"{r['boot_ci_low']:.1f} to {r['boot_ci_high']:.1f}", "p-value": f"{z['p_value']:.3f}"})
        st.markdown(cards.html_table(rows), unsafe_allow_html=True)
        st.caption("Both intervals exclude zero for both tests, so the effects are unlikely to be luck. They are wide (roughly 1–11 points), "
                   "which is why the business case uses the lower end as well as the middle.")
    with c2, cards.card("Forecast back-test: 4 start dates × 28 days ahead"):
        bt = bt.assign(origin=pd.to_datetime(bt["origin"]).dt.strftime("from %d %b"))
        piv = bt.pivot(index="origin", columns="model", values="mape").reindex(bt["origin"].unique()).reset_index()
        fig = charts.grouped_bar(piv, "origin",
                                 ["Seasonal ARIMA", "Time-series regression", "Repeat last week"], y_title="MAPE (lower is better)",
                                 value_fmt="{:.1%}", colours=[theme.BLUE, theme.GOLD, theme.SLATE])
        charts.show(fig, "val_backtest", bt, height=320)
        mean = bt.groupby("model")["mape"].mean()
        st.caption(f"Average error: SARIMA {C.pct(mean.get('Seasonal ARIMA', np.nan))}, regression {C.pct(mean.get('Time-series regression', np.nan))}, "
                   f"repeat-last-week {C.pct(mean.get('Repeat last week', np.nan))}. 'Repeat last week' is close in quiet weeks but fails badly "
                   "after the outage and around events; the two models trade places, so both are kept.")


def _response() -> None:
    co = D.table("sim_response_models")
    names = {"p1_rep": "Problem 1 · bought again in 60 days", "p1_ticket": "Problem 1 · complaint ticket",
             "p2_rep": "Problem 2 · bought again in 90 days", "p2_ticket": "Problem 2 · refund-chase ticket",
             "p3_churn": "Problem 3 · churn in 90 days", "p3_csat": "Problem 3 · CSAT score (linear)"}
    pick = st.selectbox("Response model", list(names), format_func=names.get, key="val_resp")
    d = co[(co["model"] == pick) & (co["feature"] != "(intercept)")].copy()
    with cards.card("What the simulator learned from the data", "Effect of each input, with its 80% bootstrap range"):
        fig = go.Figure()
        fig.add_bar(y=d["feature"], x=d["coef"], orientation="h", marker_color=[theme.GREEN if v > 0 else theme.RED for v in d["coef"]],
                    error_x=dict(type="data", symmetric=False, array=d["high"] - d["coef"], arrayminus=d["coef"] - d["low"], color=theme.SLATE))
        fig.update_layout(xaxis_title="Coefficient (log-odds; points of CSAT for the linear model)", yaxis=dict(autorange="reversed"))
        charts.show(fig, "val_resp_chart", d, height=340)
        cards.explanation("The simulator (page 8) does not use the formulas that generated the data. It uses these small models, fitted on "
                          f"the recorded data ({int(d['n'].iloc[0]):,} rows), and refits them 100 times on resampled data to show uncertainty.",
                          "Signs match common sense: lateness and rejected refunds lower repeat buying; updates, instant refunds and "
                          "membership raise it; extra contacts and slow resolution raise churn.",
                          "Every simulated gain on page 8 carries an 80% range from these refits.",
                          "Credit and instant-refund effects come from randomised tests (causal). The rest are associations in "
                          "observational data and may include confounding — the real A/B test is the final word.")
