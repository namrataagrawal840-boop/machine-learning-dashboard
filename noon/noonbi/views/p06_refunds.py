"""Page 6 — Problem 2: complicated returns and refunds → Trust-Tiered Instant Refunds."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from noonbi import cards, charts, theme
from noonbi import data as D
from noonbi.views import common as C


def render(f: D.Filters) -> None:
    cards.page_header("Problem 2 · Complicated returns and slow refunds",
                      "Solution: Trust-Tiered Instant Refunds — trusted customers are refunded when the courier collects the "
                      "item, risky returns are still checked, and return data stops bad products at the source.",
                      "Classification · clustering · association · regression · A/B", [("Returns", "synthetic"), ("Models", "derived")])
    r = f.apply(D.table("returns"), "order_date")
    if r.empty:
        cards.notice("warn", "No returns match the filters.")
        return
    tiers, m, ab = D.table("p2_tiers"), D.table("p2_cls_metrics"), D.table("ab_tests").iloc[1]
    S = D.summary().get("p2", {})
    cards.kpi_grid([
        cards.kpi("Median refund time today", f"{r['refund_days'].median():.0f} days", "Request → money back (passed QC)", "synthetic", theme.GOLD),
        cards.kpi("Returns rejected at quality check", C.pct(1 - r["qc_passed"].mean()), "Most then turn into a complaint", "synthetic", theme.PEACH),
        cards.kpi("Returns that qualify for Green tier", C.pct(tiers.loc[tiers.tier == "Green", "share"].iloc[0], 0),
                  "Refund when the courier collects", "derived", theme.GREEN),
        cards.kpi("Instant-refund test uplift", f"+{ab['uplift_pts']:.1f} pts", f"repeat purchase in 90 days, p = {ab['p_value']:.3f}", "derived", theme.GREEN),
    ])
    st.caption("▶ To watch this solution run record by record, open **8 · Solution Simulator** in the sidebar.")
    tabs = st.tabs(["1 · Evidence", "2 · Who returns (segments)", "3 · Trust tiers", "4 · Fix the source", "5 · Refund time & keep-it", "6 · Test"])
    with tabs[0]:
        _evidence(r)
    with tabs[1]:
        _segments()
    with tabs[2]:
        _tiers(m, tiers, S)
    with tabs[3]:
        _rules()
    with tabs[4]:
        _time_keepit(S)
    with tabs[5]:
        with cards.card("A/B test: instant refund vs standard refund for low-risk customers"):
            C.ab_card(ab, "p2_ab", f"Guardrail: QC failures in the instant group were {C.pct(ab['guardrail_fail_rate'])} of returns "
                                   f"({C.pct(ab['guardrail_value_share'])} of refund value). The test used a simple rule for "
                                   "'low risk'; the model-based Green tier on tab 3 is stricter, which is how fraud is kept under the cap. "
                                   "A bootstrap check of the same uplift is on page 11 · Model Quality.")


def _evidence(r: pd.DataFrame) -> None:
    c1, c2 = st.columns(2)
    with c1, cards.card("Why customers return", "Which return reasons are most common?"):
        fig, d = charts.count_bar(r["return_reason"], horizontal=True, colour=theme.LAVENDER, sort=True)
        fig.update_xaxes(title="% of returns")
        charts.show(fig, "p2_reasons", d)
    with c2, cards.card("Return rate by category"):
        o = D.table("orders")
        g = o.groupby("category").size().rename("orders").to_frame().join(r.groupby("category").size().rename("returns")).fillna(0).reset_index()
        g["rate"] = g["returns"] / g["orders"] * 100
        g = g.sort_values("rate", ascending=False)
        fig = charts.bar(g["category"], g["rate"], g["returns"], g["orders"], horizontal=True, colour=theme.LAVENDER, axis_title="% of orders returned")
        charts.show(fig, "p2_cat", g)
    with cards.card("Slow refunds cost repeat business", "Do customers who wait longer come back less?"):
        ok = r[r["qc_passed"] == 1].copy()
        ok["wait"] = pd.cut(ok["refund_days"], [-0.1, 1, 5, 10, 15, 60], labels=["≤1 day", "2–5 days", "6–10 days", "11–15 days", "16+ days"])
        g = ok.groupby("wait", observed=True).agg(returns=("return_id", "size"), rep=("repurchase_90d", "mean")).reset_index()
        rej = r[r["qc_passed"] == 0]["repurchase_90d"].mean()
        g = pd.concat([g, pd.DataFrame([{"wait": "Refund rejected", "returns": int((r["qc_passed"] == 0).sum()), "rep": rej}])])
        fig = charts.bar(g["wait"].astype(str), g["rep"] * 100, g["returns"], g["returns"], colour=theme.GREEN,
                         axis_title="% bought again within 90 days", hover_label="Repeat %")
        charts.show(fig, "p2_wait", g)
        cards.explanation("Repeat purchase within 90 days, grouped by how long the refund took, plus customers whose refund was rejected.",
                          "The faster the refund, the more likely the customer buys again, and a rejected refund is the worst outcome.",
                          "Speed matters most for good customers, which is why the solution gives instant refunds to the trusted "
                          "group rather than to everyone.", "Observational; the randomised test on tab 6 measures the real effect.")


def _segments() -> None:
    seg, k = D.table("p2_segments"), D.table("p2_cluster_k")
    with cards.card("Customer groups by return behaviour (K-Means on RFM + returns)"):
        fig = go.Figure()
        fig.add_scatter(x=seg["return_rate"], y=seg["clv"], mode="markers+text", text=seg["segment"], textposition="top center",
                        marker=dict(size=np.sqrt(seg["customers"]) * 1.1, color=theme.CATEGORICAL[:len(seg)], opacity=0.75),
                        customdata=seg[["customers", "churn"]],
                        hovertemplate="%{text}<br>Returns per order %{x:.2f}<br>CLV AED %{y:.0f}<br>%{customdata[0]:,} customers<extra></extra>")
        fig.update_layout(xaxis_title="Returns per order", yaxis_title="Average next-12-month profit (AED)", showlegend=False)
        charts.show(fig, "p2_seg", seg, height=380)
        C.k_chart(k, 5, "p2_k")
        cards.explanation("Clustering with recency, frequency, spend, return rate and rejected refunds. Bubble size = number of customers.",
                          "Loyal low-returners are the most valuable and safest. Frequent returners mostly swap sizes. A small "
                          "group with many rejected refunds is where possible abuse sits.",
                          "Clustering lets the data show the groups, including ones we did not expect, instead of guessing rules.",
                          "Synthetic data; segment labels are interpretations of the cluster averages.")


def _tiers(m: pd.DataFrame, tiers: pd.DataFrame, S: dict) -> None:
    roc, imp, cm, sc = D.table("p2_roc"), D.table("p2_importance"), D.table("p2_confusion"), D.table("p2_test_scores")
    sel, cal, cur = D.table("p2_model_selection"), D.table("p2_calibration"), D.table("p2_cutoff_curve")
    g_cut, r_cut = S.get("green_cutoff", 0.04), S.get("red_cutoff", 0.24)
    ci = S.get("auc_ci", [np.nan, np.nan])
    with cards.card("Will this return fail the quality check? (classification)"):
        c1, c2 = st.columns([1.1, 1])
        with c1:
            C.metrics_table(m, "p2_m", "roc_auc")
            C.roc_chart(roc, m, "p2_roc")
        with c2:
            C.importance_chart(imp, "p2_imp", colour=theme.LAVENDER)
            C.confusion_chart(cm, ("Passes QC", "Fails QC"), "p2_cm")
        cards.explanation("Three models predict whether a return will fail inspection, from what noon can see: the customer's return "
                          "history and account age, the product, seller, reason, timing and whether photos were uploaded.",
                          f"Only about {C.pct(sc['qc_failed'].mean(), 0)} of returns fail, so accuracy is misleading (always saying 'pass' "
                          f"scores {C.pct(1 - sc['qc_failed'].mean(), 0)}). The models are judged on ranking: ROC-AUC around "
                          f"{m['roc_auc'].max():.2f}. The confusion matrix is drawn at the Red cut-off ({r_cut:.2f}): the model finds "
                          f"{C.pct(S.get('red_recall', np.nan), 0)} of the failures while checking only {C.pct((sc['p_fail'] >= r_cut).mean(), 0)} of returns.",
                          "The model is a good ranker and a weak yes/no judge, which is exactly how it is used: it decides how FAST to "
                          "refund, never whether to refund. Red means 'inspect first', not 'refuse'.",
                          "The table above uses a fixed 70/30 split; the cross-validated scores below are the fairer comparison.")
    with cards.card("Which model, and can its scores be trusted? (5-fold cross-validation + calibration)"):
        c1, c2 = st.columns([1, 1.1])
        with c1:
            charts.table(sel[["model", "cv_auc_mean", "cv_auc_sd", "chosen"]], "p2_sel", "p2_model_selection.csv", column_config={
                "cv_auc_mean": st.column_config.ProgressColumn("CV ROC-AUC (mean)", min_value=0.5, max_value=1, format="%.3f"),
                "cv_auc_sd": st.column_config.NumberColumn("± sd across folds", format="%.3f"),
                "chosen": st.column_config.CheckboxColumn("Chosen")})
            st.caption(f"Seven model settings were compared on the training returns only. Chosen: **{S.get('chosen_model', '—')}**. "
                       f"Test ROC-AUC {S.get('test_auc_chosen', np.nan):.2f} (95% bootstrap CI {ci[0]:.2f}–{ci[1]:.2f}).")
        with c2:
            fig = go.Figure()
            for i, (v, d) in enumerate(cal.groupby("version", sort=False)):
                fig.add_scatter(x=d["mean_pred"], y=d["observed"], mode="lines+markers", name=f"{v} (Brier {d['brier'].iloc[0]:.3f})",
                                line=dict(color=[theme.SLATE, theme.LAVENDER][i % 2], width=2.4))
            mx = float(max(cal["mean_pred"].max(), cal["observed"].max())) * 1.05
            fig.add_scatter(x=[0, mx], y=[0, mx], mode="lines", name="Perfectly calibrated", line=dict(dash="dot", color=theme.SLATE, width=1))
            fig.update_layout(xaxis_title="Predicted chance of failing QC", yaxis_title="Share that actually failed", xaxis_tickformat=".0%",
                              yaxis_tickformat=".0%", legend=dict(orientation="h", y=-0.38), margin=dict(b=90))
            charts.show(fig, "p2_calib", cal, height=370)
        cards.explanation("Cross-validation trains and tests five times on different slices, so one lucky split cannot flatter a model. "
                          "Calibration checks that a score of 10% really means about 1 in 10 fail.",
                          "The models are close; differences smaller than the ± sd are noise. Points near the dotted line mean the "
                          "scores can be read as real probabilities, which the money-at-risk calculation needs.",
                          "Because the scores are probabilities, the Green cut-off can be set from the fraud budget directly.",
                          f"Calibration kept only if it lowers the out-of-fold Brier score (here: {'kept' if S.get('calibrated') else 'not needed'}).")
    with cards.card("Set the Green-tier cut-off", "How much fraud risk is acceptable for instant refunds?"):
        thr = st.slider("Green if predicted fail risk is below", 0.01, 0.30, float(round(g_cut, 3)), 0.005, key="p2_thr", format="%.3f")
        red = st.slider("Red if predicted fail risk is above", 0.10, 0.60, float(round(r_cut, 2)), 0.01, key="p2_red")
        g = sc["p_fail"] < thr
        rd = sc["p_fail"] >= red
        green_val = sc.loc[g, "item_value_aed"].sum()
        risk_val = sc.loc[g & (sc["qc_failed"] == 1), "item_value_aed"].sum()
        all_val = sc["item_value_aed"].sum()
        cards.kpi_grid([
            cards.kpi("Returns refunded instantly (Green)", C.pct(g.mean(), 0), f"{C.num(g.sum())} of {C.num(len(sc))} test returns", "derived", theme.GREEN),
            cards.kpi("Green returns that would fail QC", C.pct(sc.loc[g, "qc_failed"].mean()), "Refunded before inspection", "derived", theme.PEACH),
            cards.kpi("Money at risk (share of ALL refunds)", C.pct(risk_val / all_val if all_val else np.nan, 2),
                      f"Cap 0.50% · {C.pct(risk_val / green_val if green_val else np.nan)} of Green value", "derived",
                      theme.GREEN if all_val and risk_val / all_val <= 0.005 else theme.RED),
            cards.kpi("Red: full check before refund", C.pct(rd.mean(), 0), f"Fail rate {C.pct(sc.loc[rd, 'qc_failed'].mean(), 0)}", "derived", theme.SLATE),
        ])
        fig = go.Figure()
        fig.add_scatter(x=cur["cutoff"], y=cur["green_share"] * 100, name="Returns refunded instantly (%)", line=dict(color=theme.GREEN, width=3))
        fig.add_scatter(x=cur["cutoff"], y=cur["loss_share_all_refunds"] * 100, name="Money at risk, % of all refunds", yaxis="y2",
                        line=dict(color=theme.RED, width=3))
        fig.add_hline(y=0.5, line=dict(color=theme.RED, dash="dot"), yref="y2", annotation_text="0.5% cap")
        fig.add_vline(x=thr, line=dict(color=theme.SLATE, dash="dash"), annotation_text="your cut-off")
        fig.update_layout(xaxis_title="Green cut-off (predicted fail risk)", yaxis_title="% refunded instantly",
                          yaxis2=dict(title="% of refund value at risk", overlaying="y", side="right", showgrid=False, tickformat=".1f",
                                      rangemode="tozero"),
                          legend=dict(orientation="h", y=-0.3), margin=dict(b=80))
        charts.show(fig, "p2_cutoff_curve", cur, height=330)
        st.caption(f"The default cut-off ({g_cut:.3f}) was chosen on the TRAINING returns (out-of-fold scores) as the highest value that keeps "
                   "money paid early on failed items at or under 0.5% of all refund value; the chart checks it on unseen returns. "
                   "Not every QC failure is fraud, and some value is recovered from sellers, so this is an upper bound.")
    with cards.card(f"Tiers at the default cut-offs ({g_cut:.3f} / {r_cut:.2f})"):
        charts.table(tiers, "p2_tiers", "trust_tiers.csv", column_config={
            "actual_fail_rate": st.column_config.ProgressColumn("Actual fail rate", min_value=0, max_value=1, format="%.3f"),
            "share": st.column_config.ProgressColumn("Share of returns", min_value=0, max_value=1, format="%.2f"),
            "avg_value": st.column_config.NumberColumn("Avg value (AED)", format="%.0f"),
            "value": st.column_config.NumberColumn("Total value (AED)", format="%.0f"),
            "refund_timing": st.column_config.TextColumn("When the refund arrives", width="large")})
        st.caption("Fairness check (new accounts and emirates) is on page 11 · Model Quality & Fairness.")


def _rule(rules, lhs):
    r = rules[rules["if"] == lhs]
    return r.iloc[0] if len(r) else None


def _conf(rules, lhs):
    r = _rule(rules, lhs)
    return np.nan if r is None else r["confidence"]


def _lift(rules, lhs):
    r = _rule(rules, lhs)
    return np.nan if r is None else r["lift"]


def _rules() -> None:
    rules = D.table("p2_rules")
    with cards.card("What goes with a return? (Apriori)", "Which products, sellers and habits lead to returns?"):
        c1, c2 = st.columns([1.2, 1])
        with c1:
            C.rules_table(rules, "p2_rules", 10)
        with c2:
            C.rules_chart(rules, "p2_rules_chart", 12)
        cards.explanation("Apriori finds 'if this, then that' patterns across all orders, with 'Returned' as the result.",
                          f"Ordering the same item in two sizes is followed by a return in {C.pct(_conf(rules, 'Ordered two sizes'), 0)} of cases "
                          f"(lift ≈ {_lift(rules, 'Ordered two sizes'):.1f}). Fashion from poor-quality sellers and late deliveries also raise the chance of a return.",
                          "Show a size guide or fit advice when two sizes go in the basket, and send sellers with high return "
                          "patterns to a quality check. Fewer returns is cheaper than faster refunds.",
                          "Support = how common the pattern is; small-support rules are real but affect few orders.")


def _time_keepit(S: dict) -> None:
    reg, co, k = D.table("p2_refund_reg"), D.table("p2_refund_coefs"), D.table("p2_keep_it")
    c1, c2 = st.columns(2)
    with c1, cards.card("What makes a refund slow? (linear regression)"):
        co = co.sort_values("extra_days")
        fig = charts.bar(co["factor"], co["extra_days"], horizontal=True, colour=theme.GOLD, value_fmt="{:+.1f}",
                         axis_title="Extra days vs baseline (wallet, Fulfilled by noon)", hover_label="Days")
        charts.show(fig, "p2_coef", co)
        rr = reg.iloc[0]
        st.caption(f"Average error {rr['mae_days']:.1f} days, R² {rr['r2']:.2f}. This gives each customer a realistic refund date in the tracker.")
    with c2, cards.card("'Keep it' partial refund", "When is it cheaper to let the customer keep the item?"):
        k = k.sort_values("cost_share_of_value", ascending=False)
        fig = charts.grouped_bar(k, "category", ["avg_value", "avg_cost", "avg_offer"], ["Item value", "Cost to take back", "Keep-it offer"],
                                 y_title="AED", value_fmt="{:,.0f}", horizontal=True, colours=[theme.SLATE, theme.PEACH, theme.GREEN])
        charts.show(fig, "p2_keep", k)
        st.caption(f"Return-cost model R² {S.get('keep_it_r2', float('nan')):.2f}. The offer is 80% of the predicted cost of a pickup, "
                   "inspection and write-down, so both the customer and noon come out ahead.")
    cards.explanation("Regression predicts a number: refund days for the tracker, and the cost of a return for the keep-it offer.",
                      "Card refunds and seller-shipped items add the most days. For low-value groceries and beauty items, taking the "
                      "item back can cost more than the item is worth.",
                      "Wallet refunds by default, a live tracker with a guaranteed date, and a keep-it offer for damaged low-value items.",
                      "Return costs use assumed pickup and handling rates.")
