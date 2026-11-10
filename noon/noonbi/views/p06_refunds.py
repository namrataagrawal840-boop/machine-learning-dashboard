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
    rf = m[m["model"] == "Random forest"].iloc[0]
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
        _tiers(m, tiers)
    with tabs[3]:
        _rules()
    with tabs[4]:
        _time_keepit(S)
    with tabs[5]:
        with cards.card("A/B test: instant refund vs standard refund for low-risk customers"):
            C.ab_card(ab, "p2_ab", f"Guardrail: QC failures in the instant group were {C.pct(ab['guardrail_fail_rate'])} of returns "
                                   f"({C.pct(ab['guardrail_value_share'])} of refund value). The test used a simple rule for "
                                   "'low risk'; the model-based Green tier on tab 3 is stricter, which is how fraud is kept under the cap.")


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


def _tiers(m: pd.DataFrame, tiers: pd.DataFrame) -> None:
    roc, imp, cm, sc = D.table("p2_roc"), D.table("p2_importance"), D.table("p2_confusion"), D.table("p2_test_scores")
    with cards.card("Will this return fail the quality check? (classification)"):
        c1, c2 = st.columns([1.1, 1])
        with c1:
            C.metrics_table(m, "p2_m", "f1")
            C.roc_chart(roc, m, "p2_roc")
        with c2:
            C.importance_chart(imp, "p2_imp", colour=theme.LAVENDER)
            C.confusion_chart(cm, ("Passes QC", "Fails QC"), "p2_cm")
        cards.explanation("Random forest predicts whether a return will fail inspection, from the customer's history, product, "
                          "seller, reason, timing and whether photos were uploaded.",
                          "Prior returns, account age and how fast the item came back matter most. Only about 8% of returns fail, "
                          "so the model is judged on ranking (ROC-AUC) and F1, not accuracy.",
                          "The score turns into a tier. Because random forest shows which factors drove a decision, a rejected "
                          "refund can be explained to the customer.", "Synthetic data.")
    with cards.card("Set the Green-tier cut-off", "How much fraud risk is acceptable for instant refunds?"):
        thr = st.slider("Green if predicted fail risk is below", 0.02, 0.40, 0.15, 0.01, key="p2_thr")
        red = st.slider("Red if predicted fail risk is above", 0.20, 0.90, 0.50, 0.05, key="p2_red")
        g = sc["p_fail"] < thr
        rd = sc["p_fail"] >= red
        green_val = sc.loc[g, "item_value_aed"].sum()
        risk_val = sc.loc[g & (sc["qc_failed"] == 1), "item_value_aed"].sum()
        cards.kpi_grid([
            cards.kpi("Returns refunded instantly (Green)", C.pct(g.mean(), 0), f"{C.num(g.sum())} of {C.num(len(sc))} test returns", "derived", theme.GREEN),
            cards.kpi("Green returns that would fail QC", C.pct(sc.loc[g, "qc_failed"].mean()), "Refunded before inspection", "derived", theme.PEACH),
            cards.kpi("Value at risk in Green", C.pct(risk_val / green_val if green_val else np.nan), "Share of Green refund value", "derived", theme.RED),
            cards.kpi("Red: full check before refund", C.pct(rd.mean(), 0), f"Fail rate {C.pct(sc.loc[rd, 'qc_failed'].mean(), 0)}", "derived", theme.SLATE),
        ])
        st.caption("Not every QC failure is fraud (many are used or incomplete items), so 'value at risk' is an upper bound. Some of it is "
                   "recovered from sellers through the dispute process. Lower the Green cut-off until the risk sits under the agreed cap.")
    with cards.card("Tiers at the default cut-offs (0.15 / 0.50)"):
        charts.table(tiers, "p2_tiers", "trust_tiers.csv", column_config={
            "actual_fail_rate": st.column_config.ProgressColumn("Actual fail rate", min_value=0, max_value=1, format="%.3f"),
            "share": st.column_config.ProgressColumn("Share of returns", min_value=0, max_value=1, format="%.2f"),
            "avg_value": st.column_config.NumberColumn("Avg value (AED)", format="%.0f"),
            "value": st.column_config.NumberColumn("Total value (AED)", format="%.0f"),
            "refund_timing": st.column_config.TextColumn("When the refund arrives", width="large")})


def _rules() -> None:
    rules = D.table("p2_rules")
    with cards.card("What goes with a return? (Apriori)", "Which products, sellers and habits lead to returns?"):
        c1, c2 = st.columns([1.2, 1])
        with c1:
            C.rules_table(rules, "p2_rules", 10)
        with c2:
            C.rules_chart(rules, "p2_rules_chart", 12)
        cards.explanation("Apriori finds 'if this, then that' patterns across all orders, with 'Returned' as the result.",
                          "Ordering the same item in two sizes is followed by a return every time (lift ≈ 6). Fashion from "
                          "poor-quality sellers and late deliveries also raise the chance of a return.",
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
