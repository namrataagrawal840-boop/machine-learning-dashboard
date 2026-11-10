"""Page 4 — customer foundations shared by all three solutions (blogs 2–5)."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from noonbi import cards, charts, theme
from noonbi import data as D
from noonbi.views import common as C


def render(f: D.Filters) -> None:
    cards.page_header("Customer foundations: segments, lifetime value, churn, next purchase",
                      "The three solutions all need to know who a customer is worth keeping and who is about to leave. "
                      "These models are built once and shared.",
                      "Blogs 2–5 · clustering, regression, classification", [("Customers", "synthetic"), ("Models", "derived")])
    S = D.summary().get("customers", {})
    seg, segk, samp = D.table("p2_segments"), D.table("p2_cluster_k"), D.table("p2_segment_sample")
    clv, clvt, clvs = D.table("cust_clv_metrics"), D.table("cust_clv_tiers"), D.table("cust_clv_sample")
    ch, chroc, chimp = D.table("cust_churn_metrics"), D.table("cust_churn_roc"), D.table("cust_churn_importance")
    npd = D.table("cust_npd_confusion")
    best_clv = clv.sort_values("r2").iloc[-1]
    best_ch = ch.sort_values("roc_auc").iloc[-1]
    cards.kpi_grid([
        cards.kpi("Customer segments found", f"{len(seg)}", "K-Means on recency, frequency, spend and returns", "derived", theme.LAVENDER),
        cards.kpi("CLV model accuracy (R²)", f"{best_clv['r2']:.2f}", f"{best_clv['model']}; average error {C.aed(best_clv['mae_aed'])}",
                  "derived", theme.GREEN),
        cards.kpi("Churn model ROC-AUC", f"{best_ch['roc_auc']:.2f}", f"{best_ch['model']}; 0.5 = guessing, 1 = perfect", "derived", theme.BLUE),
        cards.kpi("Next-purchase window accuracy", C.pct(S.get("npd_accuracy", float("nan")), 0),
                  f"vs {C.pct(S.get('npd_baseline', float('nan')), 0)} by always guessing the most common window", "derived", theme.GOLD),
    ])
    tabs = st.tabs(["Segments (Blog 2)", "Lifetime value (Blog 3)", "Churn (Blog 4)", "Next purchase (Blog 5)"])
    with tabs[0]:
        c1, c2 = st.columns([1.3, 1])
        with c1, cards.card("Customer segments", "Which groups of customers behave alike?"):
            fig = go.Figure()
            for i, (s, d) in enumerate(samp.groupby("segment")):
                fig.add_scattergl(x=d["recency_days"], y=d["spend_12m_aed"], mode="markers", name=s,
                                  marker=dict(size=6, opacity=0.6, color=theme.CATEGORICAL[i]),
                                  hovertemplate=f"{s}<br>Last order %{{x}} days ago<br>Spend AED %{{y:,.0f}}<extra></extra>")
            fig.update_layout(xaxis_title="Days since last order (recency)", yaxis_title="Spend in last 12 months (AED)", yaxis_type="log")
            charts.show(fig, "cu_seg_scatter", samp, height=400)
        with c2, cards.card("Choosing the number of groups"):
            C.k_chart(segk, 5, "cu_seg_k")
            st.caption("The silhouette score checks how clearly the groups are separated. Five groups were kept because "
                       "the score is close to the best and every group has a clear business meaning.")
        with cards.card("Segment profiles"):
            show = seg[["segment", "customers", "recency_days", "orders_12m", "spend_12m_aed", "return_rate", "rejected_refunds",
                        "clv", "churn"]].copy()
            charts.table(show, "cu_seg_profile", "segments.csv", column_config={
                "recency_days": st.column_config.NumberColumn("Days since last order", format="%.0f"),
                "orders_12m": st.column_config.NumberColumn("Orders / yr", format="%.1f"),
                "spend_12m_aed": st.column_config.NumberColumn("Spend / yr (AED)", format="%.0f"),
                "return_rate": st.column_config.NumberColumn("Returns per order", format="%.2f"),
                "rejected_refunds": st.column_config.NumberColumn("Rejected refunds", format="%.2f"),
                "clv": st.column_config.NumberColumn("Next-12m profit (AED)", format="%.0f"),
                "churn": st.column_config.ProgressColumn("Churn next 90d", min_value=0, max_value=1, format="%.2f")})
            cards.explanation("K-Means groups customers by how recently, how often and how much they buy (RFM), plus how often "
                              "they return and how many refunds were rejected. Nobody tells it the groups.",
                              "Loyal low-returners bring the most profit and rarely churn. A small group of disputed / high-risk "
                              "returners has many rejected refunds and high churn. Lapsing customers have not bought for months.",
                              "Each segment gets a different policy: instant refunds and fast escalation for loyal customers, "
                              "full checks for high-risk returners, win-back timing for lapsing customers.",
                              "Segment names are the group's reading of each cluster's averages; synthetic data.")
    with tabs[1]:
        c1, c2 = st.columns(2)
        with c1, cards.card("Predicted vs actual next-12-month profit", "Can we tell in advance who is worth keeping?"):
            fig = charts.scatter(clvs, "actual", "predicted", x_title="Actual profit next 12 months (AED)",
                                 y_title="Predicted (AED)", diagonal=True, opacity=0.35)
            charts.show(fig, "cu_clv_scatter", clvs, height=360)
        with c2, cards.card("CLV tiers"):
            charts.table(clv, "cu_clv_m", "clv_metrics.csv")
            fig = C.charts.grouped_bar(clvt, "clv_tier", ["avg_pred_clv"], ["Average predicted CLV (AED)"], y_title="AED",
                                       value_fmt="{:,.0f}")
            charts.show(fig, "cu_clv_tiers", clvt, height=260)
        cards.explanation("Regression predicts a number: the profit each customer will bring in the next 12 months, from what "
                          "they did in the last 12 (orders, spend, recency, bad experiences).",
                          "The model explains about three quarters of the differences between customers. Points close to the "
                          "dotted line are predicted well; the biggest spenders are the hardest to predict exactly.",
                          "CLV decides how much noon should spend to keep someone: who gets a credit, an instant refund or a "
                          "senior agent.", "Profit uses an assumed 11% margin; synthetic data.")
    with tabs[2]:
        c1, c2 = st.columns(2)
        with c1, cards.card("Who will stop buying in the next 90 days?"):
            C.metrics_table(ch, "cu_churn_m")
            C.roc_chart(chroc, ch, "cu_churn_roc")
        with c2, cards.card("What drives churn"):
            C.importance_chart(chimp, "cu_churn_imp", colour=theme.ROSE)
        cards.explanation("Classification predicts yes/no: will this customer stop buying? ROC-AUC shows how well the model ranks "
                          "customers from most to least at risk.",
                          "Recency and engagement matter most, but late orders, rejected refunds and repeat contacts also raise "
                          "the risk, which links churn directly to the three problems.",
                          "Customers with a bad experience and high churn risk are the ones the recovery actions should reach first.",
                          "Synthetic data; with real data the drivers' ranking may change.")
    with tabs[3]:
        with cards.card("When will an active customer buy again?"):
            mat = npd.set_index("actual")
            fig = charts.heatmap(mat, text_fmt="{:,.0f}", value_name="Customers", x_title="Predicted window", y_title="Actual window")
            charts.show(fig, "cu_npd", npd, height=300)
            cards.explanation("A random forest puts each active customer into a window: next order within 14 days, 15–45 days, or later.",
                              f"The model is right {C.pct(S.get('npd_accuracy', float('nan')), 0)} of the time, against "
                              f"{C.pct(S.get('npd_baseline', float('nan')), 0)} for always guessing the most common window.",
                              "A recovery message or follow-up works best just before the customer would normally reorder.",
                              "Modest accuracy: use it to time messages, not to make big spending decisions.")
