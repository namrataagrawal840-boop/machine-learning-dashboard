"""Small building blocks reused across pages: model scorecards, ROC curves, rules tables."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from noonbi import cards, charts, theme

esc = cards.esc


def pct(x, d: int = 1) -> str:
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.{d}f}%"


def num(x, d: int = 0) -> str:
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:,.{d}f}"


def aed(x, d: int = 0) -> str:
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"AED {x:,.{d}f}"


def metrics_table(m: pd.DataFrame, key: str, highlight: str = "roc_auc") -> None:
    show = m[["model", "accuracy", "precision", "recall", "f1", "roc_auc"]].copy()
    best = show[highlight].idxmax()
    show.insert(0, "", ["★ best" if i == best else "" for i in show.index])
    cfg = {c: st.column_config.ProgressColumn(c.upper() if c == "f1" else c.replace("_", " ").title(), min_value=0,
                                              max_value=1, format="%.3f") for c in ["accuracy", "precision", "recall", "f1", "roc_auc"]}
    charts.table(show, key=key, filename=f"{key}.csv", column_config=cfg)


def roc_chart(roc: pd.DataFrame, metrics: pd.DataFrame, key: str) -> None:
    fig = go.Figure()
    for i, (name, d) in enumerate(roc.groupby("model", sort=False)):
        auc = metrics.loc[metrics["model"] == name, "roc_auc"]
        lab = f"{name} (AUC {auc.iloc[0]:.2f})" if len(auc) else name
        fig.add_scatter(x=d["fpr"], y=d["tpr"], mode="lines", name=lab, line=dict(width=2.2, color=theme.CATEGORICAL[i]))
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random guess", line=dict(dash="dot", color=theme.SLATE, width=1))
    fig.update_layout(xaxis_title="False alarms (false-positive rate)", yaxis_title="Caught (true-positive rate)",
                      xaxis_range=[0, 1], yaxis_range=[0, 1.02])
    charts.show(fig, key, roc, height=360)


def importance_chart(imp: pd.DataFrame, key: str, top: int = 12, colour: str = theme.BLUE) -> None:
    d = imp.sort_values("importance", ascending=False).head(top).copy()
    d["share"] = d["importance"] / imp["importance"].sum() * 100
    fig = charts.bar(d["feature"], d["share"], horizontal=True, colour=colour, axis_title="Share of model importance (%)",
                     hover_label="Importance %")
    charts.show(fig, key, d)


def confusion_chart(cm: pd.DataFrame, labels=("On time", "Late"), key: str = "cm") -> None:
    r = cm.iloc[0]
    mat = pd.DataFrame([[r["true_neg"], r["false_pos"]], [r["false_neg"], r["true_pos"]]],
                       index=[f"Actual: {labels[0]}", f"Actual: {labels[1]}"],
                       columns=[f"Predicted: {labels[0]}", f"Predicted: {labels[1]}"])
    fig = charts.heatmap(mat, text_fmt="{:,.0f}", value_name="Orders", colorbar_title="Count")
    fig.update_layout(height=300)
    charts.show(fig, key, mat.reset_index().rename(columns={"index": "actual"}))


def rules_table(rules: pd.DataFrame, key: str, top: int = 12) -> None:
    d = rules.head(top)[["if", "then", "support", "confidence", "lift"]].copy()
    cfg = {"support": st.column_config.NumberColumn("Support", format="%.3f", help="How often the whole pattern appears"),
           "confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1, format="%.2f",
                                                          help="When the IF happens, how often the THEN follows"),
           "lift": st.column_config.NumberColumn("Lift", format="%.2f", help="How many times stronger than chance (1 = no link)"),
           "if": st.column_config.TextColumn("IF these happen together", width="large"),
           "then": st.column_config.TextColumn("THEN")}
    charts.table(d, key=key, filename=f"{key}.csv", column_config=cfg)


def rules_chart(rules: pd.DataFrame, key: str, top: int = 12) -> None:
    d = rules.head(top).copy()
    fig = go.Figure(go.Scatter(x=d["support"] * 100, y=d["confidence"] * 100, mode="markers",
                               marker=dict(size=d["lift"] * 9, color=d["lift"], colorscale=[[0, theme.SOFT_BLUE], [1, theme.BLUE]],
                                           line=dict(width=1, color=theme.BORDER), showscale=True,
                                           colorbar=dict(title="Lift", thickness=10)),
                               text=d["if"], hovertemplate="%{text}<br>Support %{x:.1f}% · Confidence %{y:.0f}%<extra></extra>"))
    fig.update_layout(xaxis_title="Support (% of all records)", yaxis_title="Confidence (%)")
    charts.show(fig, key, d[["if", "then", "support", "confidence", "lift"]], height=340)


def k_chart(k: pd.DataFrame, chosen: int, key: str) -> None:
    fig = go.Figure()
    fig.add_bar(x=k["k"].astype(str), y=k["silhouette"], marker_color=[theme.BLUE if v == chosen else theme.SOFT_BLUE
                                                                        for v in k["k"]], name="Silhouette",
                text=[f"{v:.2f}" for v in k["silhouette"]], textposition="outside", cliponaxis=False)
    fig.update_layout(xaxis_title="Number of groups (k)", yaxis_title="Silhouette score (higher = better separated)",
                      yaxis_range=[0, max(k["silhouette"]) * 1.25], showlegend=False)
    charts.show(fig, key, k, height=280)


def ab_card(ab: pd.Series, key: str, guardrail: str = "") -> None:
    fig = go.Figure()
    fig.add_bar(x=["Control (today)", "Treatment (new)"], y=[ab["control_rate"] * 100, ab["treatment_rate"] * 100],
                marker_color=[theme.SLATE, theme.GREEN], text=[pct(ab["control_rate"]), pct(ab["treatment_rate"])],
                textposition="outside", cliponaxis=False,
                customdata=[[int(ab["control_n"])], [int(ab["treatment_n"])]], hovertemplate="%{x}<br>%{y:.1f}%<br>n = %{customdata[0]:,}<extra></extra>")
    fig.update_layout(yaxis_title=f"% {ab['metric'].lower()}", yaxis_range=[0, max(ab["control_rate"], ab["treatment_rate"]) * 125])
    c1, c2 = st.columns([1.2, 1])
    with c1:
        charts.show(fig, key, pd.DataFrame([ab]), height=300)
    with c2:
        sig = ab["p_value"] < 0.05
        cards.kpi_grid([cards.kpi("Uplift", f"+{ab['uplift_pts']:.1f} pts", f"95% CI {ab['ci_low_pts']:.1f} to {ab['ci_high_pts']:.1f} pts",
                                  "derived", theme.GREEN),
                        cards.kpi("p-value", f"{ab['p_value']:.3f}", "Significant at 5%" if sig else "Not significant at 5%",
                                  "derived", theme.BLUE if sig else theme.PEACH)], per_row=2)
        cards.notice("ok" if sig else "warn", "Reading the test:",
                     f"customers were split at random, so the only difference between the groups is the new process. "
                     f"Control n = {int(ab['control_n']):,}, treatment n = {int(ab['treatment_n']):,}. {guardrail}")
