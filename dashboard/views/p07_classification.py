"""Page 7 — Classification results (reads prepared model outputs; no training here)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.components import cards, charts
from dashboard.components import stats as S
from dashboard.config import theme
from dashboard.config.schema import LEAKAGE_FIELDS
from dashboard.data.mapping import normalise_name
from dashboard.views import common

esc = cards.esc
POS = {"1", "1.0", "true", "yes", "high", "high_intent", "interested", "positive", "booked"}
NEG = {"0", "0.0", "false", "no", "low", "not_interested", "negative", "not_booked"}
METRIC_COLS = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]


def to01(s: pd.Series) -> pd.Series:
    def m(v):
        if pd.isna(v):
            return np.nan
        t = normalise_name(v)
        if t in POS:
            return 1.0
        if t in NEG:
            return 0.0
        try:
            x = float(v)
            return 1.0 if x >= 0.5 else 0.0
        except (TypeError, ValueError):
            return np.nan
    return s.map(m).astype(float)


def _target_kind(v) -> str:
    t = str(v).lower()
    return "actual_booking" if any(w in t for w in ("book", "actual", "pilot", "conversion", "transaction")) else "intent"


def render(ctx) -> None:
    mr, pr = ctx.r("classification_metrics"), ctx.r("classification_predictions")
    cards.page_header("Classification — predicting customer interest",
                      "How well can pre-intent traveller characteristics predict high stated booking intent?",
                      "Group-stage analytics · classification", [mr, pr])
    cards.text_card("Target definition",
                    "<p><b>High_Intent = 1</b> when booking_intent_score (Q23) is 8–10; <b>0</b> when 0–7.</p>"
                    "<p>This is <b>stated purchase intent from a survey, not actual conversion</b>. Once pilot bookings "
                    "exist, an actual-booking target should replace or supplement it and be evaluated separately.</p>"
                    "<p>To avoid leakage, Q23 itself and the later questions Q24–Q30 (first service, pricing, WTP, "
                    "features, barriers, pilot interest) must not be used as predictors.</p>", tint="blue")
    if not mr.ok and not pr.ok:
        cards.missing_dataset(mr, "Model comparison needs the saved metrics from the classification pipeline.")
        cards.missing_dataset(pr, "Confusion matrices and actual-vs-predicted need saved predictions.")
        cards.notice("pending", "Classification results are pending.",
                     "This page does not train models. Run the existing classification pipeline and save its outputs "
                     "to <code>data/</code>.")
        return
    _leakage(mr)
    targets = ["intent"]
    if pr.ok and "target" in pr.df:
        targets = sorted({_target_kind(v) for v in pr.df["target"].dropna().unique()}) or ["intent"]
    elif mr.ok and "target" in mr.df:
        targets = sorted({_target_kind(v) for v in mr.df["target"].dropna().unique()}) or ["intent"]
    tab_names = {"intent": "Stated intent (High_Intent)", "actual_booking": "Actual pilot booking"}
    tabs = st.tabs([tab_names[t] for t in targets])
    for t, tab in zip(targets, tabs):
        with tab:
            if t == "actual_booking":
                cards.notice("info", "Actual-booking target — evaluated separately from survey intent.")
            _metrics(ctx, mr, t)
            _predictions(ctx, pr, t)


def _subset_target(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    if "target" not in df:
        return df if kind == "intent" else df.iloc[0:0]
    return df[df["target"].map(_target_kind) == kind]


def _leakage(mr) -> None:
    if not mr.ok:
        return
    fcols = [c for c in mr.df.columns if c in ("features", "features_used", "predictors")]
    if not fcols:
        cards.notice("warn", "Feature list not supplied with the metrics.",
                     "Add a <code>features</code> column (semicolon-separated) to classification_metrics.csv so the "
                     "dashboard can confirm Q23–Q30 were excluded. Until then, leakage cannot be checked here.")
        return
    leaks = set()
    for v in mr.df[fcols[0]].dropna().astype(str):
        for f in v.replace(",", ";").split(";"):
            n = normalise_name(f)
            if any(n == l or n.startswith(l + "_") for l in LEAKAGE_FIELDS):
                leaks.add(f.strip())
    if leaks:
        cards.notice("error", "Possible target leakage detected in the supplied feature list.",
                     "Features derived from Q23–Q30 were listed: " + ", ".join(f"<code>{esc(x)}</code>" for x in sorted(leaks))
                     + ". Metrics may be inflated — review the pipeline.")
    else:
        cards.notice("ok", "Leakage check passed on the supplied feature list.",
                     "No Q23–Q30 fields were found among the listed predictors.")


def _metrics(ctx, mr, kind) -> None:
    with cards.card("Model comparison", "Which model best separates high- from lower-intent respondents?"):
        if not mr.ok:
            cards.missing_dataset(mr)
            return
        m = _subset_target(mr.df, kind).copy()
        if m.empty:
            cards.notice("pending", "No metrics for this target.")
            return
        if "split" in m and m["split"].nunique() > 1:
            splits = list(m["split"].dropna().astype(str).unique())
            sp = st.segmented_control("Data split", splits, default="test" if "test" in splits else splits[0],
                                      key=f"cls_split_{kind}") or splits[0]
            m = m[m["split"].astype(str) == sp]
        present = [c for c in METRIC_COLS if c in m and pd.to_numeric(m[c], errors="coerce").notna().any()]
        if not present:
            cards.missing_columns(METRIC_COLS, "No recognised metric columns.")
            return
        for c in present:
            m[c] = pd.to_numeric(m[c], errors="coerce")
        names = {"accuracy": "Accuracy", "precision": "Precision", "recall": "Recall", "f1": "F1", "roc_auc": "ROC-AUC",
                 "pr_auc": "PR-AUC"}
        fig = charts.grouped_bar(m, "model", present, [names[c] for c in present], y_title="Score (0–1)",
                                 value_fmt="{:.2f}")
        fig.update_yaxes(range=[0, 1.08])
        charts.show(fig, f"cls_metrics_{kind}", m)
        missing = [names[c] for c in METRIC_COLS if c not in present]
        rank_by = "f1" if "f1" in present else present[0]
        best = m.sort_values(rank_by, ascending=False).iloc[0]
        base = ""
        if ctx.consumer is not None and "high_intent" in ctx.consumer and kind == "intent":
            p = S.share(ctx.consumer["high_intent"])[0]
            base = f" Base rate of High_Intent in the full survey: {p:.1f}%, so accuracy of {100 - p:.1f}% is achievable by always predicting 0."
        cards.explanation(
            "Metrics exactly as saved by the pipeline. Precision = share of predicted high-intent who are; recall = share "
            "of actual high-intent found; F1 balances both; ROC-AUC/PR-AUC measure ranking quality across thresholds.",
            esc(f"Best by {names[rank_by]}: {best['model']} ({best[rank_by]:.3f})." + base
                + (f" Not supplied: {', '.join(missing)}." if missing else "")),
            esc("Prefer F1/PR-AUC over accuracy when high-intent respondents are a minority. A model is only useful for "
                "targeting if its precision clearly beats the base rate."),
            common.limitation(ctx, ctx.n_total, "Model metrics are not affected by the dashboard filters. Trained on "
                                                 "survey data, so they predict stated intent only."))


def _predictions(ctx, pr, kind) -> None:
    if not pr.ok:
        cards.missing_dataset(pr, "Confusion matrices and actual-vs-predicted need saved predictions.")
        return
    p = _subset_target(pr.df, kind).copy()
    if p.empty:
        cards.notice("pending", "No predictions for this target.")
        return
    p["y"] = to01(p["actual"])
    p["yhat"] = to01(p["predicted"])
    has_prob = "probability" in p and pd.to_numeric(p["probability"], errors="coerce").notna().any()
    if has_prob:
        p["prob"] = pd.to_numeric(p["probability"], errors="coerce")
    with cards.card("Evaluate a model", help_text="Confusion matrix and report are computed from the saved predictions "
                                                  "file. If the filters are active, only matching respondents are used."):
        models = list(p["model"].dropna().astype(str).unique()) if "model" in p else ["(single model)"]
        c1, c2, c3 = st.columns([1.3, 1, 1.3])
        model = c1.selectbox("Model", models, key=f"cls_model_{kind}")
        if "model" in p:
            p = p[p["model"].astype(str) == model]
        if "split" in p and p["split"].nunique() > 1:
            splits = list(p["split"].dropna().astype(str).unique())
            sp = c2.selectbox("Split", splits, index=splits.index("test") if "test" in splits else 0,
                              key=f"cls_psplit_{kind}")
            p = p[p["split"].astype(str) == sp]
        ids = ctx.filter_ids()
        if ids is not None and "respondent_id" in p:
            p = p[p["respondent_id"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True).isin(ids)]
        thr = 0.5
        if has_prob:
            thr = c3.slider("Decision threshold (what-if)", 0.05, 0.95, 0.5, 0.05, key=f"cls_thr_{kind}",
                            help="0.5 reproduces a standard classifier. Other values show the trade-off between false "
                                 "positives and false negatives; they are not the pipeline's saved predictions.")
            if abs(thr - 0.5) > 1e-9:
                p["yhat"] = (p["prob"] >= thr).astype(float)
                st.caption(f"What-if: predictions re-derived from saved probabilities at threshold {thr:.2f}.")
    p = p.dropna(subset=["y", "yhat"])
    if p.empty:
        cards.notice("warn", "No usable prediction rows (labels could not be read as 0/1 or no rows match filters).")
        return
    y, yh = p["y"].astype(int), p["yhat"].astype(int)
    tp, tn = int(((y == 1) & (yh == 1)).sum()), int(((y == 0) & (yh == 0)).sum())
    fp, fn = int(((y == 0) & (yh == 1)).sum()), int(((y == 1) & (yh == 0)).sum())
    n = len(p)
    c1, c2 = st.columns([1, 1], gap="medium")
    with c1:
        with cards.card("Confusion matrix", "Where does the model get it right and wrong?"):
            z = np.array([[tn, fp], [fn, tp]])
            txt = [[f"{v:,}<br>({v / n * 100:.1f}%)" for v in row] for row in z]
            fig = go.Figure(go.Heatmap(z=z, x=["Predicted 0 (not high intent)", "Predicted 1 (high intent)"],
                                       y=["Actual 0", "Actual 1"], text=txt, texttemplate="%{text}",
                                       colorscale=[[0, "#F4F8FC"], [1, theme.BLUE]], showscale=False,
                                       hovertemplate="%{y} · %{x}<br>%{z:,}<extra></extra>", xgap=3, ygap=3))
            fig.update_layout(yaxis=dict(autorange="reversed"), height=320)
            charts.show(fig, f"cls_cm_{kind}_{model}")
            cards.kpi_grid([cards.kpi("True positives", f"{tp:,}", "correctly identified interested", accent=theme.GREEN),
                            cards.kpi("True negatives", f"{tn:,}", "correctly identified not interested", accent=theme.GREEN),
                            cards.kpi("False positives", f"{fp:,}", "predicted interested, stated not", accent=theme.PEACH),
                            cards.kpi("False negatives", f"{fn:,}", "predicted not, stated interested", accent=theme.RED)],
                           per_row=2)
    with c2:
        with cards.card("Classification report", "How does performance differ by class?"):
            rows = []
            for cls, tpp, fpp, fnn in ((0, tn, fn, fp), (1, tp, fp, fn)):
                prec = tpp / (tpp + fpp) if tpp + fpp else np.nan
                rec = tpp / (tpp + fnn) if tpp + fnn else np.nan
                f1 = 2 * prec * rec / (prec + rec) if prec and rec and not np.isnan(prec + rec) else np.nan
                rows.append({"class": "1 · High intent" if cls else "0 · Not high intent", "precision": prec,
                             "recall": rec, "f1": f1, "support": int((y == cls).sum())})
            rep = pd.DataFrame(rows)
            sup = rep["support"].sum()
            rep = pd.concat([rep, pd.DataFrame([
                {"class": "accuracy", "precision": np.nan, "recall": np.nan, "f1": (tp + tn) / n, "support": n},
                {"class": "macro avg", "precision": rep["precision"].mean(), "recall": rep["recall"].mean(),
                 "f1": rep["f1"].mean(), "support": sup},
                {"class": "weighted avg", **{k: float((rep[k] * rep["support"]).sum() / sup) for k in ("precision", "recall", "f1")},
                 "support": sup}])], ignore_index=True)
            rep_show = rep.copy()
            for c in ("precision", "recall", "f1"):
                rep_show[c] = rep_show[c].map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
            charts.table(rep_show, key=f"cls_report_{kind}_{model}", filename="classification_report.csv")
            st.caption(f"n = {n:,} prediction rows" + (" (filtered)" if ctx.filter_ids() is not None else "") + ".")
            if has_prob:
                _curves(p, kind, model)
    with cards.card("What the errors mean for the business"):
        cards.explanation(
            "Counts from the confusion matrix for the selected model and threshold.",
            esc(f"{fp:,} false positives and {fn:,} false negatives out of {n:,}. Precision for high intent: "
                f"{(tp / (tp + fp) * 100) if tp + fp else float('nan'):.1f}%; recall: "
                f"{(tp / (tp + fn) * 100) if tp + fn else float('nan'):.1f}%."),
            esc("False positive → marketing spend, discounts or partner capacity aimed at people who are not actually "
                "keen; overstated demand forecasts. False negative → interested travellers left out of campaigns; "
                "under-estimated demand in a zone. If outreach is cheap (e.g. in-app prompts), favour recall; if it is "
                "expensive (e.g. paid acquisition, reserved partner space), favour precision."),
            esc("Labels are stated survey intent. A 'true positive' is someone who said 8–10, not someone who booked. "
                "Replace with pilot booking outcomes when available."))
    with cards.card("Actual vs predicted (respondent level)"):
        show = p[[c for c in ("respondent_id", "model", "split", "actual", "predicted", "probability") if c in p]].copy()
        show["actual_01"], show["predicted_01"] = y.values, yh.values
        show["outcome"] = np.select([(y == 1) & (yh == 1), (y == 0) & (yh == 0), (y == 0) & (yh == 1)],
                                    ["TP", "TN", "FP"], default="FN")
        charts.table(show, key=f"cls_rows_{kind}_{model}", filename=f"predictions_{model}.csv", height=320)


def _curve_points(y: np.ndarray, s: np.ndarray):
    order = np.argsort(-s)
    y, s = y[order], s[order]
    tps = np.cumsum(y)
    fps = np.cumsum(1 - y)
    P, N = y.sum(), len(y) - y.sum()
    tpr = np.concatenate([[0], tps / P]) if P else None
    fpr = np.concatenate([[0], fps / N]) if N else None
    prec = tps / np.maximum(tps + fps, 1)
    rec = tps / P if P else None
    return fpr, tpr, prec, rec


def _curves(p, kind, model):
    d = p.dropna(subset=["prob"])
    y = d["y"].astype(int).to_numpy()
    s = d["prob"].to_numpy(dtype=float)
    if len(np.unique(y)) < 2:
        st.caption("ROC/PR curves need both classes present.")
        return
    fpr, tpr, prec, rec = _curve_points(y, s)
    auc = float(np.trapezoid(tpr, fpr)) if hasattr(np, "trapezoid") else float(np.trapz(tpr, fpr))
    ap = float(np.sum(np.diff(np.concatenate([[0], rec])) * prec))
    fig = go.Figure()
    fig.add_scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC {auc:.3f})", line=dict(color=theme.BLUE, width=2))
    fig.add_scatter(x=rec, y=prec, mode="lines", name=f"Precision–recall (AP {ap:.3f})",
                    line=dict(color=theme.GREEN, width=2))
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color=theme.MUTED, dash="dot", width=1))
    fig.update_layout(xaxis_title="FPR (ROC) · Recall (PR)", yaxis_title="TPR (ROC) · Precision (PR)", height=300,
                      xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1.02]))
    charts.show(fig, f"cls_curves_{kind}_{model}")
    st.caption("Curves and AUC/AP here are recomputed from the saved probabilities (for the filtered rows); the "
               "pipeline's own metrics are in the comparison chart above.")
