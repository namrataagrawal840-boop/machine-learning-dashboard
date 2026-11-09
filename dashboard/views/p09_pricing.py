"""Page 9 — Pricing (stated WTP) and regression results."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.components import cards, charts
from dashboard.components import stats as S
from dashboard.config import theme
from dashboard.views import common

esc = cards.esc
TARGETS = {"storage": ("storage_wtp_inr", "Storage WTP — one medium suitcase, 3 hours, verified location ≤ 10 min walk (Q26)"),
           "pickup": ("pickup_delivery_wtp_inr", "Additional pickup/delivery WTP per suitcase, ≤ 5 km (Q27)")}
SEGMENTS = {"Travel purpose": "travel_purpose__label", "Inconvenience (1–5)": "inconvenience_score__label",
            "Travel frequency": "trips_12m__label", "Bags": "bag_count__label", "City": "city__label",
            "Microzone": "microzone__label", "Accommodation": "accommodation_type__label",
            "Transport mode": "travel_mode__label", "Checkout gap": "checkout_gap_hours__label", "Cluster": "cluster",
            "High stated intent": "high_intent"}


def _target_key(v) -> str | None:
    t = str(v).lower()
    if "pick" in t or "deliver" in t or "q27" in t:
        return "pickup"
    if "storage" in t or "q26" in t:
        return "storage"
    return None


def render(ctx) -> None:
    cards.page_header("Pricing & regression — willingness to pay",
                      "What do travellers say they would pay for storage and for pickup/delivery, and what explains it?",
                      "Group-stage analytics · regression (with descriptive pricing)",
                      [ctx.r("consumer_clean"), ctx.r("regression_metrics"), ctx.r("regression_predictions")])
    cards.notice("warn", "Stated willingness to pay is not a transaction price.",
                 "Survey WTP is hypothetical and usually overstates what people pay. Real prices must be tested with "
                 "paid bookings. High WTP does not by itself mean the business is profitable — see the contribution "
                 "calculator on page 10.")
    tabs = st.tabs(["Storage WTP", "Pickup / delivery WTP"])
    for key, tab in zip(TARGETS, tabs):
        with tab:
            col, desc = TARGETS[key]
            cards.text_card(None, f"<b>Target:</b> {esc(desc)}", tint="blue")
            _descriptive(ctx, key, f"{col}__value")
            _regression(ctx, key)


# ------------------------------------------------------------------ descriptive
def _descriptive(ctx, key, col):
    if ctx.consumer is None:
        cards.missing_dataset(ctx.r("consumer_clean"), "WTP descriptives need the consumer survey.")
        return
    if not common.require_consumer(ctx):
        return
    d = ctx.cf
    if col not in d or d[col].dropna().empty:
        cards.missing_columns([col.replace("__value", "")], "")
        return
    desc = S.describe_numeric(d[col])
    flag, lo, hi = S.iqr_outliers(d[col])
    prov = ctx.cprov
    cards.kpi_grid([
        cards.kpi("Valid answers", f"{desc['n']:,}", f"of {ctx.n_filtered:,} filtered respondents", prov),
        cards.kpi("Median", S.fmt_inr(desc["median"]), f"mean {S.fmt_inr(desc['mean'])}", prov, theme.BLUE),
        cards.kpi("Interquartile range", f"{S.fmt_inr(desc['q1'])}–{S.fmt_inr(desc['q3'])}", "middle 50% of answers",
                  prov, theme.GREEN),
        cards.kpi("Zero WTP", f"{desc['zero_share']:.1f}%", f"{int((d[col] == 0).sum()):,} said ₹0 — kept in all figures",
                  prov, theme.PEACH),
        cards.kpi("Maximum", S.fmt_inr(desc["max"]), f"{int(flag.sum()):,} above the 1.5×IQR fence (₹{hi:,.0f}); "
                                                       "flagged, not removed", prov, theme.LAVENDER),
    ], per_row=5)
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        with cards.card("WTP distribution", "How are stated amounts spread?"):
            cap = st.toggle("Zoom to 99th percentile", value=False, key=f"wtp_zoom_{key}",
                            help="Display only; extreme answers stay in all statistics.")
            v = d[col].dropna()
            vv = v[v <= v.quantile(0.99)] if cap else v
            fig = charts.histogram(vv, nbins=40, x_title="₹", colour=theme.BLUE if key == "storage" else theme.GREEN,
                                   marker_lines={"Q1": desc["q1"], "Median": desc["median"], "Q3": desc["q3"]})
            charts.show(fig, f"wtp_hist_{key}", d[["respondent_id", col]].dropna() if "respondent_id" in d else d[[col]].dropna())
            cards.explanation("Histogram of stated maximum WTP with quartiles.",
                              esc(f"Median ₹{desc['median']:,.0f}; mean ₹{desc['mean']:,.0f} "
                                  f"({'right-skewed' if desc['mean'] > desc['median'] * 1.1 else 'roughly symmetric'}); "
                                  f"{desc['zero_share']:.1f}% zero."),
                              esc("Quote medians and quartiles rather than the mean. The zero group needs a different "
                                  "value proposition (or is outside the market)."),
                              common.limitation(ctx, desc["n"], "Legitimate extreme values are retained."))
    with c2:
        with cards.card("Stated acceptance curve", "What share would accept each price?"):
            v = np.sort(d[col].dropna().to_numpy())
            grid = np.unique(np.quantile(v, np.linspace(0, 0.99, 60)))
            share = [(v >= p).mean() * 100 for p in grid]
            data = pd.DataFrame({"price_inr": grid.round(0), "pct_willing": np.round(share, 1)})
            fig = go.Figure(go.Scatter(x=grid, y=share, mode="lines", line=dict(color=theme.BLUE, width=2.5,
                                                                                shape="hv"),
                                       hovertemplate="₹%{x:,.0f}: %{y:.1f}% would pay at least this<extra></extra>"))
            for q, nm in ((desc["q1"], "Q1"), (desc["median"], "Median"), (desc["q3"], "Q3")):
                fig.add_vline(x=q, line_dash="dot", line_color=theme.MUTED, line_width=1, annotation_text=nm,
                              annotation_font_color=theme.MUTED)
            fig.update_layout(xaxis_title="Price (₹)", yaxis_title="% stating WTP ≥ price", yaxis=dict(range=[0, 101]))
            charts.show(fig, f"wtp_curve_{key}", data)
            at_med = (v >= desc["median"]).mean() * 100
            at_q3 = (v >= desc["q3"]).mean() * 100
            cards.explanation("Share of respondents whose stated maximum is at or above each price (a stated demand curve).",
                              esc(f"{at_med:.0f}% state ≥ ₹{desc['median']:,.0f} (median); {at_q3:.0f}% state ≥ "
                                  f"₹{desc['q3']:,.0f} (Q3)."),
                              esc("Steep sections show price points where many respondents drop out — good candidates "
                                  "for A/B price tests (e.g. test arms at Q1, median and Q3)."),
                              common.limitation(ctx, desc["n"], "Ignores substitution with free hotel storage; real "
                                                                 "acceptance will be lower."))
    with cards.card("WTP by traveller segment", "Which groups state higher WTP?"):
        opts = {k: v for k, v in SEGMENTS.items() if v in d and d[v].notna().any()}
        c1, c2 = st.columns([2, 1])
        pick = c1.selectbox("Segment by", list(opts), key=f"wtp_seg_{key}")
        g = opts[pick]
        sub = d[[g, col]].dropna().copy()
        sub[g] = sub[g].astype(str)
        stats = S.group_median(sub, g, col, ctx.min_n)
        hide_small = c2.toggle(f"Hide groups with n < {ctx.min_n}", value=True, key=f"wtp_hide_{key}")
        if hide_small:
            keep = stats.loc[stats["reliable"], g].tolist()
            sub = sub[sub[g].isin(keep)]
        if isinstance(d[opts[pick]].dtype, pd.CategoricalDtype):
            order = [str(c) for c in d[opts[pick]].cat.categories if str(c) in set(sub[g])]
        else:
            order = stats.sort_values("median", ascending=False)[g].astype(str).tolist()
            order = [o for o in order if o in set(sub[g])]
        if sub.empty:
            cards.notice("info", f"No group reaches n ≥ {ctx.min_n}. Lower the minimum in Analysis settings or switch off "
                                 "the toggle.")
            return
        fig = charts.box_by(sub, g, col, order=order, y_title="₹", horizontal=False)
        charts.show(fig, f"wtp_seg_{key}_{g}", stats.round(1))
        rel = stats[stats["reliable"]].sort_values("median", ascending=False)
        obs = (esc(f"Highest median: {rel.iloc[0][g]} (₹{rel.iloc[0]['median']:,.0f}, n={int(rel.iloc[0]['n'])}); lowest: "
                   f"{rel.iloc[-1][g]} (₹{rel.iloc[-1]['median']:,.0f}, n={int(rel.iloc[-1]['n'])}).")
               if len(rel) > 1 else esc(f"Fewer than two groups have n ≥ {ctx.min_n}."))
        cards.explanation(esc(f"WTP distribution within each {pick.lower()} group; table lists n, median, mean and "
                              "quartiles with a reliability flag."), obs,
                          esc("Large, consistent gaps between reliable groups justify segment-specific price tests "
                              "(e.g. multi-bag or business bundles); overlapping boxes suggest one price."),
                          common.limitation(ctx, int(stats["n"].sum()),
                                            "Small groups (city × microzone especially) are unreliable."))


# ------------------------------------------------------------------ regression
def _regression(ctx, key):
    mr, pr, cr = ctx.r("regression_metrics"), ctx.r("regression_predictions"), ctx.r("regression_coefficients")
    with cards.card("Regression model comparison (Linear · Ridge · Lasso)",
                    "How well do traveller characteristics explain stated WTP?"):
        if not mr.ok and not pr.ok:
            cards.missing_dataset(mr, "Model comparison needs saved metrics from the regression pipeline.")
            cards.missing_dataset(pr, "Actual-vs-predicted and residual plots need saved predictions.")
            cards.notice("pending", "Regression results are pending.", "The dashboard does not fit models.")
            return
        if mr.ok:
            m = mr.df.copy()
            if "target" in m:
                m = m[m["target"].map(_target_key) == key]
            if m.empty:
                cards.notice("pending", "No metrics for this target in regression_metrics.csv.",
                             "Include a <code>target</code> column naming storage or pickup/delivery WTP.")
            else:
                if "split" in m and m["split"].nunique() > 1:
                    splits = list(m["split"].astype(str).unique())
                    sp = st.segmented_control("Split", splits, default="test" if "test" in splits else splits[0],
                                              key=f"reg_split_{key}") or splits[0]
                    m = m[m["split"].astype(str) == sp]
                for c in ("mae", "rmse", "r2"):
                    if c in m:
                        m[c] = pd.to_numeric(m[c], errors="coerce")
                err = [c for c in ("mae", "rmse") if c in m]
                c1, c2 = st.columns(2)
                with c1:
                    if err:
                        fig = charts.grouped_bar(m, "model", err, [c.upper() for c in err], y_title="₹ (lower is better)",
                                                 value_fmt="₹{:,.0f}", colours=[theme.BLUE, theme.LAVENDER])
                        charts.show(fig, f"reg_err_{key}", m)
                with c2:
                    if "r2" in m:
                        fig = charts.bar(m["model"], m["r2"], value_fmt="{:.3f}", colour=theme.GREEN,
                                         axis_title="R² (higher is better)", hover_label="R²")
                        charts.show(fig, f"reg_r2_{key}")
                best = m.sort_values("rmse" if "rmse" in m else "mae").iloc[0] if err else m.iloc[0]
                med = ctx.consumer[f"{TARGETS[key][0]}__value"].median() if ctx.consumer is not None and \
                    f"{TARGETS[key][0]}__value" in ctx.consumer else np.nan
                cards.explanation(
                    "Metrics as saved by the pipeline. MAE = average absolute error in ₹; RMSE penalises large errors; "
                    "R² = share of variance explained.",
                    esc(f"Lowest error: {best['model']}" + (f" (RMSE ₹{best['rmse']:,.0f}" if "rmse" in m else "")
                        + (f", MAE ₹{best['mae']:,.0f}" if "mae" in m else "") + (f", R² {best['r2']:.3f})" if "r2" in m
                                                                                  else ")")
                        + (f". For scale, median stated WTP is ₹{med:,.0f}." if not np.isnan(med) else ".")),
                    esc("Low R² is normal for WTP — it means price sensitivity is mostly individual. Use the model to "
                        "understand drivers (coefficients), not to set individual prices."),
                    esc("Pipeline outputs on survey data; not affected by dashboard filters. Ridge/Lasso shrink "
                        "coefficients, so compare them only within the same scaling."))
    if pr.ok:
        p = pr.df.copy()
        if "target" in p:
            p = p[p["target"].map(_target_key) == key]
        p["actual"] = pd.to_numeric(p["actual"], errors="coerce")
        p["predicted"] = pd.to_numeric(p["predicted"], errors="coerce")
        p = p.dropna(subset=["actual", "predicted"])
        ids = ctx.filter_ids()
        if ids is not None and "respondent_id" in p:
            p = p[p["respondent_id"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True).isin(ids)]
        if p.empty:
            cards.notice("pending", "No usable predictions for this target (or none match the filters).")
        else:
            models = list(p["model"].astype(str).unique()) if "model" in p else ["(model)"]
            with cards.card("Actual vs predicted & residuals", "Where does the model miss?"):
                model = st.selectbox("Model", models, key=f"reg_model_{key}")
                q = p[p["model"].astype(str) == model] if "model" in p else p
                if "split" in q and q["split"].nunique() > 1:
                    q = q[q["split"].astype(str) == ("test" if "test" in set(q["split"].astype(str)) else
                                                     q["split"].astype(str).iloc[0])]
                q = q.assign(residual=q["actual"] - q["predicted"])
                c1, c2 = st.columns(2)
                with c1:
                    fig = charts.scatter(q, "actual", "predicted", hover=["respondent_id"] if "respondent_id" in q else None,
                                         x_title="Actual WTP (₹)", y_title="Predicted WTP (₹)", diagonal=True)
                    charts.show(fig, f"reg_avp_{key}_{model}", q)
                with c2:
                    fig = charts.histogram(q["residual"], nbins=40, colour=theme.LAVENDER, x_title="Residual (₹, actual − predicted)",
                                           marker_lines={"Mean": float(q["residual"].mean())})
                    charts.show(fig, f"reg_resid_{key}_{model}")
                fig = charts.scatter(q, "predicted", "residual", x_title="Predicted (₹)", y_title="Residual (₹)")
                fig.add_hline(y=0, line_dash="dot", line_color=theme.MUTED)
                charts.show(fig, f"reg_rvp_{key}_{model}", height=300)
                mae = q["residual"].abs().mean()
                rmse = float(np.sqrt((q["residual"] ** 2).mean()))
                zero_pred = q.loc[q["actual"] == 0, "predicted"].mean() if (q["actual"] == 0).any() else np.nan
                cards.explanation("Points on the dotted diagonal are perfect predictions; residual = actual − predicted.",
                                  esc(f"On these {len(q):,} rows: MAE ₹{mae:,.0f}, RMSE ₹{rmse:,.0f}, mean residual "
                                      f"₹{q['residual'].mean():,.0f}."
                                      + (f" Respondents who said ₹0 are predicted ₹{zero_pred:,.0f} on average."
                                         if not np.isnan(zero_pred) else "")),
                                  esc("A funnel shape (residuals widening with predictions) or a cluster of zero answers "
                                      "far above the diagonal suggests a two-part model (pay anything? then how much) "
                                      "would describe the data better."),
                                  esc("Recomputed from saved predictions for the selected rows; filters apply when "
                                      "respondent_id is present."))
    with cards.card("Coefficients", "Which characteristics are associated with higher or lower WTP?"):
        if not cr.ok:
            cards.missing_dataset(cr, "Optional: save model coefficients to show drivers.")
            return
        c = cr.df.copy()
        if "target" in c:
            c = c[c["target"].map(_target_key) == key]
        c["coefficient"] = pd.to_numeric(c["coefficient"], errors="coerce")
        c = c.dropna(subset=["coefficient"])
        if c.empty:
            cards.notice("pending", "No coefficients for this target.")
            return
        if "model" in c:
            ms = list(c["model"].astype(str).unique())
            mm = st.selectbox("Model", ms, key=f"reg_coef_model_{key}")
            c = c[c["model"].astype(str) == mm]
        c = c.reindex(c["coefficient"].abs().sort_values(ascending=False).index).head(25)
        fig = go.Figure(go.Bar(y=c["feature"].astype(str), x=c["coefficient"], orientation="h",
                               marker_color=[theme.GREEN if v > 0 else theme.PEACH for v in c["coefficient"]],
                               hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
        fig.update_layout(yaxis=dict(autorange="reversed"), xaxis_title="Coefficient", height=max(300, 26 * len(c) + 80))
        charts.show(fig, f"reg_coef_{key}", c)
        zeroed = int((c["coefficient"] == 0).sum())
        cards.explanation("Largest coefficients by absolute size (green = associated with higher WTP).",
                          esc(f"Largest: {c.iloc[0]['feature']} ({c.iloc[0]['coefficient']:+.3f})."
                              + (f" {zeroed} coefficient(s) shrunk to zero (Lasso)." if zeroed else "")),
                          esc("Drivers with large positive coefficients suggest where premium tiers might work."),
                          esc("Coefficients are associations conditional on the other predictors, not causal effects; "
                              "their size depends on how features were scaled."))
