"""Page 2 — Data quality and preparation (read-only inspection of the supplied files)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.components import cards, charts
from dashboard.components import stats as S
from dashboard.config import theme
from dashboard.config.schema import CONSUMER_COLUMNS, NOMINAL_FIELDS, label
from dashboard.data.transforms import norm_text, prepare_consumer
from dashboard.views import common

esc = cards.esc


def render(ctx) -> None:
    raw, clean = ctx.r("consumer_raw"), ctx.r("consumer_clean")
    cards.page_header("Data quality & preparation",
                      "Is the consumer dataset complete, consistent and suitable for analysis? Files are inspected "
                      "read-only — nothing is modified or overwritten.",
                      "Individual submission · data cleaning & transformation", [raw, clean])
    available = {k: r for k, r in (("Cleaned dataset", clean), ("Raw dataset", raw)) if r.ok}
    if not available:
        cards.missing_dataset(clean, "Data-quality checks need at least one consumer file.")
        cards.missing_dataset(raw, "The raw file enables the raw-versus-cleaned comparison.")
        return
    with cards.card("Dataset to inspect", help_text="Quality checks run on the whole file (global filters are not "
                                                     "applied here, because quality is a property of the file)."):
        which = st.segmented_control("Dataset", list(available), default=list(available)[0], key="dq_which",
                                     label_visibility="collapsed") or list(available)[0]
        r = available[which]
        st.caption(f"{r.source} · {r.message} · provenance: {r.provenance_note}")
    df = r.df
    prep, meta = prepare_consumer(df)

    _summary(r, df, prep, meta)
    t1, t2, t3, t4, t5 = st.tabs(["Missing values", "Types & validity", "Distributions & outliers",
                                  "Raw vs cleaned", "Cleaning decisions"])
    with t1:
        _missing(ctx, r)
    with t2:
        _types(r, df, prep, meta)
    with t3:
        _outliers(ctx, prep)
    with t4:
        _compare(ctx, raw, clean)
    with t5:
        _cleaning_log(ctx)


def _summary(r, df, prep, meta) -> None:
    n, m = df.shape
    dup_rows = int(df.duplicated().sum())
    dup_ids = int(df["respondent_id"].duplicated().sum()) if "respondent_id" in df else None
    cells = n * m
    miss = int(r.raw.isna().sum().sum())
    issues = sum(v["count"] for v in meta["parse_issues"].values())
    missing_expected = [c for c in CONSUMER_COLUMNS if c not in df.columns]
    items = [
        cards.kpi("Records", f"{n:,}", r.spec.title),
        cards.kpi("Columns", f"{m:,}", f"{len(CONSUMER_COLUMNS) - len(missing_expected)} of {len(CONSUMER_COLUMNS)} "
                                       "expected fields recognised"),
        cards.kpi("Duplicate rows", f"{dup_rows:,}", "identical across every column", accent=theme.PEACH),
        cards.kpi("Duplicate respondent IDs", f"{dup_ids:,}" if dup_ids is not None else None,
                  "respondent_id repeated" if dup_ids is not None else "no respondent_id column", accent=theme.PEACH),
        cards.kpi("Completeness", f"{100 * (1 - miss / cells):.1f}%" if cells else None,
                  f"{miss:,} empty cells of {cells:,}", accent=theme.GREEN),
        cards.kpi("Columns with gaps", f"{int((r.raw.isna().sum() > 0).sum()):,}", "at least one empty cell",
                  accent=theme.GREEN),
        cards.kpi("Invalid / unrecognised answers", f"{issues:,}", "outside expected scales or answer options",
                  accent=theme.RED),
        cards.kpi("'Not applicable' answers", f"{_na_count(prep):,}",
                  "kept as their own category, not treated as missing", accent=theme.LAVENDER),
    ]
    with cards.card("Quality summary"):
        cards.kpi_grid(items, per_row=4)
        if missing_expected:
            st.caption("Expected fields not found: " + ", ".join(missing_expected) +
                       ". Pages that need them will say so. Column aliases are listed in docs/DATA_CONTRACT.md.")
        if r.renamed:
            with st.expander(f"{len(r.renamed)} column names were mapped to the contract (display only)"):
                charts.table(pd.DataFrame({"column in file": list(r.renamed), "used as": list(r.renamed.values())}),
                             key="dq_renamed", filename="column_mapping.csv")


def _na_count(prep: pd.DataFrame) -> int:
    return int(sum((prep[c] == "not_applicable").sum() for c in prep.columns if c.endswith("__status")))


def _missing(ctx, r) -> None:
    raw = r.raw
    miss = raw.isna().mean().mul(100).sort_values(ascending=False)
    data = pd.DataFrame({"column": miss.index.astype(str), "missing_pct": miss.values.round(2),
                         "missing_n": raw.isna().sum().reindex(miss.index).values.astype(int),
                         "dtype": raw.dtypes.reindex(miss.index).astype(str).values})
    c1, c2 = st.columns([1, 1], gap="medium")
    with c1:
        with cards.card("Missingness by column", "Which questions are most often unanswered?"):
            show = data[data["missing_pct"] > 0]
            if show.empty:
                cards.notice("ok", "No empty cells in this file.")
            else:
                fig = charts.bar(show["column"], show["missing_pct"], show["missing_n"], [len(raw)] * len(show),
                                 horizontal=True, colour=theme.PEACH, axis_title="% of records empty")
                charts.show(fig, "dq_missing_bar", data)
                top = show.iloc[0]
                cards.explanation(
                    "Share of records with an empty cell in each column (raw values as read from the file).",
                    f"{len(show)} of {len(data)} columns have gaps; highest is <b>{esc(top['column'])}</b> at "
                    f"{top['missing_pct']:.1f}% ({int(top['missing_n']):,} of {len(raw):,}).",
                    esc("Columns with heavy missingness weaken any chart or model that relies on them. Check whether "
                        "gaps are skip-logic (e.g. future-trip respondents skipping past-trip questions) or "
                        "non-response before interpreting."),
                    esc("Empty cells only — answers such as 'Not applicable' or 'I do not remember' are valid "
                        "responses and are not counted as missing."))
    with c2:
        with cards.card("Missing-value map", "Is missingness concentrated in particular records or questions?"):
            nrows = min(len(raw), 400)
            idx = np.linspace(0, len(raw) - 1, nrows).astype(int) if len(raw) else []
            sample = raw.iloc[idx].isna().astype(int)
            if sample.values.sum() == 0:
                cards.notice("ok", "Nothing to map — the file has no empty cells.")
            else:
                fig = go.Figure(go.Heatmap(z=sample.values, x=[str(c) for c in sample.columns],
                                           y=list(range(len(sample))), colorscale=[[0, "#F4F8FC"], [1, theme.PEACH]],
                                           showscale=False, hovertemplate="record %{y}<br>%{x}<br>missing=%{z}"
                                                                          "<extra></extra>"))
                fig.update_layout(yaxis=dict(title=f"records (evenly spaced sample of {nrows})", showticklabels=False,
                                             autorange="reversed"), xaxis=dict(tickangle=-60, tickfont=dict(size=9)),
                                  height=460)
                charts.show(fig, "dq_missing_map")
                cards.explanation(
                    "Each cell is one record × one column; shaded cells are empty.",
                    f"Showing {nrows:,} of {len(raw):,} records. Vertical bands indicate question-level gaps; "
                    "horizontal bands indicate partially completed responses.",
                    esc("Horizontal bands suggest respondents who abandoned the survey — consider whether they should be "
                        "excluded in the cleaning pipeline (not here)."),
                    esc("Sampled rows for readability; use the bar chart for exact percentages."))


def _types(r, df, prep, meta) -> None:
    rows = []
    for c in r.raw.columns:
        s = r.raw[c]
        rows.append({"column": str(c), "used_as": r.renamed.get(c, normalise(c)), "dtype": str(s.dtype),
                     "non_empty": int(s.notna().sum()), "unique_values": int(s.nunique(dropna=True)),
                     "examples": "; ".join(map(str, s.dropna().astype(str).unique()[:3]))[:120]})
    with cards.card("Data types & recognised fields", "Is each column stored and interpreted as expected?"):
        charts.table(pd.DataFrame(rows), key="dq_types", filename="data_types.csv", height=320)
    issues = []
    for f in meta["ordinal"] + meta["score"] + meta["numeric"]:
        stc = prep[f"{f}__status"]
        issues.append({"field": f, "valid": int((stc == "valid").sum()), "not_applicable": int((stc == "not_applicable").sum()),
                       "other_special": int((stc == "special").sum()), "missing": int((stc == "missing").sum()),
                       "unrecognised": int((stc == "unrecognised").sum()),
                       "out_of_range": int((stc == "out_of_range").sum()),
                       "examples_of_problems": "; ".join(meta["parse_issues"].get(f, {}).get("examples", []))})
    with cards.card("Invalid or inconsistent responses",
                    "Do answers fall inside the questionnaire's scales and answer options?"):
        if issues:
            idf = pd.DataFrame(issues)
            charts.table(idf, key="dq_validity", filename="validity_checks.csv")
            bad = idf[(idf["unrecognised"] + idf["out_of_range"]) > 0]
            if bad.empty:
                cards.notice("ok", "Every scale and single-choice answer matched the questionnaire.")
            else:
                cards.notice("warn", f"{len(bad)} field(s) contain unrecognised or out-of-range values.",
                             "They are kept visible as their own category and excluded from scale statistics. "
                             "Fix them in the cleaning pipeline, not in the dashboard.")
        st.markdown("**Category spelling / formatting variants**")
        var_rows = []
        for f in NOMINAL_FIELDS + tuple(meta["ordinal"]):
            if f in df.columns:
                vals = df[f].dropna().astype(str)
                groups = vals.groupby(vals.map(norm_text)).unique()
                for k, v in groups.items():
                    if len(v) > 1:
                        var_rows.append({"field": f, "normalised": k, "variants": " | ".join(v[:6]),
                                         "records": int(vals.map(norm_text).eq(k).sum())})
        if var_rows:
            charts.table(pd.DataFrame(var_rows), key="dq_variants", filename="category_variants.csv")
            st.caption("These variants differ only in case, spacing, dashes or trailing punctuation. The dashboard "
                       "treats them as the same answer for display; the source file is unchanged.")
        else:
            cards.notice("ok", "No case/spacing/punctuation variants found in categorical answers.")
        if meta["multi"]:
            st.caption("Multi-select questions detected: " + "; ".join(f"{label(k)} ({v})" for k, v in meta["multi"].items()))


def normalise(c) -> str:
    from dashboard.data.mapping import normalise_name
    return normalise_name(c)


def _outliers(ctx, prep) -> None:
    numeric = {label(f): f"{f}__value" for f in ("storage_wtp_inr", "pickup_delivery_wtp_inr", "booking_intent_score",
                                                  "inconvenience_score") if f"{f}__value" in prep}
    for f in ("checkout_gap_hours", "luggage_management_hours", "bag_count", "trips_12m", "group_size"):
        if f in ctx.cmeta.get("numeric_source", set()) and f"{f}__value" in prep:
            numeric[label(f)] = f"{f}__value"
    if not numeric:
        cards.missing_columns(["storage_wtp_inr", "pickup_delivery_wtp_inr", "booking_intent_score"],
                              "Numeric checks need at least one numeric survey field.")
        return
    rows = []
    for name, col in numeric.items():
        d = S.describe_numeric(prep[col])
        if d["n"] == 0:
            continue
        flag, lo, hi = S.iqr_outliers(prep[col])
        rows.append({"variable": name, "n": d["n"], "min": d["min"], "q1": d["q1"], "median": d["median"],
                     "mean": round(d["mean"], 2), "q3": d["q3"], "max": d["max"], "zero_pct": round(d["zero_share"], 1),
                     "iqr_low_fence": round(lo, 2), "iqr_high_fence": round(hi, 2), "flagged_outliers": int(flag.sum())})
    with cards.card("Numerical ranges & outlier flags", "Are numeric answers plausible?"):
        charts.table(pd.DataFrame(rows), key="dq_ranges", filename="numeric_ranges.csv")
        st.caption("Outliers are flagged with the 1.5 × IQR rule for review only. Legitimate extreme answers (e.g. a "
                   "very high stated WTP) are never deleted automatically.")
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        with cards.card("Distribution", "What does each numeric answer look like?"):
            pick = st.selectbox("Variable", list(numeric), key="dq_hist_var")
            col = numeric[pick]
            d = S.describe_numeric(prep[col])
            fig = charts.histogram(prep[col], nbins=30, x_title=pick,
                                   marker_lines={"Median": d.get("median"), "Mean": d.get("mean")})
            charts.show(fig, f"dq_hist_{col}", prep[[col]].dropna().rename(columns={col: pick}))
            cards.explanation(esc(f"Frequency of each value of {pick}."),
                              esc(f"n = {d['n']:,}; median {d['median']:,.1f}, mean {d['mean']:,.1f}, range "
                                  f"{d['min']:,.0f}–{d['max']:,.0f}; {d['zero_share']:.1f}% are zero."),
                              esc("A mean well above the median signals right-skew — report medians for WTP."
                                  if d["mean"] > d["median"] * 1.15 else
                                  "Mean and median are close, so the distribution is not strongly skewed."),
                              common.limitation(ctx, d["n"], "Unfiltered file."))
    with c2:
        with cards.card("Outlier comparison (box plots)", "Which variables have extreme values?"):
            sel = st.multiselect("Variables", list(numeric), default=list(numeric)[:2], key="dq_box_vars")
            if sel:
                long = pd.concat([pd.DataFrame({"variable": s, "value": prep[numeric[s]]}) for s in sel]).dropna()
                fig = charts.box_by(long, "variable", "value", order=sel, y_title="Value", points="outliers")
                charts.show(fig, "dq_box", long)
                cards.explanation("Median (line), mean (dashed), interquartile box and points beyond 1.5 × IQR.",
                                  esc("Points beyond the whiskers are flagged; see the table above for counts."),
                                  esc("Investigate flagged values against the questionnaire before deciding to cap or "
                                      "exclude — high WTP from frequent business travellers may be genuine."),
                                  esc("Scales differ between variables; compare shapes, not absolute heights."))


def _compare(ctx, raw, clean) -> None:
    if not (raw.ok and clean.ok):
        cards.notice("pending", "Raw-versus-cleaned comparison needs both files.",
                     "Supply <code>consumer_survey_raw.csv</code> (or <code>consumer_survey_raw_synthetic.csv</code>) "
                     "and <code>consumer_survey_clean.csv</code>.")
        return
    a, b = raw.df, clean.df
    rows = [
        {"measure": "Records", "raw": len(a), "cleaned": len(b)},
        {"measure": "Columns", "raw": a.shape[1], "cleaned": b.shape[1]},
        {"measure": "Duplicate rows", "raw": int(a.duplicated().sum()), "cleaned": int(b.duplicated().sum())},
        {"measure": "Empty cells", "raw": int(raw.raw.isna().sum().sum()), "cleaned": int(clean.raw.isna().sum().sum())},
    ]
    if "respondent_id" in a and "respondent_id" in b:
        ia, ib = set(a["respondent_id"].astype(str)), set(b["respondent_id"].astype(str))
        rows.append({"measure": "Respondent IDs removed in cleaning", "raw": len(ia - ib), "cleaned": "—"})
        rows.append({"measure": "Respondent IDs only in cleaned file", "raw": "—", "cleaned": len(ib - ia)})
    comp = pd.DataFrame(rows)
    added = sorted(set(b.columns) - set(a.columns))
    removed = sorted(set(a.columns) - set(b.columns))
    with cards.card("Raw vs cleaned: what changed", "What did the cleaning pipeline change?"):
        charts.table(comp.astype(str), key="dq_compare", filename="raw_vs_clean.csv")
        st.caption(f"Columns added in cleaning: {', '.join(added) or 'none'} · removed: {', '.join(removed) or 'none'}")
    common_cols = [c for c in a.columns if c in b.columns]
    if common_cols:
        mp = pd.DataFrame({"column": common_cols,
                           "raw_missing_pct": [a[c].isna().mean() * 100 for c in common_cols],
                           "clean_missing_pct": [b[c].isna().mean() * 100 for c in common_cols]})
        mp = mp[(mp["raw_missing_pct"] > 0) | (mp["clean_missing_pct"] > 0)]
        with cards.card("Missingness before and after cleaning", "Did cleaning reduce gaps, and where?"):
            if mp.empty:
                cards.notice("ok", "Neither file has empty cells in shared columns.")
            else:
                fig = charts.grouped_bar(mp, "column", ["raw_missing_pct", "clean_missing_pct"], ["Raw", "Cleaned"],
                                         y_title="% empty", value_fmt="{:.1f}", horizontal=True,
                                         colours=[theme.PEACH, theme.GREEN])
                charts.show(fig, "dq_missing_compare", mp.round(2))
                cards.explanation("Share of empty cells per shared column in each file.",
                                  esc(f"Total empty cells: {rows[3]['raw']:,} raw → {rows[3]['cleaned']:,} cleaned."),
                                  esc("Reductions should be explained in the cleaning log — imputation, exclusion or "
                                      "recoding each has different consequences for the analysis."),
                                  esc("Empty cells only; recoded categories are not visible in this comparison."))


def _cleaning_log(ctx) -> None:
    log = ctx.r("cleaning_log")
    if log.ok:
        with cards.card("Cleaning decisions (from the supplied log)", "What was changed, and why?"):
            charts.table(log.df, key="dq_log", filename="data_cleaning_log.csv")
    else:
        cards.notice("pending", "No cleaning log supplied.",
                     "The cleaning itself is handled outside the dashboard. To document each decision here, add "
                     "<code>data/data_cleaning_log.csv</code> with columns <code>step, column, action, rows_affected, "
                     "reason</code>. Until then, use the raw-vs-cleaned tab to see the net effect.")
    cards.text_card("Dashboard handling rules (presentation only)",
                    "<p>• Source CSVs are read into memory; nothing is written back.</p>"
                    "<p>• 'Not applicable', 'Did not search', 'Do not remember' and similar answers are kept as their own "
                    "categories and are excluded from denominators where the question does not apply — they are "
                    "never converted to zero or to missing.</p>"
                    "<p>• Empty cells stay missing; zero WTP stays zero.</p>"
                    "<p>• Multi-select answers are read from semicolon-separated text or binary-encoded columns.</p>"
                    "<p>• Outliers are flagged, not removed.</p>", tint="blue")
