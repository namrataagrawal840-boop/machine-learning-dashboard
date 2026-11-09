"""Page 6 — Customer segmentation (uses prepared cluster assignments / profiles)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from dashboard.components import cards, charts
from dashboard.components import stats as S
from dashboard.config import theme
from dashboard.config.schema import label
from dashboard.data.transforms import MS_SEP, ms_columns, ms_option
from dashboard.views import common

esc = cards.esc

RADAR_VARS = {  # column -> (label, min, max)
    "trips_12m__rank": ("Travel frequency", 0, 4), "checkout_gap_hours__rank": ("Checkout gap", 0, 6),
    "bag_count__rank": ("Bags", 0, 3), "inconvenience_score__value": ("Inconvenience", 1, 5),
    "storage_search_difficulty__rank": ("Search difficulty", 0, 4), "problem_frequency__rank": ("Gap frequency", 0, 4),
    "group_size__rank": ("Group size", 0, 4),
}
PROFILE_CATS = {"City": "city__label", "Travel purpose": "travel_purpose__label", "Transport mode": "travel_mode__label",
                "Accommodation": "accommodation_type__label", "Age group": "age_group__label",
                "First service": "first_service_preference__label", "Non-use barrier": "main_nonuse_barrier__label",
                "Pricing model": "preferred_pricing_model__label", "Checkout gap": "checkout_gap_hours__label"}
PROFILE_MULTI = {"Current methods": "existing_storage_methods", "Security requirements": "security_requirements",
                 "Acceptable venues": "acceptable_venue_types", "Preferred features": "preferred_features",
                 "Storage barriers": "existing_barriers"}
PRE_INTENT_HINT = ("Clusters should be built from pre-intent characteristics (Q1–Q22). Intent, WTP and preferences "
                   "(Q23–Q30) are overlaid afterwards to evaluate segments, not to define them.")


def render(ctx) -> None:
    ca, cp = ctx.r("cluster_assignments"), ctx.r("cluster_profiles")
    cards.page_header("Customer segmentation",
                      "Which traveller segments exist, what does each need, and which service should each receive?",
                      "Group-stage analytics · clustering", [ca, cp, ctx.r("consumer_clean")])
    cards.notice("info", "Segments come from the prepared clustering pipeline.",
                 esc(PRE_INTENT_HINT + " Cluster numbers are arbitrary labels; no persona names are invented."))
    if cp.ok:
        with cards.card("Cluster profiles (as supplied by the pipeline)", "What did the clustering output report?"):
            charts.table(cp.df, key="seg_profiles", filename="cluster_profiles.csv")
            st.caption(f"Source: {cp.source}. Shown exactly as supplied.")
    else:
        cards.missing_dataset(cp, "Optional: the pipeline's own cluster summary.")
    if not ca.ok:
        cards.missing_dataset(ca, "Respondent-level assignments are needed to profile segments against survey answers.")
        return
    if ctx.consumer is None or "cluster" not in ctx.consumer:
        cards.notice("pending", "Cluster assignments could not be linked to consumer responses.",
                     "Both files need a matching <code>respondent_id</code>.")
        return
    if not common.require_consumer(ctx):
        return
    d = ctx.cf[ctx.cf["cluster"].notna()].copy()
    matched = ctx.cmeta.get("cluster_join", 0)
    if d.empty:
        cards.notice("warn", "No clustered respondents match the current filters.")
        return
    clusters = sorted(d["cluster"].unique(), key=lambda s: (len(s), s))
    _sizes(ctx, d, clusters, matched, ca)
    _overlay(ctx, d, clusters)
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        _radar(ctx, d, clusters)
    with c2:
        _categorical(ctx, d, clusters)
    _wtp(ctx, d, clusters)
    _segment_cards(ctx, d, clusters)


def _sizes(ctx, d, clusters, matched, ca):
    with cards.card("Cluster sizes", "How large is each segment?"):
        vc = d["cluster"].value_counts().reindex(clusters)
        data = pd.DataFrame({"cluster": clusters, "respondents": vc.values, "share_pct": (vc / vc.sum() * 100).round(2).values})
        fig = charts.bar(data["cluster"], data["share_pct"], data["respondents"], [int(vc.sum())] * len(data),
                         colour=theme.BLUE, axis_title="% of clustered respondents")
        charts.show(fig, "seg_sizes", data)
        unmatched = len(ca.df) - matched
        cards.explanation("Respondents per cluster after filters.",
                          esc(f"{len(clusters)} clusters; largest {data.sort_values('respondents').iloc[-1]['cluster']} "
                              f"({data['share_pct'].max():.1f}%), smallest {data['share_pct'].min():.1f}%. "
                              f"{matched:,} of {len(ca.df):,} assignments matched consumer records"
                              + (f" ({unmatched:,} unmatched)." if unmatched else ".")),
                          esc("Very small clusters may be outliers rather than markets; very large ones may hide "
                              "sub-segments. Size alone does not indicate value — see the intent overlay."),
                          common.limitation(ctx, len(d)))


def _overlay(ctx, d, clusters):
    with cards.card("High-intent and unmet-need overlay", "Which segments show conversion potential?"):
        metrics = {k: v for k, v in {"High stated intent %": "high_intent", "Unmet need %": "unmet_need",
                                     "Gap ≥ 3 h %": "long_gap", "Accept partner venues %": "accepts_partner",
                                     "Pilot: yes %": None}.items() if v is None or v in d}
        rows = []
        for c in clusters:
            g = d[d["cluster"] == c]
            row = {"cluster": c, "n": len(g)}
            for name, col in metrics.items():
                if col:
                    row[name] = S.share(g[col])[0]
                elif "pilot_interest__rank" in g:
                    row[name] = S.share(g["pilot_interest__rank"].dropna() >= 2)[0]
            rows.append(row)
        data = pd.DataFrame(rows)
        ycols = [c for c in data.columns if c.endswith("%") and data[c].notna().any()]
        data["label"] = [f"{c} (n={n})" for c, n in zip(data["cluster"], data["n"])]
        fig = charts.grouped_bar(data, "label", ycols, y_title="% within cluster",
                                 colours=[theme.GOLD, theme.LAVENDER, theme.BLUE, theme.GREEN, theme.PEACH])
        charts.show(fig, "seg_overlay", data.drop(columns="label").round(2))
        if "High stated intent %" in data:
            b = data.sort_values("High stated intent %", ascending=False).iloc[0]
            overall = S.share(d["high_intent"])[0]
            obs = (f"Highest stated intent: <b>{esc(b['cluster'])}</b> at {b['High stated intent %']:.1f}% "
                   f"(n={int(b['n'])}) vs {overall:.1f}% across all clustered respondents.")
        else:
            obs = "Intent not available."
        cards.explanation("Outcome indicators calculated within each cluster after clustering.", obs,
                          esc("Prioritise segments combining high intent with high unmet need; high intent without "
                              "unmet need may reflect enthusiasm that existing options already satisfy."),
                          common.limitation(ctx, len(d), "Stated intent, not conversion."))


def _radar(ctx, d, clusters):
    with cards.card("Numeric profile comparison", "How do segments differ on problem intensity?"):
        avail = {k: v for k, v in RADAR_VARS.items() if k in d and d[k].notna().any()}
        if len(avail) < 3:
            cards.missing_columns([k.split("__")[0] for k in RADAR_VARS], "The radar needs at least three scale fields.")
            return
        sel = st.multiselect("Clusters to compare", clusters, default=clusters[:4], key="seg_radar_sel")
        rows = []
        for c in sel:
            g = d[d["cluster"] == c]
            row = {"cluster": c}
            for k, (lbl, lo, hi) in avail.items():
                row[lbl] = (g[k].mean() - lo) / (hi - lo) * 100 if g[k].notna().any() else np.nan
            rows.append(row)
        if not rows:
            cards.notice("info", "Select clusters to compare.")
            return
        data = pd.DataFrame(rows)
        labels_ = [v[0] for v in avail.values()]
        fig = charts.radar(data, "cluster", labels_, labels_)
        charts.show(fig, "seg_radar", data.round(1))
        overall = {v[0]: (d[k].mean() - v[1]) / (v[2] - v[1]) * 100 for k, v in avail.items()}
        diffs = [(r["cluster"], l, r[l] - overall[l]) for _, r in data.iterrows() for l in labels_ if pd.notna(r[l])]
        big = max(diffs, key=lambda t: abs(t[2])) if diffs else None
        cards.explanation("Cluster mean on each ordered scale, rescaled so the scale minimum = 0 and maximum = 100.",
                          esc(f"Biggest departure from the overall mean: {big[0]} on {big[1]} ({big[2]:+.0f} points).")
                          if big else "—",
                          esc("Segments far above average on gap length and inconvenience are natural first targets; "
                              "those high on bags or group size suit multi-bag packages."),
                          common.limitation(ctx, len(d), "Means of ordinal bands are approximate summaries."))


def _categorical(ctx, d, clusters):
    with cards.card("Categorical profile by cluster", "What answers characterise each cluster?"):
        opts = {k: v for k, v in PROFILE_CATS.items() if v in d}
        opts.update({k: v for k, v in PROFILE_MULTI.items() if ms_columns(d, v)})
        if not opts:
            cards.missing_columns(list(PROFILE_CATS.values()), "")
            return
        pick = st.selectbox("Variable", list(opts), key="seg_cat_var")
        v = opts[pick]
        if v in PROFILE_MULTI.values():
            pct, n = S.multi_share_by(d, "cluster", ms_columns(d, v))
            pct.columns = [ms_option(c) for c in pct.columns]
        else:
            pct, cnt = S.crosstab_share(d, "cluster", v)
            n = cnt.sum(axis=1)
        if pct.empty:
            cards.notice("info", "No data.")
            return
        pct = pct.reindex([c for c in clusters if c in pct.index])
        disp = pct.copy()
        disp.index = [f"{i} (n={n[i]})" for i in pct.index]
        fig = charts.heatmap(disp.T, text_fmt="{:.0f}%", zmin=0, zmax=100, value_name="% of cluster",
                             colorbar_title="%")
        charts.show(fig, f"seg_cat_{v}", pct.round(1).reset_index(names="cluster"))
        overall = pct.mean()
        dev = pct.sub(overall, axis=1)
        idx = dev.stack().abs().idxmax()
        cards.explanation(esc(f"% of each cluster choosing each {pick.lower()} option (columns = clusters)."),
                          esc(f"Most distinctive: {idx[0]} — '{idx[1]}' at {pct.loc[idx]:.0f}% "
                              f"({dev.loc[idx]:+.0f} pts vs the cluster average)."),
                          esc("Distinctive answers are the hooks for targeting and messaging for that segment."),
                          common.limitation(ctx, int(n.sum())))


def _wtp(ctx, d, clusters):
    cols = [c for c in ("storage_wtp_inr__value", "pickup_delivery_wtp_inr__value") if c in d]
    if not cols:
        return
    with cards.card("Willingness to pay by cluster", "Do segments differ in what they say they would pay?"):
        long = pd.concat([pd.DataFrame({"cluster": d["cluster"], "measure": label(c.replace("__value", "")),
                                        "value": d[c]}) for c in cols]).dropna()
        pick = st.segmented_control("Measure", list(long["measure"].unique()), default=long["measure"].iloc[0],
                                    key="seg_wtp_measure") or long["measure"].iloc[0]
        sub = long[long["measure"] == pick]
        fig = charts.box_by(sub, "cluster", "value", order=clusters, y_title="₹")
        charts.show(fig, "seg_wtp", sub)
        med = sub.groupby("cluster")["value"].median()
        cards.explanation("Distribution of stated WTP within each cluster (box = IQR, line = median, dashed = mean).",
                          esc(f"Highest median: {med.idxmax()} (₹{med.max():,.0f}); lowest: {med.idxmin()} "
                              f"(₹{med.min():,.0f})."),
                          esc("Differences suggest segment-specific price tests rather than segment-specific prices at "
                              "launch."),
                          common.limitation(ctx, len(sub), "Hypothetical WTP."))
    if "city__label" in d:
        with cards.card("Segments by city", "Are segments concentrated in particular cities?"):
            pct, cnt = S.crosstab_share(d, "city__label", "cluster")
            rn = cnt.sum(axis=1)
            disp = pct.copy()
            disp.index = [f"{i} (n={rn[i]})" for i in pct.index]
            fig = charts.stacked_share(disp, cnt.set_axis(disp.index), horizontal=True, legend_title="Cluster")
            charts.show(fig, "seg_city", pct.round(1).reset_index(names="city"))
            v, p, n = S.cramers_v(d["city__label"], d["cluster"])
            cards.explanation("Cluster mix within each city (row %).",
                              esc(f"City × cluster Cramér's V = {S.fmt_num(v)} ({S.strength_word(v, 'v')}, n={n:,})."),
                              esc("A strong association means the launch city partly determines which segment you "
                                  "serve first."),
                              common.limitation(ctx, n))


# ------------------------------------------------------------------ segment cards
def _features(d: pd.DataFrame) -> dict[str, pd.Series]:
    """Binary indicators used to find distinctive characteristics."""
    feats = {}
    for col, nm in (("long_gap", "gap ≥ 3 h"), ("has_problem", "managed luggage ≥ 1 h after a long gap"),
                    ("search_difficulty", "found storage hard / unaware"), ("unmet_need", "unmet storage need"),
                    ("accepts_partner", "accepts partner venues")):
        if col in d:
            feats[nm] = d[col]
    for nm, col in (("frequent traveller (4+ trips)", "trips_12m__rank"), ("2+ bags", "bag_count__rank"),
                    ("inconvenience 4–5", "inconvenience_score__value")):
        if col in d:
            thr = {"trips_12m__rank": 3, "bag_count__rank": 1, "inconvenience_score__value": 4}[col]
            feats[nm] = (d[col] >= thr).astype("boolean").where(d[col].notna())
    for f in ("travel_purpose__label", "travel_mode__label", "accommodation_type__label", "age_group__label"):
        if f in d:
            for v in d[f].dropna().unique():
                feats[f"{label(f.split('__')[0]).lower()}: {v}"] = (d[f] == v).astype("boolean").where(d[f].notna())
    for f in ("existing_storage_methods", "existing_barriers"):
        for c in ms_columns(d, f):
            feats[f"{label(f).lower()}: {ms_option(c)}"] = d[c]
    return feats


def _top_in(g: pd.DataFrame, d: pd.DataFrame, col: str) -> tuple[str, float, float] | None:
    if col not in g or g[col].dropna().empty:
        return None
    vc = g[col].value_counts(normalize=True)
    top = vc.index[0]
    return str(top), float(vc.iloc[0] * 100), float((d[col] == top).mean() * 100)


def _top_multi(g: pd.DataFrame, field: str) -> list[tuple[str, float]]:
    cols = ms_columns(g, field)
    if not cols:
        return []
    ans = g[cols].notna().any(axis=1)
    if not ans.any():
        return []
    s = g.loc[ans, cols].fillna(False).astype(float).mean().sort_values(ascending=False) * 100
    return [(ms_option(c), v) for c, v in s.head(3).items()]


def _segment_cards(ctx, d, clusters):
    feats = _features(d)
    if not feats:
        return
    overall = {k: S.share(v)[0] for k, v in feats.items()}
    with cards.card("Segment interpretation", "Who is each segment, what do they need, and how should we reach them?",
                    help_text="Every sentence is generated from the filtered data; distinctive traits are those whose "
                              "share differs most from the clustered sample overall (min. 5 respondents)."):
        pick = st.selectbox("Segment", clusters, key="seg_card_pick")
        g = d[d["cluster"] == pick]
        diffs = []
        for k, v in feats.items():
            vv = v.loc[g.index]
            p, kk, n = S.share(vv)
            if n >= 5 and not np.isnan(overall[k]):
                diffs.append((k, p, overall[k], p - overall[k], n))
        diffs.sort(key=lambda t: t[3], reverse=True)
        pos = [x for x in diffs if x[3] > 0][:4]
        neg = sorted([x for x in diffs if x[3] < 0], key=lambda t: t[3])[:2]
        who = "; ".join(f"{k} {p:.0f}% (vs {o:.0f}%)" for k, p, o, _, _ in pos) or "No trait differs markedly."
        lower = "; ".join(f"{k} {p:.0f}% (vs {o:.0f}%)" for k, p, o, _, _ in neg)
        need_bits = []
        for k in ("unmet storage need", "gap ≥ 3 h", "found storage hard / unaware", "2+ bags", "inconvenience 4–5"):
            if k in feats:
                p = S.share(feats[k].loc[g.index])[0]
                need_bits.append(f"{k}: {p:.0f}% (overall {overall[k]:.0f}%)")
        barrier = _top_in(g, d, "main_nonuse_barrier__label")
        service = _top_in(g, d, "first_service_preference__label")
        sec = _top_multi(g, "security_requirements")
        feat = _top_multi(g, "preferred_features")
        hi = S.share(g["high_intent"]) if "high_intent" in g else (np.nan, 0, 0)
        med = g["storage_wtp_inr__value"].median() if "storage_wtp_inr__value" in g else np.nan
        rows = [
            ("Who is this segment?", f"{len(g):,} respondents ({len(g) / len(d) * 100:.1f}% of clustered). Over-represented: "
                                     f"{who}." + (f" Under-represented: {lower}." if lower else "")),
            ("What is its main need?", "; ".join(need_bits) + "." if need_bits else "Problem indicators unavailable."),
            ("What stops it from purchasing?",
             f"Most cited non-use reason: '{barrier[0]}' ({barrier[1]:.0f}% vs {barrier[2]:.0f}% overall)." if barrier
             else "Q29 not available."),
            ("Which service could we introduce?",
             (f"Most chosen first service: '{service[0]}' ({service[1]:.0f}%). " if service else "") +
             (f"High stated intent {S.fmt_pct(*hi)}; median storage WTP ₹{med:,.0f}." if not np.isnan(med) else "")),
            ("How should we market to it?",
             ("Lead with the trust features this segment rates essential: " + ", ".join(f"{o} ({v:.0f}%)" for o, v in sec)
              + ". " if sec else "") +
             ("Highlight: " + ", ".join(f"{o} ({v:.0f}%)" for o, v in feat) + "." if feat else "")),
        ]
        st.markdown(cards.html_table([{"Question": f"<b>{esc(q)}</b>", "Data-derived answer": esc(a)} for q, a in rows]),
                    unsafe_allow_html=True)
        st.caption("Suggested actions are hypotheses for testing. Associations within a cluster do not show what causes "
                   "the behaviour.")
        if diffs:
            tbl = pd.DataFrame(diffs, columns=["trait", "cluster_pct", "overall_pct", "difference_pts", "n_in_cluster"])
            with st.expander("All traits ranked by distinctiveness"):
                charts.table(tbl.round(2), key=f"seg_traits_{pick}", filename=f"traits_{pick}.csv")
