"""Page 8 — Association rule mining (reads prepared rules)."""
from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.components import cards, charts
from dashboard.config import theme
from dashboard.views import common

esc = cards.esc


def parse_items(v) -> tuple[str, ...]:
    if pd.isna(v):
        return ()
    s = str(v).strip()
    quoted = re.findall(r"'([^']*)'|\"([^\"]*)\"", s)
    if quoted:
        items = [a or b for a, b in quoted]
    else:
        s = re.sub(r"^frozenset\(|\)$", "", s).strip("{}[]() ")
        sep = ";" if ";" in s else "|" if "|" in s else ","
        items = [x.strip() for x in s.split(sep)]
    return tuple(sorted(i.strip() for i in items if i and i.strip()))


def render(ctx) -> None:
    ar = ctx.r("association_rules")
    cards.page_header("Association rule mining — service combinations",
                      "Which services, features and trust requirements do travellers tend to choose together?",
                      "Group-stage analytics · association rules", [ar])
    cards.notice("warn", "Associations are not causal.",
                 "A rule A → B means respondents who selected A also selected B more often than average. It does not "
                 "show that offering A causes demand for B, nor the order of a customer journey.")
    if not ar.ok:
        cards.missing_dataset(ar, "Rules come from the association-mining pipeline (e.g. Apriori on Q15, Q21, Q22, Q24, "
                                  "Q28). The dashboard does not mine rules itself.")
        return
    r = ar.df.copy()
    for c in ("support", "confidence", "lift"):
        r[c] = pd.to_numeric(r[c], errors="coerce")
    r = r.replace([np.inf, -np.inf], np.nan).dropna(subset=["support", "confidence", "lift"])
    r["A"] = r["antecedents"].map(parse_items)
    r["B"] = r["consequents"].map(parse_items)
    r = r[(r["A"].map(len) > 0) & (r["B"].map(len) > 0)]
    if r.empty:
        cards.notice("error", "No usable rules", "Antecedents/consequents could not be parsed.")
        return
    r["antecedent"] = r["A"].map(lambda t: " + ".join(t))
    r["consequent"] = r["B"].map(lambda t: " + ".join(t))
    n_resp = None
    if "count" in r and pd.to_numeric(r["count"], errors="coerce").notna().any():
        r["supporting_respondents"] = pd.to_numeric(r["count"], errors="coerce")
        count_note = "Supporting respondents as supplied in the rules file."
    elif ctx.consumer is not None:
        n_resp = ctx.n_total
        r["supporting_respondents"] = (r["support"] * n_resp).round().astype(int)
        count_note = (f"Supporting respondents estimated as support × {n_resp:,} consumer records (the rules file has "
                      "no count column).")
    else:
        r["supporting_respondents"] = np.nan
        count_note = "No count column and no consumer file — respondent counts unavailable."

    with cards.card("Rule filters", help_text="Rules are mined on the full survey; the global respondent filters do "
                                              "not apply to this page."):
        c1, c2, c3, c4 = st.columns(4)
        smax = float(max(r["support"].max(), 0.01))
        min_s = c1.slider("Minimum support", 0.0, round(smax, 3), 0.0, step=max(round(smax / 50, 4), 0.001),
                          key="ar_sup", format="%.3f")
        min_c = c2.slider("Minimum confidence", 0.0, 1.0, 0.0, 0.05, key="ar_conf")
        lmin, lmax = float(r["lift"].min()), float(r["lift"].max())
        min_l = c3.slider("Minimum lift", round(min(lmin, 1.0), 2), round(max(lmax, 1.01), 2), round(min(max(1.0, lmin), lmax), 2),
                          0.05, key="ar_lift", help="Lift > 1: items co-occur more than expected by chance.")
        q = c4.text_input("Search items", key="ar_q", placeholder="e.g. CCTV")
        st.caption(count_note)
    f = r[(r["support"] >= min_s) & (r["confidence"] >= min_c) & (r["lift"] >= min_l)]
    if q:
        f = f[f["antecedent"].str.contains(q, case=False, regex=False) | f["consequent"].str.contains(q, case=False,
                                                                                                      regex=False)]
    f = f.sort_values(["lift", "confidence"], ascending=False)
    cards.kpi_grid([cards.kpi("Rules supplied", f"{len(r):,}", ar.source, ar.provenance),
                    cards.kpi("Rules after filters", f"{len(f):,}", "support / confidence / lift / search"),
                    cards.kpi("Rules with lift > 1", f"{int((r['lift'] > 1).sum()):,}", "positive association"),
                    cards.kpi("Highest lift", f"{r['lift'].max():.2f}", f"{r.loc[r['lift'].idxmax(), 'antecedent']} → "
                                                                         f"{r.loc[r['lift'].idxmax(), 'consequent']}")],
                   per_row=4)
    if f.empty:
        cards.notice("info", "No rules meet the current thresholds — lower them to see results.")
        return
    with cards.card("Rule table", "Which rules are strongest?"):
        show = f[["antecedent", "consequent", "support", "confidence", "lift", "supporting_respondents"] +
                 [c for c in ("leverage", "conviction") if c in f]].reset_index(drop=True)
        charts.table(show, key="ar_table", filename="association_rules_filtered.csv", height=360 if len(show) > 9 else None, column_config={
            "support": st.column_config.ProgressColumn("Support", min_value=0, max_value=float(r["support"].max()),
                                                       format="%.3f"),
            "confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=1, format="%.2f"),
            "lift": st.column_config.NumberColumn("Lift", format="%.2f"),
            "supporting_respondents": st.column_config.NumberColumn("Respondents", format="%d")})
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        _strength(ctx, f)
    with c2:
        _network(ctx, f)
    _bundles(ctx, f, n_resp)


def _strength(ctx, f):
    with cards.card("Rule-strength comparison", "How do support, confidence and lift trade off?"):
        size = (f["lift"].clip(lower=0.1) / f["lift"].max() * 28 + 6).to_numpy()
        fig = go.Figure(go.Scatter(
            x=f["support"], y=f["confidence"], mode="markers",
            marker=dict(size=size, color=f["lift"], colorscale=[[0, "#DDE7F3"], [1, theme.BLUE]], showscale=True,
                        colorbar=dict(title="lift", thickness=10), line=dict(width=0.5, color="white")),
            text=f["antecedent"] + " → " + f["consequent"],
            customdata=np.column_stack([f["lift"], f["supporting_respondents"].fillna(-1)]),
            hovertemplate="%{text}<br>support %{x:.3f} · confidence %{y:.2f}<br>lift %{customdata[0]:.2f} · "
                          "respondents %{customdata[1]:.0f}<extra></extra>"))
        fig.update_layout(xaxis_title="Support (share of respondents with all items)", yaxis_title="Confidence")
        charts.show(fig, "ar_scatter")
        top = f.iloc[0]
        cards.explanation("Each point is a rule; size and colour = lift.",
                          esc(f"Top rule by lift: {top['antecedent']} → {top['consequent']} (lift {top['lift']:.2f}, "
                              f"confidence {top['confidence']:.2f}, support {top['support']:.3f})."),
                          esc("High-lift but tiny-support rules describe niches; for bundles, prefer rules that "
                              "combine lift > 1 with meaningful support."),
                          esc("Rule quality depends on the pipeline's minimum-support setting and item encoding."))


def _network(ctx, f):
    with cards.card("Association network", "Which items sit at the centre of co-selection?"):
        if len(f) > 5:
            top_n = st.slider("Rules in network", 5, min(80, len(f)), min(25, len(f)), key="ar_net_n")
        else:
            top_n = len(f)
        g = f.head(top_n)
        edges = [(a, b, row.lift) for row in g.itertuples() for a in row.A for b in row.B]
        nodes = sorted({a for a, _, _ in edges} | {b for _, b, _ in edges})
        if not nodes:
            cards.notice("info", "No edges.")
            return
        k = len(nodes)
        ang = np.linspace(0, 2 * np.pi, k, endpoint=False)
        pos = {n: (math.cos(a), math.sin(a)) for n, a in zip(nodes, ang)}
        deg = {n: 0 for n in nodes}
        for a, b, _ in edges:
            deg[a] += 1
            deg[b] += 1
        fig = go.Figure()
        lmax = max(l for _, _, l in edges)
        for a, b, l in edges:
            (x0, y0), (x1, y1) = pos[a], pos[b]
            fig.add_annotation(x=x1, y=y1, ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y", showarrow=True,
                               arrowhead=2, arrowsize=1, arrowwidth=0.6 + 2.4 * l / lmax, arrowcolor="rgba(102,141,186,0.45)",
                               standoff=8, startstandoff=8, text="")
        fig.add_scatter(x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes], mode="markers+text",
                        marker=dict(size=[10 + 3 * deg[n] for n in nodes], color=theme.GREEN, line=dict(color="white", width=1)),
                        text=[n for n in nodes], textposition="top center", textfont=dict(size=10, color=theme.TEXT),
                        hovertemplate="%{text}<br>connections: %{customdata}<extra></extra>",
                        customdata=[deg[n] for n in nodes], showlegend=False)
        fig.update_layout(xaxis=dict(visible=False, range=[-1.5, 1.5]), yaxis=dict(visible=False, range=[-1.35, 1.35],
                                                                                   scaleanchor="x"), height=520)
        charts.show(fig, "ar_network", pd.DataFrame(edges, columns=["from_item", "to_item", "lift"]))
        hub = max(deg, key=deg.get)
        cards.explanation("Arrows run from antecedent items to consequent items for the strongest filtered rules; "
                          "thicker arrows = higher lift; larger nodes = more connections.",
                          esc(f"Most connected item: {hub} ({deg[hub]} links among {len(nodes)} items)."),
                          esc("Hub items are natural anchors for a product page or default bundle."),
                          esc("Layout is circular for readability; distances carry no meaning. Labels and arrows can be "
                              "repositioned in the chart (editable)."))


def _bundles(ctx, f, n_resp):
    with cards.card("From rules to possible product-bundle decisions",
                    help_text="Generated from the filtered rules with lift > 1. These are hypotheses to test."):
        cand = f[f["lift"] > 1].head(8)
        if cand.empty:
            cards.notice("info", "No filtered rule has lift > 1.")
            return
        rows = []
        for row in cand.itertuples():
            nr = "" if pd.isna(row.supporting_respondents) else f", ~{int(row.supporting_respondents):,} respondents"
            rows.append({
                "Observed association": esc(f"Respondents choosing {row.antecedent} also chose {row.consequent} in "
                                            f"{row.confidence * 100:.0f}% of cases (lift {row.lift:.2f}, support "
                                            f"{row.support:.3f}{nr})."),
                "Business possibility": esc(f"Consider presenting {row.consequent} alongside {row.antecedent} — e.g. as a "
                                            "default add-on or bundled tier — and A/B test uptake."),
            })
        st.markdown(cards.html_table(rows), unsafe_allow_html=True)
        st.caption("Preference associations, not proof of a causal customer journey. Validate bundles with real "
                   "booking add-on rates.")
