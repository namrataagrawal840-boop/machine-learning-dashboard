"""Page 4 — Geographic opportunity and market gaps."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.components import cards, charts
from dashboard.components import stats as S
from dashboard.config import theme
from dashboard.views import common
from dashboard.views.market import (COMPONENTS, DEFAULT_WEIGHTS, filter_market, primary_partner_frame, score,
                                    zone_table)

esc = cards.esc
LAYER_COLOURS = {
    "Official airport / railway storage": theme.BLUE, "Smart locker operator": theme.TEAL,
    "Aggregator network": theme.LAVENDER, "Other short-term storage": theme.SLATE,
    "Prospective business partner": theme.GOLD, "Unclassified": "#A9B3C1",
    "Partner — verified": theme.GREEN, "Partner — not yet verified": theme.PEACH, "Travel node": theme.ROSE,
    "Microzone reference point": theme.TEXT,
}


def render(ctx) -> None:
    cards.page_header("Geographic opportunity & market gaps",
                      "Where might a storage marketplace be viable? Combines consumer need, competitor coverage and "
                      "partner readiness by city and microzone.",
                      "Business application · market evidence",
                      [ctx.r("consumer_clean"), ctx.r("competitors"), ctx.r("geo_audit"), ctx.r("partner_audit"),
                       ctx.r("partner_survey")])
    cards.notice("warn", "No area is labelled 'underserved' here.",
                 "A lack of competitors can mean weak demand or poor economics as easily as an opportunity. Areas are "
                 "flagged for <b>field verification</b> only when consumer samples, partner data and competitor "
                 "evidence all exist.")
    _map(ctx)
    if ctx.consumer is None:
        cards.missing_dataset(ctx.r("consumer_clean"), "The microzone table combines consumer answers with market data.")
        return
    if ctx.cf is None or ctx.cf.empty:
        common.require_consumer(ctx)
        return
    _opportunity(ctx)
    _city(ctx)


# ------------------------------------------------------------------ map
def _map(ctx) -> None:
    layers = []
    comp = filter_market(ctx, ctx.competitors)
    if len(comp) and "has_coords" in comp:
        c = comp[comp["has_coords"]].copy()
        c["layer"] = c["category__label"]
        c["name"] = c.get("competitor_name", pd.Series("Competitor", index=c.index))
        layers.append(c)
    _, pf = primary_partner_frame(ctx)
    pf = filter_market(ctx, pf)
    if len(pf) and {"latitude", "longitude"} <= set(pf.columns):
        from dashboard.data.transforms import valid_coords
        p = pf[valid_coords(pf)].copy()
        p["layer"] = np.where(p["is_verified"], "Partner — verified", "Partner — not yet verified")
        p["name"] = p.get("partner_id", pd.Series("Partner", index=p.index)).astype(str) + " · " + \
            p.get("establishment_type", pd.Series("", index=p.index)).astype(str)
        layers.append(p)
    geo = filter_market(ctx, ctx.geo)
    if len(geo) and "has_coords" in geo:
        g = geo[geo["has_coords"]].copy()
        nt = g.get("node_type", pd.Series("", index=g.index)).astype(str).str.lower()
        g["layer"] = np.where(nt.str.contains("zone|centroid|reference"), "Microzone reference point", "Travel node")
        g["name"] = g.get("node_name", pd.Series("Location", index=g.index)).astype(str) + " · " + \
            g.get("node_type", pd.Series("", index=g.index)).astype(str)
        layers.append(g)

    with cards.card("Market map", "Where are existing storage options, prospective partners and travel nodes?"):
        if not layers:
            missing = [r.spec.filenames[0] for r in (ctx.r("competitors"), ctx.r("geo_audit"), ctx.r("partner_audit"))
                       if not r.ok]
            cards.notice("pending", "No records with real coordinates are available, so no map is drawn.",
                         "The dashboard never generates locations. Supply latitude/longitude in "
                         + ", ".join(f"<code>{esc(m)}</code>" for m in missing)
                         + ". The microzone table below still works from survey answers.")
            return
        allp = pd.concat(layers, ignore_index=True, sort=False)
        allp["lat"] = pd.to_numeric(allp["latitude"], errors="coerce")
        allp["lon"] = pd.to_numeric(allp["longitude"], errors="coerce")
        c1, c2 = st.columns([3, 1])
        with c2:
            present = [l for l in LAYER_COLOURS if l in set(allp["layer"])] + \
                      sorted(set(allp["layer"]) - set(LAYER_COLOURS))
            show_layers = st.multiselect("Layers", present, default=present, key="geo_layers")
            ev = st.multiselect("Evidence status", sorted(allp["evidence__status"].dropna().unique()),
                                default=sorted(allp["evidence__status"].dropna().unique()), key="geo_evidence")
            basemap = st.selectbox("Base map", ["carto-positron", "open-street-map", "carto-voyager",
                                                "coordinate plot (offline)"], key="geo_basemap",
                                   help="Street maps load tiles from the internet in your browser and need WebGL. "
                                        "If the map stays blank, choose 'coordinate plot (offline)'.")
            st.caption("Only records supplied with coordinates are plotted. Records without a recognised status are "
                       "shown as 'Needs verification'.")
        sub = allp[allp["layer"].isin(show_layers) & allp["evidence__status"].isin(ev)]
        with c1:
            fig = go.Figure()
            street = basemap != "coordinate plot (offline)"
            for lyr, g in sub.groupby("layer"):
                hover = (g["name"].astype(str) + "<br>" + g["layer"] + "<br>Evidence: " + g["evidence__status"]
                         + g.get("city", pd.Series("", index=g.index)).astype(str).radd("<br>City: "))
                marker = dict(size=11, color=LAYER_COLOURS.get(lyr, theme.SLATE), opacity=0.85)
                trace = go.Scattermap if street else go.Scatter
                kw = dict(lat=g["lat"], lon=g["lon"]) if street else dict(x=g["lon"], y=g["lat"])
                fig.add_trace(trace(mode="markers", name=f"{lyr} ({len(g)})", marker=marker, text=hover,
                                    hovertemplate="%{text}<extra></extra>", **kw))
            if len(sub) and street:
                fig.update_layout(map=dict(style=basemap, center=dict(lat=float(sub["lat"].mean()),
                                                                     lon=float(sub["lon"].mean())),
                                           zoom=4 if sub["lat"].std() > 2 else 11), height=520,
                                  margin=dict(l=0, r=0, t=30, b=0))
            elif len(sub):
                ratio = 1 / max(np.cos(np.radians(float(sub["lat"].mean()))), 0.2)
                fig.update_layout(xaxis=dict(title="Longitude"), yaxis=dict(title="Latitude", scaleanchor="x",
                                                                            scaleratio=ratio), height=520)
            charts.show(fig, "geo_map", sub.drop(columns=[c for c in sub.columns if c.endswith("__label")
                                                           and c != "category__label"], errors="ignore")
                        [[c for c in ("layer", "name", "city", "microzone", "lat", "lon", "evidence__status",
                                      "source_url", "verification_date") if c in sub.columns]])
        by = sub.groupby(["layer", "evidence__status"]).size().rename("records").reset_index()
        vshare = (sub["evidence__status"] == "Verified").mean() * 100 if len(sub) else 0
        cards.explanation("Points are supplied records with coordinates, coloured by type. Hover for evidence status.",
                          esc(f"{len(sub):,} points shown; {vshare:.0f}% carry 'Verified' status. "
                              + "; ".join(f"{r.layer}: {r.records}" for r in by.groupby('layer')['records'].sum()
                                          .reset_index().itertuples())),
                          esc("Clusters of travel nodes with few verified storage points are candidates for field "
                              "visits — not conclusions. Distances on a map are straight-line, not walking routes."),
                          esc("Coverage depends entirely on what has been audited; missing points may simply be "
                              "unrecorded."))
        if "nearest_storage_distance_m" in sub:
            st.caption("Distances come from geographic_market_audit.csv; the 'distance_type' column states whether "
                       "they are walking, road or straight-line, and only rows with evidence 'Verified' should be "
                       "treated as verified.")


# ------------------------------------------------------------------ opportunity
def _opportunity(ctx) -> None:
    with cards.card("Opportunity score settings",
                    help_text="Weights are PROVISIONAL assumptions for exploration, not statistically established."):
        cols = st.columns(5)
        weights = {}
        for col, (k, lbl, desc) in zip(cols, COMPONENTS):
            weights[k] = col.slider(f"{lbl} weight", 0, 100, DEFAULT_WEIGHTS[k], 5, key=f"w_{k}", help=desc)
        c1, c2, c3 = st.columns([1.2, 1.2, 1])
        scaling = c1.radio("Normalisation to 0–100", ["absolute", "relative"], horizontal=True, key="geo_scaling",
                           format_func=lambda x: {"absolute": "Absolute % (default)",
                                                  "relative": "Relative min–max across zones"}[x],
                           help="Absolute keeps percentages as scores; relative stretches the best zone to 100 and the "
                                "worst to 0, which exaggerates small differences.")
        window = c2.slider("Service window partners must cover (hour of day)", 6, 23, (11, 17), key="geo_window",
                           help="Partner opening-hour compatibility = opens by the start and closes after the end.")
        allow_partial = c3.toggle("Score zones with missing components", value=False, key="geo_partial",
                                  help="Off: a combined score needs all weighted components. On: re-weights the "
                                       "available ones (marked 'partial').")
        tot = sum(weights.values())
        st.caption(f"Weights total {tot} → normalised to 100%. Defaults: demand 25, unmet need 25, customer "
                   "acceptance 20, partner readiness 15, operational feasibility 15.")
    zt, info = zone_table(ctx, window=(float(window[0]), float(window[1])))
    if zt.empty:
        cards.missing_columns(["city", "microzone (or locality)"], "Microzone analysis needs both location fields.")
        return
    z = score(zt, weights, scaling, ctx.min_n, allow_partial)
    zd = score(zt, DEFAULT_WEIGHTS, scaling, ctx.min_n, allow_partial)
    z["rank_current"] = z["opportunity_score"].rank(ascending=False, method="min")
    z["rank_default_weights"] = zd["opportunity_score"].rank(ascending=False, method="min")
    if not info["has_partner"]:
        cards.notice("pending", "Partner readiness and operational feasibility cannot be scored.",
                     "Supply <code>partner_audit.csv</code> or <code>partner_survey.csv</code> with city and "
                     "microzone. Combined scores stay blank unless partial scoring is switched on.")
    if not info["has_competitor"]:
        cards.notice("pending", "Competitor evidence missing.",
                     "Without <code>competitor_locations.csv</code>, no zone can be assessed for market gaps.")

    with cards.card("Component and combined scores by microzone",
                    "Which zones combine demand, unmet need, acceptance and partner readiness?"):
        sc = [f"score_{k}" for k, _, _ in COMPONENTS] + ["opportunity_score"]
        names = [lbl for _, lbl, _ in COMPONENTS] + ["Combined"]
        zz = z.sort_values("opportunity_score", ascending=False, na_position="last")
        mat = zz[sc].copy()
        mat.columns = names
        mat.index = [f"{c} · {m} (n={n})" for c, m, n in zip(zz["city"], zz["microzone"], zz["respondents"])]
        fig = charts.heatmap(mat, text_fmt="{:.0f}", zmin=0, zmax=100, value_name="Score (0–100)",
                             colorbar_title="score")
        charts.show(fig, "geo_scores")
        rel = zz[(zz["respondents"] >= ctx.min_n) & zz["opportunity_score"].notna()]
        if len(rel):
            t = rel.iloc[0]
            weak = dict((k, l) for k, l, _ in COMPONENTS).get(t["weakest_component"], t["weakest_component"])
            obs = (f"Highest combined score: <b>{esc(t['city'])} · {esc(t['microzone'])}</b> "
                   f"({t['opportunity_score']:.0f}/100, n={int(t['respondents'])}, {esc(t['score_basis'])}); its weakest "
                   f"component is <b>{esc(weak)}</b> ({t['score_' + t['weakest_component']]:.0f}). "
                   f"{int(zz['opportunity_score'].notna().sum())} of {len(zz)} zones could be scored.")
        else:
            obs = esc(f"No zone with n ≥ {ctx.min_n} has a complete score. Adjust the minimum sample, supply partner "
                      "data, or enable partial scoring.")
        cards.explanation(
            "Each component is a percentage (0–100) from the filtered consumer data and the partner file; the combined "
            "score is their weighted average. Rows show all five so a strong total cannot hide a weak component.",
            obs,
            esc("Use the ranking to choose where to do field verification and partner recruitment first — not as a "
                "launch decision. A high score with weak feasibility means 'recruit partners', not 'launch'."),
            common.limitation(ctx, int(zz["respondents"].sum()),
                              f"Partner components from: {info['partner_source'] or 'not available'}. Weights are "
                              "provisional. Zones below the minimum sample are shown but not ranked."))

    with cards.card("Microzone comparison table", "How do zones compare on each underlying measure?"):
        show = z.sort_values("opportunity_score", ascending=False, na_position="last")
        cols = ["city", "microzone", "respondents", "long_gap_pct", "unmet_pct", "high_intent_pct",
                "median_storage_wtp", "acceptance_pct", "partners_assessed", "partners_willing", "verified_partners",
                "capacity_willing", "hours_compatible_pct", "secure_space_pct", "digital_handover_pct",
                "competitor_records", "competitors_verified", "nearest_storage_option", "nearest_storage_distance_m",
                "distance_type", "opportunity_score", "score_basis", "weakest_component", "rank_current",
                "rank_default_weights", "evidence_status"]
        show = show[[c for c in cols if c in show.columns]]
        if "nearest_storage_option" not in show:
            show["nearest_storage_option"] = "Not available — needs verified distance data"
        pct_cfg = {c: st.column_config.ProgressColumn(c.replace("_pct", " %").replace("_", " "), min_value=0,
                                                      max_value=100, format="%.0f%%")
                   for c in show.columns if c.endswith("_pct")}
        pct_cfg["opportunity_score"] = st.column_config.ProgressColumn("Combined score", min_value=0, max_value=100,
                                                                       format="%.0f")
        pct_cfg["median_storage_wtp"] = st.column_config.NumberColumn("Median storage WTP", format="₹%.0f")
        charts.table(show, key="geo_zone_table", filename="microzone_opportunity.csv", column_config=pct_cfg)
        st.caption("Percentages use the valid answers in each zone (respondent counts shown). Verified partners = "
                   "evidence status 'Verified' and willing. Capacity sums P6 for willing partners only.")


# ------------------------------------------------------------------ cities
def _city(ctx) -> None:
    d = ctx.cf
    if "city__label" not in d:
        return
    rows = []
    for city, g in d.groupby("city__label", observed=True):
        row = {"city": city, "respondents": len(g)}
        for col, name in (("long_gap", "Gap ≥ 3 h %"), ("unmet_need", "Unmet need %"),
                          ("accepts_partner", "Accept partner venues %"), ("high_intent", "High intent %")):
            if col in g:
                row[name] = S.share(g[col])[0]
        if "storage_wtp_inr__value" in g:
            row["Median storage WTP ₹"] = g["storage_wtp_inr__value"].median()
        rows.append(row)
    cdf = pd.DataFrame(rows)
    _, pf = primary_partner_frame(ctx)
    pf = filter_market(ctx, pf)
    if len(pf) and "city" in pf:
        from dashboard.views.market import _key
        cnt = pf.groupby(pf["city"].map(_key)).agg(partners=("city", "size"),
                                                   willing=("is_willing", lambda s: int(s.fillna(False).sum()))
                                                   if "is_willing" in pf else ("city", "size"))
        cdf["partners assessed"] = cdf["city"].map(lambda c: cnt["partners"].get(_key(c), 0))
        if "is_willing" in pf:
            cdf["partners willing"] = cdf["city"].map(lambda c: cnt["willing"].get(_key(c), 0))
    comp = filter_market(ctx, ctx.competitors)
    if len(comp) and "city" in comp:
        from dashboard.views.market import _key
        vc = comp.assign(k=comp["city"].map(_key)).groupby("k")["evidence__status"].apply(lambda s: (s == "Verified").sum())
        cdf["verified competitors"] = cdf["city"].map(lambda c: int(vc.get(_key(c), 0)))
    with cards.card("City comparison", "How do the four study cities differ?"):
        metric_cols = [c for c in cdf.columns if c.endswith("%")]
        if metric_cols:
            fig = charts.grouped_bar(cdf, "city", metric_cols, y_title="% of valid answers")
            charts.show(fig, "geo_city", cdf.round(2))
        best = cdf.sort_values("Unmet need %", ascending=False).iloc[0] if "Unmet need %" in cdf and len(cdf) else None
        obs = (f"Highest unmet-need share: <b>{esc(best['city'])}</b> ({best['Unmet need %']:.1f}%, n={int(best['respondents'])})."
               if best is not None else "Unmet need not available.")
        cards.explanation(
            "City-level shares of the main demand indicators, with partner and verified-competitor counts in the table.",
            obs,
            esc("The best first-pilot city is not necessarily the one with the most tourists. A pilot needs a dense "
                "cluster of willing partners near transport nodes, a sizeable unmet-need group, acceptable prices and "
                "manageable competition — a smaller city with strong partner readiness can beat a larger one where "
                "official cloakrooms and lockers already cover demand. Drill into microzones above before deciding."),
            common.limitation(ctx, int(cdf["respondents"].sum()), "City samples follow the study design, not market size."))
