"""Page 5 — Partner network and local acceptance (supply side)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from dashboard.components import cards, charts
from dashboard.components import stats as S
from dashboard.config import theme
from dashboard.config.schema import PROVENANCE, TRUST_CAPABILITY_PAIRS
from dashboard.data.transforms import MS_SEP, ms_columns, ms_option
from dashboard.views import common
from dashboard.views.market import filter_market, partner_frames, zone_column

esc = cards.esc


def render(ctx) -> None:
    cards.page_header("Partner network & local acceptance",
                      "Can enough suitable, willing and verified local establishments support the service?",
                      "Business application · supply-side evidence",
                      [ctx.r("partner_survey"), ctx.r("partner_audit"), ctx.r("consumer_clean")])
    frames = partner_frames(ctx)
    if not frames:
        cards.missing_dataset(ctx.r("partner_survey"), "Supply-side analysis needs the partner survey (P1–P14).")
        cards.missing_dataset(ctx.r("partner_audit"), "The audit adds field-verified status, capacity and coordinates.")
        return
    with cards.card("Partner dataset",
                    help_text="Surveyed establishments are potential partners. Only records with evidence status "
                              "'Verified' AND willingness to store non-customers count as verified operational."):
        names = [t for t, _, _ in frames]
        pick = st.segmented_control("Dataset", names, default=names[0], key="pt_which",
                                    label_visibility="collapsed") or names[0]
        title, df, prov = next(f for f in frames if f[0] == pick)
        df = filter_market(ctx, df)
        meta = ctx.pmeta.get("partner_audit" if pick == "Partner audit" else "partner_survey", {})
        st.markdown(f"{cards.badge(prov)} <span class='lc-muted'>{esc(PROVENANCE.get(prov, prov))} · "
                    f"{len(df):,} records after city/microzone filters</span>", unsafe_allow_html=True)
        if prov == "unclassified":
            st.caption("Provenance not declared — set it in data/sources.toml (e.g. partner_survey = \"synthetic\").")
    if df.empty:
        cards.notice("warn", "No partner records match the city/microzone filters.")
        return
    _kpis(df, prov, meta)
    _pipeline(df)
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        _willingness_by_type(ctx, df, prov)
    with c2:
        _capacity_by_zone(ctx, df, prov)
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        _hours(ctx, df, prov)
    with c2:
        _readiness_by_zone(ctx, df, prov)
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        _simple(ctx, df, prov, "commercial_preference", "Preferred commercial arrangement",
                "Which payment model do partners prefer?", "P9 preferred arrangement.", theme.GOLD)
    with c2:
        _payment(ctx, df, prov)
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        _simple(ctx, df, prov, "digital_handover_acceptance__label", "Digital handover acceptance",
                "Will partners follow digital handover procedures?", "P11.", theme.TEAL)
    with c2:
        _simple(ctx, df, prov, "pickup_delivery_acceptance__label", "Pickup / delivery participation",
                "Can partners support courier operations (Phase 2)?", "P12.", theme.LAVENDER)
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        _multi(ctx, df, prov, "security_features", "Security facilities available",
               "What security can partners offer today?", theme.BLUE)
    with c2:
        _multi(ctx, df, prov, "partner_barriers", "Main reasons partners may decline",
               "What stops establishments from joining?", theme.ROSE)
    _trust_vs_capability(ctx, df, prov)


def _lim(prov: str, n: int, extra: str = "") -> str:
    s = f"n = {n:,} partner records. Source: {PROVENANCE.get(prov, prov)}."
    if prov == "synthetic":
        s += " Simulated records — not real establishments."
    s += " Stated willingness is not a signed agreement. " + extra
    return esc(s)


def _kpis(df, prov, meta) -> None:
    items = [cards.kpi("Establishments assessed", f"{len(df):,}", "after filters", prov, theme.SLATE)]
    if "is_willing" in df:
        p, k, n = S.share(df["is_willing"])
        items.append(cards.kpi("Willing to store non-customers", S.fmt_pct(p), f"{k:,} of {n:,} · P8 definitely/yes "
                                                                                "with procedures", prov, theme.GREEN))
    else:
        items.append(cards.kpi("Willing to store non-customers", None, "P8 partner_willingness not present", prov))
    if "evidence_status" in df:
        ver = int((df["is_verified"] & df.get("is_willing", pd.Series(True, index=df.index)).fillna(False)).sum())
        items.append(cards.kpi("Verified operational partners", f"{ver:,}", "evidence 'Verified' and willing", prov,
                               theme.GREEN))
    else:
        items.append(cards.kpi("Verified operational partners", None, "no evidence_status column — 0 can be treated "
                                                                       "as verified", prov))
    if "storage_capacity__value" in df:
        w = df[df["is_willing"].fillna(False).astype(bool)] if "is_willing" in df else df
        items.append(cards.kpi("Bag capacity (willing partners)", f"{w['storage_capacity__value'].sum():,.0f}",
                               f"P6 total across {w['storage_capacity__value'].notna().sum():,} willing establishments",
                               prov, theme.BLUE))
    if "expected_payment_inr__value" in df:
        d = S.describe_numeric(df["expected_payment_inr__value"])
        items.append(cards.kpi("Median expected payment", S.fmt_inr(d.get("median")) if d["n"] else None,
                               f"n = {d['n']:,} · unit varies (see P10 note)", prov, theme.GOLD))
    if "digital_handover_acceptance__rank" in df:
        p, k, n = S.share(df["digital_handover_acceptance__rank"].dropna() >= 2)
        items.append(cards.kpi("Accept digital handover", S.fmt_pct(p), f"{k:,} of {n:,} · yes or yes with training",
                               prov, theme.TEAL))
    if "pickup_delivery_acceptance__rank" in df:
        p, k, n = S.share(df["pickup_delivery_acceptance__rank"].dropna() >= 2)
        items.append(cards.kpi("Support courier pickup/delivery", S.fmt_pct(p), f"{k:,} of {n:,}", prov,
                               theme.LAVENDER))
    if "pilot_participation__rank" in df:
        p, k, n = S.share(df["pilot_participation__rank"].dropna() >= 2)
        items.append(cards.kpi("Would join a pilot (yes)", S.fmt_pct(p), f"{k:,} of {n:,} · P14", prov, theme.PEACH))
    with cards.card("Supply-side indicators"):
        cards.kpi_grid(items, per_row=4)
        if meta.get("willing_definition"):
            st.caption(meta["willing_definition"])


def _pipeline(df) -> None:
    stages = [("Assessed", len(df))]
    if "is_willing" in df:
        w = df["is_willing"].fillna(False).astype(bool)
        stages.append(("Willing (P8)", int(w.sum())))
        if "secure_storage_available__rank" in df:
            ws = w & (df["secure_storage_available__rank"] >= 2)
            stages.append(("Willing + secure space", int(ws.sum())))
        else:
            ws = w
        if "evidence_status" in df:
            stages.append(("Verified operational", int((ws & df["is_verified"]).sum())))
    with cards.card("Partner pipeline: surveyed → willing → verified",
                    "How many surveyed establishments could actually operate?"):
        fig = charts.funnel([s for s, _ in stages], [v for _, v in stages], colour=theme.GREEN)
        charts.show(fig, "pt_pipeline", pd.DataFrame(stages, columns=["stage", "establishments"]), height=300)
        last = stages[-1]
        cards.explanation("Each stage is a subset of the one above.",
                          esc(f"{last[1]:,} of {stages[0][1]:,} assessed establishments reach '{last[0]}'."),
                          esc("Recruitment targets should be set on the bottom of this funnel, not on establishments "
                              "listed online. 'Verified' requires a recorded field check."),
                          esc("Secure space = dedicated room or restricted-access area (P5)."))


def _willingness_by_type(ctx, df, prov):
    with cards.card("Partner willingness by establishment type", "Which establishment types are most open?"):
        if common.missing_cols(df, "establishment_type", "partner_willingness__label"):
            cards.missing_columns(["establishment_type", "partner_willingness"], "")
            return
        d = df.assign(est=df["establishment_type"].astype(str).str.rstrip("."))
        pct, cnt = S.crosstab_share(d, "est", "partner_willingness__label")
        rn = cnt.sum(axis=1)
        disp = pct.copy()
        disp.index = [f"{i} (n={rn[i]})" for i in pct.index]
        fig = charts.stacked_share(disp, cnt.set_axis(disp.index), horizontal=True, legend_title="Willingness",
                                   colours=[theme.RED, theme.GOLD, theme.TEAL, theme.GREEN])
        charts.show(fig, "pt_will_type", pct.round(1).reset_index(names="establishment_type"))
        g = S.group_share(d, "est", "is_willing", 1) if "is_willing" in d else pd.DataFrame()
        obs = ""
        if len(g):
            g = g.sort_values("pct", ascending=False)
            obs = (f"Most willing: <b>{esc(g.iloc[0]['est'])}</b> ({g.iloc[0]['pct']:.1f}%, n={int(g.iloc[0]['n'])}); "
                   f"least: <b>{esc(g.iloc[-1]['est'])}</b> ({g.iloc[-1]['pct']:.1f}%).")
        cards.explanation("Row % of P8 answers within each establishment type.", obs,
                          esc("Recruit first from the most willing types; adapt the offer (e.g. liability cover) for "
                              "types that answer 'maybe, depending on compensation'."),
                          _lim(prov, int(rn.sum())))


def _capacity_by_zone(ctx, df, prov):
    with cards.card("Bag capacity by locality", "Where is storage capacity concentrated?"):
        zc = zone_column(df)
        if zc is None or "storage_capacity__value" not in df:
            cards.missing_columns(["microzone", "storage_capacity"], "")
            return
        d = df.copy()
        d["willing"] = d["is_willing"].fillna(False).astype(bool) if "is_willing" in d else True
        g = d.groupby([d.get("city", pd.Series("", index=d.index)).astype(str), zc]).apply(
            lambda x: pd.Series({"capacity_all": x["storage_capacity__value"].sum(),
                                 "capacity_willing": x.loc[x["willing"], "storage_capacity__value"].sum(),
                                 "establishments": len(x), "willing": int(x["willing"].sum())}),
            include_groups=False).reset_index()
        g.columns = ["city", "zone", "capacity_all", "capacity_willing", "establishments", "willing"]
        g = g.sort_values("capacity_willing", ascending=False)
        g["label"] = g["city"] + " · " + g["zone"].astype(str) + " (" + g["willing"].astype(int).astype(str) + "/" + \
            g["establishments"].astype(int).astype(str) + " willing)"
        fig = charts.grouped_bar(g, "label", ["capacity_willing", "capacity_all"], ["Willing partners", "All assessed"],
                                 y_title="Medium suitcases (P6)", value_fmt="{:,.0f}", horizontal=True,
                                 colours=[theme.GREEN, "#C9D3DF"])
        charts.show(fig, "pt_cap_zone", g.drop(columns="label"))
        top = g.iloc[0] if len(g) else None
        cards.explanation("Sum of stated simultaneous suitcase capacity per microzone.",
                          esc(f"Largest willing capacity: {top['label']} — {top['capacity_willing']:,.0f} bags.")
                          if top is not None else "—",
                          esc("Compare with demand in the same zone (page 4). Capacity in one large venue is fragile; "
                              "several mid-size partners give coverage and redundancy."),
                          _lim(prov, len(df), "Self-reported capacity; verify on site."))


def _hours(ctx, df, prov):
    with cards.card("Opening-hour compatibility", "Are partners open when travellers need them (after checkout)?"):
        if common.missing_cols(df, "opening_time__hour", "closing_time__hour"):
            cards.missing_columns(["opening_time", "closing_time"], "Times like 09:00, 9 AM or 21:30 are accepted.")
            return
        d = df.dropna(subset=["opening_time__hour", "closing_time__hour"])
        if "is_willing" in d:
            only = st.toggle("Willing partners only", value=True, key="pt_hours_willing")
            if only:
                d = d[d["is_willing"].fillna(False).astype(bool)]
        hours = np.arange(6, 24)
        open_share = [float(((d["opening_time__hour"] <= h) & (d["closing_time__hour"] >= h + 1)).mean() * 100)
                      if len(d) else np.nan for h in hours]
        data = pd.DataFrame({"hour": [f"{h:02d}:00" for h in hours], "pct_open": np.round(open_share, 1),
                             "partners": len(d)})
        fig = charts.bar(data["hour"], data["pct_open"], [int(round(p * len(d) / 100)) for p in open_share],
                         [len(d)] * len(data), colour=theme.TEAL, axis_title="% of partners open for the full hour")
        fig.add_vrect(x0="11:00", x1="17:00", fillcolor=theme.SOFT_YELLOW, opacity=0.6, line_width=0, layer="below",
                      annotation_text="typical checkout → departure", annotation_position="top left",
                      annotation_font_color=theme.MUTED)
        charts.show(fig, "pt_hours", data)
        win = ((d["opening_time__hour"] <= 11) & (d["closing_time__hour"] >= 17)) if len(d) else pd.Series(dtype=bool)
        p, k, n = S.share(win)
        cards.explanation("Share of partners open for each full hour, from P3 opening and closing times.",
                          esc(f"{S.fmt_pct(p, k, n)} cover the whole 11:00–17:00 window."),
                          esc("Late-evening departures need partners open past 20:00; morning-only coverage leaves "
                              "the main post-checkout window unserved."),
                          _lim(prov, n, "Days of operation and holidays are not modelled."))


def _readiness_by_zone(ctx, df, prov):
    with cards.card("Partner readiness by microzone", "Which zones have a ready partner base?"):
        zc = zone_column(df)
        if zc is None or "is_willing" not in df:
            cards.missing_columns(["microzone", "partner_willingness"], "")
            return
        d = df.assign(zone=df.get("city", pd.Series("", index=df.index)).astype(str) + " · " + df[zc].astype(str))
        g = S.group_share(d, "zone", "is_willing", 1).sort_values("pct", ascending=False)
        fig = charts.bar(g["zone"], g["pct"], g["k"], g["n"], horizontal=True, colour=theme.GREEN,
                         axis_title="% willing (P8)")
        charts.show(fig, "pt_ready_zone", g)
        cards.explanation("Share of assessed establishments in each zone willing to store non-customers' luggage.",
                          esc(f"{len(g)} zones; highest {g.iloc[0]['zone']} ({g.iloc[0]['pct']:.1f}%, "
                              f"{int(g.iloc[0]['k'])}/{int(g.iloc[0]['n'])}).") if len(g) else "—",
                          esc("Zones with few assessed establishments need more field interviews before readiness can "
                              "be judged."),
                          _lim(prov, int(g["n"].sum()) if len(g) else 0))


def _simple(ctx, df, prov, col, title, question, measures, colour):
    with cards.card(title, question):
        if col not in df or df[col].dropna().empty:
            cards.missing_columns([col.split("__")[0]], "")
            return
        s = df[col] if col.endswith("__label") else df[col].astype(str).str.rstrip(".")
        order = list(s.cat.categories) if isinstance(s.dtype, pd.CategoricalDtype) else None
        fig, data = charts.count_bar(s, horizontal=True, order=order, sort=order is None, colour=colour)
        charts.show(fig, f"pt_{col}", data)
        top = data.sort_values("respondents", ascending=False).iloc[0]
        cards.explanation(esc(measures + " Share of establishments answering."),
                          f"Most common: <b>{esc(top['category'])}</b> ({top['share_pct']:.1f}%, "
                          f"{int(top['respondents'])} of {int(data['respondents'].sum())}).",
                          esc("Design the partner offer around the dominant answer while keeping alternatives for "
                              "sizeable minorities."), _lim(prov, int(data["respondents"].sum())))


def _payment(ctx, df, prov):
    with cards.card("Expected partner payment", "What would make the arrangement attractive?"):
        if "expected_payment_inr__value" not in df or df["expected_payment_inr__value"].dropna().empty:
            cards.missing_columns(["expected_payment_inr"], "")
            return
        d = S.describe_numeric(df["expected_payment_inr__value"])
        fig = charts.histogram(df["expected_payment_inr__value"], nbins=20, colour=theme.GOLD, x_title="₹",
                               marker_lines={"Median": d["median"]})
        charts.show(fig, "pt_payment", df[[c for c in ("partner_id", "establishment_type", "commercial_preference",
                                                         "expected_payment_inr") if c in df]])
        cards.explanation("Distribution of P10 expected payment.",
                          esc(f"Median ₹{d['median']:,.0f}, IQR ₹{d['q1']:,.0f}–₹{d['q3']:,.0f} (n={d['n']:,})."),
                          esc("Partner payout is the largest variable cost — compare it with consumer storage WTP on "
                              "page 9 and in the contribution calculator on page 10."),
                          _lim(prov, d["n"], "P10 units differ (per bag, hour, day or booking); compare like with like "
                                             "before using this as a cost input."))


def _multi(ctx, df, prov, field, title, question, colour):
    with cards.card(title, question):
        cols = ms_columns(df, field)
        if not cols:
            cards.missing_columns([field], "")
            return
        answered = df[cols].notna().any(axis=1)
        sub = df.loc[answered, cols].fillna(False).astype(bool)
        n = len(sub)
        data = pd.DataFrame({"option": [ms_option(c) for c in cols], "establishments": sub.sum().values.astype(int),
                             "share_pct": (sub.mean() * 100).values.round(2)}).sort_values("share_pct", ascending=False)
        fig = charts.bar(data["option"], data["share_pct"], data["establishments"], [n] * len(data), horizontal=True,
                         colour=colour, axis_title="% of establishments (multi-select)")
        charts.show(fig, f"pt_{field}", data)
        top = data.iloc[0]
        cards.explanation("Share of establishments selecting each option (multi-select).",
                          f"Most common: <b>{esc(top['option'])}</b> ({top['share_pct']:.1f}%).",
                          esc("For barriers: the top concerns define the partner contract (liability cover, space "
                              "guarantees, payout). For facilities: gaps vs customer requirements are shown below."),
                          _lim(prov, n))


def _trust_vs_capability(ctx, df, prov):
    with cards.card("Customer trust requirements vs partner security facilities",
                    "Can partners provide the security travellers say is essential?"):
        if ctx.cf is None:
            cards.missing_dataset(ctx.r("consumer_clean"), "This comparison needs consumer Q22.")
            return
        rows = []
        for cust, part in TRUST_CAPABILITY_PAIRS:
            cc = f"security_requirements{MS_SEP}{cust}"
            pc = f"security_features{MS_SEP}{part}"
            row = {"feature": cust, "partner facility": part}
            if cc in ctx.cf:
                p, k, n = S.share(ctx.cf[cc].where(ctx.cf[ms_columns(ctx.cf, 'security_requirements')].notna().any(axis=1)))
                row["customers requiring %"], row["customers n"] = p, n
            if pc in df:
                p, k, n = S.share(df[pc].where(df[ms_columns(df, 'security_features')].notna().any(axis=1)))
                row["partners offering %"], row["partners n"] = p, n
            rows.append(row)
        data = pd.DataFrame(rows)
        ycols = [c for c in ("customers requiring %", "partners offering %") if c in data]
        if len(ycols) < 2:
            cards.missing_columns(["security_requirements (consumer Q22)", "security_features (partner P7)"],
                                  "Both sides are needed.")
            return
        fig = charts.grouped_bar(data, "feature", ycols, ["Customers requiring", "Partners offering"],
                                 y_title="%", value_fmt="{:.0f}%", colours=[theme.LAVENDER, theme.GREEN])
        charts.show(fig, "pt_trust_cap", data.round(1))
        data["gap"] = data["customers requiring %"] - data["partners offering %"]
        w = data.sort_values("gap", ascending=False).iloc[0]
        cards.explanation("Pairs of comparable items: share of consumers listing a feature as essential (Q22) vs share "
                          "of partners already offering the matching facility (P7).",
                          esc(f"Largest shortfall: {w['feature']} — {w['customers requiring %']:.0f}% of customers "
                              f"require it vs {w['partners offering %']:.0f}% of partners offering it."),
                          esc("Shortfalls are what the platform must supply itself (e.g. seals, photo logs, lockable "
                              "bags) or require in partner onboarding."),
                          esc("Feature pairs are approximate matches. Consumer figures follow the global filters; "
                              "partner figures follow the city/microzone filters. ") + _lim(prov, len(df)))
