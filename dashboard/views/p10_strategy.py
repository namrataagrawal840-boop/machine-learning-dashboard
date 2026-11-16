"""Page 10 — Business strategy, sales pipeline, unit economics and validation tracker."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from dashboard.components import cards, charts
from dashboard.components import stats as S
from dashboard.config import theme
from dashboard.config.schema import PROVENANCE
from dashboard.data.transforms import norm_text
from dashboard.views import common
from dashboard.views.common import multi_table
from dashboard.views.market import DEFAULT_WEIGHTS, filter_market, primary_partner_frame, score, zone_table

esc = cards.esc
FUNNEL_STAGES = ["Awareness", "Website visit", "Location search", "Availability check", "Booking initiation",
                 "Paid booking", "Completed handover", "Repeat purchase"]


def render(ctx) -> None:
    cards.page_header("Business strategy & sales pipeline",
                      "What do the analyses imply for launch decisions — and what evidence is still missing?",
                      "Decision support · recommendations", [ctx.r("consumer_clean"), ctx.r("partner_audit"),
                                                             ctx.r("partner_survey"), ctx.r("booking_funnel")])
    level = _evidence_level(ctx)
    cards.notice(*level)
    tabs = st.tabs(["Decision questions", "Product roadmap", "Sales funnel", "Unit economics", "Validation tracker"])
    with tabs[0]:
        _decisions(ctx)
    with tabs[1]:
        _roadmap(ctx)
    with tabs[2]:
        _funnel(ctx)
    with tabs[3]:
        _economics(ctx)
    with tabs[4]:
        _tracker(ctx)


def _evidence_level(ctx):
    fun = ctx.r("booking_funnel")
    if ctx.consumer is None:
        return ("pending", "No consumer evidence loaded.", "Recommendations cannot be generated yet.")
    if ctx.cprov == "synthetic":
        return ("warn", "Evidence level: INDICATIVE ONLY — based on synthetic survey data.",
                "Answers below show how the decision logic works on the simulated data. The business has <b>not</b> been "
                "validated; every recommendation must be re-run on real survey, partner and pilot data.")
    if not fun.ok:
        return ("info", f"Evidence level: stated preferences ({esc(PROVENANCE.get(ctx.cprov, ctx.cprov))}).",
                "No transaction data yet — recommendations rest on what respondents say, not what they do.")
    return ("ok", "Evidence includes pilot transaction data.", "Survey and transaction evidence are shown separately.")


# ------------------------------------------------------------------ decisions
def _strength(ctx, n: int) -> str:
    if ctx.cprov == "synthetic":
        return "Indicative (synthetic)"
    if n < ctx.min_n:
        return "Weak (small sample)"
    return "Stated preference" if not ctx.r("booking_funnel").ok else "Stated + transactions"


def _best_group(d, col, flag, min_n):
    if col not in d or flag not in d:
        return None
    g = S.group_share(d, col, flag, min_n)
    g = g[g["reliable"]].sort_values("pct", ascending=False)
    return g.iloc[0] if len(g) else None


def _decisions(ctx):
    if ctx.consumer is None:
        cards.missing_dataset(ctx.r("consumer_clean"), "Decision answers are calculated from the consumer survey.")
        return
    if not common.require_consumer(ctx):
        return
    d = ctx.cf
    rows = []

    # 1 target segment
    ans, miss, n = [], "", ctx.n_filtered
    if "cluster" in d and "high_intent" in d:
        g = d.dropna(subset=["cluster"]).groupby("cluster").agg(n=("cluster", "size"),
                                                                hi=("high_intent", lambda s: S.share(s)[0]),
                                                                un=("unmet_need", lambda s: S.share(s)[0])
                                                                if "unmet_need" in d else ("cluster", "size"))
        g = g[g["n"] >= ctx.min_n]
        if len(g):
            g["combo"] = g["hi"] + (g["un"] if "unmet_need" in d else 0)
            b = g.sort_values("combo", ascending=False).iloc[0]
            ans.append(f"{b.name}: {b['hi']:.0f}% high intent" + (f", {b['un']:.0f}% unmet need" if "unmet_need" in d else "")
                       + f" (n={int(b['n'])}). See page 6 for its profile.")
    for col, nm in (("travel_purpose__label", "purpose"), ("accommodation_type__label", "accommodation"),
                    ("trips_12m__label", "travel frequency")):
        b = _best_group(d, col, "unmet_need" if "unmet_need" in d else "high_intent", ctx.min_n)
        if b is not None:
            ans.append(f"by {nm}: '{b[col]}' has the highest "
                       f"{'unmet need' if 'unmet_need' in d else 'high intent'} ({b['pct']:.0f}%, n={int(b['n'])})")
    rows.append(("1. Which customer group should be targeted first?", "; ".join(ans) or "Insufficient data.",
                 "Cluster assignments; real survey responses" if "cluster" not in d else "Real survey responses"))

    # 2 city / microzone
    zt, info = zone_table(ctx)
    best_zone, recruit = None, []
    if not zt.empty:
        z = score(zt, DEFAULT_WEIGHTS, "absolute", ctx.min_n, allow_partial=not info["has_partner"])
        rel = z[(z["respondents"] >= ctx.min_n) & z["opportunity_score"].notna()].sort_values("opportunity_score",
                                                                                              ascending=False)
        if len(rel):
            best_zone = rel.iloc[0]
        if info["has_partner"]:
            hi_dem = z[(z["respondents"] >= ctx.min_n) & (z["score_demand"] >= z["score_demand"].median())]
            recruit = hi_dem[(hi_dem["score_readiness"].fillna(0) < 50) | hi_dem["partners_assessed"].isna()]
    rows.append(("2. Which city and microzone should be tested first?",
                 (f"{best_zone['city']} · {best_zone['microzone']} has the highest provisional opportunity score "
                  f"({best_zone['opportunity_score']:.0f}/100, {best_zone['score_basis']}; weakest component: "
                  f"{best_zone['weakest_component']}). Treat as the first zone to field-verify, not to launch.")
                 if best_zone is not None else "No zone has enough data to score.",
                 "" if info["has_partner"] and info["has_competitor"] else
                 "Partner audit and verified competitor locations"))

    # 3 storage first?
    if "first_service_preference__label" in d:
        fs = d["first_service_preference__label"].dropna().str.lower()
        st_share = fs.str.contains("drop my luggage|locker").mean() * 100
        pk_share = fs.str.contains("pick|deliver").mean() * 100
        a3 = (f"{st_share:.0f}% would try a storage-only option first vs {pk_share:.0f}% a pickup/delivery option (Q24, "
              f"n={len(fs):,}). " + ("Supports a storage-first launch." if st_share >= pk_share else
                                      "Pickup interest is at least as strong — but storage-first remains lower-risk "
                                      "until courier costs are known."))
    else:
        a3 = "Q24 not available."
    rows.append(("3. Should the initial service offer luggage storage only?", a3, "Courier cost data; pilot uptake"))

    # 4 & 5 pickup / delivery
    pk = d["pickup_delivery_wtp_inr__value"].dropna() if "pickup_delivery_wtp_inr__value" in d else pd.Series(dtype=float)
    det = (d["maximum_detour__label"] == "Pickup from accommodation only").mean() * 100 if "maximum_detour__label" in d else np.nan
    a4 = (f"Median extra pickup/delivery WTP ₹{pk.median():,.0f} ({(pk == 0).mean() * 100:.0f}% say ₹0, n={len(pk):,}); "
          f"{det:.0f}% would only use a service with pickup. " if len(pk) else "") + \
        "Justified only when (a) a sizeable group values pickup, (b) WTP covers courier + handling cost per bag, and " \
        "(c) partners accept courier handovers. (a) is indicated here; (b) needs real courier quotes."
    rows.append(("4. What would justify introducing pickup?", a4, "Courier cost per trip/bag; pilot add-on uptake"))
    pf_name, pf = primary_partner_frame(ctx)
    pdacc = ""
    if pf is not None and "pickup_delivery_acceptance__rank" in pf:
        p, k, n2 = S.share(filter_market(ctx, pf)["pickup_delivery_acceptance__rank"].dropna() >= 2)
        pdacc = f"{S.fmt_pct(p, k, n2)} of assessed partners can receive/release bags for couriers. "
    fsd = d["first_service_preference__label"].dropna().str.lower().str.contains("airport|railway|station|deliver").mean() * 100 \
        if "first_service_preference__label" in d else np.nan
    rows.append(("5. What would justify airport / railway delivery?",
                 (f"{fsd:.0f}% would first try a delivery option. " if not np.isnan(fsd) else "") + pdacc +
                 "Requires reliable scheduled delivery within departure deadlines — validate on-time performance and "
                 "cost in a Phase 2 pickup pilot before committing.", "Delivery time/cost trials; liability terms"))

    # 6 trust features
    sec, nsec = multi_table(d, "security_requirements")
    rows.append(("6. Which trust features are essential?",
                 ", ".join(f"{r.option} ({r.share_pct:.0f}%)" for r in sec.head(5).itertuples()) + f" — of {nsec:,} answering."
                 if len(sec) else "Q22 not available.", ""))

    # 7 partners to approach first
    a7 = "Partner data not available."
    if pf is not None:
        pff = filter_market(ctx, pf)
        if "establishment_type" in pff and "is_willing" in pff:
            g = S.group_share(pff.assign(est=pff["establishment_type"].astype(str).str.rstrip(".")), "est", "is_willing", 5)
            g = g.sort_values("pct", ascending=False)
            if len(g):
                a7 = (f"Most willing type: {g.iloc[0]['est']} ({g.iloc[0]['pct']:.0f}% of {int(g.iloc[0]['n'])}); "
                      "prioritise willing establishments with secure space, long opening hours and digital handover "
                      f"acceptance in the top-scoring zone ({pf_name}).")
    rows.append(("7. Which partners should be approached first?", a7, "Verified field audit"))

    # 8 pricing experiments
    a8 = ""
    if "storage_wtp_inr__value" in d:
        ds = S.describe_numeric(d["storage_wtp_inr__value"])
        if ds["n"]:
            a8 = (f"A/B test storage price arms around stated Q1/median/Q3: ₹{ds['q1']:,.0f} · ₹{ds['median']:,.0f} · "
                  f"₹{ds['q3']:,.0f} per bag (3 h). ")
    if "preferred_pricing_model__label" in d:
        top = d["preferred_pricing_model__label"].value_counts(normalize=True).head(2)
        a8 += "Test pricing structures: " + " vs ".join(f"'{i}' ({v * 100:.0f}%)" for i, v in top.items()) + ". "
    a8 += "Measure paid conversion, not stated intent."
    rows.append(("8. What pricing experiments should be conducted?", a8, "Landing-page price tests; paid pilot bookings"))

    # 9 partner recruitment locations
    if info.get("has_partner") and len(recruit):
        a9 = "; ".join(f"{r.city} · {r.microzone} (demand {r.score_demand:.0f}, readiness "
                       f"{'n/a' if pd.isna(r.score_readiness) else f'{r.score_readiness:.0f}'})"
                       for r in recruit.head(5).itertuples())
    elif info.get("has_partner"):
        a9 = "No high-demand zone has low partner readiness under current filters."
    else:
        a9 = "Cannot be determined without partner data by microzone."
    rows.append(("9. Which locations need additional partner recruitment?", a9, "Partner audit per microzone"))

    # 10 missing evidence
    missing = [r.spec.title for r in ctx.results.values() if not r.ok and r.key != "cleaning_log"]
    prov_issue = [r.spec.title for r in ctx.results.values() if r.ok and r.provenance == "unclassified"]
    a10 = ("Missing inputs: " + ", ".join(missing) + ". " if missing else "") + \
          ("Provenance undeclared: " + ", ".join(prov_issue) + ". " if prov_issue else "") + \
          ("Consumer data is synthetic — real responses are required. " if ctx.cprov == "synthetic" else "") + \
          "See the validation tracker."
    rows.append(("10. What additional data is required before launch?", a10, ""))

    with cards.card("Evidence-supported answers", help_text="Each answer is generated from the currently filtered data. "
                                                            "'Evidence still needed' lists what would make it decision-grade."):
        st.markdown(cards.html_table([{"Decision question": f"<b>{esc(q)}</b>", "What the data indicates": esc(a),
                                       "Evidence strength": esc(_strength(ctx, ctx.n_filtered)),
                                       "Evidence still needed": esc(m or "—")} for q, a, m in rows]),
                    unsafe_allow_html=True)


# ------------------------------------------------------------------ roadmap
def _roadmap(ctx):
    d = ctx.cf if ctx.cf is not None else pd.DataFrame()
    have = {k: r.ok for k, r in ctx.results.items()}
    synthetic = ctx.cprov == "synthetic"

    def status(ok_data: bool, ok_signal: bool | None) -> str:
        if not ok_data:
            return "⚪ Evidence missing"
        if synthetic:
            return "🟣 Indicative only (synthetic)"
        if ok_signal is None:
            return "🟡 Partially evidenced"
        return "🟢 Signal supports" if ok_signal else "🔴 Signal does not support"

    unmet = S.share(d["unmet_need"])[0] if "unmet_need" in d else np.nan
    acc = S.share(d["accepts_partner"])[0] if "accepts_partner" in d else np.nan
    pk_med = d["pickup_delivery_wtp_inr__value"].median() if "pickup_delivery_wtp_inr__value" in d else np.nan
    phases = [
        ("Phase 1 · Basic verified luggage storage", "Online booking, transparent pricing, digital handover records",
         "Unmet need present; customers accept partner venues; ≥ 5–10 verified willing partners in one microzone; "
         "positive contribution per booking at tested prices",
         status(have["consumer_clean"] and (have["partner_audit"] or have["partner_survey"]),
                (unmet >= 10 and acc >= 50) if not np.isnan(unmet) and not np.isnan(acc) else None),
         f"Unmet need {S.fmt_pct(unmet)}; partner-venue acceptance {S.fmt_pct(acc)}"),
        ("Phase 2 · Pickup + storage", "Scheduled collection from accommodation to a partner",
         "Phase 1 paid bookings; pickup add-on uptake in pilot; courier cost < pickup price",
         status(have["booking_funnel"], None),
         f"Stated median pickup WTP ₹{pk_med:,.0f}" if not np.isnan(pk_med) else "Pickup WTP n/a"),
        ("Phase 3 · Scheduled delivery to airport / station", "Time-critical delivery to departure point",
         "Reliable on-time pickup in Phase 2; liability/insurance terms; delivery cost covered",
         status(False, None), "Requires Phase 2 operational data"),
        ("Phase 4 · Multi-city integrated network", "Consistent partner standards across cities",
         "Repeatable unit economics in the first city; partner density in additional microzones",
         status(False, None), "Requires Phase 1–3 results"),
    ]
    with cards.card("Phased product roadmap", "When should each service be introduced?",
                    help_text="Advance a phase only when its gate is met with real customer, partner and financial "
                              "evidence. Pickup and delivery are not assumed to launch with storage."):
        st.markdown(cards.html_table([{"Phase": f"<b>{esc(a)}</b>", "Scope": esc(b), "Gate to advance": esc(c),
                                       "Current status": esc(s), "Current evidence": esc(e)}
                                      for a, b, c, s, e in phases]), unsafe_allow_html=True)


# ------------------------------------------------------------------ funnel
def _funnel(ctx):
    fun = ctx.r("booking_funnel")
    if fun.ok:
        _actual_funnel(ctx, fun)
    else:
        cards.notice("pending", "No actual sales or analytics data — the funnel below is a SURVEY-INTENT funnel.",
                     "It shows how many respondents pass successive survey conditions. It is not sales performance. "
                     "Add <code>booking_funnel.csv</code> from the pilot to measure real conversion.")
    if ctx.cf is not None and not ctx.cf.empty:
        d = ctx.cf
        stages = [("All respondents (filtered)", len(d))]
        cur = pd.Series(True, index=d.index)
        for col, nm in (("long_gap", "Gap ≥ 3 h"), ("unmet_need", "Unmet need"),
                        ("accepts_partner", "Accepts partner venues"), ("high_intent", "High stated intent (8–10)")):
            if col in d:
                cur = cur & d[col].fillna(False).astype(bool)
                stages.append((nm, int(cur.sum())))
        if "pilot_interest__rank" in d:
            cur = cur & (d["pilot_interest__rank"] >= 2)
            stages.append(("Pilot: yes, definitely", int(cur.sum())))
        with cards.card("Survey-intent funnel (not sales)", "How many respondents meet each successive condition?"):
            fig = charts.funnel([s for s, _ in stages], [v for _, v in stages], colour=theme.LAVENDER)
            charts.show(fig, "st_intent_funnel", pd.DataFrame(stages, columns=["condition", "respondents"]), height=360)
            cards.explanation("Each stage keeps respondents who also meet the next condition.",
                              esc(f"{stages[-1][1]:,} of {stages[0][1]:,} respondents ({stages[-1][1] / max(stages[0][1], 1) * 100:.1f}%) "
                                  "meet every condition."),
                              esc("This is an upper bound on an early-adopter pool for outreach — real conversion will "
                                  "be lower."),
                              common.limitation(ctx, stages[0][1]))
    with cards.card("Proposed sales-funnel measurement framework",
                    help_text="Stages to instrument once the pilot runs. Nothing here is measured yet unless booking "
                              "data is supplied."):
        measured = set()
        if fun.ok and "stage" in fun.df:
            measured = {norm_text(s) for s in fun.df["stage"].dropna().unique()}
        st.markdown(cards.html_table([{"Stage": esc(s), "Example metric": esc(m),
                                       "Status": "Measured" if norm_text(s) in measured else "Not yet measured"}
                                      for s, m in zip(FUNNEL_STAGES, [
                                          "Impressions / reach", "Unique visitors", "Searches with a location",
                                          "Searches returning an available slot", "Checkout started", "Paid bookings",
                                          "Bags handed over and collected", "Customers with ≥ 2 bookings"])]),
                    unsafe_allow_html=True)


def _actual_funnel(ctx, fun):
    df = fun.df.copy()
    with cards.card("Actual booking funnel (pilot data)", "Where do real users drop out?"):
        if "stage" not in df:
            cards.missing_columns(["stage"], "booking_funnel.csv needs a stage column.")
            return
        if "count" in df:
            agg = df.groupby("stage")["count"].sum()
        else:
            agg = df["stage"].value_counts()
        order = [s for s in FUNNEL_STAGES if s in agg.index] + [s for s in agg.index if s not in FUNNEL_STAGES]
        agg = agg.reindex(order)
        fig = charts.funnel(list(agg.index), list(agg.values.astype(float)), colour=theme.GREEN)
        charts.show(fig, "st_actual_funnel", agg.reset_index(), height=380)
        conv = (agg.iloc[-1] / agg.iloc[0] * 100) if agg.iloc[0] else np.nan
        cards.explanation("Counts per funnel stage from booking_funnel.csv.",
                          esc(f"{agg.iloc[-1]:,.0f} at '{agg.index[-1]}' from {agg.iloc[0]:,.0f} at '{agg.index[0]}' "
                              f"({conv:.1f}%)."),
                          esc("The largest stage-to-stage drop is the first thing to fix."),
                          esc(f"Source: {PROVENANCE.get(fun.provenance, fun.provenance)} ({fun.provenance_note})."))


# ------------------------------------------------------------------ economics
def _economics(ctx):
    d = ctx.cf if ctx.cf is not None else pd.DataFrame()
    fun = ctx.r("booking_funnel")
    cost_cols = ["booking_revenue", "partner_payment", "transport_cost", "payment_fee", "other_variable_cost"]
    if fun.ok and any(c in fun.df for c in cost_cols):
        f = fun.df.copy()
        for c in cost_cols:
            f[c] = pd.to_numeric(f.get(c, 0), errors="coerce").fillna(0)
        f["contribution"] = f["booking_revenue"] - f[cost_cols[1:]].sum(axis=1)
        paid = f[f["booking_revenue"] > 0]
        with cards.card("Actual contribution per booking (pilot data)"):
            cards.kpi_grid([cards.kpi("Paid bookings", f"{len(paid):,}", "rows with revenue", fun.provenance),
                            cards.kpi("Avg revenue", S.fmt_inr(paid["booking_revenue"].mean()), "", fun.provenance),
                            cards.kpi("Avg contribution", S.fmt_inr(paid["contribution"].mean()), "", fun.provenance),
                            cards.kpi("Contribution margin", f"{paid['contribution'].sum() / paid['booking_revenue'].sum() * 100:.1f}%"
                                      if paid["booking_revenue"].sum() else None, "", fun.provenance)], per_row=4)
    with cards.card("Contribution-margin scenario calculator",
                    "Could a booking cover its variable costs?",
                    help_text="ILLUSTRATIVE FINANCIAL SCENARIO. Every input is an assumption you enter; nothing here is "
                              "a forecast or measured result."):
        st.markdown(cards.badge("illustrative") + " <span class='lc-muted'>All values below are user assumptions.</span>",
                    unsafe_allow_html=True)
        med_s = d["storage_wtp_inr__value"].median() if "storage_wtp_inr__value" in d else np.nan
        med_p = d["pickup_delivery_wtp_inr__value"].median() if "pickup_delivery_wtp_inr__value" in d else np.nan
        c1, c2, c3, c4 = st.columns(4)
        bookings = c1.number_input("Paid bookings (scenario)", 0, 1_000_000, 0, 10, key="ue_bookings",
                                   help="Assumption. Leave 0 until you have a pilot estimate.")
        rev_s = c2.number_input("Storage revenue / booking (₹)", 0.0, 100000.0, 0.0, 10.0, key="ue_rev_s",
                                help=f"Assumption. Stated median WTP for reference: "
                                     f"{'n/a' if np.isnan(med_s) else f'₹{med_s:,.0f}'} (stated, not a price).")
        pickup_share = c3.slider("Bookings adding pickup (%)", 0, 100, 0, 5, key="ue_pick_share")
        rev_p = c4.number_input("Pickup revenue / pickup booking (₹)", 0.0, 100000.0, 0.0, 10.0, key="ue_rev_p",
                                help=f"Assumption. Stated median extra WTP: {'n/a' if np.isnan(med_p) else f'₹{med_p:,.0f}'}.")
        c1, c2, c3, c4 = st.columns(4)
        payout = c1.number_input("Partner payout / booking (₹)", 0.0, 100000.0, 0.0, 5.0, key="ue_payout")
        courier = c2.number_input("Courier cost / pickup booking (₹)", 0.0, 100000.0, 0.0, 10.0, key="ue_courier")
        fee_pct = c3.number_input("Payment fees (% of revenue)", 0.0, 20.0, 0.0, 0.1, key="ue_fee")
        refunds = c4.number_input("Refunds & claims / booking (₹)", 0.0, 100000.0, 0.0, 5.0, key="ue_refunds")
        c1, c2, c3, c4 = st.columns(4)
        handling = c1.number_input("Other variable cost / booking (₹)", 0.0, 100000.0, 0.0, 5.0, key="ue_other",
                                   help="Insurance/protection, seals, SMS, handling.")
        cac = c2.number_input("Customer acquisition cost / booking (₹)", 0.0, 100000.0, 0.0, 10.0, key="ue_cac")
        ps = pickup_share / 100
        revenue = rev_s + ps * rev_p
        var_cost = payout + ps * courier + revenue * fee_pct / 100 + refunds + handling
        contrib = revenue - var_cost
        after_cac = contrib - cac
        cards.kpi_grid([
            cards.kpi("Revenue / booking", S.fmt_inr(revenue), "storage + share × pickup", "illustrative", theme.BLUE),
            cards.kpi("Variable cost / booking", S.fmt_inr(var_cost), "payout + courier + fees + refunds + other",
                      "illustrative", theme.PEACH),
            cards.kpi("Contribution / booking", S.fmt_inr(contrib),
                      f"margin {contrib / revenue * 100:.1f}%" if revenue else "enter revenue", "illustrative", theme.GREEN),
            cards.kpi("After acquisition cost", S.fmt_inr(after_cac),
                      f"× {bookings:,} bookings = {S.fmt_inr(after_cac * bookings)}" if bookings else
                      "enter a booking scenario", "illustrative", theme.LAVENDER),
        ], per_row=4)
        st.caption("Contribution per booking = booking revenue − partner payment − transportation cost − payment fees − "
                   "refunds/claims − other variable operating costs. High stated WTP does not imply profitability; fixed "
                   "costs (salaries, technology, marketing overheads) are not included.")
        if revenue:
            items = pd.DataFrame({"item": ["Revenue", "Partner payout", "Courier (expected)", "Payment fees",
                                           "Refunds & claims", "Other variable", "Contribution"],
                                  "inr": [revenue, -payout, -ps * courier, -revenue * fee_pct / 100, -refunds, -handling,
                                          contrib]})
            import plotly.graph_objects as go
            fig = go.Figure(go.Waterfall(x=items["item"], y=items["inr"],
                                         measure=["absolute"] + ["relative"] * 5 + ["total"],
                                         connector=dict(line=dict(color=theme.BORDER)),
                                         increasing=dict(marker=dict(color=theme.GREEN)),
                                         decreasing=dict(marker=dict(color=theme.PEACH)),
                                         totals=dict(marker=dict(color=theme.BLUE))))
            fig.update_layout(yaxis_title="₹ per booking (assumed)")
            charts.show(fig, "ue_waterfall", items)


# ------------------------------------------------------------------ tracker
def _tracker(ctx):
    real_consumer = ctx.cprov == "primary" and ctx.consumer is not None
    n_real = ctx.n_total if real_consumer else 0
    _, pf = primary_partner_frame(ctx)
    real_partner = ctx.r("partner_audit").ok and ctx.r("partner_audit").provenance in ("primary", "secondary")
    n_part = len(ctx.partner_audit) if real_partner else 0
    comp = ctx.competitors
    n_comp_ver = int((comp["evidence__status"] == "Verified").sum()) if comp is not None else 0
    fun = ctx.r("booking_funnel")
    n_paid = 0
    if fun.ok:
        if "booking_revenue" in fun.df:
            n_paid = int((pd.to_numeric(fun.df["booking_revenue"], errors="coerce") > 0).sum())
        elif "stage" in fun.df and "count" in fun.df:
            n_paid = int(pd.to_numeric(fun.df.loc[fun.df["stage"].map(norm_text) == "paid booking", "count"],
                                       errors="coerce").sum())
    rows = [
        ("Consumer", "Collect real survey responses", "600–800 across four cities (~150–200 per city)", f"{n_real:,}",
         min(n_real / 600, 1.0)),
        ("Consumer", "Compare actual need with synthetic assumptions", "Side-by-side once real data exists",
         "Pending" if not real_consumer else "Possible", 1.0 if real_consumer else 0.0),
        ("Consumer", "Test willingness to pay with real prices", "Landing-page price test", "Not started" if not fun.ok else "Data present",
         0.0 if not fun.ok else 0.5),
        ("Partner", "Conduct establishment interviews / audits", "40–60 per city", f"{n_part:,} real audit records",
         min(n_part / 160, 1.0)),
        ("Partner", "Agree commercial, security & liability terms", "Signed terms with pilot partners", "Not tracked in data",
         0.0),
        ("Geographic", "Verify competitor locations", "All listed competitors verified", f"{n_comp_ver:,} verified",
         min(n_comp_ver / max(len(comp), 1), 1.0) if comp is not None else 0.0),
        ("Geographic", "Measure real walking / road access", "Verified distances per microzone",
         "Present" if ctx.geo is not None and "nearest_storage_distance_m" in ctx.geo else "Not supplied",
         0.5 if ctx.geo is not None and "nearest_storage_distance_m" in ctx.geo else 0.0),
        ("Commercial", "Pilot storage-only bookings in one microzone", "5–10 vetted partners; first 50–100 paid bookings",
         f"{n_paid:,} paid bookings", min(n_paid / 50, 1.0)),
        ("Commercial", "Then test pickup / delivery", "After storage pilot", "Not started", 0.0),
        ("Commercial", "Calculate real contribution margins", "From pilot revenue and cost records",
         "Possible" if fun.ok and "booking_revenue" in fun.df else "Not possible yet",
         1.0 if fun.ok and "booking_revenue" in fun.df else 0.0),
    ]
    df = pd.DataFrame(rows, columns=["area", "validation step", "exploratory target", "current status", "progress"])
    with cards.card("Real-world validation tracker", "What must happen before launch?",
                    help_text="Targets are research and pilot-planning guides, not claims of statistical "
                              "representativeness or guaranteed success. Status is read from the supplied files."):
        charts.table(df, key="st_tracker", filename="validation_tracker.csv", column_config={
            "progress": st.column_config.ProgressColumn("Progress", min_value=0, max_value=1, format="percent")})
        if ctx.cprov == "synthetic":
            st.caption("Synthetic consumer records do not count towards the real-response target.")
