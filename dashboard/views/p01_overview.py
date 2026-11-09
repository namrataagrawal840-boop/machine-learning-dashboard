"""Page 1 — Executive business overview."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.components import cards, charts
from dashboard.components import stats as S
from dashboard.config import theme
from dashboard.config.schema import PROVENANCE, label
from dashboard.data.transforms import ms_columns, ms_option
from dashboard.views import common
from dashboard.views.market import filter_market, partner_frames

esc = cards.esc


def render(ctx) -> None:
    cards.page_header("Executive business overview",
                      "Does the checkout-to-departure luggage problem exist, how widespread is it, and is there "
                      "stated demand for a verified storage marketplace?",
                      "Individual submission · descriptive", [ctx.r("consumer_clean"), ctx.r("partner_survey"),
                                                              ctx.r("partner_audit")])
    _provenance_banner(ctx)
    if not common.require_consumer(ctx, "The overview KPIs and charts are calculated from the cleaned consumer survey."):
        _partner_kpis_only(ctx)
        return
    _kpis(ctx)
    _charts(ctx)


def _provenance_banner(ctx) -> None:
    r = ctx.r("consumer_clean")
    if not r.ok:
        return
    if ctx.cprov == "synthetic":
        cards.notice("warn", "All consumer figures on this page come from SYNTHETIC survey data.",
                     "They test whether the analysis works and illustrate possible patterns. They are not evidence "
                     "about real travellers. The equal city allocation (e.g. 500 per city) is an experimental design "
                     "choice and does not represent market size. No actual transactions are shown.")
    elif ctx.cprov == "unclassified":
        cards.notice("warn", "Consumer data provenance is not declared.",
                     "Declare it in <code>data/sources.toml</code> (synthetic / primary) before citing these figures.")
    else:
        cards.notice("info", f"Consumer data source: {esc(PROVENANCE[ctx.cprov])}.",
                     f"{esc(r.provenance_note)}. Survey answers are stated intentions, not transactions.")


def _kpis(ctx) -> None:
    d = ctx.cf
    prov = ctx.cprov
    items = []
    items.append(cards.kpi("Valid consumer responses", f"{ctx.n_filtered:,}",
                           f"of {ctx.n_total:,} in the cleaned file · {ctx.filter_note()}", prov))
    if "checkout_gap_hours__status" in d:
        rel = int((d["checkout_gap_hours__status"] == "valid").sum())
        na = int((d["checkout_gap_hours__status"] == "not_applicable").sum())
        items.append(cards.kpi("Relevant respondents", f"{rel:,}",
                               f"answered Q11 with an actual gap · {na:,} 'not applicable' kept separate", prov))
    for col, lbl, accent in [("long_gap", "Checkout gap ≥ 3 hours", theme.BLUE),
                             ("search_difficulty", "Difficulty finding storage", theme.PEACH),
                             ("unmet_need", "Unmet storage need (working definition)", theme.LAVENDER),
                             ("accepts_partner", "Would accept partner venues", theme.GREEN),
                             ("high_intent", "High stated booking intent (8–10)", theme.GOLD)]:
        if col in d:
            p, k, n = S.share(d[col])
            items.append(cards.kpi(lbl, S.fmt_pct(p), f"{k:,} of {n:,} answering", prov, accent))
        else:
            items.append(cards.kpi(lbl, None, "required survey fields not present", prov, accent))
    for col, lbl in [("storage_wtp_inr__value", "Median storage WTP (3 h, 1 bag)"),
                     ("pickup_delivery_wtp_inr__value", "Median extra pickup/delivery WTP")]:
        if col in d:
            desc = S.describe_numeric(d[col])
            items.append(cards.kpi(lbl, S.fmt_inr(desc.get("median")) if desc["n"] else None,
                                   f"n = {desc['n']:,} · stated, not paid prices" if desc["n"] else "no valid answers",
                                   prov, theme.TEAL))
        else:
            items.append(cards.kpi(lbl, None, "column not present", prov, theme.TEAL))
    items += _partner_kpi_items(ctx)
    with cards.card("Key indicators", help_text="Percentages exclude missing and 'not applicable' answers; each "
                                                "card shows its numerator and denominator."):
        cards.kpi_grid(items, per_row=4)
        with st.expander("How are these indicators defined?"):
            defs = ctx.cmeta.get("definitions", {})
            rows = [{"Indicator": esc(label(k)), "Working definition": esc(v)} for k, v in defs.items()]
            st.markdown(cards.html_table(rows) or "No derived indicators available.", unsafe_allow_html=True)
            st.caption("These are dashboard working definitions, applied transparently to the supplied answers. "
                       "They are not part of the questionnaire and can be changed in dashboard/data/transforms.py.")


def _partner_kpi_items(ctx) -> list[str]:
    items = []
    frames = partner_frames(ctx)
    if not frames:
        items.append(cards.kpi("Potential partners assessed", None, "partner_survey.csv / partner_audit.csv not supplied"))
        items.append(cards.kpi("Verified operational partners", None, "requires verification status in partner data"))
        return items
    total = sum(len(filter_market(ctx, f)) for _, f, _ in frames)
    srcs = ", ".join(f"{t} ({len(filter_market(ctx, f)):,})" for t, f, _ in frames)
    prov = frames[0][2]
    items.append(cards.kpi("Potential partners assessed", f"{total:,}", srcs, prov, theme.SLATE))
    ver = 0
    has_status = False
    for _, f, _ in frames:
        f = filter_market(ctx, f)
        if "evidence_status" in f.columns:
            has_status = True
            ver += int((f["is_verified"] & f.get("is_willing", pd.Series(True, index=f.index)).fillna(False)).sum())
    items.append(cards.kpi("Verified operational partners", f"{ver:,}" if has_status else None,
                           "evidence_status = Verified and willing (P8)" if has_status
                           else "no evidence_status column — none can be counted as verified", prov, theme.SLATE))
    return items


def _partner_kpis_only(ctx) -> None:
    items = _partner_kpi_items(ctx)
    with cards.card("Supply-side indicators"):
        cards.kpi_grid(items, per_row=2)


# ------------------------------------------------------------------ charts
def _dist_card(ctx, title, question, col, measures, implication, horizontal=False, order=None, sort=False,
               colour=theme.BLUE, key=None):
    d = ctx.cf
    with cards.card(title, question):
        if col not in d or d[col].dropna().empty:
            cards.missing_columns([col.split("__")[0]], "This chart needs the answer column.")
            return
        s = d[col]
        if order is None and isinstance(s.dtype, pd.CategoricalDtype):
            order = [c for c in s.cat.categories]
        fig, data = charts.count_bar(s.astype(object) if order is None else s, horizontal=horizontal, order=order,
                                     sort=sort, colour=colour)
        charts.show(fig, key or col, data)
        n = int(data["respondents"].sum())
        top = data.sort_values("respondents", ascending=False).iloc[0] if len(data) else None
        obs = (f"Most common answer: <b>{esc(top['category'])}</b> — {top['share_pct']:.1f}% "
               f"({int(top['respondents']):,} of {n:,}).") if top is not None else "No answers."
        cards.explanation(esc(measures), obs, implication(data, n) if top is not None else "—",
                          common.limitation(ctx, n, common.small_sample_note(n, ctx.min_n).strip()))


def _charts(ctx) -> None:
    d = ctx.cf
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        _dist_card(ctx, "Respondents by city", "How are the analysed responses spread across the four study cities?",
                   "city__label", "Count and share of respondents per city after filters.",
                   lambda data, n: esc(
                       "City shares reflect the sampling design, not market size. "
                       + ("Groups are roughly balanced, so city comparisons are not dominated by one city."
                          if len(data) > 1 and data["respondents"].min() >= 0.8 * data["respondents"].max()
                          else "Groups are unbalanced, so compare percentages within cities rather than counts.")),
                   sort=True)
    with c2:
        with cards.card("Traveller profile", "Who are the respondents?"):
            opts = {k: v for k, v in {"Age group": "age_group__label", "Travel purpose": "travel_purpose__label",
                                      "Travel group": "group_size__label", "Transport mode": "travel_mode__label",
                                      "Accommodation": "accommodation_type__label"}.items() if v in d}
            if not opts:
                cards.missing_columns(["age_group", "travel_purpose", "group_size", "travel_mode",
                                       "accommodation_type"], "Profile charts need at least one of these.")
            else:
                pick = st.segmented_control("Profile variable", list(opts), default=list(opts)[0], key="ov_profile",
                                            label_visibility="collapsed") or list(opts)[0]
                col = opts[pick]
                s = d[col]
                order = list(s.cat.categories) if isinstance(s.dtype, pd.CategoricalDtype) else None
                fig, data = charts.count_bar(s, horizontal=True, order=order, sort=order is None, colour=theme.GREEN)
                charts.show(fig, f"profile_{col}", data)
                n = int(data["respondents"].sum())
                top = data.sort_values("respondents", ascending=False).iloc[0]
                cards.explanation(esc(f"Distribution of {pick.lower()} among respondents answering."),
                                  f"Largest group: <b>{esc(top['category'])}</b> ({top['share_pct']:.1f}%, "
                                  f"{int(top['respondents']):,} of {n:,}).",
                                  esc("Profile composition tells you which travellers dominate the sample — "
                                      "later percentages are weighted toward this group."),
                                  common.limitation(ctx, n))

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        _dist_card(ctx, "Travel frequency", "How often do respondents travel within India?", "trips_12m__label",
                   "Trips within India in the last 12 months (Q7).",
                   lambda data, n: esc(
                       f"{data.loc[data['category'].isin(['4–6', '7+']), 'share_pct'].sum():.1f}% report 4+ trips; "
                       "frequent travellers are candidates for repeat use, which matters for acquisition cost."))
    with c2:
        _dist_card(ctx, "Checkout-to-departure gap", "How long is the gap between checkout and departure?",
                   "checkout_gap_hours__label", "Q11 gap on the most recent trip; 'Not applicable' shown separately.",
                   lambda data, n: _gap_implication(data, n), colour=theme.LAVENDER)

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        _methods_card(ctx)
    with c2:
        _prevalence_card(ctx)

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        _dist_card(ctx, "Inconvenience of managing luggage", "How much did luggage affect the trip?",
                   "inconvenience_score__label", "Q17, 1 = not at all … 5 = extremely.",
                   lambda data, n: esc(
                       f"{data.loc[data['category'].isin(['4', '5']), 'share_pct'].sum():.1f}% rate the impact 4–5 "
                       "(significant/extreme). A high share indicates a felt pain point rather than a minor nuisance."),
                   colour=theme.PEACH)
    with c2:
        _dist_card(ctx, "Awareness & ease of finding storage", "Can travellers find existing storage on their route?",
                   "storage_search_difficulty__label", "Q18 ease of finding a conveniently located facility.",
                   lambda data, n: esc(
                       f"{data.loc[data['category'].isin(['Difficult', 'Very difficult']), 'share_pct'].sum():.1f}% "
                       f"found it difficult and "
                       f"{data.loc[data['category'] == 'Unaware of any facility', 'share_pct'].sum():.1f}% were "
                       "unaware of any facility — awareness gaps may be addressed by discovery/marketing, "
                       "difficulty gaps need actual supply."), colour=theme.TEAL)

    _dist_card(ctx, "Which service would travellers try first?", "What should the launch service be?",
               "first_service_preference__label", "Q24 single choice after reading the service concept.",
               lambda data, n: _service_implication(data), horizontal=True, sort=True, colour=theme.GOLD)


def _gap_implication(data: pd.DataFrame, n: int) -> str:
    valid = data[data["category"] != "Not applicable"]
    nv = int(valid["respondents"].sum())
    k = int(valid.loc[valid["category"].isin(["3–4 h", "5–6 h", "7–8 h", "> 8 h"]), "respondents"].sum())
    if nv == 0:
        return "No applicable answers."
    p = 100 * k / nv
    return esc(f"Among applicable answers, {p:.1f}% ({k:,} of {nv:,}) had ≥ 3 hours to fill — the window in which "
               "a storage booking is plausible. " + ("This is a material share of travellers." if p >= 30 else
                                                      "This is a minority, so the addressable group is narrower."))


def _service_implication(data: pd.DataFrame) -> str:
    def s(pat):
        return data.loc[data["category"].str.lower().str.contains(pat, regex=True), "share_pct"].sum()
    storage = s(r"drop my luggage|locker")
    pickup = s(r"pick|deliver")
    none = s(r"^none")
    return esc(f"Storage-only options: {storage:.1f}%; options involving pickup/delivery: {pickup:.1f}%; "
               f"'none of these': {none:.1f}%. "
               + ("Stated first choice leans to storage, consistent with a storage-first Phase 1."
                  if storage > pickup else
                  "Pickup/delivery is chosen at least as often as storage — test it early, but validate costs first."))


def _methods_card(ctx) -> None:
    d = ctx.cf
    with cards.card("How travellers currently manage luggage", "What existing alternatives do we compete with?"):
        cols = ms_columns(d, "existing_storage_methods")
        if not cols:
            cards.missing_columns(["existing_storage_methods"], "Needs Q15 (semicolon-separated or binary columns).")
            return
        answered = d[cols].notna().any(axis=1)
        n = int(answered.sum())
        sub = d.loc[answered, cols].fillna(False)
        pct = sub.mean() * 100
        k = sub.sum()
        data = pd.DataFrame({"method": [ms_option(c) for c in cols], "respondents": k.values.astype(int),
                             "share_pct": pct.values.round(2)}).sort_values("share_pct", ascending=False)
        fig = charts.bar(data["method"], data["share_pct"], data["respondents"], [n] * len(data), horizontal=True,
                         colour=theme.BLUE, axis_title="% of respondents (multi-select)")
        charts.show(fig, "methods", data)
        top = data.iloc[0]
        carry = data.loc[data["method"] == "Carried it", "share_pct"].sum()
        cards.explanation(
            esc("Share of respondents selecting each method (Q15, select all that apply; totals exceed 100%)."),
            f"Most used: <b>{esc(top['method'])}</b> ({top['share_pct']:.1f}%, {int(top['respondents']):,} of {n:,}). "
            f"Carried luggage themselves: {carry:.1f}%.",
            esc("The most common method is the incumbent alternative the service must beat on convenience, trust "
                "or price. A high 'carried it' share suggests travellers lacking a usable option."),
            common.limitation(ctx, n, "Multi-select shares cannot be added together."))


def _prevalence_card(ctx) -> None:
    d = ctx.cf
    with cards.card("Problem prevalence cascade", "How many travellers have a problem vs an unmet need?"):
        steps = [("long_gap", "Gap ≥ 3 h"), ("has_problem", "Gap ≥ 3 h and managed luggage ≥ 1 h"),
                 ("search_difficulty", "Found storage hard / unaware"), ("access_barrier", "Cites an access barrier"),
                 ("unmet_need", "Unmet need (working definition)")]
        rows = []
        for col, lbl in steps:
            if col in d:
                p, k, n = S.share(d[col])
                rows.append({"indicator": lbl, "share_pct": round(p, 2), "respondents": k, "denominator": n})
        if not rows:
            cards.missing_columns(["checkout_gap_hours", "storage_search_difficulty", "existing_barriers"],
                                  "The cascade needs Q11, Q12, Q18 and Q19.")
            return
        data = pd.DataFrame(rows)
        fig = charts.bar(data["indicator"], data["share_pct"], data["respondents"], data["denominator"],
                         horizontal=True, colour=theme.LAVENDER, axis_title="% of respondents with a valid answer")
        charts.show(fig, "prevalence", data)
        um = data.loc[data["indicator"].str.startswith("Unmet"), "share_pct"]
        lg = data.loc[data["indicator"] == "Gap ≥ 3 h", "share_pct"]
        obs = " · ".join(f"{r.indicator}: <b>{r.share_pct:.1f}%</b> ({r.respondents:,}/{r.denominator:,})"
                         for r in data.itertuples())
        imp = ("Experiencing a long gap is not the same as needing a new service — many travellers solve it "
               "with hotel storage or by carrying bags. ")
        if len(um) and len(lg) and lg.iloc[0] > 0:
            imp += f"Here the unmet-need share is {um.iloc[0]:.1f}% versus {lg.iloc[0]:.1f}% with a long gap."
        cards.explanation(esc("Successively stricter definitions, from having time to fill to lacking a usable "
                              "storage option. Each bar uses its own valid-answer denominator."), obs, esc(imp),
                          common.limitation(ctx, int(data["denominator"].max()),
                                            "The unmet-need definition is a provisional dashboard rule (see "
                                            "'How are these indicators defined?')."))
