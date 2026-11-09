"""Page 3 — Descriptive and diagnostic analytics (individual-submission core)."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats as ss

from dashboard.components import cards, charts
from dashboard.components import stats as S
from dashboard.config import theme
from dashboard.config.schema import label
from dashboard.data.transforms import ms_columns, ms_option
from dashboard.views import common
from dashboard.views.common import dist_card, multi_card

esc = cards.esc


def render(ctx) -> None:
    cards.page_header("Descriptive & diagnostic analytics",
                      "Who experiences the luggage problem, how they cope today, and which factors move together with "
                      "difficulty, intent and willingness to pay.",
                      "Individual submission · descriptive, diagnostic & correlation analysis",
                      [ctx.r("consumer_clean")])
    if not common.require_consumer(ctx):
        return
    cards.notice("info", f"{esc(ctx.filter_note())} n = {ctx.n_filtered:,} respondents.",
                 "Every observation below is calculated live from the filtered data. Associations describe "
                 "co-occurrence in survey answers and do not establish cause and effect.")
    tabs = st.tabs(["A · Traveller profile", "B · Luggage problem", "C · Current alternatives",
                    "D · Customer experience", "E · Demand & price", "Diagnostic questions 1–10", "Correlation"])
    with tabs[0]:
        _profile(ctx)
    with tabs[1]:
        _problem(ctx)
    with tabs[2]:
        _alternatives(ctx)
    with tabs[3]:
        _experience(ctx)
    with tabs[4]:
        _demand(ctx)
    with tabs[5]:
        _diagnostics(ctx)
    with tabs[6]:
        _correlation(ctx)


def _share_of(data: pd.DataFrame, cats) -> float:
    return float(data.loc[data["category"].isin(cats), "share_pct"].sum())


def _two(ctx, items):
    for i in range(0, len(items), 2):
        cols = st.columns(2, gap="medium")
        for c, fn in zip(cols, items[i:i + 2]):
            with c:
                fn()


# ------------------------------------------------------------------ A
def _profile(ctx):
    _two(ctx, [
        lambda: dist_card(ctx, "Age group", "Which age groups dominate the sample?", "age_group__label",
                          "Q2 age band.", lambda d, n: esc(
                              f"{_share_of(d, ['18–24', '25–34']):.1f}% are under 35. Younger travellers are typically "
                              "more app-native, which affects the booking channel."), key="a_age"),
        lambda: dist_card(ctx, "Travel purpose", "Why are respondents travelling?", "travel_purpose__label",
                          "Q6 primary purpose.", lambda d, n: esc(
                              "Purpose shapes the checkout-gap pattern: leisure itineraries tend to have flexible "
                              "afternoons, business trips tighter schedules. Compare purposes in Diagnostic Q3."),
                          horizontal=True, sort=True, colour=theme.GREEN, key="a_purpose"),
        lambda: dist_card(ctx, "Travel frequency", "How often do they travel?", "trips_12m__label",
                          "Q7 trips within India in the last 12 months.", lambda d, n: esc(
                              f"{_share_of(d, ['4–6', '7+']):.1f}% took 4+ trips — the group most likely to "
                              "become repeat customers."), colour=theme.LAVENDER, key="a_trips"),
        lambda: dist_card(ctx, "Travel group size", "Do they travel alone or in groups?", "group_size__label",
                          "Q8 usual travel group.", lambda d, n: esc(
                              f"{100 - _share_of(d, ['Alone']):.1f}% travel with others — relevant to multi-bag "
                              "and group pricing."), colour=theme.PEACH, key="a_group"),
        lambda: dist_card(ctx, "Transport mode", "How do they arrive and depart?", "travel_mode__label",
                          "Q9 most frequent transport mode.", lambda d, n: esc(
                              "Mode determines the departure point (airport, station, bus terminal) the service "
                              "must connect to."), horizontal=True, sort=True, colour=theme.TEAL, key="a_mode"),
        lambda: dist_card(ctx, "Accommodation type", "Where do they stay?", "accommodation_type__label",
                          "Q10 usual accommodation.", lambda d, n: esc(
                              "Hotels often hold luggage for free; short-term rentals and homestays usually cannot — "
                              "a key driver of unmet need (see Diagnostic Q4)."), horizontal=True, sort=True,
                          colour=theme.GOLD, key="a_accom"),
    ])


# ------------------------------------------------------------------ B
def _problem(ctx):
    _two(ctx, [
        lambda: dist_card(ctx, "Checkout-to-departure gap", "How long is the gap?", "checkout_gap_hours__label",
                          "Q11 most recent relevant trip.", lambda d, n: esc(
                              f"{_share_of(d, ['3–4 h', '5–6 h', '7–8 h', '> 8 h']):.1f}% of all answers (including "
                              "'not applicable') report ≥ 3 hours."), key="b_gap"),
        lambda: dist_card(ctx, "Time personally managing luggage", "How long did they carry or manage bags?",
                          "luggage_management_hours__label", "Q12 duration after checkout.", lambda d, n: esc(
                              f"{_share_of(d, ['3–4 h', '5–6 h', '7–8 h', '> 8 h']):.1f}% managed luggage for ≥ 3 h; "
                              f"{_share_of(d, ['Did not manage']):.1f}% did not have to."),
                          colour=theme.LAVENDER, key="b_mgmt"),
        lambda: dist_card(ctx, "Bags to store", "How many bags would need storage?", "bag_count__label",
                          "Q13 bags in this situation.", lambda d, n: esc(
                              f"{_share_of(d, ['2', '3', '4+']):.1f}% need 2+ bags — per-bag pricing multiplies "
                              "revenue but also partner space needs."), colour=theme.PEACH, key="b_bags"),
        lambda: dist_card(ctx, "How often the gap occurs", "Is this a recurring problem?", "problem_frequency__label",
                          "Q14 frequency of a multi-hour gap after checkout.", lambda d, n: esc(
                              f"{_share_of(d, ['Frequently', 'Almost every trip']):.1f}% say frequently or almost "
                              "every trip — recurring pain supports repeat purchase."), colour=theme.GREEN,
                          key="b_freq"),
        lambda: dist_card(ctx, "Inconvenience score", "How much did luggage affect the trip?",
                          "inconvenience_score__label", "Q17, 1 = not at all … 5 = extremely.", lambda d, n: esc(
                              f"{_share_of(d, ['4', '5']):.1f}% rate 4–5. "
                              + ("A sizeable high-impact group exists." if _share_of(d, ['4', '5']) >= 30 else
                                 "High impact is limited to a minority.")), colour=theme.GOLD, key="b_inconv"),
    ])


# ------------------------------------------------------------------ C
def _alternatives(ctx):
    multi_card(ctx, "Current luggage-management methods", "What do travellers do today?", "existing_storage_methods",
               "Q15 methods used in these situations.", lambda d, n: esc(
                   "Each incumbent method sets a benchmark: hotel storage is free but limited to guests; station and "
                   "airport facilities are fixed in location; carrying bags is the 'do nothing' option the service must "
                   "beat on convenience."), key="c_methods")
    d = ctx.cf
    cols = ms_columns(d, "existing_storage_methods")
    if cols and "travel_purpose__label" in d:
        with cards.card("Methods by city", "Do alternatives differ between cities?"):
            grp = st.segmented_control("Compare by", ["City", "Travel purpose", "Accommodation"], default="City",
                                       key="c_methods_by") or "City"
            gcol = {"City": "city__label", "Travel purpose": "travel_purpose__label",
                    "Accommodation": "accommodation_type__label"}[grp]
            if gcol not in d:
                cards.missing_columns([gcol.split("__")[0]], "")
                return
            pct, n = S.multi_share_by(d, gcol, cols)
            if pct.empty:
                cards.notice("info", "No data for this breakdown.")
                return
            pct.columns = [ms_option(c) for c in pct.columns]
            pct.index = [f"{i} (n={n[i]})" for i in pct.index]
            fig = charts.heatmap(pct, text_fmt="{:.0f}%", zmin=0, zmax=100, value_name="% selecting",
                                 colorbar_title="%")
            charts.show(fig, f"c_methods_{grp}", pct.round(1).reset_index(names=grp))
            flat = pct.stack()
            hi = flat.idxmax()
            cards.explanation(esc(f"% of respondents in each {grp.lower()} selecting each method (rows are groups)."),
                              f"Highest: <b>{esc(hi[1])}</b> in <b>{esc(hi[0])}</b> ({flat.max():.1f}%).",
                              esc("City-specific incumbents mean the value proposition may need local positioning "
                                  "(e.g. against station cloakrooms in rail-heavy cities)."),
                              common.limitation(ctx, int(n.sum()), "Groups below the minimum sample are shown but "
                                                                   "should not be compared."))


# ------------------------------------------------------------------ D
def _experience(ctx):
    _two(ctx, [
        lambda: dist_card(ctx, "Ease of finding existing storage", "Is existing storage accessible on travellers' routes?",
                          "storage_search_difficulty__label", "Q18 ease of finding conveniently located storage.",
                          lambda d, n: esc(
                              f"Difficult/very difficult: {_share_of(d, ['Difficult', 'Very difficult']):.1f}%; unaware: "
                              f"{_share_of(d, ['Unaware of any facility']):.1f}%; did not search: "
                              f"{_share_of(d, ['Did not search']):.1f}%."), colour=theme.PEACH, key="d_search"),
        lambda: dist_card(ctx, "Maximum detour accepted", "How close must storage be?", "maximum_detour__label",
                          "Q20 maximum extra distance to drop off luggage.", lambda d, n: esc(
                              f"{_share_of(d, ['No extra distance', '≤ 5-min walk', '≤ 10-min walk']):.1f}% accept at "
                              f"most a 10-minute walk; {_share_of(d, ['Pickup from accommodation only']):.1f}% would only "
                              "use pickup. Short tolerance means partner density matters more than partner count."),
                          horizontal=True, colour=theme.TEAL, key="d_detour"),
    ])
    _two(ctx, [
        lambda: multi_card(ctx, "Difficulties with existing storage", "What goes wrong with today's options?",
                           "existing_barriers", "Q19 difficulties (up to 3).", lambda d, n: esc(
                               "Location/booking barriers point to a supply-and-discovery gap; trust barriers point to "
                               "verification and protection features; price barriers point to pricing tests."),
                           colour=theme.PEACH, key="d_barriers"),
        lambda: multi_card(ctx, "Acceptable partner venues", "Where would travellers leave their bags?",
                           "acceptable_venue_types", "Q21 venues where respondents feel comfortable.", lambda d, n: esc(
                               "Venues accepted by many respondents are the safest first partner categories; "
                               "a large 'none' share signals a trust barrier for the marketplace model."),
                           colour=theme.GREEN, key="d_venues"),
    ])
    multi_card(ctx, "Essential security & trust features", "What must be in place before travellers trust the service?",
               "security_requirements", "Q22 features essential before trusting the service.", lambda d, n: esc(
                   "Top-ranked features are launch requirements rather than upsells; low-ranked features can wait."),
               colour=theme.LAVENDER, key="d_security")


# ------------------------------------------------------------------ E
def _demand(ctx):
    d = ctx.cf
    _two(ctx, [
        lambda: dist_card(ctx, "Booking-intent distribution", "How likely are respondents to book?",
                          "booking_intent_score__label", "Q23, 0 = definitely not … 10 = definitely would book.",
                          lambda dd, n: esc(
                              f"{_share_of(dd, ['8', '9', '10']):.1f}% score 8–10 (High_Intent). Stated intent "
                              "typically overstates real conversion, so treat this as an upper bound."),
                          colour=theme.GOLD, key="e_intent"),
        lambda: dist_card(ctx, "Preferred pricing structure", "How should storage be priced?",
                          "preferred_pricing_model__label", "Q25 preferred pricing structure.", lambda dd, n: esc(
                              "Use the leading structures as A/B test arms on the landing page rather than "
                              "choosing one on survey answers alone."), horizontal=True, sort=True, colour=theme.BLUE,
                          key="e_pricing"),
    ])
    c1, c2 = st.columns(2, gap="medium")
    for cont, col, ttl, clr in ((c1, "storage_wtp_inr__value", "Storage WTP (₹, one suitcase, 3 h)", theme.BLUE),
                                (c2, "pickup_delivery_wtp_inr__value", "Extra pickup/delivery WTP (₹ per suitcase)",
                                 theme.GREEN)):
        with cont:
            with cards.card(ttl, "What would travellers say they pay?"):
                if col not in d or d[col].dropna().empty:
                    cards.missing_columns([col.replace("__value", "")], "Needs the numeric WTP column.")
                    continue
                desc = S.describe_numeric(d[col])
                fig = charts.histogram(d[col], nbins=30, colour=clr, x_title="₹",
                                       marker_lines={"Median": desc["median"]})
                charts.show(fig, f"e_{col}", d[[col]].dropna())
                cards.explanation("Distribution of stated maximum willingness to pay; zero answers kept.",
                                  esc(f"Median ₹{desc['median']:,.0f}, mean ₹{desc['mean']:,.0f}, IQR "
                                      f"₹{desc['q1']:,.0f}–₹{desc['q3']:,.0f}; {desc['zero_share']:.1f}% said ₹0."),
                                  esc("The interquartile range is a reasonable band for initial price tests; the zero "
                                      "share is the group to reach with free alternatives or bundles, not price cuts."),
                                  common.limitation(ctx, desc["n"], "Hypothetical WTP is not a transaction price."))
    dist_card(ctx, "Main reason NOT to use the service", "What is the biggest purchase barrier?",
              "main_nonuse_barrier__label", "Q29 single biggest reason for not using the service.", lambda dd, n: esc(
                  "Trust/loss barriers call for protection and verification features; price barriers for pricing "
                  "tests; 'free hotel storage' and 'not needed often' define the non-addressable group."),
              horizontal=True, sort=True, colour=theme.ROSE, key="e_barrier")


# ------------------------------------------------------------------ diagnostics
DIAG = [
    ("Q1", "Checkout gap vs luggage-management duration"),
    ("Q2", "Travel frequency vs luggage difficulties"),
    ("Q3", "Travel purpose vs luggage-management methods"),
    ("Q4", "Accommodation type vs storage availability"),
    ("Q5", "Transport mode vs checkout gap"),
    ("Q6", "Number of bags vs demand for storage"),
    ("Q7", "Inconvenience vs likelihood of using storage"),
    ("Q8", "Existing alternatives vs difficulty"),
    ("Q9", "Security concerns vs acceptable partner types"),
    ("Q10", "Travel frequency vs spending and WTP"),
]


def _diagnostics(ctx):
    with cards.card("Choose diagnostic questions", help_text="Select one or more to display."):
        sel = st.pills("Questions", [f"{q} · {t}" for q, t in DIAG], selection_mode="multi",
                       default=[f"{q} · {t}" for q, t in DIAG[:3]], key="diag_sel", label_visibility="collapsed")
    fns = {"Q1": _d1, "Q2": _d2, "Q3": _d3, "Q4": _d4, "Q5": _d5, "Q6": _d6, "Q7": _d7, "Q8": _d8, "Q9": _d9,
           "Q10": _d10}
    if not sel:
        cards.notice("info", "Select at least one question above.")
    for s in sel or []:
        q = s.split(" · ")[0]
        try:
            fns[q](ctx)
        except Exception as e:  # noqa: BLE001 — one failing analysis must not break the page
            cards.notice("error", f"{q} could not be calculated with the current data.", esc(str(e)))


def _rho_text(x: pd.Series, y: pd.Series) -> tuple[str, float, int]:
    dd = pd.concat([x, y], axis=1).dropna()
    if len(dd) < 5 or dd.iloc[:, 0].nunique() < 2 or dd.iloc[:, 1].nunique() < 2:
        return "Spearman ρ not computable (too few varying answers).", math.nan, len(dd)
    r, p = ss.spearmanr(dd.iloc[:, 0], dd.iloc[:, 1])
    return (f"Spearman ρ = {r:+.2f} ({S.strength_word(r)}; p {'< 0.001' if p < 0.001 else f'= {p:.3f}'}; "
            f"n = {len(dd):,})."), float(r), len(dd)


def _need(ctx, *cols) -> bool:
    miss = common.missing_cols(ctx.cf, *cols)
    if miss:
        cards.missing_columns([m.split("__")[0].split("::")[0] for m in miss], "This analysis needs these fields.")
        return False
    return True


def _crosstab_card(ctx, key, title, question, row, col, measures, implication_fn, extra_obs=""):
    pct, cnt = S.crosstab_share(ctx.cf, row, col)
    if pct.empty:
        cards.notice("info", "No overlapping answers for this pair after filters.")
        return None, None
    rn = cnt.sum(axis=1)
    pct_disp = pct.copy()
    pct_disp.index = [f"{i} (n={rn[i]})" for i in pct.index]
    fig = charts.stacked_share(pct_disp, cnt.set_axis(pct_disp.index), horizontal=True, legend_title=label(col))
    charts.show(fig, key, pct.round(1).reset_index(names=label(row)))
    return pct, cnt


def _d1(ctx):
    with cards.card("Q1 · Checkout gap vs time spent managing luggage",
                    "Do longer gaps translate into longer periods of carrying luggage?"):
        if not _need(ctx, "checkout_gap_hours__label", "luggage_management_hours__label"):
            return
        d = ctx.cf
        valid = d[(d["checkout_gap_hours__status"] == "valid") & (d["luggage_management_hours__status"] == "valid")]
        pct, cnt = S.crosstab_share(valid, "checkout_gap_hours__label", "luggage_management_hours__label")
        if pct.empty:
            cards.notice("info", "No valid answer pairs after filters.")
            return
        rn = cnt.sum(axis=1)
        disp = pct.copy()
        disp.index = [f"{i} (n={rn[i]})" for i in pct.index]
        fig = charts.heatmap(disp, text_fmt="{:.0f}%", zmin=0, zmax=100, counts=cnt.set_axis(disp.index),
                             x_title="Time managing luggage", y_title="Checkout gap", value_name="% of row",
                             colorbar_title="% of row")
        charts.show(fig, "d1", pct.round(1).reset_index(names="checkout_gap"))
        rho_t, rho, n = _rho_text(valid["checkout_gap_hours__rank"], valid["luggage_management_hours__rank"])
        lg = valid[valid["checkout_gap_hours__rank"] >= 3]
        p3, k3, n3 = S.share(lg["luggage_management_hours__rank"] >= 3) if len(lg) else (math.nan, 0, 0)
        cards.explanation(
            "Row percentages: for each gap band, how long respondents personally managed luggage. Both are ordinal, "
            "so Spearman rank correlation is used.",
            esc(f"{rho_t} Among travellers with a gap ≥ 3 h, {S.fmt_pct(p3, k3, n3)} managed luggage for ≥ 3 h."),
            esc("A positive association means long gaps are not being absorbed by existing storage — travellers end up "
                "carrying bags. A weak association suggests many already have a workaround."
                if not math.isnan(rho) else "Insufficient data to judge."),
            common.limitation(ctx, n, "'Not applicable' answers are excluded from both variables."))


def _d2(ctx):
    with cards.card("Q2 · Travel frequency vs luggage difficulties",
                    "Do frequent travellers find storage harder (or easier) to access?"):
        if not _need(ctx, "trips_12m__label", "storage_search_difficulty__label"):
            return
        d = ctx.cf
        pct, cnt = _crosstab_card(ctx, "d2", "", "", "trips_12m__label", "storage_search_difficulty__label", "", None)
        if pct is None:
            return
        valid = d[d["storage_search_difficulty__status"] == "valid"]
        rho_t, rho, n = _rho_text(valid["trips_12m__rank"], valid["storage_search_difficulty__rank"])
        freq = d[d["trips_12m__rank"] >= 3]
        rare = d[d["trips_12m__rank"] <= 1]
        pf = S.share(freq["search_difficulty"]) if "search_difficulty" in d else (math.nan, 0, 0)
        pr = S.share(rare["search_difficulty"]) if "search_difficulty" in d else (math.nan, 0, 0)
        cards.explanation(
            "Distribution of Q18 answers within each travel-frequency band (row %). Spearman uses the 5-point ease "
            "scale only (unaware / did not search excluded).",
            esc(f"{rho_t} Difficulty or unawareness: {S.fmt_pct(*pf)} among 4+ trips vs {S.fmt_pct(*pr)} among 0–1 trips."),
            esc("If frequent travellers still struggle, the gap is structural (supply), not inexperience — they are a "
                "strong early-adopter target. If only infrequent travellers struggle, discovery/awareness is the lever."),
            common.limitation(ctx, n))


def _d3(ctx):
    with cards.card("Q3 · Travel purpose vs luggage-management methods",
                    "Do business, leisure and other travellers cope differently?"):
        d = ctx.cf
        cols = ms_columns(d, "existing_storage_methods")
        if not cols or not _need(ctx, "travel_purpose__label"):
            if not cols:
                cards.missing_columns(["existing_storage_methods"], "")
            return
        pct, n = S.multi_share_by(d, "travel_purpose__label", cols)
        if pct.empty:
            cards.notice("info", "No data after filters.")
            return
        overall = d.loc[d[cols].notna().any(axis=1), cols].fillna(False).astype(float).mean() * 100
        pct.columns = [ms_option(c) for c in pct.columns]
        overall.index = pct.columns
        disp = pct.copy()
        disp.index = [f"{i} (n={n[i]})" for i in pct.index]
        fig = charts.heatmap(disp, text_fmt="{:.0f}%", zmin=0, zmax=100, value_name="% selecting", colorbar_title="%")
        charts.show(fig, "d3", pct.round(1).reset_index(names="travel_purpose"))
        reliable = pct[n.reindex(pct.index) >= ctx.min_n]
        if len(reliable):
            dev = reliable.sub(overall, axis=1)
            idx = dev.stack().abs().idxmax()
            obs = (f"Largest deviation from the overall pattern: <b>{esc(idx[0])}</b> travellers select "
                   f"<b>{esc(idx[1])}</b> {reliable.loc[idx]:.1f}% vs {overall[idx[1]]:.1f}% overall "
                   f"({dev.loc[idx]:+.1f} pts; purposes with n ≥ {ctx.min_n} only).")
        else:
            obs = esc(f"No purpose group reaches the minimum sample of {ctx.min_n}.")
        cards.explanation("% of respondents in each travel purpose selecting each method (multi-select).", obs,
                          esc("Purposes that rely on carrying bags or on friends are candidates for targeted messaging; "
                              "purposes relying on hotel storage are harder to convert."),
                          common.limitation(ctx, int(n.sum()), "No significance test is applied to multi-select shares."))


def _d4(ctx):
    with cards.card("Q4 · Accommodation type vs storage availability",
                    "Do some accommodation types leave travellers without storage?"):
        if not _need(ctx, "accommodation_type__label"):
            return
        d = ctx.cf
        hotel_col = [c for c in ms_columns(d, "existing_storage_methods") if ms_option(c) == "Hotel / accommodation held it"]
        metrics = {}
        if hotel_col:
            metrics["Accommodation held luggage (%)"] = hotel_col[0]
        if "search_difficulty" in d:
            metrics["Found storage hard / unaware (%)"] = "search_difficulty"
        if "unmet_need" in d:
            metrics["Unmet need (%)"] = "unmet_need"
        if not metrics:
            cards.missing_columns(["existing_storage_methods", "storage_search_difficulty"], "")
            return
        frames = []
        for name, col in metrics.items():
            g = S.group_share(d, "accommodation_type__label", col, ctx.min_n)
            g["metric"] = name
            frames.append(g)
        long = pd.concat(frames)
        wide = long.pivot_table(index="accommodation_type__label", columns="metric", values="pct", observed=True)
        nn = long.groupby("accommodation_type__label", observed=True)["n"].max()
        wide = wide.reset_index()
        wide["label"] = [f"{a} (n={nn[a]})" for a in wide["accommodation_type__label"]]
        fig = charts.grouped_bar(wide, "label", list(metrics), y_title="% of respondents", horizontal=True,
                                 colours=[theme.GREEN, theme.PEACH, theme.LAVENDER])
        charts.show(fig, "d4", long.round(2))
        rel = long[long["reliable"]]
        if "Accommodation held luggage (%)" in metrics and len(rel):
            h = rel[rel["metric"] == "Accommodation held luggage (%)"].sort_values("pct")
            obs = (f"Lowest accommodation-held share: <b>{esc(h.iloc[0]['accommodation_type__label'])}</b> "
                   f"({h.iloc[0]['pct']:.1f}%, n={int(h.iloc[0]['n'])}); highest: "
                   f"<b>{esc(h.iloc[-1]['accommodation_type__label'])}</b> ({h.iloc[-1]['pct']:.1f}%).") if len(h) else ""
        else:
            obs = esc("Groups below the minimum sample are not ranked.")
        cards.explanation("Per accommodation type: share whose accommodation held luggage (Q15), share finding storage "
                          "hard/unaware (Q18), and unmet-need share.", obs,
                          esc("Accommodation types with low 'held luggage' and high difficulty are where a third-party "
                              "storage network adds the most value — a targeting and partnership signal (e.g. "
                              "short-term-rental hosts as referral partners)."),
                          common.limitation(ctx, int(nn.sum())))


def _d5(ctx):
    with cards.card("Q5 · Transport mode vs checkout gap", "Which arrival/departure modes create longer gaps?"):
        if not _need(ctx, "travel_mode__label", "checkout_gap_hours__label"):
            return
        d = ctx.cf
        sub = d[d["checkout_gap_hours__status"] == "valid"]
        pct, cnt = S.crosstab_share(sub, "travel_mode__label", "checkout_gap_hours__label")
        if pct.empty:
            cards.notice("info", "No data after filters.")
            return
        rn = cnt.sum(axis=1)
        disp = pct.copy()
        disp.index = [f"{i} (n={rn[i]})" for i in pct.index]
        fig = charts.stacked_share(disp, cnt.set_axis(disp.index), horizontal=True, legend_title="Checkout gap",
                                   colours=theme.SEQUENTIAL[1:] + [theme.LAVENDER])
        charts.show(fig, "d5", pct.round(1).reset_index(names="travel_mode"))
        v, p, n = S.cramers_v(sub["travel_mode__label"], sub["checkout_gap_hours__label"])
        lg = S.group_share(sub, "travel_mode__label", "long_gap", ctx.min_n) if "long_gap" in sub else pd.DataFrame()
        lg = lg[lg["reliable"]].sort_values("pct", ascending=False) if len(lg) else lg
        top = (f" Highest ≥ 3 h share: <b>{esc(lg.iloc[0]['travel_mode__label'])}</b> ({lg.iloc[0]['pct']:.1f}%, "
               f"n={int(lg.iloc[0]['n'])}).") if len(lg) else ""
        cards.explanation("Distribution of checkout gaps within each transport mode. Mode is nominal, so association "
                          "is measured with bias-corrected Cramér's V.",
                          esc(f"Cramér's V = {S.fmt_num(v)} ({S.strength_word(v, 'v')}; χ² p "
                              f"{'< 0.001' if p < 0.001 else f'= {p:.3f}'}; n = {n:,}).") + top,
                          esc("Modes with long gaps indicate which departure points (airport vs station) the first "
                              "partner cluster should be near."),
                          common.limitation(ctx, n, "'Not applicable' excluded."))


def _d6(ctx):
    with cards.card("Q6 · Number of bags vs demand for storage", "Do travellers with more bags want storage more?"):
        if not _need(ctx, "bag_count__label"):
            return
        d = ctx.cf
        metrics = {k: v for k, v in {"High stated intent (%)": "high_intent", "Unmet need (%)": "unmet_need",
                                     "Gap ≥ 3 h (%)": "long_gap"}.items() if v in d}
        if not metrics:
            cards.missing_columns(["booking_intent_score"], "")
            return
        frames = []
        for name, col in metrics.items():
            g = S.group_share(d, "bag_count__label", col, ctx.min_n)
            g["metric"] = name
            frames.append(g)
        long = pd.concat(frames)
        wide = long.pivot_table(index="bag_count__label", columns="metric", values="pct", observed=True).reset_index()
        nn = long.groupby("bag_count__label", observed=True)["n"].max()
        wide["label"] = [f"{b} bag(s) (n={nn[b]})" for b in wide["bag_count__label"]]
        fig = charts.grouped_bar(wide, "label", list(metrics), y_title="% of respondents",
                                 colours=[theme.GOLD, theme.LAVENDER, theme.BLUE])
        charts.show(fig, "d6", long.round(2))
        rho_t, rho, n = _rho_text(d["bag_count__rank"], d.get("booking_intent_score__value"))
        cards.explanation("Share with high intent, unmet need and long gaps by number of bags.",
                          esc(f"Bags vs intent score: {rho_t}"),
                          esc("If intent rises with bag count, multi-bag and family packages deserve priority; if flat, "
                              "per-bag pricing need not be discounted to win larger parties."),
                          common.limitation(ctx, n))


def _d7(ctx):
    with cards.card("Q7 · Inconvenience vs likelihood of using storage",
                    "Do travellers who suffer more say they are more likely to book?"):
        if not _need(ctx, "inconvenience_score__value", "booking_intent_score__value"):
            return
        d = ctx.cf
        order = [str(i) for i in range(1, 6)]
        sub = d.dropna(subset=["inconvenience_score__label", "booking_intent_score__value"]).copy()
        sub["inconv"] = sub["inconvenience_score__label"].astype(str)
        fig = charts.box_by(sub, "inconv", "booking_intent_score__value", order=order, y_title="Booking intent (0–10)")
        fig.update_layout(xaxis_title="Inconvenience score (1–5)")
        g = S.group_share(d, "inconvenience_score__label", "high_intent", ctx.min_n) if "high_intent" in d else None
        if g is not None and len(g):
            fig.add_scatter(x=[f"{l} (n={int(n)})" for l, n in zip(g["inconvenience_score__label"].astype(str), g["n"])],
                            y=g["pct"] / 10, mode="lines+markers", name="High-intent % ÷ 10", yaxis="y",
                            line=dict(color=theme.GOLD, width=2), showlegend=True,
                            hovertemplate="High intent: %{customdata:.1f}%<extra></extra>", customdata=g["pct"])
        charts.show(fig, "d7", sub[["inconvenience_score__label", "booking_intent_score__value"]].rename(
            columns={"inconvenience_score__label": "inconvenience", "booking_intent_score__value": "intent"}))
        rho_t, rho, n = _rho_text(d["inconvenience_score__value"], d["booking_intent_score__value"])
        hi = S.share(d.loc[d["inconvenience_score__value"] >= 4, "high_intent"]) if "high_intent" in d else (math.nan, 0, 0)
        lo = S.share(d.loc[d["inconvenience_score__value"] <= 2, "high_intent"]) if "high_intent" in d else (math.nan, 0, 0)
        cards.explanation("Box plots of the 0–10 intent score for each inconvenience level, with the high-intent share "
                          "(gold line, scaled ÷ 10 to share the axis).",
                          esc(f"{rho_t} High intent: {S.fmt_pct(*hi)} among inconvenience 4–5 vs {S.fmt_pct(*lo)} "
                              "among 1–2."),
                          esc("A clear gradient supports targeting messaging at moments of pain (late checkout, "
                              "long layovers). A flat pattern means pain alone does not drive intent — trust or price "
                              "may be the binding constraint."),
                          common.limitation(ctx, n, "Both are self-reported in the same survey, so shared response "
                                                    "style can inflate the correlation."))


def _d8(ctx):
    with cards.card("Q8 · Existing alternatives vs difficulty", "Which current methods leave travellers worst off?"):
        d = ctx.cf
        cols = ms_columns(d, "existing_storage_methods")
        if not cols:
            cards.missing_columns(["existing_storage_methods"], "")
            return
        rows = []
        for c in cols:
            users = d[d[c].fillna(False).astype(bool)]
            row = {"method": ms_option(c), "users": len(users)}
            if "inconvenience_score__value" in d:
                p, k, n = S.share(users["inconvenience_score__value"].dropna() >= 4)
                row["high_inconvenience_pct"] = p
            if "search_difficulty" in d:
                row["search_difficulty_pct"] = S.share(users["search_difficulty"])[0]
            rows.append(row)
        data = pd.DataFrame(rows)
        data = data[data["users"] > 0].sort_values("users", ascending=False)
        ycols = [c for c in ("high_inconvenience_pct", "search_difficulty_pct") if c in data]
        if not ycols:
            cards.missing_columns(["inconvenience_score", "storage_search_difficulty"], "")
            return
        data["label"] = [f"{m} (n={u})" for m, u in zip(data["method"], data["users"])]
        fig = charts.grouped_bar(data, "label", ycols, ["Inconvenience 4–5 (%)", "Found storage hard / unaware (%)"][:len(ycols)],
                                 y_title="% of users of each method", horizontal=True,
                                 colours=[theme.PEACH, theme.LAVENDER])
        charts.show(fig, "d8", data.drop(columns="label").round(2))
        rel = data[data["users"] >= ctx.min_n]
        if len(rel) and ycols:
            w = rel.sort_values(ycols[0], ascending=False).iloc[0]
            obs = (f"Highest high-inconvenience share among users of <b>{esc(w['method'])}</b> "
                   f"({w[ycols[0]]:.1f}%, n={int(w['users'])}).")
        else:
            obs = esc(f"No method has ≥ {ctx.min_n} users after filters.")
        cards.explanation("For users of each method (multi-select, so groups overlap): share rating inconvenience 4–5 and "
                          "share finding storage hard/unaware.", obs,
                          esc("Methods associated with high pain are the switching opportunity; methods with low pain "
                              "(often free hotel storage) are sticky and hard to displace."),
                          common.limitation(ctx, int(data["users"].max()) if len(data) else 0,
                                            "Groups overlap; differences are descriptive only."))


def _d9(ctx):
    with cards.card("Q9 · Security concerns vs acceptable partner types",
                    "Are travellers with stricter security needs less willing to use local businesses?"):
        d = ctx.cf
        sec = ms_columns(d, "security_requirements")
        ven = ms_columns(d, "acceptable_venue_types")
        if not sec or not ven:
            cards.missing_columns([x for x, c in (("security_requirements", sec), ("acceptable_venue_types", ven)) if not c], "")
            return
        both = d[d[sec].notna().any(axis=1) & d[ven].notna().any(axis=1)]
        V = both[ven].fillna(False).astype(bool)
        rows, ns = {}, {}
        rows["All respondents"] = V.mean() * 100
        ns["All respondents"] = len(both)
        for c in sec:
            m = both[c].fillna(False).astype(bool)
            if m.sum() >= 1:
                rows[ms_option(c)] = V[m].mean() * 100
                ns[ms_option(c)] = int(m.sum())
        mat = pd.DataFrame(rows).T
        mat.columns = [ms_option(c) for c in ven]
        disp = mat.copy()
        disp.index = [f"{i} (n={ns[i]})" for i in mat.index]
        fig = charts.heatmap(disp, text_fmt="{:.0f}%", zmin=0, zmax=100, x_title="Venue accepted",
                             y_title="Respondents requiring…", value_name="% accepting", colorbar_title="%")
        charts.show(fig, "d9", mat.round(1).reset_index(names="security_requirement"))
        partner_venues = [c for c in mat.columns if c.startswith("Verified") or c.startswith("Any registered")]
        diffs = (mat[partner_venues].mean(axis=1) - mat.loc["All respondents", partner_venues].mean()) if partner_venues else None
        if diffs is not None and len(diffs) > 1:
            dd = diffs.drop("All respondents")
            dd = dd[[ns[i] >= ctx.min_n for i in dd.index]]
            obs = (f"Average acceptance of partner-type venues is {mat.loc['All respondents', partner_venues].mean():.1f}% "
                   "overall. " + (f"Largest gap: respondents requiring <b>{esc(dd.abs().idxmax())}</b> differ by "
                                  f"{dd[dd.abs().idxmax()]:+.1f} pts." if len(dd) else ""))
        else:
            obs = esc("Insufficient overlap to compare.")
        cards.explanation("Row = respondents who list a security feature as essential; cell = % of them who accept each "
                          "venue type. Top row is the baseline.", obs,
                          esc("If strict-security respondents still accept partner venues, the features they require "
                              "can unlock trust; if acceptance drops, they may prefer dedicated or official facilities."),
                          common.limitation(ctx, len(both), "Multi-select overlap; descriptive only."))


def _d10(ctx):
    with cards.card("Q10 · Travel frequency vs spending and WTP", "Do frequent travellers spend and pay more?"):
        if not _need(ctx, "trips_12m__label"):
            return
        d = ctx.cf
        frames = []
        for col, name in (("storage_wtp_inr__value", "Median storage WTP (₹)"),
                          ("pickup_delivery_wtp_inr__value", "Median pickup/delivery WTP (₹)")):
            if col in d:
                g = S.group_median(d, "trips_12m__label", col, ctx.min_n)
                g["metric"] = name
                frames.append(g)
        if not frames:
            cards.missing_columns(["storage_wtp_inr", "pickup_delivery_wtp_inr"], "")
            return
        long = pd.concat(frames)
        wide = long.pivot_table(index="trips_12m__label", columns="metric", values="median", observed=True).reset_index()
        nn = long.groupby("trips_12m__label", observed=True)["n"].max()
        wide["label"] = [f"{t} trips (n={nn[t]})" for t in wide["trips_12m__label"]]
        metric_cols = [c for c in wide.columns if c.startswith("Median")]
        fig = charts.grouped_bar(wide, "label", metric_cols, y_title="₹", value_fmt="₹{:,.0f}",
                                 colours=[theme.BLUE, theme.GREEN])
        charts.show(fig, "d10", long.round(2))
        rho_t, rho, n = _rho_text(d["trips_12m__rank"], d.get("storage_wtp_inr__value"))
        spend = ""
        if "past_spend_inr__rank" in d:
            r2, _, n2 = _rho_text(d["trips_12m__rank"], d["past_spend_inr__rank"])
            spend = f" Frequency vs past spend band: {r2}"
        cards.explanation("Median stated WTP by travel frequency (bars) and rank correlations.",
                          esc(f"Frequency vs storage WTP: {rho_t}{spend}"),
                          esc("A positive gradient would justify a subscription or repeat-discount offer for frequent "
                              "travellers; a flat one suggests a single public price is adequate."),
                          common.limitation(ctx, n, "Past-spend bands exclude 'did not use' and 'do not remember'."))


# ------------------------------------------------------------------ correlation
SPEARMAN_VARS = {
    "trips_12m__rank": "Trips (12 m)", "group_size__rank": "Group size", "checkout_gap_hours__rank": "Checkout gap",
    "luggage_management_hours__rank": "Luggage-mgmt time", "bag_count__rank": "Bags",
    "problem_frequency__rank": "Gap frequency", "inconvenience_score__value": "Inconvenience",
    "storage_search_difficulty__rank": "Search difficulty", "maximum_detour__rank": "Max detour",
    "booking_intent_score__value": "Booking intent", "pilot_interest__rank": "Pilot interest",
    "storage_wtp_inr__value": "Storage WTP", "pickup_delivery_wtp_inr__value": "Pickup WTP", "age_group__rank": "Age",
}
NOMINAL_VARS = {
    "city__label": "City", "travel_purpose__label": "Purpose", "travel_mode__label": "Transport",
    "accommodation_type__label": "Accommodation", "first_service_preference__label": "First service",
    "preferred_pricing_model__label": "Pricing model", "main_nonuse_barrier__label": "Non-use barrier",
    "high_intent": "High intent", "pilot_interest__label": "Pilot interest", "age_group__label": "Age group",
}


def _corr_card(ctx, kind: str, mat_r: pd.DataFrame, mat_p: pd.DataFrame, mat_n: pd.DataFrame, names: dict, key: str,
               measures: str, impl: str, limit_extra: str) -> None:
    r = mat_r.rename(index=names, columns=names)
    n = mat_n.rename(index=names, columns=names)
    p = mat_p.rename(index=names, columns=names)
    diverging = kind != "v"
    fig = charts.heatmap(r, text_fmt="{:.2f}", zmin=-1 if diverging else 0, zmax=1, diverging=diverging, counts=n,
                         value_name="Cramér's V" if kind == "v" else "ρ" if kind == "spearman" else "r",
                         colorbar_title="V" if kind == "v" else "r")
    long = pd.DataFrame([{"var_1": a, "var_2": b, "coefficient": r.loc[a, b], "p_value": p.loc[a, b], "n": n.loc[a, b]}
                         for i, a in enumerate(r.index) for b in r.columns[i + 1:]])
    charts.show(fig, key, long.round(4))
    tp = S.top_pair(r)
    if tp:
        a, b, v = tp
        obs = (f"Strongest pair: <b>{esc(a)}</b> × <b>{esc(b)}</b> = {v:+.2f} "
               f"({S.strength_word(v, 'v' if kind == 'v' else 'corr')}; n = {int(n.loc[a, b]):,}). "
               f"{int((long['coefficient'].abs() >= (0.2 if kind == 'v' else 0.3)).sum())} of {len(long)} pairs reach "
               f"{'V ≥ 0.2' if kind == 'v' else '|r| ≥ 0.3'}.")
    else:
        obs = "No computable pairs."
    cards.explanation(esc(measures), obs, esc(impl),
                      common.limitation(ctx, ctx.n_filtered, limit_extra + " Many pairs are tested at once, so some "
                                                                           "'significant' p-values will occur by chance."))


def _correlation(ctx):
    d = ctx.cf
    with cards.card("Which correlation measure is used where?"):
        st.markdown(cards.html_table([
            {"Measure": "Pearson r", "Used for": "Genuinely continuous numbers (₹ WTP; hours/counts only if the file "
                                                 "stores numbers rather than answer bands)",
             "Why": "Assumes interval data and a linear relationship."},
            {"Measure": "Spearman ρ", "Used for": "Ordered answer scales (frequency bands, 1–5 and 0–10 scores, "
                                                  "gap bands) and WTP", "Why": "Uses ranks only — no assumption "
                                                                                "that bands are equally spaced."},
            {"Measure": "Cramér's V (bias-corrected)", "Used for": "Unordered categories (city, purpose, mode, …)",
             "Why": "Category codes have no numeric meaning, so Pearson on them would be meaningless."},
        ]), unsafe_allow_html=True)
    t1, t2, t3 = st.tabs(["Spearman (ordinal)", "Pearson (continuous)", "Cramér's V (categorical)"])
    with t1:
        with cards.card("Spearman rank correlation", "Which ordered answers move together?"):
            avail = {k: v for k, v in SPEARMAN_VARS.items() if k in d and d[k].notna().sum() > 2}
            pick = st.multiselect("Variables", list(avail.values()), default=list(avail.values()), key="sp_vars")
            cols = [k for k, v in avail.items() if v in pick]
            if len(cols) < 2:
                cards.notice("info", "Select at least two variables.")
            else:
                r, p, n = S.corr_matrix(d, cols, "spearman")
                _corr_card(ctx, "spearman", r, p, n, avail, "corr_spearman",
                           "Spearman ρ between ordinal ranks (pairwise-complete; special answers such as 'unaware' or "
                           "'not applicable' are excluded).",
                           "Use strong pairs to choose which drivers to examine in the diagnostic questions and "
                           "segmentation; weak pairs mean the variables carry distinct information.",
                           "Hover a cell for its n.")
    with t2:
        with cards.card("Pearson correlation", "Do the continuous ₹ answers move together linearly?"):
            num = {"storage_wtp_inr__value": "Storage WTP", "pickup_delivery_wtp_inr__value": "Pickup WTP"}
            for f in ctx.cmeta.get("numeric_source", set()):
                num[f"{f}__value"] = label(f)
            avail = {k: v for k, v in num.items() if k in d and d[k].notna().sum() > 2}
            if len(avail) < 2:
                cards.notice("info", "Pearson needs at least two genuinely continuous variables.",
                             "Only the WTP columns qualify unless hours/counts are supplied as numbers. Ordered answer "
                             "bands are analysed with Spearman instead.")
            else:
                r, p, n = S.corr_matrix(d, list(avail), "pearson")
                _corr_card(ctx, "pearson", r, p, n, avail, "corr_pearson",
                           "Pearson r on numeric values (zeros included).",
                           "A strong storage-vs-pickup WTP correlation means the same travellers value both — "
                           "supporting a bundle; a weak one means separate audiences.",
                           "Pearson is sensitive to outliers; compare with the Spearman tab.")
                if "storage_wtp_inr__value" in avail and "pickup_delivery_wtp_inr__value" in avail:
                    fig = charts.scatter(d, "storage_wtp_inr__value", "pickup_delivery_wtp_inr__value",
                                         color="city__label" if "city__label" in d else None,
                                         hover=["respondent_id"], x_title="Storage WTP (₹)",
                                         y_title="Pickup/delivery WTP (₹)", jitter=3)
                    charts.show(fig, "corr_scatter_wtp")
    with t3:
        with cards.card("Cramér's V association", "Which categorical answers are associated?"):
            avail = {k: v for k, v in NOMINAL_VARS.items() if k in d and d[k].nunique(dropna=True) >= 2}
            pick = st.multiselect("Variables", list(avail.values()), default=list(avail.values()), key="cv_vars")
            cols = [k for k, v in avail.items() if v in pick]
            if len(cols) < 2:
                cards.notice("info", "Select at least two variables.")
            else:
                v, p, n = S.cramers_matrix(d, cols)
                _corr_card(ctx, "v", v, p, n, avail, "corr_cramers",
                           "Bias-corrected Cramér's V from chi-square tests (0 = no association, 1 = perfect). V has "
                           "no direction.",
                           "Associations with High intent identify categorical profiles worth testing as targeting "
                           "variables; associations among profile variables (e.g. purpose × accommodation) warn of "
                           "confounding when interpreting single cross-tabs.",
                           "Sparse cells inflate χ² — check n in the hover.")
