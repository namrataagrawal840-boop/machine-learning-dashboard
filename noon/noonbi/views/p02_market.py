"""Page 2 — market and competitive analysis (all secondary research, with links)."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from noonbi import cards, charts, theme
from noonbi import data as D

esc = cards.esc


def _link(url: str, text: str = "source") -> str:
    return f'<a href="{esc(url)}" target="_blank">{esc(text)}</a>'


def render(f: D.Filters) -> None:
    cards.page_header("Market & competitive analysis",
                      "How big the market is, who noon competes with, and what rivals already do about delivery, refunds "
                      "and service. Every figure links to where it came from.",
                      "Secondary research · 10 Nov 2026", [("Published sources", "secondary")])
    ms, mf, nf = D.table("market_size"), D.table("market_facts"), D.table("noon_facts")
    comp, bench = D.table("competitors"), D.table("benchmarks")

    cards.kpi_grid([
        cards.kpi("UAE e-commerce 2025", "USD 11.5 bn", "AED 42.2 bn; ×2.4 since 2020", "secondary", theme.BLUE),
        cards.kpi("UAE forecast 2030", "USD 18.3 bn", "9.9% a year 2026–2030 (EZDubai)", "secondary", theme.BLUE),
        cards.kpi("UAE online retail penetration", "15%+", "Online retail passed AED 50 bn in 2024 (Redseer)", "secondary", theme.GREEN),
        cards.kpi("noon on Trustpilot", "1.2 / 5", "~5,000 reviews, mostly delivery, refund and service complaints", "secondary", theme.RED),
    ])

    c1, c2 = st.columns([1.15, 1])
    with c1, cards.card("Market size by region (USD bn)", "How fast is the market noon plays in growing?"):
        reg = st.segmented_control("Region", ["UAE", "MENASA", "Saudi Arabia"], default="UAE", key="mk_reg")
        d = ms[ms["region"] == (reg or "UAE")]
        fig = go.Figure(go.Bar(x=d["year"].astype(str), y=d["value_usd_bn"],
                               marker_color=[theme.BLUE if k == "Actual" else theme.SOFT_BLUE if k == "Forecast" else theme.TEAL
                                             for k in d["kind"]],
                               marker_line=dict(color=theme.BLUE, width=1), text=[f"{v:.1f}" for v in d["value_usd_bn"]],
                               textposition="outside", cliponaxis=False, customdata=d[["kind", "source"]],
                               hovertemplate="%{x}: USD %{y:.1f} bn<br>%{customdata[0]}<br>%{customdata[1]}<extra></extra>"))
        fig.update_layout(yaxis_title="USD billion", yaxis_range=[0, d["value_usd_bn"].max() * 1.2])
        charts.show(fig, "mk_size", d)
        cards.explanation("Total e-commerce sales value; dark bars are reported actuals, light bars are forecasts.",
                          "The UAE market more than doubled between 2020 and 2025 and is still expected to grow by about "
                          "10% a year. Saudi Arabia is roughly twice the size of the UAE.",
                          "Growth is slowing from the Covid boom, so winning means keeping customers, not only finding new ones. "
                          "That makes repeat purchase after a bad experience the key number.",
                          "Research firms define e-commerce differently (some include travel, food and B2B), so figures from "
                          "different firms are shown separately and never added together.")
    with c2, cards.card("Market facts that matter for the three problems"):
        st.markdown(cards.html_table([{"Fact": esc(r.fact), "Source": _link(r.url, r.source)} for r in mf.itertuples()]),
                    unsafe_allow_html=True)

    with cards.card("Competitor map", "Who competes with noon, and on what?"):
        view = comp[["competitor", "type", "fast_delivery", "membership", "returns_signal", "review_signal", "threat_to_noon"]]
        st.markdown(cards.html_table([{
            "Competitor": f"<b>{esc(r.competitor)}</b><br><span class='lc-muted'>{esc(r.type)}</span>",
            "Fast delivery": esc(r.fast_delivery), "Membership": esc(r.membership),
            "Signals from research": esc(r.review_signal), "Threat": esc(r.threat_to_noon), "Source": _link(r.source)}
            for r in comp.itertuples()]), unsafe_allow_html=True)
        with st.expander("View, sort & download the competitor table"):
            charts.table(view, "mk_comp", "competitors.csv")
        cards.explanation(
            "Each rival's delivery offer, loyalty programme and what public sources say about them.",
            "Amazon's 15-minute Amazon Now and Talabat's dark stores mean speed alone no longer sets noon apart. "
            "Every big player has a membership. Reviews of almost all of them complain about delivery and support.",
            "noon can stand out on <b>reliability and trust</b>: keeping the promised date, fast refunds for good customers, "
            "and complaints that actually get fixed.",
            "Threat ratings are the group's judgement from the cited facts, not a measured score. Market shares by company "
            "are not published consistently, so none are shown.")

    with cards.card("What rivals already do about our three problems (benchmarks)"):
        st.markdown(cards.html_table([{"Problem": esc(r.problem), "Who": f"<b>{esc(r.who)}</b>", "What they do": esc(r.what_they_do),
                                       "Lesson for noon": esc(r.lesson_for_noon), "Source": _link(r.url)} for r in bench.itertuples()]),
                    unsafe_allow_html=True)

    with cards.card("noon at a glance"):
        st.markdown(cards.html_table([{"Fact": esc(r.fact), "Source": _link(r.url, r.source)} for r in nf.itertuples()]),
                    unsafe_allow_html=True)
        cards.notice("info", "Why reviews were not copied in.",
                     "Public review text belongs to its authors, so this dashboard does not store copies. The text model on the "
                     "Problem 3 page is trained on synthetic tickets written in the same style; noon would run the same "
                     "code on its own ticket and review exports.")
