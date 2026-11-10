"""Page 1 — overview: the three problems, how they link, and the headline numbers."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from noonbi import cards, charts, theme
from noonbi import data as D
from noonbi.views.common import aed, num, pct


def render(f: D.Filters) -> None:
    cards.page_header("Data-driven solutions for noon's three biggest customer problems",
                      "Orders arrive late, refunds are slow and complaints go nowhere. This dashboard shows the evidence, "
                      "the algorithms that fix each problem, and the business case.",
                      "Overview", [("Orders, returns, tickets", "synthetic"), ("Market & reviews", "secondary")])
    o, r, t = f.apply(D.table("orders"), "order_date"), f.apply(D.table("returns"), "order_date"), D.table("tickets")
    t = t[t["emirate"].isin(f.emirates)] if f.emirates else t
    if o.empty:
        cards.notice("warn", "No orders match the filters.", "Clear the sidebar filters to see the full data.")
        return
    cards.notice("info", "About the numbers.",
                 "noon's internal data is not public, so the orders, returns and tickets here are <b>synthetic</b>: built "
                 "with realistic cause-and-effect rules so every method can be shown end to end. Market size, competitor "
                 "facts and review ratings are <b>real secondary research</b> with links. Badges on every number say which is which.")
    k = [
        cards.kpi("Orders delivered after the promised date", pct(o["late"].mean()), f"{num(o['late'].sum())} of {num(len(o))} orders",
                  "synthetic", theme.PEACH),
        cards.kpi("Late orders where the customer got an update", pct(o.loc[o.late == 1, "proactive_update_sent"].mean()),
                  "Customers mostly find out by themselves", "synthetic", theme.PEACH),
        cards.kpi("Median refund time today", f"{r['refund_days'].median():.0f} days", "From return request to money back",
                  "synthetic", theme.GOLD),
        cards.kpi("Tickets needing a repeat contact", pct(t["repeat_contact"].mean()), f"{num(len(t))} tickets in the year",
                  "synthetic", theme.LAVENDER),
        cards.kpi("noon Trustpilot TrustScore", "1.2 / 5", "~5,000 reviews; 97–98% of negative reviews get a reply",
                  "secondary", theme.RED),
        cards.kpi("UAE e-commerce market 2025", "USD 11.5 bn", "AED 42.2 bn, more than double 2020 (EZDubai/Euromonitor)",
                  "secondary", theme.BLUE),
        cards.kpi("Bought again within 60 days: on time vs late", f"{pct(o.loc[o.late == 0, 'repurchase_60d'].mean(), 0)} vs "
                  f"{pct(o.loc[o.late == 1, 'repurchase_60d'].mean(), 0)}", "A late order costs repeat business",
                  "synthetic", theme.GREEN),
        cards.kpi("Customers who complained and then stopped buying", pct(t["churn_90d"].mean()), "Within 90 days of the ticket",
                  "synthetic", theme.ROSE),
    ]
    cards.kpi_grid(k)

    with cards.card("How the three problems feed each other",
                    "Does a bad delivery or refund turn into a complaint and then a lost customer?"):
        _chain(o, r, t)
        cards.explanation(
            "Each band follows customers from what went wrong, to whether they raised a ticket, to whether they "
            "stopped buying within 90 days.",
            "Most tickets start with a delivery or refund failure, and customers who had to complain churn far more often. "
            "That is why the three solutions are built as one system rather than three projects.",
            "Fixing delivery first removes the biggest source of tickets; fixing service stops the remaining "
            "complaints from becoming churn.",
            "Synthetic data built to reflect the pattern described in public reviews. With noon's own data the band "
            "sizes will change, but the method stays the same.")

    c1, c2 = st.columns(2)
    with c1, cards.card("The plan in one table"):
        st.markdown(cards.html_table([
            {"Solution": "<b>1 · Delivery Promise & Recovery Engine</b>", "Fixes": "Promise only what we can deliver; warn early; credit only persuadable customers",
             "Main techniques": "Classification, regression (P90), K-Means, Apriori, uplift, A/B test"},
            {"Solution": "<b>2 · Trust-Tiered Instant Refunds</b>", "Fixes": "Instant refunds for trusted customers; stop bad returns at source",
             "Main techniques": "Random forest tiers, K-Means on RFM, Apriori, regression, A/B test"},
            {"Solution": "<b>3 · Case Ownership Engine</b>", "Fixes": "One owner per complaint, visible deadline, early contact",
             "Main techniques": "TF-IDF text classification, K-Means themes, regression, churn model"},
        ]), unsafe_allow_html=True)
    with c2, cards.card("Where to look next"):
        st.markdown(
            '<div class="lc-body"><p><b>Market & competitors</b> — who noon is up against and what rivals already do.</p>'
            '<p><b>Data & cleaning</b> — how the data was built, what was wrong with it and how it was fixed.</p>'
            '<p><b>Customer foundations</b> — segments, lifetime value, churn and next purchase (blogs 2–5).</p>'
            '<p><b>Problems 1–3</b> — one page per problem: evidence, root causes, models and the test.</p>'
            '<p><b>Business case</b> — nine pillars, KPIs, and a gain calculator you can change.</p></div>',
            unsafe_allow_html=True)


def _chain(o: pd.DataFrame, r: pd.DataFrame, t: pd.DataFrame) -> None:
    trig = t["trigger"].replace({"Late order": "Late delivery", "Failed delivery": "Failed delivery", "Slow refund": "Slow refund",
                                 "Refund rejected": "Refund rejected", "Defect": "Other issues", "Wrong/damaged": "Other issues",
                                 "Payment": "Other issues"})
    late_no_ticket = int(o["late"].sum() - (t["trigger"] == "Late order").sum())
    flows = []
    for src, g in t.assign(src=trig).groupby("src"):
        flows.append((src, "Raised a ticket", len(g)))
    flows.append(("Late delivery", "No ticket", max(late_no_ticket, 0)))
    tk = t
    flows.append(("Raised a ticket", "Stopped buying (90 days)", int(tk["churn_90d"].sum())))
    flows.append(("Raised a ticket", "Kept buying", int((1 - tk["churn_90d"]).sum())))
    labels = list(dict.fromkeys([a for a, _, _ in flows] + [b for _, b, _ in flows]))
    idx = {l: i for i, l in enumerate(labels)}
    colours = {"Late delivery": theme.PEACH, "Failed delivery": theme.GOLD, "Slow refund": theme.LAVENDER,
               "Refund rejected": theme.ROSE, "Other issues": theme.SLATE, "Raised a ticket": theme.BLUE,
               "No ticket": "#D5DDE6", "Stopped buying (90 days)": theme.RED, "Kept buying": theme.GREEN}
    fig = go.Figure(go.Sankey(
        node=dict(label=labels, pad=18, thickness=16, color=[colours.get(l, theme.BLUE) for l in labels], line=dict(width=0)),
        link=dict(source=[idx[a] for a, _, _ in flows], target=[idx[b] for _, b, _ in flows], value=[v for *_, v in flows],
                  color="rgba(102,141,186,0.18)", hovertemplate="%{source.label} → %{target.label}: %{value:,}<extra></extra>")))
    charts.show(fig, "ov_chain", pd.DataFrame(flows, columns=["from", "to", "count"]), height=380)
