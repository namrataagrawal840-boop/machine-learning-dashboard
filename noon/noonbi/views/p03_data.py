"""Page 3 — how the data was built, what was wrong with it and how it was cleaned."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from noonbi import cards, charts, theme
from noonbi import data as D
from noonbi.views.common import num

DATASETS = [
    ("orders", "Orders", "One row per order: zone, seller, courier, category, sale day, promised vs actual delivery, "
     "failed attempts, recovery test group, repeat purchase.", "Problem 1, customer models"),
    ("returns", "Returns", "One row per return: reason, customer history, photos, QC result, refund time, instant-refund "
     "test group, repeat purchase.", "Problem 2"),
    ("tickets", "Support tickets", "One row per complaint: text, topic, anger, contacts, resolution time, reopened, CSAT, "
     "fix applied, churn.", "Problem 3"),
    ("customers", "Customers", "One row per customer: RFM, returns, tickets, next-12-month value, churn, next purchase.",
     "Segmentation, CLV, churn"),
    ("daily_demand", "Daily demand", "Daily orders by region over two years, with sale events.", "Demand forecast"),
    ("sellers", "Sellers", "Seller id and fulfilment type.", "Seller tiers"),
]


def render(f: D.Filters) -> None:
    cards.page_header("Data: how it was created and cleaned",
                      "Step 1 builds realistic synthetic data; step 2 cleans and reshapes it so the models can use it.",
                      "Steps 1–2", [("All operational data", "synthetic")])
    q, log = D.table("quality_summary"), D.table("cleaning_log")
    cards.kpi_grid([cards.kpi(f"{r.dataset.title()}: raw → clean rows", f"{num(r.raw_rows)} → {num(r.clean_rows)}",
                              f"{r.columns} columns after cleaning", "synthetic", theme.BLUE) for r in q.itertuples()]
                   + [cards.kpi("Fixes logged", num(log["rows_affected"].where(log["step"] != "Derived fields", 0).sum()),
                                f"{len(log)} cleaning steps, each with a reason", "derived", theme.GREEN)])

    with cards.card("Step 1 · Why synthetic data, and how it was made realistic"):
        st.markdown(
            '<div class="lc-body"><p><b>Why:</b> noon does not publish its orders, refunds or tickets. To show the solutions '
            'working end to end, we generated a stand-in for the data noon already holds. The code is fixed-seed, so anyone '
            'gets exactly the same data.</p>'
            '<p><b>How:</b> answers are not random. Each record follows cause-and-effect rules that match what customers say '
            'in public reviews: slow sellers, bulky items, sale days, evening orders, 3PL couriers, far zones and weak '
            'addresses make orders later; a late order with no update is more likely to become a complaint; slow or '
            'rejected refunds reduce repeat buying; repeat contacts push customers to leave. Two randomised tests were '
            'built in (a recovery credit and instant refunds) so A/B testing and uplift modelling can be shown.</p>'
            '<p><b>Scale:</b> 30,000 orders, ~4,900 returns, ~8,500 tickets and ~8,100 customers over Nov 2025 – Oct 2026, '
            'across 22 real UAE neighbourhoods. This is a sample, not noon\'s real volume.</p></div>',
            unsafe_allow_html=True)
        st.markdown(cards.html_table([{"Dataset": f"<b>{n}</b>", "What one row is": d, "Used for": u}
                                      for _, n, d, u in DATASETS]), unsafe_allow_html=True)

    with cards.card("Step 2 · Cleaning log", "What was wrong with the raw exports, and what was done about it?"):
        ds = st.pills("Dataset", ["orders", "returns", "tickets"], default=None, selection_mode="multi", key="dq_ds")
        lg = log[log["dataset"].isin(ds)] if ds else log
        lg2 = lg[lg["step"] != "Derived fields"]
        fig = charts.bar(lg2["dataset"] + " · " + lg2["step"], lg2["rows_affected"], horizontal=True, colour=theme.PEACH,
                         value_fmt="{:,.0f}", axis_title="Rows affected", hover_label="Rows")
        charts.show(fig, "dq_log_chart")
        charts.table(lg, "dq_log", "cleaning_log.csv",
                     column_config={"why": st.column_config.TextColumn("Why", width="large"),
                                    "action": st.column_config.TextColumn("Action", width="large")})
        cards.explanation(
            "Every problem found in the raw files, how many rows it touched, and the rule used to fix it.",
            "Duplicates, two date formats, spelling variants of emirate names, prices typed as text, impossible negative "
            "times and out-of-range scores were all present, as they would be in a real export.",
            "Cleaning first means a late order is counted once and a 'DXB' order is still a Dubai order, so the models "
            "learn from the truth rather than from typing errors.",
            "Missing values were <b>not</b> guessed: a missing zone stays 'Unknown' and a missing refund time stays blank, "
            "because inventing values would invent findings.")

    with cards.card("Preview the clean data"):
        name = st.selectbox("Table", [d[0] for d in DATASETS], format_func=lambda x: dict((a, b) for a, b, *_ in DATASETS)[x],
                            key="dq_prev")
        df = D.table(name)
        charts.table(df.head(500), f"dq_prev_{name}", f"{name}_first500.csv", height=320)
        st.caption(f"Showing the first 500 of {len(df):,} rows. Full files are in noon/data/.")
