"""Page 8 — one system: nine pillars, KPIs, incremental gain and test design."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from noonbi import cards, charts, theme
from noonbi import data as D
from noonbi.views import common as C

PILLARS = [
    ("1", "Know your metrics", "Pick the few numbers that matter", "Promise kept, refund days, repeat contacts, repeat purchase"),
    ("2", "Customer segmentation", "Split customers into similar groups", "Seller tiers, returner groups, zone groups"),
    ("3", "CLV prediction", "Profit a customer will bring", "Who gets credits, instant refunds, fast escalation"),
    ("4", "Churn prediction", "Who is about to stop buying", "Customers with late orders, rejected refunds, open cases"),
    ("5", "Next purchase day", "When a customer will buy next", "Timing of win-back messages and follow-ups"),
    ("6", "Predicting sales", "Forecast demand ahead", "Courier and agent capacity for Yellow Friday, Ramadan"),
    ("7", "Market response models", "How customers react to what we change", "Effect of contacts and resolution time on churn"),
    ("8", "Uplift modelling", "Who changes because of our action", "Credits only for persuadable customers"),
    ("9", "A/B testing", "A fair experiment to prove cause", "Every solution is tested against today's process"),
]
KPIS = [
    ("K-Means", "Silhouette score, elbow", "Retention and CLV by segment"),
    ("XGBoost / random forest classification", "Precision, recall, F1, ROC-AUC", "On-time delivery, fraud rate, churn"),
    ("Regression (XGBoost, linear, quantile)", "MAE, R², P90 coverage", "Promise accuracy, refund days, resolution time"),
    ("Apriori", "Support, confidence, lift", "Fewer repeat delays and returns at source"),
    ("TF-IDF + logistic regression", "Accuracy, macro F1", "First-contact resolution"),
    ("Uplift (two-model)", "Observed uplift by group", "Credit cost per customer kept"),
]


def render(f: D.Filters) -> None:
    cards.page_header("Business case: one system, measurable gain",
                      "How the three solutions share data and models, which KPI each algorithm moves, and what the gain could be. "
                      "Change the assumptions to see the effect.",
                      "Integration · KPIs · incremental gain", [("Gain inputs", "illustrative"), ("Test results", "derived")])
    ab = D.table("ab_tests")
    c1, c2 = st.columns([1, 1])
    with c1, cards.card("Nine pillars, in simple words"):
        st.markdown(cards.html_table([{"#": a, "Pillar": f"<b>{b}</b>", "Meaning": c, "Where we use it": d} for a, b, c, d in PILLARS]),
                    unsafe_allow_html=True)
    with c2, cards.card("How each algorithm is judged, and the KPI it moves"):
        st.markdown(cards.html_table([{"Algorithm": f"<b>{a}</b>", "Model metric": b, "Business KPI": c} for a, b, c in KPIS]),
                    unsafe_allow_html=True)
        cards.notice("info", "Why both columns matter.", "A model can be accurate and still not help the business, so each one is "
                     "tied to a KPI and proven with a randomised test.")
    with cards.card("Root-cause chains across the problems (Apriori on customer histories)"):
        ch = D.table("chain_rules")
        C.rules_table(ch, "bc_chain", 12)
        st.caption("Customers with a late delivery and no update, or a slow or rejected refund, almost always complain; a rejected "
                   "refund then raises churn. One data platform therefore serves all three solutions.")

    with cards.card("Incremental gain calculator", "What could each solution be worth per month? (assumptions from our proposal)"):
        cards.notice("warn", "These are assumptions, not noon's real figures.",
                     "Defaults come from the group's proposal. The (?) help on each uplift box shows what the synthetic A/B test measured. "
                     "noon would plug in its own numbers; the formula stays: extra customers kept × value per customer − cost.")
        value = st.number_input("Profit per kept customer over 12 months (AED)", 50, 2000, 240, 10, key="g_val")
        t1, t2, t3, t4 = st.tabs(["Solution 1 · Delivery", "Solution 2 · Refunds", "Solution 3 · Service", "Total"])
        with t1:
            a, b = st.columns(2)
            late = a.number_input("Late orders per month", 1000, 1_000_000, 80_000, 1000, key="g1_late")
            up1 = a.number_input("Extra repeat purchase (pts)", 0.0, 30.0, 5.0, 0.5, key="g1_up",
                                 help=f"Measured in the synthetic A/B test: +{ab.iloc[0]['uplift_pts']:.1f} pts")
            cr = b.number_input("Customers given a credit", 0, 1_000_000, 30_000, 1000, key="g1_cr")
            cv = b.number_input("Credit per customer (AED)", 0, 200, 15, 1, key="g1_cv")
            kept1 = late * up1 / 100
            g1 = kept1 * value - cr * cv
            _gain([("Extra customers kept", kept1, ""), ("Value", kept1 * value, "AED"), ("Cost of credits", cr * cv, "AED"),
                   ("Net gain / month", g1, "AED")])
        with t2:
            a, b = st.columns(2)
            gr = a.number_input("Green-tier returns per month", 1000, 1_000_000, 90_000, 1000, key="g2_ret")
            up2 = a.number_input("Extra repeat purchase (pts)", 0.0, 30.0, 6.0, 0.5, key="g2_up",
                                 help=f"Measured in the synthetic A/B test: +{ab.iloc[1]['uplift_pts']:.1f} pts")
            fr = b.number_input("Extra fraud loss from faster refunds (AED / month)", 0, 10_000_000, 90_000, 5000, key="g2_fr")
            kept2 = gr * up2 / 100
            g2 = kept2 * value - fr
            _gain([("Extra customers kept", kept2, ""), ("Value", kept2 * value, "AED"), ("Extra fraud loss", fr, "AED"),
                   ("Net gain / month", g2, "AED")])
        with t3:
            a, b = st.columns(2)
            con = a.number_input("Support contacts per month", 1000, 5_000_000, 200_000, 5000, key="g3_con")
            rep0 = a.number_input("Repeat contacts today (%)", 0.0, 100.0, 35.0, 1.0, key="g3_r0")
            rep1 = a.number_input("Repeat contacts with the solution (%)", 0.0, 100.0, 20.0, 1.0, key="g3_r1")
            cpc = b.number_input("Cost per contact (AED)", 0.0, 200.0, 12.0, 1.0, key="g3_cpc")
            base = b.number_input("Customers who contacted support", 0, 5_000_000, 140_000, 5000, key="g3_base")
            dch = b.number_input("Drop in churn (pts)", 0.0, 30.0, 2.0, 0.5, key="g3_dch")
            saved = con * (rep0 - rep1) / 100
            kept3 = base * dch / 100
            g3 = saved * cpc + kept3 * value
            _gain([("Contacts avoided", saved, ""), ("Contact cost saved", saved * cpc, "AED"), ("Customers kept", kept3, ""),
                   ("Total gain / month", g3, "AED")])
        with t4:
            tot = pd.DataFrame({"solution": ["1 · Delivery", "2 · Refunds", "3 · Service"], "gain_aed": [g1, g2, g3]})
            fig = charts.bar(tot["solution"], tot["gain_aed"] / 1e6, colour=theme.GREEN, value_fmt="AED {:.2f} m",
                             axis_title="Net gain per month (AED millions)", hover_label="AED m")
            charts.show(fig, "bc_total", tot, height=300)
            st.markdown(cards.kpi("Combined net gain per month", C.aed(tot["gain_aed"].sum()), "Sum of the three solutions with the inputs above",
                                  "illustrative", theme.GREEN), unsafe_allow_html=True)

    with cards.card("Plan a test: how many customers are needed?", "Sample size for an A/B test on a repeat-purchase rate"):
        a, b, c = st.columns(3)
        p0 = a.number_input("Today's rate (%)", 1.0, 99.0, 45.0, 1.0, key="ss_p0") / 100
        mde = b.number_input("Smallest uplift worth detecting (pts)", 0.5, 30.0, 5.0, 0.5, key="ss_mde") / 100
        power = c.select_slider("Power", [0.7, 0.8, 0.9], 0.8, key="ss_pow")
        p1 = min(p0 + mde, 0.999)
        za, zb = stats.norm.ppf(0.975), stats.norm.ppf(power)
        n = ((za * math.sqrt(2 * ((p0 + p1) / 2) * (1 - (p0 + p1) / 2)) + zb * math.sqrt(p0 * (1 - p0) + p1 * (1 - p1))) ** 2) / (mde ** 2)
        st.markdown(cards.kpi("Customers needed in each group", C.num(math.ceil(n)), f"Two-sided test at 5%, power {power:.0%}",
                              "derived", theme.BLUE), unsafe_allow_html=True)
        st.caption("Solution 3 compares three routing options, so it uses a one-way ANOVA (or a chi-square test on rates) with the "
                   "same sample per arm. Tests are stratified by RFM segment so each arm has the same mix of customers.")

    with cards.card("What to do first"):
        st.markdown(cards.html_table([
            {"When": "<b>Weeks 1–4</b>", "Action": "GPS lock on 'customer unavailable', WhatsApp pin request, early-update message",
             "Why first": "Almost no modelling, removes fake failed deliveries and silence"},
            {"When": "<b>Months 2–3</b>", "Action": "Late-order model + P90 promise in one city; recovery-credit A/B test with uplift targeting",
             "Why first": "Delivery causes most tickets, so it shrinks Problem 3 too"},
            {"When": "<b>Months 3–5</b>", "Action": "Trust tiers with instant wallet refunds for Green; refund tracker; size-guide fixes from Apriori",
             "Why first": "Needs the CLV and churn models, which are built once and shared"},
            {"When": "<b>Months 5–6</b>", "Action": "Ticket reading, named case owner, Resolution Desk; weekly theme report to operations",
             "Why first": "Uses the same customer scores to route urgent, high-value cases"},
        ]), unsafe_allow_html=True)
        cards.notice("info", "Biggest risk: bad data.", "If handover times or return reasons are recorded wrongly the models learn the wrong "
                     "things, so cleaning and checking come first, and every rollout starts as an A/B test.")


def _gain(items) -> None:
    cards.kpi_grid([cards.kpi(lbl, (f"AED {v:,.0f}" if unit == "AED" else f"{v:,.0f}"), "",
                              "illustrative", theme.GREEN if i == len(items) - 1 else theme.BLUE) for i, (lbl, v, unit) in enumerate(items)])
