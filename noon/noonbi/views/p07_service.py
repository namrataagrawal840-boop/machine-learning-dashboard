"""Page 7 — Problem 3: poor customer service → Case Ownership Engine."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from noonbi import cards, charts, theme
from noonbi import data as D
from noonbi.views import common as C


def render(f: D.Filters) -> None:
    cards.page_header("Problem 3 · Poor customer service and complaint resolution",
                      "Solution: the Case Ownership Engine — read every ticket automatically, give it one named owner and a "
                      "visible deadline, and contact the customer before they have to chase.",
                      "Text classification · clustering · regression · churn model",
                      [("Tickets", "synthetic"), ("Models", "derived")])
    t = D.table("tickets")
    if f.emirates:
        t = t[t["emirate"].isin(f.emirates)]
    if f.months:
        m = t["created_date"].dt.to_period("M").astype(str)
        t = t[(m >= f.months[0]) & (m <= f.months[1])]
    if t.empty:
        cards.notice("warn", "No tickets match the filters.")
        return
    S = D.summary().get("p3", {})
    cards.kpi_grid([
        cards.kpi("Tickets needing a repeat contact", C.pct(t["repeat_contact"].mean()), "Today: no single owner", "synthetic", theme.LAVENDER),
        cards.kpi("Answered with a template reply", C.pct(t["template_reply"].mean()), f"Average CSAT {t['csat'].mean():.1f} / 5", "synthetic", theme.PEACH),
        cards.kpi("Ticket topic read correctly", C.pct(S.get("topic_accuracy", np.nan)), "TF-IDF + logistic regression", "derived", theme.BLUE),
        cards.kpi("Extra churn per extra contact", f"+{S.get('churn_per_contact_pts', np.nan):.1f} pts", "Market response model", "derived", theme.RED),
    ])
    st.caption("▶ To watch this solution run record by record, open **8 · Solution Simulator** in the sidebar.")
    tabs = st.tabs(["1 · Evidence", "2 · Read every ticket", "3 · Complaint themes", "4 · Deadline, fix & churn", "5 · Cost of slow service"])
    with tabs[0]:
        _evidence(t)
    with tabs[1]:
        _read(S)
    with tabs[2]:
        _themes(S)
    with tabs[3]:
        _resolve(S)
    with tabs[4]:
        _cost(S)


def _evidence(t: pd.DataFrame) -> None:
    c1, c2 = st.columns(2)
    with c1, cards.card("What customers complain about"):
        fig, d = charts.count_bar(t["topic"], horizontal=True, colour=theme.LAVENDER, sort=True)
        fig.update_xaxes(title="% of tickets")
        charts.show(fig, "p3_topics", d)
    with c2, cards.card("Hours to resolve, by topic"):
        fig = charts.box_by(t, "topic", "resolution_hours", y_title="Hours to resolve", horizontal=True, points=False)
        charts.show(fig, "p3_res_box", t.groupby("topic")["resolution_hours"].describe().reset_index(), height=380)
    with cards.card("Template replies and repeat contacts drag satisfaction down"):
        g = t.groupby(["template_reply", "repeat_contact"]).agg(tickets=("ticket_id", "size"), csat=("csat", "mean"),
                                                              churn=("churn_90d", "mean")).reset_index()
        g["group"] = g["template_reply"].map({0: "Personal reply", 1: "Template reply"}) + " · " + g["repeat_contact"].map({0: "solved first time", 1: "had to contact again"})
        fig = charts.grouped_bar(g, "group", ["csat"], ["Average CSAT (1–5)"], y_title="CSAT", value_fmt="{:.2f}", colours=[theme.BLUE])
        charts.show(fig, "p3_csat", g, height=300)
        cards.explanation("Average satisfaction score (1–5) by reply type and whether the customer had to come back.",
                          "Customers who got a template and had to contact noon again are the least satisfied; personal replies "
                          "solved first time score highest.",
                          "The problem is ownership and routing, not head-count: more agents giving more templates faster will not fix it.",
                          "Synthetic data shaped by the pattern in public reviews.")


def _read(S: dict) -> None:
    cm, tw, ex = D.table("p3_topic_confusion"), D.table("p3_topic_words"), D.table("p3_ticket_examples")
    c1, c2 = st.columns([1.2, 1])
    with c1, cards.card("Reading tickets automatically (TF-IDF + logistic regression)"):
        mat = cm.set_index("actual")
        share = mat.div(mat.sum(axis=1), axis=0) * 100
        fig = charts.heatmap(share, text_fmt="{:.0f}%", value_name="% of actual topic", x_title="Predicted topic", y_title="Actual topic",
                             counts=mat, colorbar_title="%")
        charts.show(fig, "p3_cm", cm, height=380)
    with c2, cards.card("Words that give each topic away"):
        st.markdown(cards.html_table([{"Topic": f"<b>{C.esc(r.topic)}</b>", "Top words": C.esc(r.top_words)} for r in tw.itertuples()]),
                    unsafe_allow_html=True)
    with cards.card("Try it on a few test tickets"):
        charts.table(ex, "p3_ex", "ticket_examples.csv", column_config={"text": st.column_config.TextColumn("Ticket text", width="large"),
                                                                       "p_angry": st.column_config.ProgressColumn("Anger score", min_value=0, max_value=1)})
    cards.explanation("TF-IDF turns each ticket into numbers, giving more weight to telling words ('refund', 'rejected', "
                      "'driver') than to common ones ('order'). Logistic regression then learns which words go with which topic, "
                      "and a second model scores anger.",
                      f"Topic is read correctly {C.pct(S.get('topic_accuracy', np.nan))} of the time (macro F1 {S.get('topic_macro_f1', np.nan):.2f}); "
                      f"the anger model ranks tickets with ROC-AUC {S.get('anger_auc', np.nan):.2f}. Mistakes happen mainly when a "
                      "ticket mentions two issues.",
                      "Each ticket goes straight to the right team, and angry customers are flagged for a senior agent.",
                      "Synthetic tickets use a limited vocabulary, so real accuracy will be lower; noon would train on its labelled history.")


def _themes(S: dict) -> None:
    th, k, tm = D.table("p3_themes"), D.table("p3_theme_k"), D.table("p3_topic_month")
    c1, c2 = st.columns([1.3, 1])
    with c1, cards.card("Complaint themes found without labels (TF-IDF + K-Means)"):
        th = th.sort_values("tickets", ascending=False)
        fig = charts.bar(th["theme"] + ": " + th["top_terms"].str.split(", ").str[:3].str.join(", "), th["share"] * 100, th["tickets"],
                         [th["tickets"].sum()] * len(th), horizontal=True, colour=theme.LAVENDER, axis_title="% of tickets")
        charts.show(fig, "p3_themes", th)
    with c2, cards.card("How many themes?"):
        C.k_chart(k, S.get("theme_k", 7), "p3_k")
        st.caption("Silhouette keeps rising slowly; seven themes were kept because each one can be named and handed to a team.")
    with cards.card("Monthly complaint volume by topic (shared with operations every week)"):
        tm = tm[tm["month"] <= "2026-10"]  # tickets opened in early Nov belong to a month that is not complete
        piv = tm.pivot(index="month", columns="topic", values="tickets").fillna(0)
        fig = go.Figure()
        for i, c in enumerate(piv.columns):
            fig.add_bar(x=piv.index, y=piv[c], name=c, marker_color=theme.CATEGORICAL[i % len(theme.CATEGORICAL)])
        fig.update_layout(barmode="stack", yaxis_title="Tickets")
        charts.show(fig, "p3_month", tm)
    cards.explanation("K-Means groups tickets by the words they use, without being told the topics, so managers see what customers "
                      "actually complain about, including themes nobody planned for.",
                      "Most themes match one topic closely (purity above 85%). One theme is about template 'copy paste' replies "
                      "themselves, and one large theme mixes refund and delivery wording.",
                      "Each theme is sent to the team that owns the cause every week, so the source gets fixed, not just the ticket.",
                      "Unsupervised groups are imperfect; mixed themes are reviewed by a person.")


def _resolve(S: dict) -> None:
    m, roc, imp = D.table("p3_churn_metrics"), D.table("p3_churn_roc"), D.table("p3_churn_importance")
    co, best = D.table("p3_resolution_coefs"), D.table("p3_best_fix")
    c1, c2 = st.columns(2)
    with c1, cards.card("A realistic deadline (linear regression)", "What adds hours to a case?"):
        co = co.sort_values("extra_hours")
        fig = charts.bar(co["factor"], co["extra_hours"], horizontal=True, colour=theme.GOLD, value_fmt="{:+.0f} h",
                         axis_title="Extra hours vs a chat about a late delivery", hover_label="Hours")
        charts.show(fig, "p3_coef", co)
        st.caption(f"Average error {S.get('resolution_mae_h', np.nan):.0f} hours, R² {S.get('resolution_r2', np.nan):.2f}. "
                   "The customer sees: 'Your case owner: Omar, resolution by Thursday'.")
    with c2, cards.card("Suggested fix that stops repeat contacts (decision tree)"):
        charts.table(best[["topic", "suggested_fix", "recontact_7d", "csat", "tickets"]], "p3_best", "best_fix.csv", column_config={
            "recontact_7d": st.column_config.ProgressColumn("Came back within 7 days", min_value=0, max_value=0.6, format="%.2f"),
            "csat": st.column_config.NumberColumn("CSAT", format="%.2f")})
        with st.expander("Decision-tree rules (as the agent would see them)"):
            st.code(D.text("fix_tree_rules.txt")[:3000] or "Not available", language="text")
    with cards.card("Who will leave after a complaint? (churn classification)"):
        c1, c2 = st.columns([1.1, 1])
        with c1:
            C.metrics_table(m, "p3_m")
            C.roc_chart(roc, m, "p3_roc")
        with c2:
            C.importance_chart(imp, "p3_imp", colour=theme.ROSE)
        cards.explanation("Three classifiers predict whether a customer stops buying within 90 days of a ticket. Recall-heavy models "
                          "catch more leavers but raise more false alarms.",
                          f"Prediction is harder here (ROC-AUC ≈ {m['roc_auc'].max():.2f}), but repeat contacts, long resolution, "
                          "reopened tickets and low CSAT clearly push churn up.",
                          "High churn risk + high CLV → senior agent, resolved within 24 hours. A second contact moves the case "
                          "to a Resolution Desk that can approve a fix on the spot.",
                          "Churn depends on much more than one ticket, so moderate accuracy is expected.")


def _cost(S: dict) -> None:
    bc, br = D.table("p3_churn_by_contacts"), D.table("p3_churn_by_resolution")
    c1, c2 = st.columns(2)
    with c1, cards.card("Churn by number of contacts"):
        fig = charts.bar(bc["contacts_band"].astype(str), bc["churn_90d"] * 100, bc["tickets"], bc["tickets"], colour=theme.RED,
                         axis_title="% stopped buying within 90 days", hover_label="Churn %")
        fig.update_xaxes(title="Contacts on the same issue")
        charts.show(fig, "p3_bc", bc, height=320)
    with c2, cards.card("Churn by time to resolve"):
        order = ["<1 day", "1–2 days", "2–4 days", "4–7 days", "7+ days"]
        br["res_band"] = pd.Categorical(br["res_band"], order, ordered=True)
        br = br.sort_values("res_band")
        fig = charts.bar(br["res_band"].astype(str), br["churn_90d"] * 100, br["tickets"], br["tickets"], colour=theme.RED,
                         axis_title="% stopped buying within 90 days", hover_label="Churn %")
        charts.show(fig, "p3_br", br, height=320)
    cards.explanation("A market response model (multiple linear regression) measures how churn moves with each extra contact and each extra day.",
                      f"Each extra contact adds about {S.get('churn_per_contact_pts', np.nan):.1f} points of churn, and each extra day to "
                      f"resolve about {S.get('churn_per_day_pts', np.nan):.1f} points.",
                      "This puts a price on slow service: fixing a case on the second contact is cheaper than losing the customer.",
                      "Linear approximation on synthetic data.")
