"""Solution simulator: pick a problem → the problem in detail → data points → descriptive analytics → run the simulation.

The run replays real (synthetic) records the models never trained on, once under today's process and once with the
solution switched on, and animates the result record by record. With default assumptions the numbers are fixed.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from noonbi import cards, charts, sim, theme
from noonbi import data as D
from noonbi.views import common as C

esc = cards.esc
PROBLEMS = ["Problem 1 · Delivery", "Problem 2 · Refunds", "Problem 3 · Service"]


def render(f: D.Filters) -> None:
    cards.page_header("Solution simulator",
                      "Pick a problem. See it explained, the data captured and the basic analytics, then press Run to watch "
                      "the solution work on records the models have never seen — today's process and the solution side by side.",
                      "Simulation · fixed numbers", [("Records", "synthetic"), ("Model scores", "derived"), ("Assumptions", "illustrative")])
    choice = st.segmented_control("Choose a problem", PROBLEMS, default=PROBLEMS[0], key="sim_problem") or PROBLEMS[0]
    i = PROBLEMS.index(choice)
    [_p1, _p2, _p3][i]()


# =============================================================== shared pieces
def _steps(n_active: int) -> None:
    labels = ["1 · The problem", "2 · Data captured", "3 · Descriptive analytics", "4 · Run the simulation"]
    html = "".join(f'<span class="lc-chip" style="margin-right:.4rem;{"background:#668DBA;color:#fff;" if k < n_active else ""}">{l}</span>'
                   for k, l in enumerate(labels))
    st.markdown(f'<div style="margin:.2rem 0 .6rem 0">{html}</div>', unsafe_allow_html=True)


def _problem_card(title: str, paras: list[str], evidence: list[tuple[str, str]]) -> None:
    with cards.card(f"Step 1 · {title}"):
        st.markdown('<div class="lc-body">' + "".join(f"<p>{p}</p>" for p in paras) + "</div>", unsafe_allow_html=True)
        if evidence:
            st.markdown(cards.html_table([{"Public evidence": esc(t), "Source": f'<a href="{esc(u)}" target="_blank">link</a>'}
                                          for t, u in evidence]), unsafe_allow_html=True)


def _data_points(rows: list[tuple[str, str, str, str]], example: pd.Series) -> None:
    with cards.card("Step 2 · Data points captured", "What noon records for every case, and what each field is used for"):
        st.markdown(cards.html_table([{"Data point": f"<b>{esc(a)}</b>", "Example (one real record)": esc(_ex(example.get(b, "—"))),
                                       "Where it comes from": esc(c), "Used for": esc(u)} for a, b, c, u in rows]),
                    unsafe_allow_html=True)


def _ex(v) -> str:
    if isinstance(v, (float, np.floating)):
        return f"{v:,.2f}" if abs(v) >= 1 or v == 0 else f"{v:.2f}"
    if isinstance(v, pd.Timestamp):
        return f"{v:%d %b %Y}"
    return str(v)[:90]


def _assumptions(defaults: dict, spec: list[tuple], key: str) -> dict:
    a = dict(defaults)
    with st.expander("Simulation assumptions (fixed by default — change them to test 'what if')"):
        if st.button("Reset to defaults", key=f"{key}_reset"):
            for k in defaults:
                st.session_state.pop(f"{key}_{k}", None)
        cols = st.columns(3)
        for j, (k, label, kind, lo, hi, step, helptext) in enumerate(spec):
            with cols[j % 3]:
                if kind == "bool":
                    a[k] = st.toggle(label, value=defaults[k], key=f"{key}_{k}", help=helptext)
                else:
                    a[k] = st.number_input(label, lo, hi, defaults[k], step, key=f"{key}_{k}", help=helptext)
    return a


def _fmt(label: str, v) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    if any(w in label.lower() for w in ("rate", "kept", "told", "within", "needed", "reopened")) and abs(v) <= 1.0001:
        return f"{v * 100:.1f}%"
    if "aed" in label.lower():
        return f"AED {v:,.0f}"
    if "csat" in label.lower() or "median" in label.lower():
        return f"{v:,.1f}"
    return f"{v:,.0f}"


def _kpi_compare(k: dict, better_low: set[str]) -> str:
    out = []
    for label, (t, s) in list(k.items())[1:]:
        delta = ""
        try:
            if not (np.isnan(t) or np.isnan(s)) and t != s:
                good = (s < t) if label in better_low else (s > t)
                col = theme.GREEN if good else theme.RED
                delta = f'<span style="color:{col};font-weight:600">{"▼" if s < t else "▲"}</span>'
        except TypeError:
            pass
        out.append(f'<div class="lc-card lc-kpi" style="border-top:3px solid {theme.BLUE};padding:.6rem .8rem">'
                   f'<div class="lc-kpi-label">{esc(label)}</div>'
                   f'<div style="display:flex;gap:1rem;align-items:baseline;margin-top:.2rem">'
                   f'<div><div class="lc-muted" style="font-size:.7rem">TODAY</div><div style="font-size:1.15rem;font-weight:700;color:{theme.SLATE}">{_fmt(label, t)}</div></div>'
                   f'<div><div class="lc-muted" style="font-size:.7rem">WITH SOLUTION</div><div style="font-size:1.15rem;font-weight:700;color:{theme.TEXT}">{_fmt(label, s)} {delta}</div></div>'
                   f'</div></div>')
    rows = [out[i:i + 4] for i in range(0, len(out), 4)]
    return "".join(f'<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:.6rem;margin-bottom:.6rem">{"".join(r)}</div>' for r in rows)


def _run(key: str, d: pd.DataFrame, kpi_fn, event_fn, a: dict, track: tuple[str, str, str], better_low: set[str],
         date_col: str) -> None:
    """Animated replay. track = (today_col, solution_col, label) for the running chart (cumulative sum)."""
    speed = st.select_slider("Speed", ["Slow", "Normal", "Fast", "Instant"], value="Normal", key=f"{key}_speed")
    run = st.button("▶ Run the simulation", type="primary", key=f"{key}_run")
    state_key = f"{key}_result"
    if run:
        frames = 40
        delay = {"Slow": 0.25, "Normal": 0.09, "Fast": 0.02, "Instant": 0}[speed]
        bar = st.progress(0.0, text="Starting…")
        kpi_ph, chart_ph, feed_ph = st.empty(), st.empty(), st.empty()
        bounds = np.linspace(0, len(d), frames + 1).astype(int)
        cum_t, cum_s, xs = [], [], []
        if delay == 0:
            bounds = bounds[[0, -1]]
        for j in range(1, len(bounds)):
            part = d.iloc[: bounds[j]]
            cum_t.append(part[track[0]].sum())
            cum_s.append(part[track[1]].sum())
            xs.append(bounds[j])
            day = part[date_col].iloc[-1]
            bar.progress(j / (len(bounds) - 1), text=f"Processed {bounds[j]:,} of {len(d):,} · simulated date {pd.Timestamp(day):%d %b %Y}")
            kpi_ph.markdown(_kpi_compare(kpi_fn(part, a), better_low), unsafe_allow_html=True)
            if delay or j == len(bounds) - 1:
                fig = go.Figure()
                fig.add_scatter(x=xs, y=cum_t, name="Today", mode="lines", line=dict(color=theme.SLATE, width=2.5))
                fig.add_scatter(x=xs, y=cum_s, name="With solution", mode="lines", line=dict(color=theme.GREEN, width=3))
                fig.update_layout(xaxis_title="Records processed", yaxis_title=track[2], height=300, margin=dict(l=10, r=10, t=30, b=10),
                                  xaxis_range=[0, len(d)])
                chart_ph.plotly_chart(fig, width="stretch", key=f"{key}_live_{j}")
                recent = d.iloc[max(0, bounds[j] - 6): bounds[j]]
                feed_ph.markdown('<div class="lc-card" style="padding:.6rem .9rem"><div class="lc-kpi-label">LIVE FEED · last records processed</div>'
                                 + "".join(f'<div style="font-size:.82rem;padding:.2rem 0;border-bottom:1px solid #EEF1F5">{event_fn(r)}</div>'
                                           for r in recent.itertuples()) + "</div>", unsafe_allow_html=True)
            if delay:
                time.sleep(delay)
        bar.progress(1.0, text=f"Done · {len(d):,} records replayed under both processes")
        st.session_state[state_key] = True
    elif st.session_state.get(state_key):
        st.markdown(_kpi_compare(kpi_fn(d, a), better_low), unsafe_allow_html=True)
    else:
        cards.notice("info", "Ready.", f"{len(d):,} records are loaded. Press <b>Run the simulation</b> to replay them.")


def _result_table(k: dict, key: str) -> None:
    rows = [{"measure": lbl, "today": _fmt(lbl, t), "with_solution": _fmt(lbl, s)} for lbl, (t, s) in k.items()]
    charts.table(pd.DataFrame(rows), key, f"{key}.csv")


# =============================================================== Problem 1
def _p1() -> None:
    d0 = D.table("p1_orders")
    o = D.table("orders")
    _steps(4)
    _problem_card("The problem in detail: deliveries arrive late and nobody warns the customer", [
        "At checkout almost every order says <b>'tomorrow'</b>, whatever the seller, the area, the item or the season. "
        "When the order misses that date the customer is not told; they find out by checking the tracking page, then chase "
        "support, then write a review.",
        "Delays have known causes: slow sellers who ship themselves, bulky items, evening orders, 3PL couriers, villa and remote "
        "zones, sale days like Yellow Friday, and addresses the driver cannot find. Some 'customer unavailable' marks are "
        "made without the driver ever reaching the door.",
        f"In the data, <b>{C.pct(o['late'].mean())}</b> of orders arrive after the promised date and only "
        f"<b>{C.pct(o.loc[o.late == 1, 'proactive_update_sent'].mean())}</b> of those customers get an update first. "
        f"Customers with a late order buy again less often ({C.pct(o.loc[o.late == 1, 'repurchase_60d'].mean(), 0)} vs "
        f"{C.pct(o.loc[o.late == 0, 'repurchase_60d'].mean(), 0)} within 60 days).",
    ], [("noon averages about 1.2/5 on Trustpilot from ~5,000 reviews; delivery is a recurring theme", "https://jp.trustpilot.com/review/noon.com"),
        ("Amazon Now promises 15-minute delivery in the UAE, raising expectations", "https://www.aboutamazon.com/news/innovation/amazon-now-delivery-uae")])
    _data_points([
        ("Order id & date", "order_id", "Order system", "Tracking every case"),
        ("Zone / emirate", "zone", "Checkout address", "Zone clusters, distance"),
        ("Category & item size", "category", "Catalogue", "Bulky items take longer"),
        ("Seller & who ships", "fulfilment", "Seller centre", "Seller tiers, handover time"),
        ("Courier", "courier", "Logistics scans", "3PL vs in-house speed"),
        ("Sale event", "sale_event", "Marketing calendar", "Peak-day risk"),
        ("Address quality", "address_quality", "Checkout form", "Failed-attempt risk"),
        ("Promised days today", "promised_days", "Checkout page", "Compare promise vs reality"),
        ("Actual delivery hours", "actual_hours", "Courier scans", "Regression target"),
        ("Predicted late risk", "p_late", "Late-order model", "Risk tier and actions"),
        ("P90 promise (days)", "p90_promise_days", "Quantile model", "Honest date at checkout"),
        ("Predicted uplift of a credit", "pred_uplift", "Uplift model", "Who gets the credit"),
    ], d0.iloc[0])
    with cards.card("Step 3 · Descriptive analytics"):
        cards.kpi_grid([
            cards.kpi("Orders in the simulation", C.num(len(d0)), "Aug–Oct 2026, never used for training", "synthetic", theme.BLUE),
            cards.kpi("Late today", C.pct(d0["late"].mean()), f"{C.num(d0['late'].sum())} orders", "synthetic", theme.PEACH),
            cards.kpi("Failed first attempts", C.pct(d0["failed_first_attempt"].mean()), "Driver could not deliver first time", "synthetic", theme.GOLD),
            cards.kpi("High-risk orders flagged", C.pct((d0["risk_tier"] == "High risk").mean()), "By the late-order model", "derived", theme.RED),
        ])
        c1, c2 = st.columns(2)
        with c1:
            g = d0.groupby("fulfilment")["late"].mean().mul(100).reset_index()
            charts.show(charts.bar(g["fulfilment"], g["late"], colour=theme.PEACH, axis_title="% late", horizontal=True), "sim1_d1", g, height=220)
            g = d0.groupby("courier")["late"].mean().mul(100).reset_index()
            charts.show(charts.bar(g["courier"], g["late"], colour=theme.PEACH, axis_title="% late", horizontal=True), "sim1_d2", g, height=230)
        with c2:
            g = d0.groupby("risk_tier")["late"].mean().mul(100).reindex(["Low risk", "Medium risk", "High risk"]).reset_index()
            charts.show(charts.bar(g["risk_tier"], g["late"], colour=theme.RED, axis_title="% actually late"), "sim1_d3", g, height=260)
            st.caption("The model's risk tiers line up with what really happened: high-risk orders are late most of the time.")
    with cards.card("Step 4 · Run the simulation: the Delivery Promise & Recovery Engine",
                    "Every order goes through the engine at checkout and at dispatch"):
        st.markdown('<div class="lc-body"><ol>'
                    '<li><b>Checkout:</b> the P90 model sets an honest date instead of a fixed \'tomorrow\'.</li>'
                    '<li><b>Dispatch:</b> the late-order model scores the order. High risk → priority courier.</li>'
                    '<li><b>Address:</b> partial addresses get a WhatsApp pin request; drivers can only mark \'unavailable\' with GPS proof.</li>'
                    '<li><b>Before it is late:</b> customers with risk above the threshold get a message with a new date.</li>'
                    '<li><b>Recovery:</b> if still late, the uplift model picks the customers a credit will actually win back.</li>'
                    '</ol></div>', unsafe_allow_html=True)
        a = _assumptions(sim.P1_DEFAULTS, [
            ("use_p90", "Use P90 promise at checkout", "bool", None, None, None, "Off = keep today's fixed promise"),
            ("priority_hours", "Hours saved by priority courier", "num", 0.0, 24.0, 1.0, "For high-risk orders"),
            ("warn_threshold", "Warn customer if risk above", "num", 0.05, 0.95, 0.05, "Late-order model score"),
            ("credit_share", "Share of late orders given a credit", "num", 0.0, 1.0, 0.05, "Top-ranked by predicted uplift"),
            ("credit_aed", "Credit per customer (AED)", "num", 0, 100, 5, None),
            ("pin_request", "Ask for a location pin", "bool", None, None, None, "Partial addresses"),
            ("pin_fix_rate", "Share of failed attempts a pin prevents", "num", 0.0, 1.0, 0.05, None),
            ("gps_lock", "GPS lock on 'unavailable'", "bool", None, None, None, None),
        ], "sim1")
        d = sim.run_p1(d0, a)
        _run("sim1", d, sim.p1_kpis, sim.p1_event, a, ("t_late", "s_late", "Late orders (cumulative)"),
             {"Late orders", "Complaint tickets (expected)"}, "order_date")
    if st.session_state.get("sim1_result"):
        _p1_results(d, a)


def _p1_results(d: pd.DataFrame, a: dict) -> None:
    k = sim.p1_kpis(d, a)
    clv = d["clv"].mean()
    extra = k["Bought again within 60 days"][1] - k["Bought again within 60 days"][0]
    with cards.card("Simulation result", "What changed, and what it is worth"):
        c1, c2 = st.columns([1.1, 1])
        with c1:
            g = d.groupby("emirate").agg(today=("t_late", "mean"), solution=("s_late", "mean")).reset_index()
            fig = charts.grouped_bar(g, "emirate", ["today", "solution"], ["Today", "With solution"], y_title="Late rate",
                                     value_fmt="{:.0%}", colours=[theme.SLATE, theme.GREEN])
            charts.show(fig, "sim1_res_em", g, height=320)
        with c2:
            _result_table(k, "sim1_table")
        cards.explanation(
            "The same orders, customers and random draws under both processes; only the process changes.",
            f"Promise kept rises from {C.pct(k['Promise kept'][0], 0)} to {C.pct(k['Promise kept'][1], 0)}, expected complaint tickets fall "
            f"from {k['Complaint tickets (expected)'][0]:,.0f} to {k['Complaint tickets (expected)'][1]:,.0f}, and {extra:,} more customers buy again.",
            f"At an average next-12-month profit of {C.aed(clv)} per customer, the extra repeat buyers are worth about "
            f"{C.aed(extra * clv)} against {C.aed(k['Credits given (AED)'][1])} of credits, for {len(d):,} orders.",
            "The P90 promise is longer on average, and the simulation does not model customers who might not order because of "
            "a longer date. That is exactly what the real A/B test must measure.")


# =============================================================== Problem 2
def _p2() -> None:
    d0 = D.table("p2_returns")
    r = D.table("returns")
    _steps(4)
    _problem_card("The problem in detail: refunds are slow, unclear and treat every customer as a suspect", [
        "Today every return waits for a courier pickup, then travels to the warehouse, waits for a 2–5 day inspection, then "
        "the refund goes back to the card, which can take another 7–14 business days. The customer sees little of this.",
        "Most customers are honest and have a clean history, but they wait as long as the few who abuse returns. A rejected "
        "refund usually comes with no clear reason, so it turns straight into a complaint.",
        f"In the data the median refund takes <b>{r['refund_days'].median():.0f} days</b>, <b>{C.pct(1 - r['qc_passed'].mean())}</b> of "
        "returns fail inspection, and customers who wait longer or are rejected buy again less often.",
    ], [("noon return policy: inspection 2–5 working days, refund can take 7–14 business days", "https://www.noon.com/bahrain-en/return-policy/"),
        ("noon Egypt already refunds to store credit within 24 hours after inspection", "https://helpegypt.noon.com/portal/en/kb/articles/what-is-the-noon-refund-process-6-3-2024")])
    _data_points([
        ("Return id & date", "return_id", "Returns system", "Tracking every case"),
        ("Category & size", "category", "Catalogue", "Return patterns"),
        ("Return reason", "return_reason", "Return form", "Apriori, QC risk"),
        ("Item value (AED)", "item_value_aed", "Order system", "Risk exposure"),
        ("Prior returns (12 months)", "prior_returns_12m", "Customer history", "Trust score"),
        ("Account age (months)", "account_age_months", "Customer profile", "Trust score"),
        ("Photos uploaded", "photos_uploaded", "Return form", "Evidence for QC"),
        ("Refund method", "refund_method", "Payments", "Refund speed"),
        ("Refund days today", "refund_days", "Payments + warehouse", "Baseline"),
        ("QC result", "qc_passed", "Warehouse inspection", "Model target"),
        ("Predicted fail risk", "p_fail", "Random-forest model", "Green / Amber / Red tier"),
    ], d0.iloc[0])
    with cards.card("Step 3 · Descriptive analytics"):
        cards.kpi_grid([
            cards.kpi("Returns in the simulation", C.num(len(d0)), "Test split, never used for training", "synthetic", theme.BLUE),
            cards.kpi("Median refund time today", f"{d0['refund_days'].median():.0f} days", "Passed QC", "synthetic", theme.GOLD),
            cards.kpi("Fail inspection", C.pct(1 - d0["qc_passed"].mean()), "Refund rejected today", "synthetic", theme.PEACH),
            cards.kpi("Average item value", C.aed(d0["item_value_aed"].mean()), "Per return", "synthetic", theme.LAVENDER),
        ])
        c1, c2 = st.columns(2)
        with c1:
            fig, g = charts.count_bar(d0["return_reason"], horizontal=True, colour=theme.LAVENDER, sort=True)
            charts.show(fig, "sim2_d1", g, height=260)
        with c2:
            b = d0.assign(band=pd.cut(d0["p_fail"], [0, .15, .5, 1], labels=["< 0.15", "0.15–0.50", "> 0.50"])).groupby("band", observed=True).agg(
                fail=("qc_passed", lambda x: 1 - x.mean()), n=("return_id", "size")).reset_index()
            charts.show(charts.bar(b["band"].astype(str), b["fail"] * 100, b["n"], b["n"], colour=theme.RED, axis_title="% that really failed QC"),
                        "sim2_d2", b, height=260)
            st.caption("Returns the model scores as low risk almost never fail inspection, which is what makes instant refunds safe.")
    with cards.card("Step 4 · Run the simulation: Trust-Tiered Instant Refunds", "Every return is scored when the customer requests it"):
        st.markdown('<div class="lc-body"><ol>'
                    '<li><b>Request:</b> the random-forest model scores the chance the item fails inspection.</li>'
                    '<li><b>🟢 Green:</b> refund to the noon wallet as soon as the courier collects the item.</li>'
                    '<li><b>🟠 Amber:</b> refund when the warehouse scans the item (within 24 h of arrival).</li>'
                    '<li><b>🔴 Red:</b> full inspection before any refund; rejections come with the photo and the reason.</li>'
                    '<li><b>Tracker:</b> every customer sees the guaranteed refund date.</li></ol></div>', unsafe_allow_html=True)
        a = _assumptions(sim.P2_DEFAULTS, [
            ("green_threshold", "Green if fail risk below", "num", 0.01, 0.5, 0.01, None),
            ("red_threshold", "Red if fail risk above", "num", 0.2, 0.95, 0.05, None),
            ("green_days", "Days to refund for Green", "num", 0.0, 5.0, 0.5, "Courier collection"),
            ("amber_days", "Days to refund for Amber", "num", 0.5, 10.0, 0.5, "Warehouse scan"),
            ("recovery_rate", "Share of bad Green refunds recovered", "num", 0.0, 1.0, 0.05, "From the customer or the seller"),
            ("explain_rejections", "Explain rejections with photos", "bool", None, None, None, "Cuts complaint rate from 78% to 50%"),
        ], "sim2")
        d = sim.run_p2(d0, a)
        _run("sim2", d, sim.p2_kpis, sim.p2_event, a, ("t_ticket_p", "s_ticket_p", "Refund-chase tickets (cumulative, expected)"),
             {"Median refund time (days)", "Refund chase tickets (expected)", "Value paid before a failed check (AED)"}, "order_date")
    if st.session_state.get("sim2_result"):
        k = sim.p2_kpis(d, a)
        extra = k["Bought again within 90 days"][1] - k["Bought again within 90 days"][0]
        with cards.card("Simulation result"):
            c1, c2 = st.columns([1.1, 1])
            with c1:
                g = d.groupby("tier").agg(returns=("return_id", "size"), today=("t_days", "median"), solution=("s_days", "median")).reindex(["Green", "Amber", "Red"]).reset_index()
                fig = charts.grouped_bar(g, "tier", ["today", "solution"], ["Today", "With solution"], y_title="Median refund days",
                                         value_fmt="{:.1f}", colours=[theme.SLATE, theme.GREEN])
                charts.show(fig, "sim2_res", g, height=320)
            with c2:
                _result_table(k, "sim2_table")
            cards.explanation("The same returns and customers under both processes.",
                              f"Median refund time falls from {k['Median refund time (days)'][0]:.0f} to {k['Median refund time (days)'][1]:.1f} days; "
                              f"refund-chase tickets drop by about {k['Refund chase tickets (expected)'][0] - k['Refund chase tickets (expected)'][1]:,.0f}; "
                              f"{extra:,} more customers buy again.",
                              f"Extra repeat buyers are worth about {C.aed(extra * d['clv'].mean())} against {C.aed(k['Value paid before a failed check (AED)'][1])} "
                              "paid out on Green returns that later failed. Move the Green cut-off to trade speed against risk.",
                              "Not every failed check is fraud, so the loss is an upper bound.")


# =============================================================== Problem 3
def _p3() -> None:
    d0 = D.table("p3_tickets")
    t = D.table("tickets")
    _steps(4)
    _problem_card("The problem in detail: complaints get template replies and nobody owns the case", [
        "A customer with a late order or a rejected refund writes in and gets a template reply. Nothing is fixed, so they "
        "write again, often to a different agent who starts from zero. Warranty issues bounce between the seller and noon.",
        "No one is responsible for the case and the customer never sees a deadline. Each extra contact makes it more likely "
        "the customer simply stops buying.",
        f"In the data <b>{C.pct(t['repeat_contact'].mean())}</b> of tickets need another contact, <b>{C.pct(t['template_reply'].mean())}</b> get a template "
        f"reply, and <b>{C.pct(t['churn_90d'].mean())}</b> of customers who complain stop buying within 90 days.",
    ], [("noon replies to ~97–98% of negative Trustpilot reviews, mostly asking customers to email support", "https://jp.trustpilot.com/review/noon.com")])
    _data_points([
        ("Ticket id & date", "ticket_id", "Support system", "Tracking every case"),
        ("Channel", "channel", "Support system", "Routing"),
        ("Ticket text", "text", "Customer message", "TF-IDF topic + anger"),
        ("Topic (read by model)", "predicted_topic", "Text classifier", "Send to the right team"),
        ("Contacts on the same issue", "contacts", "Support system", "Repeat-contact KPI"),
        ("Hours to resolve", "resolution_hours", "Support system", "Deadline model"),
        ("Reopened", "reopened", "Support system", "Quality of the fix"),
        ("Template reply", "template_reply", "Agent tool", "Personal vs template"),
        ("CSAT (1–5)", "csat", "Post-case survey", "Satisfaction"),
        ("Fix applied", "fix_applied", "Agent tool", "Best-fix decision tree"),
        ("Customer value (CLV)", "clv", "CLV model", "Priority of the case"),
    ], d0.iloc[0])
    with cards.card("Step 3 · Descriptive analytics"):
        cards.kpi_grid([
            cards.kpi("Tickets in the simulation", C.num(len(d0)), "Test split, never used for training", "synthetic", theme.BLUE),
            cards.kpi("Repeat contact today", C.pct((d0["contacts"] > 1).mean()), "Needed more than one contact", "synthetic", theme.LAVENDER),
            cards.kpi("Read correctly by the model", C.pct(d0["routed_correctly"].mean()), "Topic from the ticket text", "derived", theme.GREEN),
            cards.kpi("Churn within 90 days", C.pct(d0["churn_90d"].mean()), "Customers who complained", "synthetic", theme.RED),
        ])
        c1, c2 = st.columns(2)
        with c1:
            fig, g = charts.count_bar(d0["topic"], horizontal=True, colour=theme.LAVENDER, sort=True)
            charts.show(fig, "sim3_d1", g, height=260)
        with c2:
            g = d0.assign(c=d0["contacts"].clip(1, 4).astype(str).replace({"4": "4+"})).groupby("c")["churn_90d"].mean().mul(100).reset_index()
            charts.show(charts.bar(g["c"], g["churn_90d"], colour=theme.RED, axis_title="% stopped buying"), "sim3_d2", g, height=260)
            st.caption("Every extra contact on the same issue raises the chance the customer leaves.")
    with cards.card("Step 4 · Run the simulation: the Case Ownership Engine", "Every new ticket goes through the engine"):
        st.markdown('<div class="lc-body"><ol>'
                    '<li><b>Read:</b> the text model reads the topic and anger and sends the ticket to the right team.</li>'
                    '<li><b>Own:</b> one named owner and a visible deadline from the resolution-time model.</li>'
                    '<li><b>Prioritise:</b> high churn risk → senior agent, personal reply, faster deadline.</li>'
                    '<li><b>Fix:</b> the decision tree suggests the fix that stops repeat contacts for that issue.</li>'
                    '<li><b>Second contact:</b> the case moves to a Resolution Desk that can approve the fix on the spot.</li></ol></div>',
                    unsafe_allow_html=True)
        a = _assumptions(sim.P3_DEFAULTS, [
            ("owner_cut", "Repeat contacts removed (correct routing)", "num", 0.0, 1.0, 0.05, None),
            ("misroute_cut", "Repeat contacts removed (mis-read)", "num", 0.0, 1.0, 0.05, None),
            ("high_risk_threshold", "Senior agent if churn risk above", "num", 0.05, 0.9, 0.05, None),
            ("priority_cut", "Time saved for priority cases", "num", 0.0, 0.9, 0.05, None),
            ("suggest_fix", "Suggest the best fix", "bool", None, None, None, None),
            ("fix_success", "Repeat contacts the fix prevents", "num", 0.0, 1.0, 0.05, None),
            ("personal_reply_for_high_risk", "Personal reply for high-risk", "bool", None, None, None, None),
            ("cost_per_contact", "Cost per agent contact (AED)", "num", 0.0, 100.0, 1.0, None),
        ], "sim3")
        d = sim.run_p3(d0, a)
        _run("sim3", d, sim.p3_kpis, sim.p3_event, a, ("t_churn", "s_churn", "Customers lost (cumulative)"),
             {"Needed a repeat contact", "Median hours to resolve", "Reopened", "Agent contacts", "Customers lost (90 days)", "Churn rate"},
             "created_date")
    if st.session_state.get("sim3_result"):
        k = sim.p3_kpis(d, a)
        saved = k["Customers lost (90 days)"][0] - k["Customers lost (90 days)"][1]
        contacts = k["Agent contacts"][0] - k["Agent contacts"][1]
        with cards.card("Simulation result"):
            c1, c2 = st.columns([1.1, 1])
            with c1:
                g = d.groupby("topic").agg(today=("t_churn", "mean"), solution=("s_churn", "mean")).reset_index()
                fig = charts.grouped_bar(g, "topic", ["today", "solution"], ["Today", "With solution"], y_title="Churn rate",
                                         value_fmt="{:.0%}", colours=[theme.SLATE, theme.GREEN], horizontal=True)
                charts.show(fig, "sim3_res", g, height=360)
            with c2:
                _result_table(k, "sim3_table")
            cards.explanation("The same tickets and customers under both processes.",
                              f"Repeat contacts drop from {C.pct(k['Needed a repeat contact'][0], 0)} to {C.pct(k['Needed a repeat contact'][1], 0)}, "
                              f"CSAT rises from {k['Average CSAT (1–5)'][0]:.2f} to {k['Average CSAT (1–5)'][1]:.2f}, and {saved:,} fewer customers leave.",
                              f"That is about {C.aed(saved * d['clv'].mean())} of next-year profit kept plus {C.aed(contacts * a['cost_per_contact'])} "
                              f"of agent time saved ({contacts:,} fewer contacts).",
                              "Effects of ownership and fixes are assumptions taken from the proposal; the real test is the 3-arm A/B on the business-case page.")
