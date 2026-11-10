"""Live platform: a running mini-noon (1,000 products, 3,000 customers) with a real database and live alerts.

Two worlds run from the same seed: today's process and the three solutions. Same customers, same carts — only the
process differs. In the dashboard the platform steps forward on a timer; under Docker a separate worker process does
the stepping and this page only reads the database (NOON_LIVE_WORKER=1).
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from live.engine import START, open_worlds, reset_all, step_all
from noonbi import cards, charts, theme
from noonbi import data as D

esc = cards.esc
WORKER = os.environ.get("NOON_LIVE_WORKER") == "1"
SEV = {"critical": theme.RED, "warning": theme.PEACH, "info": theme.BLUE}


def _folder() -> Path:
    if os.environ.get("NOON_LIVE_DIR"):
        return Path(os.environ["NOON_LIVE_DIR"])
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        sid = get_script_run_ctx().session_id[:12]
    except Exception:  # pragma: no cover
        sid = "local"
    return Path(tempfile.gettempdir()) / "noon_live" / sid  # one private platform per browser session


def _worlds():
    if "live_worlds" not in st.session_state:
        w = open_worlds(_folder(), create=not WORKER)
        if not WORKER and w["today"].t == 0:
            with st.spinner("Starting the platform: loading 1,000 products and 3,000 customers, simulating the first 36 hours…"):
                for _ in range(72):
                    step_all(w, 30, False)
        st.session_state["live_worlds"] = w
    return st.session_state["live_worlds"]


def render(f: D.Filters) -> None:
    cards.page_header("Live platform simulation",
                      "A working mini-noon: 1,000 products, 3,000 customers and 120 sellers. Orders, deliveries, returns and "
                      "complaints are written to a real database as they happen, and the platform raises alerts live. Two "
                      "copies run side by side from the same seed: today's process and the three solutions.",
                      "Live database · alerts", [("Platform events", "synthetic"), ("Rules", "illustrative")])
    w = _worlds()
    ctl = w["today"]
    if WORKER and ctl.meta("t") is None:
        cards.notice("warn", "Waiting for the simulator container.", "Start it with <code>docker compose up</code>; this page refreshes when data arrives.")
        st.button("Check again", key="live_wait")
        return
    c1, c2, c3, c4, c5 = st.columns([1.1, 1.3, 1.2, 1, 1])
    running = c1.toggle("▶ Run live", value=st.session_state.get("live_running", False), key="live_running")
    speed = c2.select_slider("Simulated time per update", [15, 30, 60, 120], value=30, key="live_speed",
                             format_func=lambda m: f"{m} min" if m < 60 else f"{m // 60} h")
    sale = c3.toggle("Yellow Friday surge (×2.4 orders)", value=False, key="live_sale")
    if c4.button("⏩ +1 day", key="live_ff", help="Fast-forward 24 simulated hours", disabled=WORKER):
        with st.spinner("Simulating 24 hours…"):
            for _ in range(48):
                step_all(w, 30, sale)
    if c5.button("↺ Reset", key="live_reset"):
        if WORKER:
            ctl.set_meta("ctl_reset", 1)
            ctl.db.commit()
        else:
            reset_all(w)
            for _ in range(72):
                step_all(w, 30, False)
    if WORKER:  # dashboard only sends controls; the worker container does the stepping
        ctl.set_meta("ctl_running", int(running))
        ctl.set_meta("ctl_speed", speed)
        ctl.set_meta("ctl_sale", int(sale))
        ctl.db.commit()
        st.caption("Docker mode: a separate simulator container is writing to the shared database; this page reads it live.")

    interval = 1.5 if running else None

    @st.fragment(run_every=interval)
    def live_panel():
        if st.session_state.get("live_running") and not WORKER:
            step_all(w, st.session_state.get("live_speed", 30), st.session_state.get("live_sale", False))
        _panel(w)

    live_panel()
    _static(w)


# ------------------------------------------------------------------ live panel (refreshes every 1.5 s when running)
def _panel(w) -> None:
    t, s = w["today"], w["solution"]
    kt, ks = t.kpis(), s.kpis()
    live = st.session_state.get("live_running")
    dot = f'<span style="color:{theme.RED if live else theme.MUTED};font-size:1.1rem">●</span>'
    st.markdown(f'<div class="lc-card" style="padding:.55rem .9rem;margin-bottom:.6rem">{dot} <b>{"LIVE" if live else "PAUSED"}</b> · '
                f'simulated time <b>{s.clock_label()}</b>{" · <b>Yellow Friday surge ON</b>" if st.session_state.get("live_sale") else ""}'
                f' · database: <code>{esc(str(s.path.parent))}</code></div>', unsafe_allow_html=True)

    def pct(x):
        return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x * 100:.1f}%"

    def days(x):
        return "pending" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.1f} d"

    pending_t = int(t.q("SELECT COUNT(*) n FROM returns WHERE status='Requested'")["n"][0])
    pending_s = int(s.q("SELECT COUNT(*) n FROM returns WHERE status='Requested'")["n"][0])
    rows = [
        ("Orders placed", f"{kt.get('orders', 0):,}", f"{ks.get('orders', 0):,}", None),
        ("Sales value (GMV)", f"AED {kt.get('gmv', 0):,.0f}", f"AED {ks.get('gmv', 0):,.0f}", None),
        ("Orders late (due so far)", pct(kt.get("late_rate")), pct(ks.get("late_rate")), "low"),
        ("Open complaint tickets", f"{kt.get('open_tickets', 0):,}", f"{ks.get('open_tickets', 0):,}", "low"),
        ("Tickets with repeat contact", f"{kt.get('repeat_contacts', 0):,}", f"{ks.get('repeat_contacts', 0):,}", "low"),
        ("Refunds paid · avg time", f"{kt.get('refunds_done', 0):,} · {days(kt.get('refund_days'))}",
         f"{ks.get('refunds_done', 0):,} · {days(ks.get('refund_days'))}", None),
        ("Customers still waiting for a refund", f"{pending_t:,}", f"{pending_s:,}", "low"),
        ("Customers at risk of leaving", f"{kt.get('at_risk', 0):,}", f"{ks.get('at_risk', 0):,}", "low"),
    ]
    tiles = []
    for lbl, a, b, _ in rows:
        tiles.append(f'<div class="lc-card lc-kpi" style="border-top:3px solid {theme.BLUE};padding:.55rem .8rem">'
                     f'<div class="lc-kpi-label">{esc(lbl)}</div><div style="display:flex;gap:1rem;margin-top:.15rem">'
                     f'<div><div class="lc-muted" style="font-size:.66rem">TODAY</div><div style="font-weight:700;color:{theme.SLATE}">{a}</div></div>'
                     f'<div><div class="lc-muted" style="font-size:.66rem">WITH SOLUTIONS</div><div style="font-weight:700;color:{theme.GREEN}">{b}</div></div>'
                     f'</div></div>')
    st.markdown(f'<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:.55rem;margin-bottom:.6rem">{"".join(tiles)}</div>',
                unsafe_allow_html=True)

    mt, ms = t.q("SELECT * FROM metrics ORDER BY t"), s.q("SELECT * FROM metrics ORDER BY t")
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        for m, nm, col in [(mt, "Today", theme.SLATE), (ms, "With solutions", theme.GREEN)]:
            fig.add_scatter(x=START + pd.to_timedelta(m["t"], unit="m"), y=m["late_rate"] * 100, name=nm, mode="lines",
                            line=dict(color=col, width=2.5))
        fig.update_layout(title=dict(text="Late-delivery rate (%)", font=dict(size=13)), height=270, margin=dict(l=10, r=10, t=36, b=10),
                          legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, width="stretch", key="live_late", config={"displayModeBar": False})
    with c2:
        fig = go.Figure()
        for m, nm, col in [(mt, "Today", theme.SLATE), (ms, "With solutions", theme.GREEN)]:
            fig.add_scatter(x=START + pd.to_timedelta(m["t"], unit="m"), y=m["open_tickets"], name=nm, mode="lines",
                            line=dict(color=col, width=2.5))
        fig.update_layout(title=dict(text="Open complaint tickets", font=dict(size=13)), height=270, margin=dict(l=10, r=10, t=36, b=10),
                          legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, width="stretch", key="live_tickets", config={"displayModeBar": False})

    c1, c2 = st.columns([1, 1.35])
    with c1:
        now = s.t
        b = s.q("SELECT category, COUNT(*) orders, SUM(value) value FROM orders WHERE t > ? GROUP BY category ORDER BY value", (now - 1440,))
        fig = go.Figure(go.Bar(y=b["category"], x=b["value"], orientation="h", marker_color=theme.BLUE,
                               text=[f"AED {v / 1000:,.0f}k · {n} orders" for v, n in zip(b["value"], b["orders"])], textposition="auto"))
        fig.update_layout(title=dict(text="Category buckets · sales value, last 24 h", font=dict(size=13)), height=300,
                          margin=dict(l=10, r=10, t=36, b=10), xaxis_title="AED")
        st.plotly_chart(fig, width="stretch", key="live_buckets", config={"displayModeBar": False})
    with c2:
        o = s.q("SELECT order_id, t, zone, n_items, value, category, risk, promise_t, status, late, warned, priority, credit "
                "FROM orders ORDER BY t DESC LIMIT 9")
        o["time"] = (START + pd.to_timedelta(o["t"], unit="m")).dt.strftime("%a %H:%M")
        o["promise"] = (START + pd.to_timedelta(o["promise_t"], unit="m")).dt.strftime("%a %d")
        o["flags"] = np.where(o["priority"] == 1, "🚚 priority ", "") + np.where(o["warned"] == 1, "💬 warned ", "") + \
            np.where(o["credit"] > 0, "🎁 credit", "")
        st.markdown('<div class="lc-kpi-label" style="margin:.2rem 0">LIVE ORDERS (with solutions)</div>', unsafe_allow_html=True)
        st.dataframe(o[["order_id", "time", "zone", "category", "n_items", "value", "risk", "promise", "status", "flags"]],
                     hide_index=True, width="stretch", height=300,
                     column_config={"value": st.column_config.NumberColumn("AED", format="%.0f"),
                                    "risk": st.column_config.ProgressColumn("Late risk", min_value=0, max_value=1, format="%.2f"),
                                    "n_items": st.column_config.NumberColumn("Items")})

    c1, c2 = st.columns(2)
    for col, world, title in [(c1, t, "Alerts today — operations finds out from customers"),
                              (c2, s, "Alerts with the solutions — the platform acts first")]:
        with col:
            a = world.q("SELECT t, severity, kind, entity, message FROM alerts ORDER BY alert_id DESC LIMIT 9")
            items = "".join(
                f'<div style="display:flex;gap:.5rem;padding:.28rem 0;border-bottom:1px solid #EEF1F5;font-size:.8rem">'
                f'<span style="min-width:4.2rem;color:{theme.MUTED}">{(START + pd.Timedelta(minutes=r.t)):%a %H:%M}</span>'
                f'<span style="min-width:.6rem;color:{SEV.get(r.severity, theme.BLUE)}">●</span>'
                f'<span><b>{esc(r.kind)}</b> · {esc(r.entity)} — {esc(r.message)}</span></div>' for r in a.itertuples())
            st.markdown(f'<div class="lc-card" style="padding:.6rem .9rem;min-height:300px"><div class="lc-kpi-label">{esc(title.upper())}</div>'
                        f'{items or "<i>No alerts yet</i>"}</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------ static sections (refresh on interaction)
def _static(w) -> None:
    t, s = w["today"], w["solution"]
    with cards.card("What each alert means and which solution raises it"):
        st.markdown(cards.html_table([
            {"Alert": "<b>High-risk order</b>", "Raised when": "Late-order risk ≥ 65% at checkout", "Action": "Priority courier", "Solution": "1"},
            {"Alert": "<b>Pin requested</b>", "Raised when": "Partial address in a risky zone", "Action": "WhatsApp location pin", "Solution": "1"},
            {"Alert": "<b>GPS lock</b>", "Raised when": "'Unavailable' marked away from the address", "Action": "Blocked, re-attempt", "Solution": "1"},
            {"Alert": "<b>Early warning</b>", "Raised when": "Order will miss its date", "Action": "Customer told before chasing", "Solution": "1"},
            {"Alert": "<b>Recovery credit</b>", "Raised when": "Late + persuadable customer", "Action": "AED 15 credit", "Solution": "1"},
            {"Alert": "<b>Red-tier return / Refund risk</b>", "Raised when": "High fail risk / Green refund later failed", "Action": "Full check / recovery", "Solution": "2"},
            {"Alert": "<b>Case owner / Resolution desk / SLA breach</b>", "Raised when": "New ticket / 2nd contact / past deadline", "Action": "Owner, fix authority, escalation", "Solution": "3"},
            {"Alert": "<b>Stock, Zone spike, Slow seller</b>", "Raised when": "10 left or sold out / zone ≥35% late / seller ≥50% late", "Action": "Restock / capacity / badge", "Solution": "Ops (both)"},
        ]), unsafe_allow_html=True)
    c1, c2 = st.columns([1.2, 1])
    with c1, cards.card("Product catalogue (1,000 products) — best sellers and stock"):
        p = s.q("SELECT sku, name, category, price, stock, start_stock, sold, seller_id FROM products ORDER BY sold DESC LIMIT 200")
        st.dataframe(p, hide_index=True, width="stretch", height=330,
                     column_config={"price": st.column_config.NumberColumn("Price (AED)", format="%.0f"),
                                    "stock": st.column_config.ProgressColumn("Stock", min_value=0, max_value=int(p["start_stock"].max() or 1), format="%d")})
    with c2, cards.card("Live database"):
        counts = []
        for tbl in ("products", "customers", "sellers", "orders", "order_items", "returns", "tickets", "alerts", "metrics"):
            counts.append({"table": tbl, "today": int(t.q(f"SELECT COUNT(*) n FROM {tbl}")["n"][0]),
                           "with_solutions": int(s.q(f"SELECT COUNT(*) n FROM {tbl}")["n"][0])})
        st.dataframe(pd.DataFrame(counts), hide_index=True, width="stretch")
        sql = st.text_area("Run a read-only SQL query (solution world)", "SELECT kind, COUNT(*) AS alerts FROM alerts GROUP BY kind ORDER BY alerts DESC",
                           height=80, key="live_sql")
        if st.button("Run query", key="live_sql_run"):
            q = sql.strip().rstrip(";")
            if not q.lower().startswith("select") or ";" in q:
                cards.notice("warn", "Only one SELECT query is allowed.", "")
            else:
                try:
                    st.dataframe(s.q(q).head(500), hide_index=True, width="stretch")
                except Exception as e:  # show SQL errors to the user
                    cards.notice("warn", "Query error:", esc(str(e)))
    cards.explanation(
        "Every order, delivery, return, ticket and alert is a row in a real database, created as simulated time passes. Both "
        "copies get exactly the same customers and carts; only the process differs.",
        "With the solutions, risky orders are caught at checkout, customers are told before an order is late, low-risk "
        "refunds arrive in a day, and every complaint has an owner. Today, the same events only show up when customers chase.",
        "This is what the operations team would watch: the alerts are the 'actions' of the three solutions, and the KPI tiles "
        "are the business case moving in real time.",
        "Event rules are simplified versions of the models on the other pages. Under Docker, a separate simulator container "
        "writes the database and this page only reads it.")
