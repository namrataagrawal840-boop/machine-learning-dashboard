"""Page 5 — Problem 1: delayed and unreliable deliveries → Delivery Promise & Recovery Engine."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from noonbi import cards, charts, theme
from noonbi import data as D
from noonbi.views import common as C

esc = cards.esc


def render(f: D.Filters) -> None:
    cards.page_header("Problem 1 · Delayed and unreliable deliveries",
                      "Solution: the Delivery Promise & Recovery Engine — promise a date noon can keep, warn the customer "
                      "before they chase, and give a credit only to customers it will win back.",
                      "Classification · regression · clustering · association · uplift · A/B",
                      [("Orders", "synthetic"), ("Models", "derived")])
    o = f.apply(D.table("orders"), "order_date")
    if o.empty:
        cards.notice("warn", "No orders match the filters.")
        return
    cmp = D.table("p1_promise_compare").set_index("promise")
    m = D.table("p1_cls_metrics")
    best = m.sort_values("roc_auc").iloc[-1]
    ab = D.table("ab_tests").iloc[0]
    cards.kpi_grid([
        cards.kpi("Late today", C.pct(o["late"].mean()), "Delivered after the date shown at checkout", "synthetic", theme.PEACH),
        cards.kpi("Promise kept with the P90 date", C.pct(cmp.loc["P90 model promise", "kept_rate"]),
                  f"vs {C.pct(cmp.loc['Today: fixed promise', 'kept_rate'])} today (test months Aug–Oct)", "derived", theme.GREEN),
        cards.kpi("Late-order model ROC-AUC", f"{best['roc_auc']:.2f}", f"{best['model']}; recall {C.pct(best['recall'], 0)}",
                  "derived", theme.BLUE),
        cards.kpi("Recovery test uplift", f"+{ab['uplift_pts']:.1f} pts", f"repeat purchase, p = {ab['p_value']:.3f}", "derived", theme.GREEN),
    ])
    st.caption("▶ To watch this solution run record by record, open **8 · Solution Simulator** in the sidebar.")
    tabs = st.tabs(["1 · Evidence", "2 · Root causes", "3 · Predict each order", "4 · Recover & test", "5 · Plan capacity"])
    with tabs[0]:
        _evidence(o)
    with tabs[1]:
        _root_causes(o)
    with tabs[2]:
        _predict()
    with tabs[3]:
        _recover(ab)
    with tabs[4]:
        _demand()


def _rate(o, col, order=None):
    g = o.groupby(col).agg(orders=("order_id", "size"), late=("late", "sum")).reset_index()
    g["late_rate"] = g["late"] / g["orders"] * 100
    if order:
        g[col] = pd.Categorical(g[col], order, ordered=True)
        g = g.sort_values(col)
    else:
        g = g.sort_values("late_rate", ascending=False)
    return g


def _evidence(o: pd.DataFrame) -> None:
    with cards.card("Late-delivery rate by month", "Is lateness constant, or does it spike?"):
        mth = o.groupby("month").agg(orders=("order_id", "size"), late=("late", "mean"), sale=("on_sale", "mean")).reset_index()
        fig = go.Figure()
        fig.add_bar(x=mth["month"], y=mth["orders"], name="Orders", marker_color=theme.SOFT_BLUE, yaxis="y2",
                    hovertemplate="%{x}<br>Orders %{y:,}<extra></extra>")
        fig.add_scatter(x=mth["month"], y=mth["late"] * 100, name="Late rate %", mode="lines+markers",
                        line=dict(color=theme.PEACH, width=3), hovertemplate="%{x}<br>Late %{y:.1f}%<extra></extra>")
        fig.update_layout(yaxis=dict(title="Late rate (%)", rangemode="tozero"),
                          yaxis2=dict(title="Orders", overlaying="y", side="right", showgrid=False))
        charts.show(fig, "p1_month", mth)
        cards.explanation("Share of orders that arrived after the promised date, month by month, with order volume behind it.",
                          "Lateness jumps in November (Yellow Friday) and stays higher in sale periods, when couriers are overloaded.",
                          "A single fixed promise ('tomorrow') cannot be right in both quiet and peak weeks; the promise has "
                          "to change with the conditions.", "Synthetic data; sale dates are approximate.")
    c1, c2 = st.columns(2)
    factors = [("fulfilment", "Who ships it"), ("courier", "Courier"), ("item_size", "Item size"),
               ("address_quality", "Address quality"), ("time_slot", "Order time"), ("zone_type", "Zone type"),
               ("sale_event", "Sale event"), ("emirate", "Emirate")]
    with c1, cards.card("What makes an order late?", "Pick a factor to compare late rates."):
        col = st.selectbox("Factor", [a for a, _ in factors], format_func=dict(factors).get, key="p1_factor")
        g = _rate(o, col)
        fig = charts.bar(g[col], g["late_rate"], g["late"], g["orders"], horizontal=True, colour=theme.PEACH,
                         axis_title="% of orders late", hover_label="Late %")
        charts.show(fig, "p1_factor_chart", g)
    with c2, cards.card("Where are late orders concentrated?"):
        z = D.table("p1_zones")
        basemap = st.segmented_control("View", ["Map", "Simple plot (offline)"], default="Map", key="p1_basemap")
        fig = go.Figure(go.Scattergeo(lat=z["lat"], lon=z["lon"], mode="markers",
                                      marker=dict(size=np.sqrt(z["orders"]) * 0.75, color=z["late_rate"] * 100,
                                                  colorscale=[[0, theme.SOFT_SAGE], [0.5, theme.GOLD], [1, theme.RED]],
                                                  showscale=True, colorbar=dict(title="Late %", thickness=10),
                                                  line=dict(width=0.5, color=theme.WHITE)),
                                      text=z["zone"] + " · " + z["zone_group"],
                                      hovertemplate="%{text}<br>Late %{marker.color:.1f}%<extra></extra>"))
        fig.update_geos(projection_type="mercator", lataxis_range=[23.9, 26.2], lonaxis_range=[53.9, 56.7],
                        showland=True, landcolor="#F4F6F9", showocean=True, oceancolor=theme.SOFT_BLUE,
                        showcountries=True, countrycolor=theme.BORDER, coastlinecolor="#C9D3DE", resolution=50,
                        showframe=False)
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
        if basemap == "Simple plot (offline)":
            fig = go.Figure(go.Scatter(x=z["lon"], y=z["lat"], mode="markers+text", text=z["zone"], textposition="top center",
                                       textfont=dict(size=9, color=theme.MUTED),
                                       marker=dict(size=np.sqrt(z["orders"]) * 0.75, color=z["late_rate"] * 100,
                                                   colorscale=[[0, theme.SOFT_SAGE], [0.5, theme.GOLD], [1, theme.RED]],
                                                   showscale=True, colorbar=dict(title="Late %", thickness=10)),
                                       hovertemplate="%{text}<br>Late %{marker.color:.1f}%<extra></extra>"))
            fig.update_layout(xaxis_title="Longitude", yaxis_title="Latitude", yaxis_scaleanchor="x")
        charts.show(fig, "p1_map", z[["zone", "emirate", "zone_type", "orders", "late_rate", "zone_group", "lat", "lon"]], height=410)
        st.caption("Bubble size = orders, colour = late rate. Zone centres are approximate.")
    with cards.card("Silence makes it worse", "Do customers get told when an order is late, and does it matter?"):
        late = o[o["late"] == 1]
        g = late.groupby("proactive_update_sent").agg(orders=("order_id", "size"), rep=("repurchase_60d", "mean")).reset_index()
        g["label"] = g["proactive_update_sent"].map({0: "No update", 1: "Got an early update"})
        allr = o.groupby("late")["repurchase_60d"].mean()
        c1, c2 = st.columns([1, 1.3])
        c1.markdown(cards.kpi("Late orders with an early update", C.pct(late["proactive_update_sent"].mean()),
                              f"{C.num(late['proactive_update_sent'].sum())} of {C.num(len(late))}", "synthetic", theme.PEACH)
                    + "<div style='height:.6rem'></div>"
                    + cards.kpi("Bought again within 60 days", f"{C.pct(allr.get(0, np.nan), 0)} on time · {C.pct(allr.get(1, np.nan), 0)} late",
                                "The cost of a late order in repeat business", "synthetic", theme.GREEN), unsafe_allow_html=True)
        with c2:
            fig = charts.bar(g["label"], g["rep"] * 100, g["orders"], g["orders"], colour=theme.GREEN,
                             axis_title="% bought again within 60 days", hover_label="Repeat %")
            charts.show(fig, "p1_update", g, height=280)
        cards.explanation("Repeat purchase for late orders, split by whether noon warned the customer first.",
                          "Only about one late order in eight gets a proactive update. Those customers come back more often.",
                          "Telling the customer first is cheap and protects repeat business; it is step one of the solution.",
                          "Observational comparison; the randomised test on tab 4 is the proper proof.")


def _root_causes(o: pd.DataFrame) -> None:
    sp, sk = D.table("p1_seller_profile"), D.table("p1_seller_k")
    z, zk, rules = D.table("p1_zones"), D.table("p1_zone_k"), D.table("p1_rules")
    c1, c2 = st.columns([1.3, 1])
    with c1, cards.card("Seller tiers (K-Means)", "Are a few slow sellers behind many delays?"):
        sel = D.table("p1_sellers")
        fig = go.Figure()
        for i, tier in enumerate(["Good", "Average", "Poor"]):
            d = sel[sel["seller_tier"] == tier]
            fig.add_scatter(x=d["avg_handover_h"], y=d["late_rate"] * 100, mode="markers", name=f"{tier} ({len(d)})",
                            marker=dict(size=np.sqrt(d["orders"]) * 1.4, color=[theme.GREEN, theme.GOLD, theme.RED][i], opacity=0.7),
                            text=d["seller_id"], hovertemplate="%{text}<br>Handover %{x:.1f} h<br>Late %{y:.0f}%<extra></extra>")
        fig.update_layout(xaxis_title="Average hours to hand the parcel to the courier", yaxis_title="Late rate (%)")
        charts.show(fig, "p1_sellers", sel)
    with c2, cards.card("Tier summary"):
        charts.table(sp, "p1_sp", "seller_tiers.csv", column_config={
            "avg_handover_h": st.column_config.NumberColumn("Handover (h)", format="%.1f"),
            "late_rate": st.column_config.ProgressColumn("Late rate", min_value=0, max_value=1, format="%.2f"),
            "return_rate": st.column_config.NumberColumn("Returns/order", format="%.2f")})
        C.k_chart(sk, 3, "p1_sk")
    cards.explanation("K-Means groups sellers by how long they take to hand parcels over, their late rate and their return rate. "
                      "No labels are given; the algorithm finds the groups.",
                      f"About {C.pct((sp.set_index('seller_tier').loc['Poor', 'sellers']) / sp['sellers'].sum(), 0)} of sellers form a 'Poor' "
                      "group with handover times several times longer than the rest and most of their orders late.",
                      "Poor sellers lose the 'express' badge and get a longer promised date until they improve, so noon stops "
                      "promising what these sellers cannot deliver.", "Three groups were kept because they are easy to act on.")
    c1, c2 = st.columns([1, 1.25])
    with c1, cards.card("Problem zones (K-Means)"):
        zz = z.sort_values("late_rate", ascending=False)
        fig = charts.bar(zz["zone"] + " (" + zz["zone_group"] + ")", zz["late_rate"] * 100, zz["orders"], zz["orders"],
                         horizontal=True, colour=theme.PEACH, axis_title="% late", hover_label="Late %")
        charts.show(fig, "p1_zones_bar", zz)
    with c2, cards.card("Which conditions happen together before a delay? (Apriori)"):
        C.rules_table(rules, "p1_rules", 10)
        C.rules_chart(rules, "p1_rules_chart", 15)
    cards.explanation("Apriori looks for combinations of conditions that appear together with a late order. Lift above 1 "
                      "means the link is stronger than chance; lift 2.7 means 2.7× more likely than an average order.",
                      "The strongest patterns combine a poor seller who ships directly with a 3PL courier, an evening order, "
                      "a bulky item or Yellow Friday. Single factors matter less than their combinations.",
                      "Operations can fix the causes: move poor sellers to Fulfilled-by-noon, add courier capacity on sale "
                      "days, and avoid giving bulky items to 3PLs in villa zones.",
                      "Association shows what happens together, not proof of cause; the A/B test checks cause.")


def _predict() -> None:
    m, roc, imp, cm = D.table("p1_cls_metrics"), D.table("p1_roc"), D.table("p1_importance"), D.table("p1_confusion")
    tiers, reg, cmp, bye = D.table("p1_risk_tiers"), D.table("p1_reg_metrics"), D.table("p1_promise_compare"), D.table("p1_promise_by_emirate")
    ps, am, ac, abt, un = D.table("p1_pred_sample"), D.table("p1_address_metrics"), D.table("p1_address_capture"), D.table("p1_address_by_type"), D.table("p1_unavailable")
    cards.notice("info", "How the models were tested.",
                 "Trained on orders from Nov 2025 – Jul 2026 and tested on Aug – Oct 2026, which the models never saw. "
                 "Seller history uses only the training months, so nothing from the future leaks in.")
    with cards.card("Will this order be late? (classification)"):
        c1, c2 = st.columns([1.1, 1])
        with c1:
            C.metrics_table(m, "p1_m")
            C.roc_chart(roc, m, "p1_roc")
        with c2:
            C.importance_chart(imp, "p1_imp", colour=theme.PEACH)
            C.confusion_chart(cm, key="p1_cm")
        cards.explanation("Three classifiers predict late / on time. Recall = share of truly late orders the model catches; "
                          "precision = share of flagged orders that really are late.",
                          f"Boosted trees and random forest beat the simple logistic regression slightly (ROC-AUC ≈ {m['roc_auc'].max():.2f}). "
                          "The biggest drivers are the seller's handover history, who ships the item, and the courier.",
                          "Flagged orders get action before they are late. The threshold is set at 0.4 rather than 0.5 to catch "
                          "more late orders, because missing one costs more than a false alarm.",
                          "Accuracy on real data will differ; the method and the comparison stay the same.")
    with cards.card("Acting by risk level"):
        charts.table(tiers, "p1_tiers", "risk_tiers.csv", column_config={
            "actual_late_rate": st.column_config.ProgressColumn("Actually late", min_value=0, max_value=1, format="%.2f"),
            "share_of_orders": st.column_config.ProgressColumn("Share of orders", min_value=0, max_value=1, format="%.2f"),
            "share_of_all_late": st.column_config.ProgressColumn("Share of all late orders", min_value=0, max_value=1, format="%.2f"),
            "avg_p_late": st.column_config.NumberColumn("Avg predicted risk", format="%.2f"),
            "action": st.column_config.TextColumn("Action", width="large")})
        st.caption("High-risk orders are about a fifth of all orders but hold more than half of the late ones, so effort goes where it pays.")
    c1, c2 = st.columns([1.1, 1])
    with c1, cards.card("An honest delivery date (regression, P90)", "How long will this order really take?"):
        d = ps.sort_values("actual_hours").reset_index(drop=True)
        fig = go.Figure()
        fig.add_scattergl(x=d["pred_hours"] / 24, y=d["actual_hours"] / 24, mode="markers", name="Orders",
                          marker=dict(size=5, opacity=0.35, color=theme.BLUE), hovertemplate="Predicted %{x:.1f} d<br>Actual %{y:.1f} d<extra></extra>")
        mx = float(max(d["pred_hours"].max(), d["actual_hours"].max()) / 24)
        fig.add_scatter(x=[0, mx], y=[0, mx], mode="lines", name="Perfect", line=dict(dash="dot", color=theme.SLATE))
        fig.update_layout(xaxis_title="Predicted days (average)", yaxis_title="Actual days")
        charts.show(fig, "p1_reg", d, height=340)
        r0 = reg.iloc[0]
        st.caption(f"Average error {r0['mae_hours']:.1f} hours ({r0['mae_days']:.2f} days), R² {r0['r2']:.2f}. "
                   f"The P90 model gives a time by which ~90% of similar orders arrive (actual coverage {C.pct(reg.iloc[1]['coverage'], 0)}).")
    with c2, cards.card("Promise kept: today vs P90 date"):
        fig = charts.grouped_bar(bye, "emirate", ["today_kept", "p90_kept"], ["Today", "P90 promise"], y_title="Promise kept (share)",
                                 value_fmt="{:.0%}", colours=[theme.SLATE, theme.GREEN])
        charts.show(fig, "p1_kept", bye, height=340)
    t0, t1 = cmp.iloc[0], cmp.iloc[1]
    cards.explanation("A regressor predicts delivery hours; a second, quantile model predicts the 90th percentile (P90): the time "
                      "by which 9 out of 10 similar orders arrive.",
                      f"Today noon keeps its promise {C.pct(t0['kept_rate'], 0)} of the time. Promising the P90 date would keep it "
                      f"{C.pct(t1['kept_rate'], 0)} of the time, at the cost of a longer promise on average "
                      f"({t0['avg_promised_days']:.1f} → {t1['avg_promised_days']:.1f} days).",
                      "A customer told 'Tue–Wed' who gets it on Tuesday is happy; one told 'tomorrow' who gets it on Wednesday "
                      "writes a 1-star review. The A/B test would check whether the longer date costs any sales.",
                      "If you promise the average time you are late about half the time; that is why P90 is used.")
    with cards.card("Addresses drivers can't find (logistic regression + GPS rule)"):
        c1, c2, c3 = st.columns(3)
        c1.markdown(cards.kpi("Failed first attempts", C.pct(am.iloc[0]["base_rate"]), "Driver could not deliver first time",
                              "synthetic", theme.PEACH), unsafe_allow_html=True)
        c2.markdown(cards.kpi("Caught by asking 20% for a pin", C.pct(ac.iloc[0]["failed_caught"], 0),
                              f"Model ROC-AUC {am.iloc[0]['roc_auc']:.2f}", "derived", theme.BLUE), unsafe_allow_html=True)
        u = un.iloc[0]
        c3.markdown(cards.kpi("'Customer unavailable' with no GPS proof", C.pct(u["no_gps_proof"] / u["failed_attempts"], 0),
                              f"{C.num(u['no_gps_proof'])} marks the GPS rule would block", "synthetic", theme.RED), unsafe_allow_html=True)
        piv = abt.pivot(index="zone_type", columns="address_quality", values="failed_rate")
        fig = charts.heatmap(piv * 100, text_fmt="{:.0f}%", value_name="Failed %", colorbar_title="Failed %")
        charts.show(fig, "p1_addr", abt, height=280)
        st.caption("Partial addresses in villa communities fail most. Those customers get a WhatsApp request for a location pin "
                   "before dispatch. Drivers can only mark 'unavailable' if GPS shows them at the address and they called through the app.")


def _recover(ab: pd.Series) -> None:
    up = D.table("p1_uplift_groups")
    with cards.card("A/B test: early message + credit for likely-late orders",
                    "Did customers who got the new treatment buy again more often?"):
        C.ab_card(ab, "p1_ab")
    with cards.card("Who should get the credit? (uplift modelling, two-model approach)"):
        fig = go.Figure()
        fig.add_bar(x=up["decile"], y=up["observed_uplift_pts"],
                    marker_color=[theme.GREEN if v > 0 else theme.RED for v in up["observed_uplift_pts"]],
                    text=[f"{v:+.1f}" for v in up["observed_uplift_pts"]], textposition="outside", cliponaxis=False)
        fig.update_layout(xaxis_title="Customers ranked by predicted uplift", yaxis_title="Observed extra repeat buying (pts)")
        charts.show(fig, "p1_uplift", up, height=320)
        top = up.iloc[:2]
        cards.explanation("One model predicts repeat buying <i>with</i> a credit, another <i>without</i>; the difference is each "
                          "customer's uplift. Customers are ranked by it and the real test results are checked in each group.",
                          f"The top 40% gain about {top['observed_uplift_pts'].mean():.0f} points from the credit. The bottom group "
                          "gains nothing or less (they would come back anyway, or will not come back at all).",
                          "Give credits only to the top groups: most of the benefit for well under half the cost.",
                          "Uplift is estimated from one 3-month test; it should be re-checked every few months.")


def _demand() -> None:
    d, ev = D.table("p1_demand"), D.table("p1_demand_events")
    sa, grid, acf = D.table("p1_sarima"), D.table("p1_sarima_grid"), D.table("p1_sarima_acf")
    cmp = D.table("p1_forecast_compare")
    S = D.summary().get("demand", {})
    with cards.card("Daily order forecast (Blog 6): Seasonal ARIMA vs time-series regression",
                    "How many orders, couriers and agents will we need, and when?"):
        view = st.segmented_control("Model", ["Seasonal ARIMA", "Regression", "Both"], default="Seasonal ARIMA", key="p1_fc_model") \
            or "Seasonal ARIMA"
        fig = go.Figure()
        fig.add_scatter(x=d["date"], y=d["orders"], name="Actual", mode="lines", line=dict(color=theme.TEXT, width=1.4))
        series = []
        if view in ("Seasonal ARIMA", "Both"):
            series.append((sa, "Seasonal ARIMA", theme.LAVENDER, "rgba(155,140,200,0.18)"))
        if view in ("Regression", "Both"):
            series.append((d, "Regression", theme.GREEN, "rgba(120,164,141,0.18)"))
        for df, name, col, band in series:
            ahead = df[df["part"] != "Train"]
            fig.add_scatter(x=pd.concat([ahead["date"], ahead["date"][::-1]]), y=pd.concat([ahead["high"], ahead["low"][::-1]]),
                            fill="toself", fillcolor=band, line=dict(width=0), name=f"{name} · 80% range", hoverinfo="skip")
            fig.add_scatter(x=df.loc[df["part"] == "Train", "date"], y=df.loc[df["part"] == "Train", "fitted"], name=f"{name} · fit",
                            mode="lines", line=dict(color=col, width=1.2, dash="dot"), opacity=0.8)
            fig.add_scatter(x=ahead["date"], y=ahead["fitted"], name=f"{name} · forecast", mode="lines", line=dict(color=col, width=2.6))
        fig.add_vline(x=pd.Timestamp("2026-08-01"), line_dash="dot", line_color=theme.MUTED)
        fig.add_vline(x=pd.Timestamp("2026-11-01"), line_dash="dot", line_color=theme.MUTED)
        fig.update_layout(yaxis_title="Orders per day (all regions)", legend=dict(orientation="h", y=-0.18),
                          xaxis_range=[pd.Timestamp("2025-10-01"), pd.Timestamp("2026-12-16")])
        charts.show(fig, "p1_demand", sa if view == "Seasonal ARIMA" else d, height=420)
        st.caption("Zoomed to the last 14 months; drag or double-click the chart to see all two years. Left of the first dotted line: training data (Nov 2024 – Jul 2026). Between the lines: the test months, forecast 92 days ahead "
                   "without seeing them. Right of the second line: the forecast for the next 45 days, including Yellow Friday.")
        cards.kpi_grid([
            cards.kpi("Seasonal ARIMA error (Aug–Oct)", C.pct(S.get("sarima_mape_test", np.nan)), f"Model {S.get('sarima_order', '')}",
                      "derived", theme.LAVENDER),
            cards.kpi("Regression error (Aug–Oct)", C.pct(S.get("mape_test", np.nan)), "Trend + weekday + yearly waves + events",
                      "derived", theme.GREEN),
            cards.kpi("Benchmark error", C.pct(S.get("naive_mape_test", np.nan)), "'Same weekday last week'", "derived", theme.SLATE),
            cards.kpi("Forecast peak", f"{S.get('sarima_peak_forecast', 0):,.0f} orders",
                      f"on {pd.Timestamp(S.get('sarima_peak_date', '2026-11-27')):%a %d %b} (Yellow Friday week)", "derived", theme.GOLD),
        ])
    c1, c2 = st.columns([1.2, 1])
    with c1, cards.card("What Seasonal ARIMA does, in simple words"):
        st.markdown(
            '<div class="lc-body">'
            '<p><b>ARIMA</b> predicts tomorrow from the series itself: recent values (<b>AR</b>, auto-regressive), the change from day '
            'to day (<b>I</b>, integrated / differencing) and its own recent mistakes (<b>MA</b>, moving average).</p>'
            '<p><b>Seasonal</b> adds the same idea one week back, because Thursdays look like last Thursday. The <b>X</b> in SARIMAX '
            'means extra inputs: the yearly shape and the sale-event flags (Ramadan, Eid, DSF, Yellow Friday, 11.11).</p>'
            f'<p><b>Chosen model: {esc(S.get("sarima_order", ""))}.</b> Read it as: compare each day with the day before and with the '
            'same weekday last week, then correct with the last day\'s and last week\'s error. This is the classic "airline model" '
            'and it had the lowest AIC of the six models tried.</p></div>', unsafe_allow_html=True)
        charts.table(grid, "p1_sarima_grid", "sarima_model_search.csv", column_config={
            "aic": st.column_config.NumberColumn("AIC (lower = better)", format="%.0f"),
            "chosen": st.column_config.CheckboxColumn("Chosen")})
    with c2, cards.card("Is anything left to learn? (residual check)"):
        fig = go.Figure()
        fig.add_bar(x=acf["lag"], y=acf["acf"], marker_color=[theme.RED if abs(v) > b else theme.LAVENDER for v, b in zip(acf["acf"], acf["bound"])],
                    hovertemplate="Lag %{x} days<br>Correlation %{y:.3f}<extra></extra>")
        b0 = float(acf["bound"].iloc[0])
        fig.add_hline(y=b0, line_dash="dot", line_color=theme.MUTED)
        fig.add_hline(y=-b0, line_dash="dot", line_color=theme.MUTED)
        fig.update_layout(xaxis_title="Lag (days)", yaxis_title="Correlation of leftover errors", showlegend=False)
        charts.show(fig, "p1_sarima_acf", acf, height=280)
        lb = S.get("sarima_ljungbox_p", np.nan)
        st.markdown(cards.kpi("Ljung-Box test (14 days)", f"p = {lb:.2f}",
                              "Above 0.05 → leftover errors look like random noise" if lb > 0.05 else "Below 0.05 → some pattern is left",
                              "derived", theme.GREEN if lb > 0.05 else theme.PEACH), unsafe_allow_html=True)
    with cards.card("Model comparison and sale-day effects"):
        c1, c2 = st.columns([1.2, 1])
        with c1:
            charts.table(cmp, "p1_fc_compare", "forecast_comparison.csv", column_config={
                "test_mape": st.column_config.NumberColumn("Error on Aug–Oct (MAPE)", format="%.3f"),
                "how_it_works": st.column_config.TextColumn("How it works", width="large")})
        with c2:
            fig = charts.bar(ev["event"], ev["uplift_pct"], horizontal=True, colour=theme.GOLD, axis_title="Extra orders vs a normal day (%)",
                             value_fmt="{:.0f}%", hover_label="Extra %")
            charts.show(fig, "p1_events", ev, height=260)
        cards.explanation("Two forecasting methods, both trained to July and tested on August–October without seeing it, plus a "
                          "simple benchmark. Error = average % gap between forecast and actual orders (MAPE).",
                          f"Both models are about {C.pct(S.get('sarima_mape_test', np.nan), 1)} off on unseen months, about four times better "
                          "than the benchmark. Both see Yellow Friday more than doubling daily orders.",
                          "Use Seasonal ARIMA for the next few weeks (it reacts to the latest days) and check it against the regression for "
                          "peaks. Courier and agent capacity can be booked weeks before the surge, when most late orders happen.",
                          "Synthetic series. A 365-day seasonal lag is too long for two years of daily data, so the yearly shape enters as "
                          "inputs. The 80% range widens the further ahead it looks.")
