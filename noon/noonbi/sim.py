"""Fixed-number solution simulator.

Each function replays the same records twice — today's process and the solution — and returns one row per record
with both outcomes. Behaviour formulas match analysis/synthetic.py. Assumptions are passed in so the user can see
and change them; with the defaults the result is always the same (fixed numbers).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _sig(x):
    return 1 / (1 + np.exp(-x))


def _frac(u, k):  # a second, independent-looking uniform derived from u (keeps runs fixed)
    return (u * k) % 1.0


# ------------------------------------------------------------------ Problem 1
P1_DEFAULTS = dict(use_p90=True, priority_hours=6.0, warn_threshold=0.25, credit_share=0.40, credit_aed=15,
                   pin_request=True, pin_fix_rate=0.5, gps_lock=True)


def run_p1(df: pd.DataFrame, a: dict) -> pd.DataFrame:
    d = df.copy()
    e = d["engagement"].to_numpy()
    # today
    d["t_late"] = d["late"]
    d["t_days_late"] = d["days_late"]
    d["t_update"] = d["proactive_update_sent"]
    d["t_credit"] = 0
    d["t_p_rep"] = _sig(-0.30 + 0.85 * e - 0.55 * d["t_late"] - 0.12 * d["t_days_late"].clip(0, 4) + 0.35 * d["t_update"])
    d["t_ticket_p"] = np.where(d["t_late"] == 1, 0.42 - 0.25 * d["t_update"] + 0.06 * d["t_days_late"].clip(0, 3), 0) + \
        np.where(d["failed_first_attempt"] == 1, np.where(d["unavailable_gps_verified"] == 1, 0.30, 0.62), 0)
    # solution
    hours = d["actual_hours"].to_numpy().copy()
    pin_fixed = (a["pin_request"] & (d["failed_first_attempt"] == 1) & (d["address_quality"] == "Partial address")
                 & (_frac(d["u"], 997) < a["pin_fix_rate"])).to_numpy()
    hours = np.where(pin_fixed, hours - 24, hours)
    high = (d["risk_tier"] == "High risk").to_numpy()
    hours = np.where(high, hours - a["priority_hours"], hours)
    days = np.maximum(1, np.ceil(np.maximum(hours, 1) / 24)).astype(int)
    promise = np.where(a["use_p90"], d["p90_promise_days"], d["promised_days"])
    d["s_promise"] = promise
    d["s_priority"] = high.astype(int)
    d["s_pin_fixed"] = pin_fixed.astype(int)
    d["s_late"] = (days > promise).astype(int)
    d["s_days_late"] = np.maximum(days - promise, 0)
    warned = (d["p_late"] >= a["warn_threshold"]).to_numpy()
    d["s_update"] = ((d["s_late"] == 1) & (warned | (d["t_update"] == 1))).astype(int)
    late_idx = d.index[d["s_late"] == 1]
    n_credit = int(round(len(late_idx) * a["credit_share"]))
    top = d.loc[late_idx, "pred_uplift"].sort_values(ascending=False).index[:n_credit]
    d["s_credit"] = 0
    d.loc[top, "s_credit"] = 1
    d["s_p_rep"] = np.clip(_sig(-0.30 + 0.85 * e - 0.55 * d["s_late"] - 0.12 * d["s_days_late"].clip(0, 4) + 0.35 * d["s_update"])
                           + d["s_credit"] * d["true_effect"], 0, 1)
    failed_s = (d["failed_first_attempt"] == 1) & ~pin_fixed
    gps = np.where(a["gps_lock"], 1, d["unavailable_gps_verified"])
    d["s_ticket_p"] = np.where(d["s_late"] == 1, 0.42 - 0.25 * d["s_update"] + 0.06 * d["s_days_late"].clip(0, 3), 0) + \
        np.where(failed_s, np.where(gps == 1, 0.30, 0.62), 0)
    d["t_rep"] = (d["u"] < d["t_p_rep"]).astype(int)
    d["s_rep"] = (d["u"] < d["s_p_rep"]).astype(int)
    return d


def p1_kpis(d: pd.DataFrame, a: dict) -> dict:
    n = max(len(d), 1)
    return {
        "Orders processed": (len(d), len(d)),
        "Promise kept": (1 - d["t_late"].mean(), 1 - d["s_late"].mean()),
        "Late orders": (int(d["t_late"].sum()), int(d["s_late"].sum())),
        "Late customers told first": (d.loc[d.t_late == 1, "t_update"].mean() if d.t_late.sum() else np.nan,
                                      d.loc[d.s_late == 1, "s_update"].mean() if d.s_late.sum() else np.nan),
        "Credits given (AED)": (0, int(d["s_credit"].sum() * a["credit_aed"])),
        "Complaint tickets (expected)": (d["t_ticket_p"].sum(), d["s_ticket_p"].sum()),
        "Bought again within 60 days": (int(d["t_rep"].sum()), int(d["s_rep"].sum())),
        "Repeat rate": (d["t_rep"].sum() / n, d["s_rep"].sum() / n),
    }


def p1_event(r) -> str:
    bits = [f"<b>{r.order_id}</b> · {r.zone} · {r.category}"]
    if r.s_priority:
        bits.append("🚚 high risk → priority courier")
    if r.s_pin_fixed:
        bits.append("📍 pin requested, first attempt saved")
    if r.s_promise != r.promised_days:
        bits.append(f"promise {r.promised_days}d → {r.s_promise}d")
    if r.s_update:
        bits.append("💬 customer warned before it was late")
    if r.s_credit:
        bits.append("🎁 AED credit (persuadable)")
    if len(bits) == 1:
        bits.append("normal flow" + (" · kept promise" if not r.s_late else " · late"))
    return " · ".join(bits)


# ------------------------------------------------------------------ Problem 2
P2_DEFAULTS = dict(green_threshold=0.15, red_threshold=0.50, green_days=1.0, amber_days=3.0, recovery_rate=0.5,
                   explain_rejections=True)


def run_p2(df: pd.DataFrame, a: dict) -> pd.DataFrame:
    d = df.copy()
    e = d["engagement"].to_numpy()
    fail = (d["qc_passed"] == 0).to_numpy()
    rd = d["refund_days"].to_numpy()
    d["tier"] = np.select([d["p_fail"] < a["green_threshold"], d["p_fail"] < a["red_threshold"]], ["Green", "Amber"], "Red")
    # today
    d["t_days"] = np.where(fail, np.nan, rd)
    d["t_p_rep"] = _sig(-0.05 + 0.7 * e - 0.012 * np.where(fail, 25, rd) - 1.1 * fail)
    d["t_ticket_p"] = np.where(fail, 0.78, np.where(rd > 8, np.clip(0.18 + 0.03 * (rd - 8), 0, 1), 0))
    # solution
    green, amber = (d["tier"] == "Green").to_numpy(), (d["tier"] == "Amber").to_numpy()
    s_days = np.where(green, np.minimum(np.nan_to_num(rd, nan=a["green_days"]), a["green_days"]),
                      np.where(amber & ~fail, np.minimum(rd, a["amber_days"]), np.where(fail, np.nan, rd)))
    s_rejected = fail & ~green  # Green is paid before inspection
    d["s_days"] = s_days
    d["s_instant"] = green.astype(int)
    d["s_p_rep"] = _sig(-0.05 + 0.7 * e - 0.012 * np.where(s_rejected, 25, np.nan_to_num(s_days, nan=25)) - 1.1 * s_rejected
                        + 0.18 * green)
    rej_p = 0.5 if a["explain_rejections"] else 0.78
    d["s_ticket_p"] = np.where(s_rejected, rej_p, np.where(np.nan_to_num(s_days) > 8, np.clip(0.18 + 0.03 * (np.nan_to_num(s_days) - 8), 0, 1), 0))
    d["s_fraud_loss"] = np.where(green & fail, d["item_value_aed"] * (1 - a["recovery_rate"]), 0)
    d["t_rep"] = (d["u"] < d["t_p_rep"]).astype(int)
    d["s_rep"] = (d["u"] < d["s_p_rep"]).astype(int)
    return d


def p2_kpis(d: pd.DataFrame, a: dict) -> dict:
    n = max(len(d), 1)
    return {
        "Returns processed": (len(d), len(d)),
        "Median refund time (days)": (np.nanmedian(d["t_days"]) if d["t_days"].notna().any() else np.nan,
                                      np.nanmedian(d["s_days"]) if d["s_days"].notna().any() else np.nan),
        "Refunded within 1 day": ((d["t_days"] <= 1).sum() / n, (d["s_days"] <= 1).sum() / n),
        "Refund chase tickets (expected)": (d["t_ticket_p"].sum(), d["s_ticket_p"].sum()),
        "Value paid before a failed check (AED)": (0, d["s_fraud_loss"].sum()),
        "Bought again within 90 days": (int(d["t_rep"].sum()), int(d["s_rep"].sum())),
        "Repeat rate": (d["t_rep"].sum() / n, d["s_rep"].sum() / n),
    }


def p2_event(r) -> str:
    colour = {"Green": "🟢", "Amber": "🟠", "Red": "🔴"}[r.tier]
    t = "rejected at QC" if r.qc_passed == 0 else f"{r.refund_days:.0f} days"
    if r.tier == "Green":
        s = "refunded when courier collected"
    elif r.tier == "Amber":
        s = "refunded at warehouse scan" if r.qc_passed else "checked, rejected with photo reason"
    else:
        s = "full check before refund" + (" · fraud stopped" if r.qc_passed == 0 else "")
    return f"<b>{r.return_id}</b> · {r.category} · {r.return_reason} · {colour} {r.tier}: {s} (today: {t})"


# ------------------------------------------------------------------ Problem 3
P3_DEFAULTS = dict(owner_cut=0.60, misroute_cut=0.20, high_risk_threshold=0.30, priority_cut=0.30, suggest_fix=True,
                   fix_success=0.60, personal_reply_for_high_risk=True, cost_per_contact=12.0)


def run_p3(df: pd.DataFrame, a: dict) -> pd.DataFrame:
    d = df.copy()
    e = d["engagement"].to_numpy()
    c, res = d["contacts"].to_numpy(), d["resolution_hours"].to_numpy()
    rej = (d["topic"] == "Refund rejected").to_numpy()

    def churn_p(c_, res_, reop_, csat_, recon_):
        return np.clip(_sig(-1.6 - 0.75 * e + 0.35 * (c_ - 1) + 0.006 * res_ + 0.6 * reop_ + 0.45 * rej
                            - 0.25 * (csat_ - 3) + 0.5 * recon_), 0, 0.97)

    d["t_p_churn"] = churn_p(c, res, d["reopened"], d["csat"], d["recontact_7d"])
    routed = d["routed_correctly"].to_numpy() == 1
    cut = np.where(routed, a["owner_cut"], a["misroute_cut"])
    c_s = 1 + np.round((c - 1) * (1 - cut))
    res_s = res / (1 + 0.35 * (c - 1)) * (1 + 0.35 * (c_s - 1))
    high = d["t_p_churn"].to_numpy() >= a["high_risk_threshold"]
    res_s = np.where(high, res_s * (1 - a["priority_cut"]), res_s)
    fix_on = a["suggest_fix"] & routed & (d["used_best_fix"].to_numpy() == 0)
    reop_s = (d["reopened"].to_numpy() == 1) & ~(fix_on & (d["u_reopen"].to_numpy() < 0.5))
    recon_s = (d["recontact_7d"].to_numpy() == 1) & ~(fix_on & (d["u_recontact"].to_numpy() < a["fix_success"]))
    tmpl_s = (d["template_reply"].to_numpy() == 1) & ~(a["personal_reply_for_high_risk"] & high)
    csat_s = np.clip(d["csat"] + 0.010 * (res - res_s) + 0.7 * (d["reopened"] - reop_s) + 0.45 * (c - c_s)
                     + 0.4 * (d["template_reply"] - tmpl_s), 1, 5)
    d["s_contacts"], d["s_res_h"], d["s_reopened"], d["s_recontact"] = c_s, res_s, reop_s.astype(int), recon_s.astype(int)
    d["s_template"], d["s_csat"], d["s_priority"], d["s_fix_suggested"] = tmpl_s.astype(int), csat_s, high.astype(int), fix_on.astype(int)
    d["s_p_churn"] = churn_p(c_s, res_s, reop_s, csat_s, recon_s)
    d["t_churn"] = (d["u"] < d["t_p_churn"]).astype(int)
    d["s_churn"] = (d["u"] < d["s_p_churn"]).astype(int)
    return d


def p3_kpis(d: pd.DataFrame, a: dict) -> dict:
    n = max(len(d), 1)
    return {
        "Tickets processed": (len(d), len(d)),
        "Needed a repeat contact": ((d["contacts"] > 1).mean() if len(d) else np.nan, (d["s_contacts"] > 1).mean() if len(d) else np.nan),
        "Median hours to resolve": (d["resolution_hours"].median(), d["s_res_h"].median()),
        "Reopened": (d["reopened"].mean(), d["s_reopened"].mean()),
        "Average CSAT (1–5)": (d["csat"].mean(), d["s_csat"].mean()),
        "Agent contacts": (int(d["contacts"].sum()), int(d["s_contacts"].sum())),
        "Customers lost (90 days)": (int(d["t_churn"].sum()), int(d["s_churn"].sum())),
        "Churn rate": (d["t_churn"].sum() / n, d["s_churn"].sum() / n),
    }


def p3_event(r) -> str:
    bits = [f"<b>{r.ticket_id}</b> · read as <i>{r.predicted_topic}</i>" + ("" if r.routed_correctly else " (mis-read)")]
    bits.append("owner assigned · deadline " + f"{r.s_res_h / 24:.1f}d")
    if r.s_priority:
        bits.append("⚠️ high churn risk → senior agent")
    if r.s_fix_suggested:
        bits.append(f"🔧 suggested fix: {r.best_fix}")
    if r.contacts > r.s_contacts:
        bits.append(f"contacts {r.contacts} → {int(r.s_contacts)}")
    return " · ".join(bits)
