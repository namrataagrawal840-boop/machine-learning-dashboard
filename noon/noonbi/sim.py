"""Fixed-number solution simulator.

Each function replays the same unseen records twice — today's process and the solution — and returns one row per
record with both outcomes. Two kinds of input drive it:

* PROCESS assumptions (the sliders): what the solution changes — hours saved, who gets a credit, refund timing.
* CUSTOMER RESPONSE models, ESTIMATED FROM THE DATA (analysis/outcome_models.py → data/sim/outcome_models.json):
  how repeat buying, complaints, CSAT and churn respond to lateness, refund speed and contacts. They never see the
  generator's hidden variables. 100 bootstrap refits give an uncertainty band for every gain.

The same random number decides each record's outcome in both runs (common random numbers), so with the default
assumptions the result is always the same (fixed numbers).
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

MODEL_FILE = Path(__file__).resolve().parents[1] / "data" / "sim" / "outcome_models.json"

FEATURES = {
    "p1_rep": ["late", "days_late (0–4)", "update sent", "credit given", "log order value", "tenure (years)", "member", "log prior orders"],
    "p1_ticket": ["late", "late × update", "late × days late (0–3)", "failed attempt", "failed × GPS proof"],
    "p2_rep": ["QC failed", "refund days (0–30)", "instant refund", "log item value", "member", "prior returns (0–10)"],
    "p2_ticket": ["QC failed", "slow refund (>8 days)", "days over 8"],
    "p3_churn": ["extra contacts", "days to resolve", "reopened", "CSAT", "CSAT missing", "recontact in 7 days",
                 "refund rejected topic", "tenure (years)", "member"],
    "p3_csat": ["days to resolve", "reopened", "extra contacts", "template reply"],
}


def _a(x):
    return np.asarray(x, dtype=float)


def features(kind: str, c: dict) -> np.ndarray:
    """One shared definition of the model inputs, used both to FIT the models and to RUN the simulator."""
    g = lambda k: _a(c[k])  # noqa: E731
    mem = lambda: (np.asarray(c["membership"]) != "Non-member").astype(float)  # noqa: E731
    if kind == "p1_rep":
        cols = [g("late"), np.clip(g("days_late"), 0, 4), g("update"), g("credit"), np.log1p(g("value")), g("tenure") / 12, mem(),
                np.log1p(g("prior_orders"))]
    elif kind == "p1_ticket":
        cols = [g("late"), g("late") * g("update"), g("late") * np.clip(g("days_late"), 0, 3), g("failed"), g("failed") * g("gps")]
    elif kind == "p2_rep":
        f = g("fail")
        cols = [f, np.where(f == 1, 0, np.clip(np.nan_to_num(g("refund_days")), 0, 30)), g("instant"), np.log1p(g("value")), mem(),
                np.clip(g("prior_returns"), 0, 10)]
    elif kind == "p2_ticket":
        f, rd = g("fail"), np.nan_to_num(g("refund_days"))
        cols = [f, (1 - f) * (rd > 8), (1 - f) * np.clip(rd - 8, 0, 20)]
    elif kind == "p3_churn":
        cols = [g("contacts") - 1, g("res_h") / 24, g("reopened"), g("csat"), g("csat_missing"), g("recontact"), g("rejected"),
                g("tenure") / 12, mem()]
    elif kind == "p3_csat":
        cols = [g("res_h") / 24, g("reopened"), g("contacts") - 1, g("template")]
    else:  # pragma: no cover
        raise KeyError(kind)
    n = max(np.size(x) for x in cols)
    return np.column_stack([np.broadcast_to(np.asarray(x, dtype=float), (n,)) for x in cols])


@lru_cache(maxsize=1)
def models() -> dict:
    m = json.loads(MODEL_FILE.read_text(encoding="utf-8"))
    for v in m.values():
        v["point"], v["boot"] = np.array(v["point"]), np.array(v["boot"])
    return m


def _lin(kind, c, boot=False):
    X = features(kind, c)
    m = models()[kind]
    if boot:
        B = m["boot"]
        return B[:, 0][None, :] + X @ B[:, 1:].T          # n × bootstrap
    return m["point"][0] + X @ m["point"][1:]


def prob(kind, c, boot=False):
    return 1 / (1 + np.exp(-_lin(kind, c, boot)))


def band(kind, today: dict, sol: dict, linear=False) -> tuple[float, float]:
    """80% range of the total change (solution − today) across the 100 bootstrap refits of the response model."""
    f = _lin if linear else prob
    diff = f(kind, sol, True).sum(0) - f(kind, today, True).sum(0)
    return float(np.percentile(diff, 10)), float(np.percentile(diff, 90))


def _frac(u, k):  # a second, independent-looking uniform derived from u (keeps runs fixed)
    return (u * k) % 1.0


# ------------------------------------------------------------------ Problem 1
P1_DEFAULTS = dict(use_p90=True, priority_hours=6.0, warn_threshold=0.25, credit_share=0.40, credit_aed=15,
                   pin_request=True, pin_fix_rate=0.5, gps_lock=True)


def _p1_cols(d, late, days_late, update, credit, failed, gps):
    return {"late": late, "days_late": days_late, "update": update, "credit": credit, "value": d["order_value_aed"],
            "tenure": d["tenure_months"], "membership": d["membership"], "prior_orders": d["prior_orders"], "failed": failed, "gps": gps}


def run_p1(df: pd.DataFrame, a: dict) -> pd.DataFrame:
    d = df.copy()
    # today
    d["t_late"] = d["late"]
    d["t_days_late"] = d["days_late"]
    d["t_update"] = d["proactive_update_sent"]
    d["t_credit"] = 0
    today = _p1_cols(d, d["t_late"], d["t_days_late"], d["t_update"], 0 * d["late"], d["failed_first_attempt"], d["unavailable_gps_verified"])
    d["t_p_rep"] = prob("p1_rep", today)
    d["t_ticket_p"] = prob("p1_ticket", today)
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
    failed_s = ((d["failed_first_attempt"] == 1) & ~pin_fixed).astype(int)
    gps = np.where(a["gps_lock"], 1, d["unavailable_gps_verified"])
    sol = _p1_cols(d, d["s_late"], d["s_days_late"], d["s_update"], d["s_credit"], failed_s, gps)
    d["s_p_rep"] = prob("p1_rep", sol)
    d["s_ticket_p"] = prob("p1_ticket", sol)
    d["t_rep"] = (d["u"] < d["t_p_rep"]).astype(int)
    d["s_rep"] = (d["u"] < d["s_p_rep"]).astype(int)
    d.attrs["bands"] = {"Bought again within 60 days": band("p1_rep", today, sol),
                        "Complaint tickets (expected)": band("p1_ticket", today, sol)}
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
        bits.append("🎁 AED credit (top predicted uplift)")
    if len(bits) == 1:
        bits.append("normal flow" + (" · kept promise" if not r.s_late else " · late"))
    return " · ".join(bits)


# ------------------------------------------------------------------ Problem 2
P2_DEFAULTS = dict(green_threshold=0.04, red_threshold=0.24, green_days=1.0, amber_days=3.0, recovery_rate=0.5,
                   explain_rejections=True, explain_cut=0.35)


def _p2_cols(d, fail, days, instant):
    return {"fail": fail, "refund_days": days, "instant": instant, "value": d["item_value_aed"], "membership": d["membership"],
            "prior_returns": d["prior_returns_12m"]}


def run_p2(df: pd.DataFrame, a: dict) -> pd.DataFrame:
    d = df.copy()
    fail = (d["qc_passed"] == 0).to_numpy()
    rd = d["refund_days"].to_numpy()
    d["tier"] = np.select([d["p_fail"] < a["green_threshold"], d["p_fail"] < a["red_threshold"]], ["Green", "Amber"], "Red")
    # today
    d["t_days"] = np.where(fail, np.nan, rd)
    today = _p2_cols(d, fail.astype(int), np.nan_to_num(rd), 0)
    d["t_p_rep"] = prob("p2_rep", today)
    d["t_ticket_p"] = prob("p2_ticket", today)
    # solution
    green, amber = (d["tier"] == "Green").to_numpy(), (d["tier"] == "Amber").to_numpy()
    s_days = np.where(green, np.minimum(np.nan_to_num(rd, nan=a["green_days"]), a["green_days"]),
                      np.where(amber & ~fail, np.minimum(rd, a["amber_days"]), np.where(fail, np.nan, rd)))
    s_rejected = fail & ~green  # Green is paid before inspection
    d["s_days"] = s_days
    d["s_instant"] = green.astype(int)
    sol = _p2_cols(d, s_rejected.astype(int), np.nan_to_num(s_days), green.astype(int))
    d["s_p_rep"] = prob("p2_rep", sol)
    tick = prob("p2_ticket", sol)
    # assumption (not in the data): a clear photo reason cuts rejection complaints by `explain_cut`
    d["s_ticket_p"] = np.where(s_rejected & a["explain_rejections"], tick * (1 - a["explain_cut"]), tick)
    d["s_fraud_loss"] = np.where(green & fail, d["item_value_aed"] * (1 - a["recovery_rate"]), 0)
    d["t_rep"] = (d["u"] < d["t_p_rep"]).astype(int)
    d["s_rep"] = (d["u"] < d["s_p_rep"]).astype(int)
    d.attrs["bands"] = {"Bought again within 90 days": band("p2_rep", today, sol),
                        "Refund chase tickets (expected)": band("p2_ticket", today, sol)}
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


def _p3_cols(d, contacts, res_h, reopened, csat, recontact, template):
    return {"contacts": contacts, "res_h": res_h, "reopened": reopened, "csat": csat, "csat_missing": d["csat_missing"],
            "recontact": recontact, "rejected": (d["topic"] == "Refund rejected").astype(int), "tenure": d["tenure_months"],
            "membership": d["membership"], "template": template}


def run_p3(df: pd.DataFrame, a: dict) -> pd.DataFrame:
    d = df.copy()
    c, res = d["contacts"].to_numpy(), d["resolution_hours"].to_numpy()
    today = _p3_cols(d, c, res, d["reopened"], d["csat"], d["recontact_7d"], d["template_reply"])
    d["t_p_churn"] = prob("p3_churn", today)
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
    # CSAT responds to the faster, cleaner case through the CSAT model estimated from answered surveys
    sol_pre = _p3_cols(d, c_s, res_s, reop_s.astype(int), d["csat"], recon_s.astype(int), tmpl_s.astype(int))
    csat_s = np.clip(d["csat"] + _lin("p3_csat", sol_pre) - _lin("p3_csat", today), 1, 5)
    sol = dict(sol_pre, csat=csat_s)
    d["s_contacts"], d["s_res_h"], d["s_reopened"], d["s_recontact"] = c_s, res_s, reop_s.astype(int), recon_s.astype(int)
    d["s_template"], d["s_csat"], d["s_priority"], d["s_fix_suggested"] = tmpl_s.astype(int), csat_s, high.astype(int), fix_on.astype(int)
    d["s_p_churn"] = prob("p3_churn", sol)
    d["t_churn"] = (d["u"] < d["t_p_churn"]).astype(int)
    d["s_churn"] = (d["u"] < d["s_p_churn"]).astype(int)
    d.attrs["bands"] = {"Customers lost (90 days)": band("p3_churn", today, sol)}
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
