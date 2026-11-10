"""Synthetic data for the noon customer-experience study (Group 2).

Everything produced here is SYNTHETIC: it is a realistic stand-in for the data noon
already holds (orders, courier scans, returns, support tickets), built so that each
method in the proposal can be demonstrated end to end. None of it is noon's real data.

How it is made realistic
------------------------
Answers are not random. Each record follows simple cause-and-effect rules that match
what customers describe in public reviews and what the group agreed in the meeting:
  * slow sellers, bulky items, sale days, evening slots, 3PL couriers, far zones and
    poor addresses make an order later;
  * a late order makes a complaint more likely, especially when nobody warns the customer;
  * slow or rejected refunds make the next purchase less likely;
  * repeat contacts and long resolution times push customers to leave;
  * a recovery credit only changes behaviour for some customers ("persuadables").
Typical data-entry problems are then added on purpose so the cleaning step has real work.

Run:  python -m analysis.run_pipeline   (from the noon/ folder)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 20261110
START, END = pd.Timestamp("2025-11-01"), pd.Timestamp("2026-10-31")

# Approximate neighbourhood centroids (for mapping only). zone_type drives delay and address risk.
ZONES = [
    # zone, emirate, lat, lon, zone_type, km_from_hub
    ("Dubai Marina", "Dubai", 25.080, 55.140, "Dense urban", 28),
    ("Downtown Dubai", "Dubai", 25.197, 55.274, "Dense urban", 22),
    ("Deira", "Dubai", 25.271, 55.307, "Dense urban", 30),
    ("Al Quoz", "Dubai", 25.141, 55.243, "Industrial", 18),
    ("Al Barsha", "Dubai", 25.110, 55.200, "Dense urban", 20),
    ("Jumeirah Village Circle", "Dubai", 25.058, 55.207, "Suburban", 16),
    ("Arabian Ranches", "Dubai", 25.055, 55.268, "Villa community", 14),
    ("Mirdif", "Dubai", 25.220, 55.420, "Villa community", 34),
    ("International City", "Dubai", 25.164, 55.408, "Dense urban", 30),
    ("Dubai Silicon Oasis", "Dubai", 25.118, 55.383, "Suburban", 26),
    ("Al Reem Island", "Abu Dhabi", 24.497, 54.404, "Dense urban", 118),
    ("Khalifa City", "Abu Dhabi", 24.420, 54.580, "Villa community", 104),
    ("Mussafah", "Abu Dhabi", 24.360, 54.500, "Industrial", 108),
    ("Yas Island", "Abu Dhabi", 24.496, 54.606, "Suburban", 98),
    ("Al Ain", "Abu Dhabi", 24.207, 55.744, "Remote", 150),
    ("Al Nahda (Sharjah)", "Sharjah", 25.302, 55.375, "Dense urban", 38),
    ("Al Majaz", "Sharjah", 25.326, 55.383, "Dense urban", 41),
    ("Muwaileh", "Sharjah", 25.300, 55.460, "Suburban", 46),
    ("Al Nuaimiya", "Ajman", 25.392, 55.448, "Dense urban", 52),
    ("Al Nakheel (RAK)", "Ras Al Khaimah", 25.790, 55.950, "Remote", 112),
    ("Fujairah City", "Fujairah", 25.128, 56.326, "Remote", 128),
    ("Umm Al Quwain", "Umm Al Quwain", 25.565, 55.555, "Remote", 72),
]
ZONE_DF = pd.DataFrame(ZONES, columns=["zone", "emirate", "lat", "lon", "zone_type", "km_from_hub"])
ZONE_WEIGHTS = np.array([7, 6, 7, 3, 6, 6, 4, 5, 5, 4, 6, 5, 3, 3, 4, 6, 4, 3, 4, 3, 2, 1], dtype=float)
ZONE_DELAY_H = {"Dense urban": 1.5, "Suburban": 3, "Villa community": 8, "Industrial": 5, "Remote": 16}
ADDR_RISK = {"Dense urban": 0.0, "Suburban": 0.4, "Villa community": 1.2, "Industrial": 0.8, "Remote": 1.0}

CATEGORIES = {  # share, size mix (S, M, Bulky), median AED, base return rate
    "Electronics": (0.20, (0.55, 0.38, 0.07), 420, 0.06),
    "Fashion": (0.19, (0.80, 0.20, 0.00), 140, 0.17),
    "Beauty & Fragrance": (0.11, (0.95, 0.05, 0.00), 110, 0.04),
    "Home & Kitchen": (0.14, (0.35, 0.50, 0.15), 160, 0.07),
    "Grocery": (0.16, (0.40, 0.60, 0.00), 95, 0.01),
    "Baby & Toys": (0.08, (0.45, 0.45, 0.10), 130, 0.06),
    "Furniture": (0.05, (0.00, 0.20, 0.80), 900, 0.08),
    "Sports & Outdoors": (0.07, (0.45, 0.40, 0.15), 210, 0.07),
}
# Sale events in the study year (dates approximate; Ramadan 2026 ≈ 18 Feb – 19 Mar).
EVENTS = [
    ("Yellow Friday", "2025-11-20", "2025-11-30", 2.4),
    ("Dubai Shopping Festival", "2025-12-12", "2026-01-11", 1.35),
    ("Ramadan", "2026-02-18", "2026-03-19", 1.45),
    ("Eid al-Fitr", "2026-03-20", "2026-03-23", 1.6),
    ("Back to school", "2026-08-15", "2026-08-31", 1.4),
    ("11.11 sale", "2026-10-28", "2026-10-31", 1.8),
]


def _event_for(d: pd.Series) -> pd.Series:
    out = pd.Series("None", index=d.index, dtype=object)
    for name, a, b, _ in EVENTS:
        out[(d >= a) & (d <= b)] = name
    return out


def _date_weights(days: pd.DatetimeIndex) -> np.ndarray:
    w = np.ones(len(days))
    for _, a, b, m in EVENTS:
        w[(days >= a) & (days <= b)] *= m
    w *= np.where(days.dayofweek.isin([3, 4]), 1.12, 1.0)  # Thu/Fri evening shopping
    w *= 1 + 0.12 * (np.arange(len(days)) / len(days))   # mild growth across the year
    return w / w.sum()


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# ======================================================================= customers
def generate_customers(rng: np.random.Generator, n: int = 9000) -> pd.DataFrame:
    zi = rng.choice(len(ZONE_DF), n, p=ZONE_WEIGHTS / ZONE_WEIGHTS.sum())
    z = ZONE_DF.iloc[zi].reset_index(drop=True)
    tenure = np.clip(rng.gamma(2.0, 13, n), 1, 96).round().astype(int)
    engagement = rng.normal(0, 1, n) + 0.012 * tenure          # latent loyalty
    returner = rng.choice(["Typical", "Size-swapper", "Serial returner", "Possible abuser"], n,
                          p=[0.84, 0.10, 0.045, 0.015])
    member = rng.random(n) < sigmoid(-1.2 + 0.7 * engagement)  # noon One-style membership
    return pd.DataFrame({
        "customer_id": [f"C{100000 + i}" for i in range(n)],
        "emirate": z["emirate"], "home_zone": z["zone"],
        "tenure_months": tenure, "membership": np.where(member, "Member", "Non-member"),
        "_engagement": engagement, "_returner": returner,
        "age_band": rng.choice(["18–24", "25–34", "35–44", "45–54", "55+"], n, p=[.17, .38, .27, .12, .06]),
    })


# ======================================================================= sellers
def generate_sellers(rng: np.random.Generator, n: int = 320) -> pd.DataFrame:
    q = rng.choice(["Good", "Average", "Poor"], n, p=[0.55, 0.30, 0.15])
    handover = np.select([q == "Good", q == "Average"], [4, 10], 26) * rng.lognormal(0, 0.25, n)
    return pd.DataFrame({
        "seller_id": [f"S{3000 + i}" for i in range(n)],
        "_quality": q,
        "fulfilment": np.where(rng.random(n) < np.select([q == "Good", q == "Average"], [0.7, 0.5], 0.25),
                               "Fulfilled by noon", "Ships directly"),
        "_handover_mean_h": handover.round(1),
        "_defect_rate": np.select([q == "Good", q == "Average"], [0.02, 0.045], 0.10),
    })


# ======================================================================= orders
def generate_orders(rng, customers, sellers, n: int = 30000) -> pd.DataFrame:
    days = pd.date_range(START, END, freq="D")
    od = pd.to_datetime(rng.choice(days, n, p=_date_weights(days)))
    w = np.exp(0.55 * customers["_engagement"].to_numpy())
    ci = rng.choice(len(customers), n, p=w / w.sum())
    c = customers.iloc[ci].reset_index(drop=True)
    # 85% of orders go to the home zone, the rest to another zone in the same emirate (office, friend)
    zone = c["home_zone"].to_numpy().copy()
    alt = rng.random(n) < 0.15
    for i in np.where(alt)[0]:
        opts = ZONE_DF.loc[ZONE_DF["emirate"] == c.at[i, "emirate"], "zone"].to_numpy()
        zone[i] = rng.choice(opts)
    zinfo = ZONE_DF.set_index("zone").loc[zone].reset_index()

    cats = list(CATEGORIES)
    cat = rng.choice(cats, n, p=[CATEGORIES[k][0] for k in cats])
    size = np.empty(n, dtype=object)
    value = np.empty(n)
    for k in cats:
        m = cat == k
        size[m] = rng.choice(["Small", "Medium", "Bulky"], m.sum(), p=CATEGORIES[k][1])
        value[m] = CATEGORIES[k][2] * rng.lognormal(0, 0.55, m.sum())
    si = rng.choice(len(sellers), n)
    s = sellers.iloc[si].reset_index(drop=True)
    is_grocery = cat == "Grocery"
    fulfil = np.where(is_grocery, "Fulfilled by noon", s["fulfilment"].to_numpy())
    courier = np.where(rng.random(n) < 0.70, "In-house fleet", np.where(rng.random(n) < 0.6, "3PL partner A", "3PL partner B"))
    courier = np.where(zinfo["zone_type"].eq("Remote") & (rng.random(n) < 0.5), "3PL partner B", courier)
    hw = np.array([1, .5, .3, .2, .2, .3, .6, 1.2, 2, 2.6, 3, 3.2, 3.4, 3.2, 3, 3, 3.3, 3.8, 4.5, 5.2, 5.6, 5, 3.6, 2])
    hour = rng.choice(24, n, p=hw / hw.sum())
    slot = pd.cut(hour, [-1, 5, 11, 16, 20, 23], labels=["Night", "Morning", "Afternoon", "Evening", "Late evening"]).astype(str)
    event = _event_for(pd.Series(od)).to_numpy()
    on_sale = event != "None"
    addr_p = np.array([0.45, 0.35, 0.20])
    addr = np.array([rng.choice(["Map pin", "Full address", "Partial address"],
                                p=(addr_p + np.array([-0.12, -0.03, 0.15]) * ADDR_RISK[t]) /
                                (addr_p + np.array([-0.12, -0.03, 0.15]) * ADDR_RISK[t]).sum())
                     for t in zinfo["zone_type"]])

    # ---- current promise (the problem: almost everything says "tomorrow")
    promised = np.where(zinfo["emirate"].isin(["Dubai", "Sharjah", "Ajman"]), 1, 2) + (size == "Bulky")

    # ---- actual delivery time in hours (cause-and-effect)
    handover = np.where(fulfil == "Fulfilled by noon", rng.gamma(2, 1.5, n),
                        s["_handover_mean_h"].to_numpy() * rng.lognormal(0, 0.45, n))
    hours = (handover + 0.07 * zinfo["km_from_hub"].to_numpy()
             + zinfo["zone_type"].map(ZONE_DELAY_H).to_numpy()
             + np.where(on_sale, 7 * rng.random(n) + np.where(event == "Yellow Friday", 9, 0), 0)
             + np.where(size == "Bulky", 27, np.where(size == "Medium", 2, 0))
             + np.where(np.isin(slot, ["Evening", "Late evening"]), 4, 0)
             + np.where(courier == "In-house fleet", 0, np.where(courier == "3PL partner A", 4, 9))
             + rng.gamma(2.0, 1.5, n))
    # failed first attempt (driver cannot find address / customer away)
    p_fail = sigmoid(-3.3 + np.select([addr == "Partial address", addr == "Full address"], [1.7, 0.6], 0)
                     + 0.5 * zinfo["zone_type"].map(ADDR_RISK).to_numpy()
                     + np.where(np.isin(slot, ["Evening", "Late evening"]), 0.35, 0))
    failed = rng.random(n) < p_fail
    hours = hours + np.where(failed, 20 + rng.gamma(2, 3, n), 0)
    gps_at_door = np.where(failed, rng.random(n) < 0.62, True)  # ~38% of "unavailable" marks with no GPS proof
    actual_days = np.ceil(hours / 24).astype(int)
    late = actual_days > promised
    notified = np.where(late, rng.random(n) < 0.12, False)

    orders = pd.DataFrame({
        "order_id": [f"N{7000000 + i}" for i in range(n)],
        "customer_id": c["customer_id"], "order_date": od.normalize(), "order_hour": hour, "time_slot": slot,
        "emirate": zinfo["emirate"], "zone": zinfo["zone"], "zone_type": zinfo["zone_type"],
        "km_from_hub": zinfo["km_from_hub"], "category": cat, "item_size": size,
        "order_value_aed": value.round(2), "seller_id": s["seller_id"], "fulfilment": fulfil,
        "seller_handover_hours": handover.round(1), "courier": courier, "sale_event": event,
        "address_quality": addr, "promised_days": promised, "actual_hours": hours.round(1),
        "actual_days": actual_days, "late": late.astype(int), "days_late": np.maximum(actual_days - promised, 0),
        "failed_first_attempt": failed.astype(int), "unavailable_gps_verified": gps_at_door.astype(int),
        "proactive_update_sent": notified.astype(int), "membership": c["membership"],
        "tenure_months": c["tenure_months"],
    })
    orders = orders.sort_values("order_date").reset_index(drop=True)

    # ---- recovery experiment (Solution 1 A/B): late orders Apr–Jun 2026, randomised 50/50

    e = orders["customer_id"].map(customers.set_index("customer_id")["_engagement"]).to_numpy()
    in_test = (orders["late"] == 1) & orders["order_date"].between("2026-04-01", "2026-06-30")
    treat = in_test & (rng.random(n) < 0.5)
    # behaviour types for uplift: sure things, persuadables, lost causes, sleeping dogs
    first_late = ~orders.duplicated(["customer_id", "late"]) & (orders["late"] == 1)
    persuadable = (np.abs(e - 0.1) < 0.75) & (orders["membership"].eq("Non-member").to_numpy() | first_late.to_numpy())
    sure = e > 0.85
    lost = e < -1.0
    effect = np.select([persuadable, sure, lost], [0.15, 0.005, 0.0], 0.02)
    base = sigmoid(-0.30 + 0.85 * e - 0.55 * orders["late"] - 0.12 * orders["days_late"].clip(0, 4)
                   + 0.35 * orders["proactive_update_sent"])
    p_rep = np.clip(base + treat * effect, 0, 1)
    orders["recovery_test_group"] = np.where(in_test, np.where(treat, "Treatment", "Control"), "Not in test")
    orders["credit_aed"] = np.where(treat, 15, 0)
    orders["repurchase_60d"] = (rng.random(n) < p_rep).astype(int)
    return orders


# ======================================================================= returns
REASONS = ["Size or fit issue", "Defective / not working", "Not as described", "Changed my mind",
           "Wrong item delivered", "Damaged in delivery"]


def generate_returns(rng, orders, customers, sellers) -> pd.DataFrame:
    cust = customers.set_index("customer_id")
    sell = sellers.set_index("seller_id")
    o = orders.copy()
    n = len(o)
    returner = o["customer_id"].map(cust["_returner"]).to_numpy()
    mult = pd.Series(returner).map({"Typical": 1.0, "Size-swapper": 2.4, "Serial returner": 4.2,
                                    "Possible abuser": 5.0}).to_numpy()
    base = o["category"].map({k: v[3] for k, v in CATEGORIES.items()}).to_numpy()
    two_sizes = (o["category"] == "Fashion").to_numpy() & (rng.random(n) < np.where(returner == "Size-swapper", 0.6, 0.10))
    defect = o["seller_id"].map(sell["_defect_rate"]).to_numpy()
    p_ret = np.clip(base * mult + np.where(two_sizes, 0.55, 0) + defect * 0.8 + 0.05 * o["late"], 0, 0.95)
    is_ret = rng.random(n) < p_ret
    r = o.loc[is_ret].reset_index(drop=True)
    r["ordered_two_sizes"] = two_sizes[is_ret].astype(int)
    m = len(r)
    rt = returner[is_ret]
    reason = np.empty(m, dtype=object)
    for i in range(m):
        cat = r.at[i, "category"]
        if r.at[i, "ordered_two_sizes"] or (cat == "Fashion" and rng.random() < 0.55):
            reason[i] = "Size or fit issue"
            continue
        p = np.array([0.04, 0.28, 0.22, 0.26, 0.10, 0.10])
        if sell.at[r.at[i, "seller_id"], "_quality"] == "Poor":
            p = p + np.array([0, 0.12, 0.10, -0.06, 0.04, 0])
        if rt[i] == "Possible abuser":
            p = p + np.array([0, 0.25, 0.05, -0.10, 0, 0.10])
        p = np.clip(p, 0.005, None)
        reason[i] = rng.choice(REASONS, p=p / p.sum())
    r["return_reason"] = reason
    r["days_since_delivery"] = np.where(rt == "Possible abuser", rng.integers(1, 6, m), rng.integers(1, 16, m))
    r["prior_returns_12m"] = rng.poisson(pd.Series(rt).map({"Typical": 0.6, "Size-swapper": 3, "Serial returner": 6,
                                                            "Possible abuser": 5}).to_numpy())
    r["account_age_months"] = np.where(rt == "Possible abuser", rng.integers(1, 5, m), r["tenure_months"])
    r["photos_uploaded"] = (rng.random(m) < np.where(rt == "Possible abuser", 0.3, 0.58)).astype(int)
    r["seller_quality_flag"] = r["seller_id"].map(sell["_quality"]).eq("Poor").astype(int)
    # QC result: passes unless the item is used / not the original / abuse
    p_fail = (0.03 + np.where(reason == "Changed my mind", 0.13, 0) + np.where(rt == "Possible abuser", 0.55, 0)
              + np.where(rt == "Serial returner", 0.08, 0)
              + np.where(r["category"].isin(["Electronics", "Beauty & Fragrance"]), 0.05, 0)
              - 0.03 * r["photos_uploaded"])
    r["qc_passed"] = (rng.random(m) > np.clip(p_fail, 0.005, 0.95)).astype(int)
    # refund time today: pickup wait + inspection + payment rail (+ seller dispute)
    pickup = rng.integers(1, 6, m)
    inspect = rng.integers(2, 6, m)
    rail = np.where(rng.random(m) < 0.35, rng.integers(0, 2, m), rng.integers(5, 15, m))  # wallet vs card
    dispute = np.where(r["fulfilment"].eq("Ships directly") & (rng.random(m) < 0.25), rng.integers(3, 8, m), 0)
    r["refund_method"] = np.where(rail <= 1, "noon wallet", "Card / bank")
    r["refund_days"] = np.where(r["qc_passed"] == 1, pickup + inspect + rail + dispute, np.nan)
    r["refund_disputed"] = ((r["qc_passed"] == 0) & (rng.random(m) < 0.72)).astype(int)
    size_cost = r["item_size"].map({"Small": 12, "Medium": 18, "Bulky": 55}).to_numpy()
    depre = r["category"].map({"Electronics": .30, "Fashion": .18, "Beauty & Fragrance": .45, "Home & Kitchen": .22,
                               "Grocery": .9, "Baby & Toys": .25, "Furniture": .28, "Sports & Outdoors": .2}).to_numpy()
    r["return_cost_aed"] = (size_cost * 2 + 6 + depre * r["order_value_aed"] * rng.uniform(0.7, 1.3, m)).round(2)
    # Solution 2 A/B: low-risk (typical, clean history) returns Jul–Sep 2026 split 50/50 to instant refund
    low_risk = np.isin(rt, ["Typical", "Size-swapper"]) & (r["prior_returns_12m"] <= 3) & (r["account_age_months"] >= 3)
    in_test = low_risk & r["order_date"].between("2026-05-01", "2026-10-31").to_numpy()
    instant = in_test & (rng.random(m) < 0.5)
    r["instant_refund_test_group"] = np.where(in_test, np.where(instant, "Instant refund", "Standard refund"), "Not in test")
    r.loc[instant & (r["qc_passed"] == 1), "refund_days"] = rng.uniform(0, 1, (instant & (r["qc_passed"] == 1)).sum()).round(1)
    eng = r["customer_id"].map(cust["_engagement"]).to_numpy()
    rd = r["refund_days"].fillna(25).to_numpy()
    p_rep = sigmoid(-0.05 + 0.7 * eng - 0.012 * rd - 1.1 * (r["qc_passed"] == 0) + 0.18 * instant)
    r["repurchase_90d"] = (rng.random(m) < p_rep).astype(int)
    r["_returner"] = rt
    r = r.rename(columns={"order_value_aed": "item_value_aed"})
    r.insert(0, "return_id", [f"R{500000 + i}" for i in range(m)])
    keep = ["return_id", "order_id", "customer_id", "order_date", "emirate", "category", "item_size", "seller_id",
            "fulfilment", "seller_quality_flag", "item_value_aed", "ordered_two_sizes", "return_reason",
            "days_since_delivery", "prior_returns_12m", "account_age_months", "photos_uploaded", "qc_passed",
            "refund_method", "refund_days", "refund_disputed", "return_cost_aed", "instant_refund_test_group",
            "repurchase_90d", "membership", "_returner"]
    return r[keep]


# ======================================================================= tickets
PHRASES = {
    "Delivery delay": ["my order is late", "still not delivered", "delivery delayed again", "was promised tomorrow",
                       "no update on my order", "tracking not moving", "order stuck in transit", "where is my parcel",
                       "expected delivery passed", "late delivery"],
    "Failed delivery": ["driver marked me unavailable", "i was at home all day", "driver could not find my villa",
                        "delivery attempt failed", "nobody called me", "marked customer not available",
                        "driver never came", "wrong location pin"],
    "Refund delay": ["refund not received", "waiting for my refund", "refund still pending", "money not returned",
                     "returned item two weeks ago", "when will i get refund", "refund status not updated"],
    "Refund rejected": ["refund rejected", "qc failed without reason", "return rejected", "item failed inspection",
                        "they refused my refund", "rejected after pickup", "quality check rejected"],
    "Warranty / seller": ["warranty claim", "seller not responding", "product stopped working", "repair request",
                          "seller refused warranty", "passed between seller and noon", "defective after a month"],
    "Wrong / damaged item": ["wrong item delivered", "received different product", "item damaged", "box was broken",
                             "missing parts", "fake product", "not the one i ordered"],
    "Payment / billing": ["charged twice", "payment deducted but order cancelled", "coupon not applied",
                          "wallet balance missing", "cash on delivery issue", "double charge"],
}
ANGER = ["this is unacceptable", "worst service", "very disappointed", "i am fed up", "never again",
         "third time contacting", "nobody is helping", "copy paste replies", "escalate this now", "terrible experience"]
NEUTRAL = ["please help", "kindly check", "can you update me", "thank you", "order number attached",
           "hi team", "please advise", "need assistance"]


def _ticket_text(rng, topic: str, angry: bool) -> str:
    parts = list(rng.choice(PHRASES[topic], rng.integers(1, 3), replace=False))
    if rng.random() < 0.18:  # some tickets mention a second issue, which makes reading harder
        other = rng.choice([t for t in PHRASES if t != topic])
        parts.append(rng.choice(PHRASES[other]))
    # tone words are noisy on purpose: some angry customers write politely and vice versa
    pool = ANGER if (rng.random() < (0.8 if angry else 0.12)) else NEUTRAL
    parts += list(rng.choice(pool, rng.integers(1, 3), replace=False))
    rng.shuffle(parts)
    return ". ".join(parts).capitalize() + "."


def generate_tickets(rng, orders, returns, customers) -> pd.DataFrame:
    cust = customers.set_index("customer_id")
    rows = []

    def add(cid, order_id, date, topic, trigger, p):
        if rng.random() < p:
            rows.append((cid, order_id, date + pd.Timedelta(days=int(rng.integers(0, 4))), topic, trigger))

    for o in orders.itertuples(index=False):
        if o.late:
            add(o.customer_id, o.order_id, o.order_date, "Delivery delay", "Late order",
                0.42 - 0.25 * o.proactive_update_sent + 0.06 * min(o.days_late, 3))
        if o.failed_first_attempt:
            add(o.customer_id, o.order_id, o.order_date, "Failed delivery", "Failed delivery",
                0.62 if not o.unavailable_gps_verified else 0.30)
    for r in returns.itertuples(index=False):
        if r.qc_passed == 0:
            add(r.customer_id, r.order_id, r.order_date, "Refund rejected", "Refund rejected", 0.78)
        elif r.refund_days > 8:
            add(r.customer_id, r.order_id, r.order_date, "Refund delay", "Slow refund", 0.18 + 0.03 * (r.refund_days - 8))
        if r.return_reason in ("Defective / not working",) and rng.random() < 0.25:
            add(r.customer_id, r.order_id, r.order_date, "Warranty / seller", "Defect", 1.0)
        if r.return_reason in ("Wrong item delivered", "Damaged in delivery"):
            add(r.customer_id, r.order_id, r.order_date, "Wrong / damaged item", "Wrong/damaged", 0.65)
    pay = orders.sample(frac=0.018, random_state=int(rng.integers(1e6)))
    for o in pay.itertuples(index=False):
        add(o.customer_id, o.order_id, o.order_date, "Payment / billing", "Payment", 1.0)

    t = pd.DataFrame(rows, columns=["customer_id", "order_id", "created_date", "topic", "trigger"])
    m = len(t)
    eng = t["customer_id"].map(cust["_engagement"]).to_numpy()
    hard = t["topic"].isin(["Refund rejected", "Warranty / seller"]).to_numpy()
    contacts = 1 + rng.poisson(np.where(hard, 1.1, 0.45))
    angry = rng.random(m) < sigmoid(-1.0 + 0.6 * (contacts - 1) + 0.7 * hard)
    base_h = t["topic"].map({"Delivery delay": 20, "Failed delivery": 18, "Refund delay": 50, "Refund rejected": 96,
                             "Warranty / seller": 150, "Wrong / damaged item": 40, "Payment / billing": 30}).to_numpy()
    res_h = base_h * rng.lognormal(0, 0.5, m) * (1 + 0.35 * (contacts - 1))
    reopened = rng.random(m) < sigmoid(-2.2 + 0.55 * (contacts - 1) + 0.8 * hard)
    template = rng.random(m) < 0.66
    csat = np.clip(np.round(4.3 - 0.010 * res_h - 0.7 * reopened - 0.45 * (contacts - 1) - 0.4 * template
                            + rng.normal(0, 0.7, m)), 1, 5).astype(int)
    fixes = {"Delivery delay": ["Re-promise date", "Expedite", "Voucher"], "Failed delivery": ["Re-attempt", "Expedite", "Voucher"],
             "Refund delay": ["Escalate to finance", "Wallet refund now", "Explain policy"],
             "Refund rejected": ["Re-inspect", "Goodwill credit", "Explain policy"],
             "Warranty / seller": ["Escalate to seller", "Replace item", "Refund"],
             "Wrong / damaged item": ["Replace item", "Refund", "Voucher"],
             "Payment / billing": ["Reverse charge", "Explain policy", "Voucher"]}
    good_fix = {"Delivery delay": "Expedite", "Failed delivery": "Re-attempt", "Refund delay": "Wallet refund now",
                "Refund rejected": "Re-inspect", "Warranty / seller": "Replace item", "Wrong / damaged item": "Replace item",
                "Payment / billing": "Reverse charge"}
    fix = np.array([rng.choice(fixes[tp]) for tp in t["topic"]])
    good = fix == t["topic"].map(good_fix).to_numpy()
    recontact_7d = rng.random(m) < sigmoid(-0.9 + 0.5 * (contacts - 1) - 1.3 * good + 0.4 * template)
    order_val = t["order_id"].map(orders.set_index("order_id")["order_value_aed"]).to_numpy()
    p_churn = sigmoid(-1.6 - 0.75 * eng + 0.35 * (contacts - 1) + 0.006 * res_h + 0.6 * reopened
                      + 0.45 * (t["topic"] == "Refund rejected") - 0.25 * (csat - 3) + 0.5 * recontact_7d)
    t["churn_90d"] = (rng.random(m) < np.clip(p_churn, 0, 0.97)).astype(int)
    t["text"] = [_ticket_text(rng, tp, a) for tp, a in zip(t["topic"], angry)]
    t["channel"] = rng.choice(["Chat", "Email", "Phone", "Social media"], m, p=[.46, .26, .2, .08])
    t["angry"] = angry.astype(int)
    t["contacts"] = contacts
    t["resolution_hours"] = res_h.round(1)
    t["reopened"] = reopened.astype(int)
    t["template_reply"] = template.astype(int)
    t["csat"] = csat
    t["fix_applied"] = fix
    t["recontact_7d"] = recontact_7d.astype(int)
    t["order_value_aed"] = order_val
    t["tenure_months"] = t["customer_id"].map(cust["tenure_months"]).to_numpy()
    t["membership"] = t["customer_id"].map(cust["membership"]).to_numpy()
    t["emirate"] = t["customer_id"].map(cust["emirate"]).to_numpy()
    t = t.sort_values("created_date").reset_index(drop=True)
    t.insert(0, "ticket_id", [f"T{900000 + i}" for i in range(m)])
    return t


# ======================================================================= customer summary
def customer_summary(orders, returns, tickets, customers) -> pd.DataFrame:
    snap = END + pd.Timedelta(days=1)
    g = orders.groupby("customer_id")
    s = pd.DataFrame({
        "orders_12m": g.size(), "spend_12m_aed": g["order_value_aed"].sum().round(2),
        "last_order": g["order_date"].max(), "first_order": g["order_date"].min(),
        "late_orders": g["late"].sum(), "failed_deliveries": g["failed_first_attempt"].sum(),
    })
    s["recency_days"] = (snap - s["last_order"]).dt.days
    s["returns_12m"] = returns.groupby("customer_id").size().reindex(s.index).fillna(0).astype(int)
    s["rejected_refunds"] = returns.groupby("customer_id")["qc_passed"].apply(lambda x: (x == 0).sum()).reindex(s.index).fillna(0).astype(int)
    s["tickets_12m"] = tickets.groupby("customer_id").size().reindex(s.index).fillna(0).astype(int)
    s["repeat_contacts"] = tickets.groupby("customer_id")["contacts"].apply(lambda x: (x - 1).sum()).reindex(s.index).fillna(0).astype(int)
    c = customers.set_index("customer_id").loc[s.index]
    s = s.join(c[["emirate", "home_zone", "tenure_months", "membership", "age_band", "_engagement", "_returner"]])
    # next-12-month profit (CLV) and churn, driven by engagement and experience
    rng = np.random.default_rng(SEED + 7)
    margin = 0.11
    future_orders = np.maximum(0, s["orders_12m"] * np.exp(0.25 * s["_engagement"] - 0.10 * s["late_orders"]
                                                           - 0.25 * s["rejected_refunds"] - 0.08 * s["repeat_contacts"])
                               * rng.lognormal(0, 0.35, len(s)))
    s["clv_next12m_aed"] = (future_orders * (s["spend_12m_aed"] / s["orders_12m"]) * margin).round(2)
    p = sigmoid(-2.5 - 0.9 * s["_engagement"] + 0.012 * s["recency_days"] + 0.25 * s["late_orders"].clip(0, 4)
                + 0.6 * s["rejected_refunds"].clip(0, 2) + 0.2 * s["repeat_contacts"].clip(0, 5))
    s["churned_next90d"] = (rng.random(len(s)) < p).astype(int)
    gap = np.clip(rng.gamma(1.6, np.clip(40 / s["orders_12m"], 2, 120)), 1, 180)
    s["next_purchase_days"] = np.where(s["churned_next90d"] == 1, np.nan, gap.round())
    return s.reset_index().rename(columns={"index": "customer_id"})


# ======================================================================= daily demand
def daily_demand(rng) -> pd.DataFrame:
    days = pd.date_range("2024-11-01", END, freq="D")
    rows = []
    for em, share in {"Dubai": .40, "Abu Dhabi": .27, "Sharjah": .17, "Northern Emirates": .16}.items():
        t = np.arange(len(days))
        base = 52000 * share * (1 + 0.18 * t / 365)
        weekly = 1 + np.select([days.dayofweek == 3, days.dayofweek == 4, days.dayofweek == 6], [.10, .14, -.06], 0)
        evm = np.ones(len(days))
        for _, a, b, m in EVENTS:
            for yr_shift in (0, -1):
                a2, b2 = pd.Timestamp(a) + pd.DateOffset(years=yr_shift), pd.Timestamp(b) + pd.DateOffset(years=yr_shift)
                if _ == "Ramadan" and yr_shift == -1:
                    a2, b2 = pd.Timestamp("2025-03-01"), pd.Timestamp("2025-03-29")
                if _ == "Eid al-Fitr" and yr_shift == -1:
                    a2, b2 = pd.Timestamp("2025-03-30"), pd.Timestamp("2025-04-02")
                evm[(days >= a2) & (days <= b2)] = np.maximum(evm[(days >= a2) & (days <= b2)], m)
        summer = 1 - 0.10 * np.exp(-((days.dayofyear - 200) / 30) ** 2)
        y = base * weekly * evm * summer * rng.lognormal(0, 0.05, len(days))
        rows.append(pd.DataFrame({"date": days, "region": em, "orders": y.round().astype(int)}))
    return pd.concat(rows, ignore_index=True)


# ======================================================================= quality issues
def add_quality_issues(rng, orders, returns, tickets):
    """Inject the kind of mess real exports contain. Cleaning must undo all of it."""
    o, r, t = orders.copy(), returns.copy(), tickets.copy()
    o["order_date"] = o["order_date"].dt.strftime("%Y-%m-%d")
    idx = rng.choice(len(o), 400, replace=False)
    o.loc[idx[:200], "order_date"] = pd.to_datetime(o.loc[idx[:200], "order_date"]).dt.strftime("%d/%m/%Y")
    o.loc[idx[200:], "emirate"] = o.loc[idx[200:], "emirate"].map(
        {"Dubai": "DXB", "Abu Dhabi": "abu dhabi", "Sharjah": "SHARJAH", "Ajman": "ajman ",
         "Ras Al Khaimah": "RAK", "Fujairah": "fujairah", "Umm Al Quwain": "UAQ"})
    o["order_value_aed"] = o["order_value_aed"].astype(object)
    v = rng.choice(len(o), 300, replace=False)
    o.loc[v[:150], "order_value_aed"] = ["AED " + f"{x:,.2f}" for x in o.loc[v[:150], "order_value_aed"]]
    o.loc[v[150:], "order_value_aed"] = [f"{x:.2f} dhs" for x in o.loc[v[150:], "order_value_aed"]]
    o.loc[rng.choice(len(o), 60, replace=False), "actual_hours"] = -1          # scanner glitch
    o.loc[rng.choice(len(o), 90, replace=False), "zone"] = np.nan              # missing zone
    o = pd.concat([o, o.sample(180, random_state=1)], ignore_index=True)       # duplicate exports
    r["refund_days"] = r["refund_days"].astype(object)
    k = rng.choice(r.index[r["refund_days"].notna()], 40, replace=False)
    r.loc[k, "refund_days"] = -3                                                # impossible
    r.loc[rng.choice(len(r), 70, replace=False), "return_reason"] = rng.choice(
        ["size issue", "DEFECTIVE", "not as described ", "changed mind"], 70)
    r = pd.concat([r, r.sample(60, random_state=2)], ignore_index=True)
    t["csat"] = t["csat"].astype(object)
    t.loc[rng.choice(len(t), 50, replace=False), "csat"] = 7                     # out of 1–5 scale
    t.loc[rng.choice(len(t), 40, replace=False), "text"] = ""                    # empty ticket body
    t = pd.concat([t, t.sample(70, random_state=3)], ignore_index=True)
    return o, r, t
