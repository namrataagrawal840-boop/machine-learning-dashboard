"""Synthetic (hypothetical) survey data for validating the luggage-marketplace business idea.

Everything produced here is SIMULATED. It follows the 30-question consumer questionnaire and the 14-question
partner questionnaire, with plausible — but assumed — relationships between answers so that the descriptive,
diagnostic and machine-learning analyses have structure to find. None of it is evidence about real travellers
or real businesses. Assumptions are documented in docs/SYNTHETIC_DATA.md.

Design (from the project brief):
* 2,000 consumer records, 500 per city (Goa, Jaipur, Delhi, Mumbai) — an experimental balanced allocation,
  not the real market distribution.
* Four research microzones per city (investigation zones, not confirmed markets).
* Respondents who have only *planned* trips answer past-trip questions as "Not applicable" / blank, flagged by
  `past_trip_experience`, so Not Applicable stays distinct from missing.
* Latent traveller types drive correlated answers. The latent type is NOT exported, so the clustering
  pipeline has to discover structure on its own.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 20261110
N_PER_CITY = 500

MICROZONES = {
    "Goa": ["Panaji", "Calangute & Candolim", "Anjuna & Mapusa", "Mopa & Dabolim airport corridors"],
    "Jaipur": ["Jaipur Junction", "Hawa Mahal / Pink City", "MI Road / C-Scheme", "Sindhi Camp"],
    "Delhi": ["New Delhi Railway Station / Paharganj", "Connaught Place", "Old Delhi Railway Station",
              "IGI Airport / Aerocity"],
    "Mumbai": ["CSMT / Fort", "Churchgate / Marine Drive", "Bandra", "Andheri / airport corridor"],
}
ZONE_WEIGHTS = [0.3, 0.25, 0.25, 0.2]

# ---------------------------------------------------------------- questionnaire options
AGE = ["18–24", "25–34", "35–44", "45–54", "55 or above", "Prefer not to disclose"]
PURPOSE = ["Leisure / holiday.", "Business / professional.", "Visiting friends or family.", "Education / university.",
           "Events / conferences.", "Religious / pilgrimage.", "Other."]
TRIPS = ["0.", "1.", "2–3.", "4–6.", "7 or more."]
GROUP = ["Alone.", "One other person.", "3–4 people.", "5–6 people.", "More than 6 people."]
MODE = ["Aeroplane.", "Train.", "Bus.", "Personal vehicle / rental car.", "Other."]
ACCOM = ["Hotels.", "Hostels.", "Airbnb / short-term rentals.", "Homestays.", "Serviced apartments.",
         "Friends' or relatives' accommodation.", "Other."]
GAP = ["No gap.", "Less than 1 hour.", "1–2 hours.", "3–4 hours.", "5–6 hours.", "7–8 hours.", "More than 8 hours."]
MGMT = ["I did not have to manage it.", "Less than 1 hour.", "1–2 hours.", "3–4 hours.", "5–6 hours.", "7–8 hours."]
BAGS = ["1.", "2.", "3.", "4 or more."]
FREQ = ["Never.", "Rarely.", "Occasionally.", "Frequently.", "Almost every trip."]
METHODS = ["Asked my hotel / accommodation to hold it.", "Used an airport storage facility.",
           "Used a railway station cloakroom.", "Used a paid luggage-storage company or smart locker.",
           "Asked friends or relatives to keep it.", "Carried the luggage with me.", "Kept it in a car / taxi.",
           "Changed my itinerary to avoid carrying luggage.", "Other."]
SPEND = ["₹0 — Free storage / no spending.", "₹1–100.", "₹101–250.", "₹251–500.", "₹501–1,000.",
         "More than ₹1,000.", "I did not use a paid storage or handling service.", "I do not remember."]
SEARCH = ["Very easy.", "Easy.", "Neither easy nor difficult.", "Difficult.", "Very difficult.",
          "I was unaware of any such facility.", "I did not search for one."]
BARRIERS = ["No suitable storage facility nearby.", "Storage facility was too far from my route.",
            "Facility was difficult to locate or book.", "Prices were too high.", "Operating hours were inconvenient.",
            "I was uncomfortable leaving my luggage with a third party.", "Security arrangements were unclear.",
            "I was concerned about loss or damage.", "Pickup or delivery was not available.",
            "Free hotel storage was sufficient.", "I have never experienced these difficulties.", "Other."]
DETOUR = ["No additional distance.", "Up to a 5-minute walk.", "Up to a 10-minute walk.", "Up to a 15-minute walk.",
          "Up to a 30-minute journey by vehicle.", "I would only consider a service with pickup from my accommodation."]
VENUES = ["Verified hotels.", "Verified restaurants / cafés.", "Verified retail stores.",
          "Dedicated luggage-storage facilities.", "Self-service smart lockers.",
          "Existing airport / railway-station facilities.",
          "Any registered business meeting clearly defined security requirements.",
          "None of these. I would only use my own accommodation or an official transport facility."]
SECURITY = ["Verified business partners with identifiable addresses.", "CCTV surveillance.",
            "Tamper-evident luggage seals.", "Digital photographs of luggage at drop-off and collection.",
            "Unique QR code / luggage identification.", "Clear loss and damage protection terms.",
            "Real-time booking and collection status.", "Customer reviews and ratings.",
            "24/7 customer support or emergency assistance.", "Secure, individual luggage compartments.",
            "Identity verification during collection.", "Transparent pricing and refund policies.",
            "A clear process for complaints and resolving disputes."]
FIRST = ["Drop my luggage at a verified partner location and collect it later.",
         "Use a dedicated self-service luggage locker.",
         "Get my luggage picked up from my accommodation and stored at a partner location.",
         "Get my luggage stored and then delivered to my airport / railway station at a scheduled time.",
         "Arrange direct pickup from my accommodation and delivery to my airport / railway station.", "None of these."]
PRICING = ["Hourly pricing per bag.", "Fixed price for 3–4 hours.", "Fixed half-day pricing.", "Full-day pricing.",
           "Pricing based on bag size.", "Group / multiple-bag packages.",
           "Whichever option is cheapest for my required duration."]
FEATURES = ["Real-time luggage tracking.", "Digital luggage identification and security seal.",
            "Photographic proof of luggage handover.", "Scheduled luggage collection from accommodation.",
            "Scheduled luggage delivery to airport / railway station.", "Multiple-bag and family packages.",
            "Group booking for friends travelling together.", "Flexible extensions if my departure is delayed.",
            "Insurance or clearly defined luggage-loss protection.", "Priority customer support.",
            "Nearby storage recommendations based on my travel route.", "Discounts for repeat customers.",
            "None of these."]
NONUSE = ["I would not trust a third-party business with my luggage.", "The service would be too expensive.",
          "I would prefer free hotel storage.", "I would rather carry my luggage with me.",
          "Storage locations would be inconvenient.", "There are already enough alternatives in my area.",
          "Pickup and delivery might be unreliable.", "I would be concerned about loss or damage.",
          "I would not need this service frequently enough.", "Booking through another app would be inconvenient.",
          "Other."]
PILOT = ["Yes, definitely. I would like to try it.",
         "Maybe, depending on the location, price and security arrangements.", "No."]
SCREEN = ["Yes, travelled within the last 24 months.", "Yes, planning to travel within the next 12 months.", "Both."]

# ---------------------------------------------------------------- latent traveller types (not exported)
TYPES = ["business", "leisure", "backpacker", "family", "vfr_pilgrim"]
CITY_TYPE_MIX = {  # assumption: city character shifts the mix
    "Goa": [0.10, 0.42, 0.22, 0.18, 0.08],
    "Jaipur": [0.14, 0.36, 0.18, 0.16, 0.16],
    "Delhi": [0.30, 0.20, 0.18, 0.12, 0.20],
    "Mumbai": [0.34, 0.24, 0.12, 0.14, 0.16],
}
T = {  # per-type answer weights (same order as option lists)
    "age": {"business": [.05, .40, .32, .15, .06, .02], "leisure": [.18, .38, .22, .12, .08, .02],
            "backpacker": [.55, .38, .05, .01, .00, .01], "family": [.03, .25, .42, .20, .08, .02],
            "vfr_pilgrim": [.10, .22, .22, .20, .24, .02]},
    "purpose": {"business": [.05, .75, .02, .02, .15, .00, .01], "leisure": [.92, .00, .03, .00, .02, .02, .01],
                "backpacker": [.85, .00, .03, .08, .02, .01, .01], "family": [.70, .00, .22, .00, .00, .07, .01],
                "vfr_pilgrim": [.05, .00, .55, .05, .00, .33, .02]},
    "trips": {"business": [.00, .05, .20, .35, .40], "leisure": [.02, .35, .45, .14, .04],
              "backpacker": [.02, .25, .43, .22, .08], "family": [.03, .45, .40, .10, .02],
              "vfr_pilgrim": [.03, .35, .40, .16, .06]},
    "group": {"business": [.70, .20, .08, .01, .01], "leisure": [.15, .55, .22, .05, .03],
              "backpacker": [.45, .30, .18, .05, .02], "family": [.00, .08, .55, .30, .07],
              "vfr_pilgrim": [.20, .30, .30, .12, .08]},
    "accom": {"business": [.78, .00, .06, .00, .14, .01, .01], "leisure": [.50, .05, .22, .18, .03, .01, .01],
              "backpacker": [.10, .62, .12, .14, .00, .01, .01], "family": [.38, .00, .27, .20, .12, .02, .01],
              "vfr_pilgrim": [.25, .03, .04, .10, .00, .56, .02]},
    "bags": {"business": [.65, .30, .04, .01], "leisure": [.25, .50, .18, .07], "backpacker": [.55, .38, .06, .01],
             "family": [.03, .20, .37, .40], "vfr_pilgrim": [.25, .40, .22, .13]},
    "wtp_median": {"business": 260, "leisure": 190, "backpacker": 110, "family": 210, "vfr_pilgrim": 110},
    "trust": {"business": 0.35, "leisure": 0.15, "backpacker": 0.30, "family": -0.25, "vfr_pilgrim": -0.40},
}
CITY_MODE = {  # aeroplane, train, bus, personal vehicle, other
    "Goa": [.62, .14, .08, .14, .02], "Jaipur": [.30, .44, .12, .13, .01],
    "Delhi": [.45, .43, .07, .04, .01], "Mumbai": [.55, .33, .05, .06, .01],
}
CITY_WTP_FACTOR = {"Goa": 1.05, "Jaipur": 0.85, "Delhi": 1.0, "Mumbai": 1.2}


def _p(w):
    w = np.asarray(w, dtype=float)
    return w / w.sum()


def _pick_multi(rng, options, weights, k_min, k_max, exclusive: str | None = None):
    k = int(rng.integers(k_min, k_max + 1))
    w = _p(weights)
    k = min(k, int((w > 0).sum()))
    chosen = list(rng.choice(options, size=k, replace=False, p=w))
    if exclusive and exclusive in chosen and len(chosen) > 1:
        chosen = [exclusive]
    return chosen


def _sig(x):
    return 1 / (1 + np.exp(-x))


def generate_consumers(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    rid = 0
    for city, zones in MICROZONES.items():
        for _ in range(N_PER_CITY):
            rid += 1
            ttype = rng.choice(TYPES, p=_p(CITY_TYPE_MIX[city]))
            zone = rng.choice(zones, p=_p(ZONE_WEIGHTS))
            r: dict = {"respondent_id": f"R{rid:05d}", "country": "India", "city": city, "microzone": zone}
            intl = rng.random() < (0.18 if city in ("Goa", "Jaipur") else 0.10)
            r["residence_country"] = rng.choice(["United Kingdom", "United States", "Germany", "France", "Australia",
                                                 "United Arab Emirates"]) if intl else "India"
            planner = rng.random() < 0.09
            r["travel_status"] = SCREEN[1] if planner else rng.choice([SCREEN[0], SCREEN[2]], p=[.55, .45])
            r["past_trip_experience"] = "No — planning only" if planner else "Yes"
            r["age_group"] = rng.choice(AGE, p=_p(T["age"][ttype]))
            r["travel_purpose"] = rng.choice(PURPOSE, p=_p(T["purpose"][ttype]))
            trips_i = int(rng.choice(5, p=_p(T["trips"][ttype])))
            if intl:
                trips_i = min(trips_i, 2)
            r["trips_12m"] = TRIPS[trips_i]
            r["group_size"] = rng.choice(GROUP, p=_p(T["group"][ttype]))
            mode_w = np.array(CITY_MODE[city], dtype=float)
            if ttype == "business":
                mode_w = mode_w * [1.6, 0.8, 0.3, 0.5, 1]
            if ttype in ("backpacker", "vfr_pilgrim"):
                mode_w = mode_w * [0.5, 1.5, 2.2, 0.6, 1]
            if ttype == "family":
                mode_w = mode_w * [1, 1, 0.6, 2.0, 1]
            mode_i = int(rng.choice(5, p=_p(mode_w)))
            r["travel_mode"] = MODE[mode_i]
            accom_i = int(rng.choice(7, p=_p(T["accom"][ttype])))
            r["accommodation_type"] = ACCOM[accom_i]
            bags_i = int(rng.choice(4, p=_p(T["bags"][ttype])))
            r["bag_count"] = BAGS[bags_i]

            # ---- Section B (past experience) ----
            hotel_like = accom_i in (0, 4)  # hotels, serviced apartments usually hold luggage
            if planner:
                r["checkout_gap_hours"] = "Not applicable."
                r["luggage_management_hours"] = "Not applicable."
                gap_i, carried, methods = -1, False, []
                for k in ("problem_frequency", "existing_storage_methods", "past_spend_inr", "inconvenience_score",
                          "storage_search_difficulty", "existing_barriers"):
                    r[k] = np.nan
            else:
                # Flights/trains often depart afternoon/evening -> longer gaps (assumption)
                gw = {0: [.06, .08, .16, .26, .22, .12, .10], 1: [.07, .10, .18, .25, .20, .11, .09],
                      2: [.10, .14, .22, .24, .15, .09, .06], 3: [.30, .22, .22, .14, .07, .03, .02],
                      4: [.15, .15, .2, .2, .15, .1, .05]}[mode_i]
                if accom_i == 5:  # staying with friends -> flexible checkout
                    gw = [.35, .2, .2, .12, .07, .04, .02]
                na_gap = rng.random() < 0.025
                gap_i = int(rng.choice(7, p=_p(gw)))
                r["checkout_gap_hours"] = "Not applicable." if na_gap else GAP[gap_i]
                p_hold = {0: .62, 1: .42, 2: .10, 3: .18, 4: .55, 5: .05, 6: .2}[accom_i]
                methods = []
                if rng.random() < p_hold:
                    methods.append(METHODS[0])
                if mode_i == 0 and rng.random() < .10:
                    methods.append(METHODS[1])
                if mode_i == 1 and rng.random() < .22:
                    methods.append(METHODS[2])
                if rng.random() < (.12 if ttype == "business" else .05):
                    methods.append(METHODS[3])
                if (accom_i == 5 or ttype == "vfr_pilgrim") and rng.random() < .55:
                    methods.append(METHODS[4])
                if mode_i == 3 and rng.random() < .6:
                    methods.append(METHODS[6])
                elif rng.random() < .07:
                    methods.append(METHODS[6])
                carry_p = .25 if methods else .75
                if ttype == "backpacker":
                    carry_p += .15
                carried = rng.random() < carry_p
                if carried:
                    methods.append(METHODS[5])
                if rng.random() < .09:
                    methods.append(METHODS[7])
                if not methods:
                    methods = [METHODS[5]]
                    carried = True
                r["existing_storage_methods"] = "; ".join(methods)
                held = METHODS[0] in methods or METHODS[4] in methods or METHODS[6] in methods
                if na_gap or gap_i == 0:
                    mg_i = 0
                elif held and not carried:
                    mg_i = int(rng.choice([0, 1], p=[.75, .25]))
                else:
                    mg_i = max(0, min(5, gap_i - int(rng.choice([0, 1, 2], p=[.55, .35, .10]))))
                r["luggage_management_hours"] = MGMT[mg_i]
                freq_base = 1 + (1 if trips_i >= 3 else 0) + (1 if gap_i >= 3 else 0)
                r["problem_frequency"] = FREQ[int(np.clip(freq_base + rng.integers(-1, 2), 0, 4))]
                paid = any(m in methods for m in METHODS[1:4])
                if paid:
                    sp = rng.choice([1, 2, 3, 4, 5, 7], p=_p([.22, .38, .24, .08, .03, .05]))
                else:
                    sp = rng.choice([0, 6, 7], p=[.45, .48, .07])
                r["past_spend_inr"] = SPEND[sp]
                inc = (1.3 + 0.38 * max(mg_i - 1, 0) + 0.35 * bags_i + (0.9 if carried else 0)
                       + (0.4 if gap_i >= 4 else 0) + rng.normal(0, 0.7))
                r["inconvenience_score"] = int(np.clip(round(inc), 1, 5))
                aware = 0.88 - (0.22 if intl else 0) - (0.10 if trips_i <= 1 else 0)
                if rng.random() > aware:
                    s_i = 5
                elif held and not carried and rng.random() < .55:
                    s_i = 6
                else:
                    diff = 1.7 + (0.9 if accom_i in (2, 3) else 0) + (0.5 if carried else 0) - \
                        (0.6 if paid else 0) + rng.normal(0, 0.9)
                    s_i = int(np.clip(round(diff), 0, 4))
                r["storage_search_difficulty"] = SEARCH[s_i]
                if (held and not carried and rng.random() < .55) or (gap_i <= 1 and rng.random() < .6):
                    bars = ["Free hotel storage was sufficient."] if (held and rng.random() < .6) else \
                        ["I have never experienced these difficulties."]
                else:
                    bw = [3 if s_i in (3, 4, 5) else 1, 2.2 if s_i in (3, 4) else 0.8, 2 if s_i in (3, 4) else .6,
                          1.6 if paid else .7, 1.2, 1.4 if ttype in ("family", "vfr_pilgrim") else .6, 1.0, 1.4,
                          1.5 if bags_i >= 2 else .5, .3, 0, .2]
                    bars = _pick_multi(rng, BARRIERS, bw, 1, 3)
                r["existing_barriers"] = "; ".join(bars)
                r["_carried"], r["_held"], r["_gap"], r["_s"] = carried, held, gap_i, s_i
            r["_type"] = ttype

            # ---- Section C/D/E ----
            trust = T["trust"][ttype] + (0.2 if intl else 0) + rng.normal(0, 0.8)
            dw = {"business": [.15, .30, .25, .08, .04, .18], "leisure": [.05, .18, .32, .25, .10, .10],
                  "backpacker": [.03, .12, .28, .30, .22, .05], "family": [.08, .14, .20, .12, .10, .36],
                  "vfr_pilgrim": [.12, .22, .25, .16, .10, .15]}[ttype]
            r["maximum_detour"] = DETOUR[int(rng.choice(6, p=_p(dw)))]
            if trust < -0.9:
                ven = [VENUES[7]] if rng.random() < .6 else _pick_multi(rng, VENUES, [.2, 0, 0, .5, .3, 1, 0, 0], 1, 2)
            else:
                vw = [1.6 + trust, .9 + trust, .5 + .8 * trust, 1.3, 1.0, 1.1, .6 + .8 * trust, 0]
                ven = _pick_multi(rng, VENUES, np.clip(vw, .05, None), 1, 4)
            r["acceptable_venue_types"] = "; ".join(ven)
            accepts_partner = any(v in ven for v in VENUES[:3] + [VENUES[6]])
            sw = [1.4, 1.6, 1.0, 1.1, 0.9, 1.3 + (0.4 if trust < 0 else 0), 1.0, 1.0, 0.8,
                  1.0 + (0.5 if ttype == "family" else 0), 0.9, 1.2, 0.6]
            r["security_requirements"] = "; ".join(_pick_multi(rng, SECURITY, sw, 2, 7))
            difficulty = 1 if (not planner and r.get("_s") in (3, 4, 5)) else 0
            long_gap = 1 if (not planner and r.get("_gap", -1) >= 3) else 0
            incv = r["inconvenience_score"] if not planner else 3
            latent = (-0.55 + 0.45 * (incv - 3) + 0.65 * difficulty + 0.7 * accepts_partner + 0.35 * long_gap
                      + 0.45 * trust + 0.25 * (trips_i >= 3) + (0.25 if ttype == "family" else 0)
                      - (0.35 if ttype == "vfr_pilgrim" else 0) + rng.normal(0, 0.75))
            intent = int(np.clip(np.round(10 * _sig(latent) + rng.normal(0, 0.8)), 0, 10))
            r["booking_intent_score"] = intent
            if intent <= 2 and rng.random() < .65:
                fs = 5
            else:
                fw = {"business": [.35, .30, .08, .12, .15], "leisure": [.40, .22, .14, .12, .12],
                      "backpacker": [.45, .38, .06, .06, .05], "family": [.18, .10, .26, .22, .24],
                      "vfr_pilgrim": [.42, .20, .14, .12, .12]}[ttype]
                fs = int(rng.choice(5, p=_p(fw)))
            r["first_service_preference"] = FIRST[fs]
            pw = {"business": [.30, .28, .08, .06, .08, .02, .18], "leisure": [.20, .25, .15, .10, .08, .06, .16],
                  "backpacker": [.28, .14, .08, .05, .10, .05, .30], "family": [.10, .14, .12, .10, .10, .30, .14],
                  "vfr_pilgrim": [.18, .16, .14, .12, .08, .10, .22]}[ttype]
            r["preferred_pricing_model"] = PRICING[int(rng.choice(7, p=_p(pw)))]
            zero_p = 0.05 + 0.20 * (intent <= 3) + (0.05 if ttype == "vfr_pilgrim" else 0)
            if rng.random() < zero_p:
                wtp = 0.0
            else:
                med = T["wtp_median"][ttype] * CITY_WTP_FACTOR[city] * (1.25 if intl else 1) * \
                    (1 + 0.05 * (intent - 5)) * (1 + 0.06 * bags_i)
                wtp = float(np.exp(np.log(med) + rng.normal(0, 0.45)))
                wtp = round(wtp / 10) * 10 if wtp < 500 else round(wtp / 50) * 50
                if rng.random() < 0.008:
                    wtp = float(rng.choice([1500, 2000, 3000, 5000]))  # legitimate extreme answers
            r["storage_wtp_inr"] = wtp
            pick_interest = fs in (2, 3, 4) or r["maximum_detour"] == DETOUR[5]
            if rng.random() < (0.12 if pick_interest else 0.34):
                pwtp = 0.0
            else:
                base = (0.75 * max(wtp, 60) + 90) * (1.3 if pick_interest else 0.85) * \
                    (1.2 if ttype in ("family", "business") else 1)
                pwtp = float(np.exp(np.log(base) + rng.normal(0, 0.4)))
                pwtp = round(pwtp / 10) * 10 if pwtp < 500 else round(pwtp / 50) * 50
            r["pickup_delivery_wtp_inr"] = pwtp
            fw2 = [1.3, .9, .9, 1.2 if pick_interest else .4, 1.2 if pick_interest else .4,
                   1.6 if bags_i >= 2 else .3, 1.0 if ttype in ("backpacker", "leisure") else .3, 1.2,
                   1.2 + (0.5 if trust < 0 else 0), .6 + (0.5 if ttype == "business" else 0), 1.0,
                   .6 + 0.6 * (trips_i >= 3), .15 if intent > 3 else 1.2]
            r["preferred_features"] = "; ".join(_pick_multi(rng, FEATURES, fw2, 1, 4, exclusive=FEATURES[-1]))
            nw = [2.5 if trust < -0.3 else .6, 2.0 if wtp < 80 else .8, 2.2 if r.get("_held") else .4,
                  1.2 if ttype == "backpacker" else .4, 1.0, .5 if city in ("Delhi", "Mumbai") else .3,
                  1.0 if pick_interest else .3, 1.6 if trust < 0 else .7, 1.8 if trips_i <= 1 else .5, .6, .3]
            r["main_nonuse_barrier"] = NONUSE[int(rng.choice(len(NONUSE), p=_p(nw)))]
            pp = _sig((intent - 6) * 0.9)
            r["pilot_interest"] = PILOT[0] if rng.random() < pp * 0.75 else (PILOT[1] if rng.random() < 0.7 else PILOT[2])
            if intent <= 2:
                r["pilot_interest"] = PILOT[2] if rng.random() < .75 else PILOT[1]
            rows.append(r)
    df = pd.DataFrame(rows)
    order = ["respondent_id", "travel_status", "past_trip_experience", "age_group", "country", "residence_country", "city",
             "microzone", "travel_purpose", "trips_12m", "group_size", "travel_mode", "accommodation_type",
             "checkout_gap_hours", "luggage_management_hours", "bag_count", "problem_frequency",
             "existing_storage_methods", "past_spend_inr", "inconvenience_score", "storage_search_difficulty",
             "existing_barriers", "maximum_detour", "acceptable_venue_types", "security_requirements",
             "booking_intent_score", "first_service_preference", "preferred_pricing_model", "storage_wtp_inr",
             "pickup_delivery_wtp_inr", "preferred_features", "main_nonuse_barrier", "pilot_interest"]
    return df[order]


def add_quality_issues(clean_like: pd.DataFrame, seed: int = SEED + 1) -> pd.DataFrame:
    """Turn the simulated responses into a realistic *raw* export with typical survey problems."""
    rng = np.random.default_rng(seed)
    raw = clean_like.copy().astype(object)
    n = len(raw)
    log = {}

    def idx(k):
        return rng.choice(n, size=k, replace=False)

    # inconsistent text formatting
    for col in ("travel_mode", "accommodation_type", "travel_purpose", "city"):
        i = idx(40)
        raw.loc[i, col] = [str(v).upper().rstrip(".") if rng.random() < .5 else f"  {str(v).lower()} " for v in raw.loc[i, col]]
    # WTP typed as text / with currency / negative / blank
    i = idx(60)
    raw.loc[i, "storage_wtp_inr"] = [f"₹{int(v)}" if pd.notna(v) and rng.random() < .5 else f"{v} rs" for v in raw.loc[i, "storage_wtp_inr"]]
    i = idx(8)
    raw.loc[i, "storage_wtp_inr"] = -50
    i = idx(30)
    raw.loc[i, "storage_wtp_inr"] = np.nan
    i = idx(35)
    raw.loc[i, "pickup_delivery_wtp_inr"] = np.nan
    # out-of-range intent scores
    i = idx(10)
    raw.loc[i, "booking_intent_score"] = rng.choice([11, 12, -1, 15], size=len(i))
    # item non-response
    for col, k in (("age_group", 25), ("group_size", 15), ("preferred_pricing_model", 20), ("preferred_features", 30),
                   ("security_requirements", 18), ("main_nonuse_barrier", 22)):
        raw.loc[idx(k), col] = np.nan
    # partially completed responses (abandoned after Section C)
    i = idx(25)
    for col in ("booking_intent_score", "first_service_preference", "preferred_pricing_model", "storage_wtp_inr",
                "pickup_delivery_wtp_inr", "preferred_features", "main_nonuse_barrier", "pilot_interest"):
        raw.loc[i, col] = np.nan
    # screened-out respondents who should have ended at Q1
    scr = clean_like.sample(12, random_state=seed).copy().astype(object)
    scr["respondent_id"] = [f"S{k:05d}" for k in range(len(scr))]
    scr["travel_status"] = "Neither. (End survey.)"
    # duplicate submissions
    dups = raw.sample(30, random_state=seed + 2)
    raw = pd.concat([raw, scr, dups], ignore_index=True)
    raw = raw.sample(frac=1, random_state=seed + 3).reset_index(drop=True)
    raw.attrs["injected"] = log
    return raw


# ---------------------------------------------------------------- partner survey (P1–P14)
EST = ["Hotel / hostel.", "Restaurant / café.", "Retail store.", "Shopping centre / commercial establishment.",
       "Dedicated storage facility.", "Other."]
P4 = ["Yes, free of charge.", "Yes, for a fee.", "Only for existing customers.", "No, but we would consider it.",
      "No, and we are not interested."]
P5 = ["Yes, a dedicated storage room.", "Yes, a restricted-access area.", "Limited space behind the counter / reception.",
      "No suitable space currently available."]
P7 = ["CCTV.", "Restricted staff-only access.", "Dedicated storage room.", "Individual lockers / compartments.",
      "Staff responsible for recording handovers.", "Digital access or identification systems.",
      "Fire and emergency safety arrangements.", "None of these."]
P8 = ["Definitely yes.", "Yes, subject to security procedures.", "Maybe, depending on compensation.", "No."]
P9 = ["Fixed payment per bag per hour.", "Fixed daily payment per bag.", "Commission per successful booking.",
      "Monthly partnership arrangement.", "Open to negotiation."]
P11 = ["Yes.", "Yes, with basic staff training.", "Maybe.", "No."]
P12 = ["Yes, we can receive and release bags for couriers.", "Yes, we can also help arrange transportation.",
       "Storage only.", "No."]
P13 = ["Lack of space.", "Security risks.", "Liability for lost or damaged luggage.", "Additional staff workload.",
       "Insurance or regulatory concerns.", "Low expected earnings.", "Disruption to existing customers.",
       "Uncertain demand.", "Other."]
P14 = ["Yes.", "Maybe, after discussing terms.", "No."]
PER_ZONE = 15


def generate_partners(seed: int = SEED + 5) -> pd.DataFrame:
    """Anonymous synthetic partner-survey responses. No business names or coordinates are generated."""
    rng = np.random.default_rng(seed)
    rows, k = [], 0
    for city, zones in MICROZONES.items():
        for z in zones:
            transit = any(w in z for w in ("Railway", "Junction", "Sindhi", "airport", "Aerocity", "CSMT", "Mopa"))
            for _ in range(PER_ZONE):
                k += 1
                ew = [.32, .30, .20, .10, .04, .04] if not transit else [.34, .26, .16, .10, .10, .04]
                e = int(rng.choice(6, p=_p(ew)))
                space = int(rng.choice(4, p=_p({0: [.22, .38, .30, .10], 1: [.05, .20, .45, .30], 2: [.05, .22, .45, .28],
                                                3: [.20, .35, .30, .15], 4: [.80, .18, .02, 0], 5: [.1, .3, .4, .2]}[e])))
                cap_base = {0: 20, 1: 6, 2: 8, 3: 25, 4: 80, 5: 8}[e] * [1.6, 1.1, 0.5, 0.0][space]
                cap = int(max(0, round(rng.normal(cap_base, cap_base * 0.35)))) if cap_base else 0
                will_score = (0.4 if e in (0, 4) else 0) + (0.6 if space <= 1 else -0.3 if space == 3 else 0) + \
                    (0.2 if transit else 0) + rng.normal(0, 0.8)
                w = 0 if will_score > 0.9 else 1 if will_score > 0.2 else 2 if will_score > -0.5 else 3
                if space == 3 and rng.random() < .6:
                    w = 3
                open_h = rng.choice([7, 8, 9, 10, 11, 12], p=_p({0: [.6, .2, .1, .05, .03, .02], 1: [.15, .25, .25, .2, .1, .05],
                                                                2: [.02, .05, .25, .4, .23, .05], 3: [.0, .05, .2, .5, .25, 0],
                                                                4: [.5, .3, .15, .05, 0, 0], 5: [.1, .2, .3, .3, .1, 0]}[e]))
                close_h = rng.choice([18, 20, 21, 22, 23, 24], p=_p([.05, .15, .25, .3, .15, .1] if e != 0 else [0, .05, .1, .1, .15, .6]))
                if e == 0 and rng.random() < .5:
                    open_h, close_h = 0, 24
                sec = []
                if rng.random() < (.85 if e in (0, 3, 4) else .55):
                    sec.append(P7[0])
                if space <= 1 and rng.random() < .7:
                    sec.append(P7[1])
                if space == 0 and rng.random() < .8:
                    sec.append(P7[2])
                if e == 4 and rng.random() < .7:
                    sec.append(P7[3])
                if rng.random() < (.5 if e == 0 else .25):
                    sec.append(P7[4])
                if rng.random() < (.4 if e in (3, 4) else .1):
                    sec.append(P7[5])
                if rng.random() < .45:
                    sec.append(P7[6])
                if not sec:
                    sec = [P7[7]]
                pay_unit = rng.choice(["per bag per hour", "per bag per day", "per booking"], p=[.35, .3, .35])
                pay = {"per bag per hour": rng.normal(25, 8), "per bag per day": rng.normal(90, 30),
                       "per booking": rng.normal(60, 20)}[pay_unit]
                barriers = _pick_multi(rng, P13, [2.5 if space >= 2 else .6, 1.4, 2.0, 1.2, 0.9, 1.0, 0.9, 1.1, 0.2], 1, 3)
                rows.append({
                    "partner_id": f"P{k:04d}", "establishment_type": EST[e], "city": city, "microzone": z,
                    "opening_time": "00:00" if open_h == 0 else f"{open_h:02d}:00",
                    "closing_time": "23:59" if close_h == 24 else f"{close_h:02d}:00",
                    "days_of_operation": rng.choice(["All days", "Mon–Sat", "All days except holidays"], p=[.7, .2, .1]),
                    "current_storage_offer": P4[int(rng.choice(5, p=_p([.25, .05, .30, .28, .12] if e == 0 else [.04, .02, .14, .55, .25])))],
                    "secure_storage_available": P5[space], "storage_capacity": cap,
                    "security_features": "; ".join(sec), "partner_willingness": P8[w],
                    "commercial_preference": P9[int(rng.choice(5, p=_p([.25, .15, .30, .10, .20])))],
                    "expected_payment_inr": max(5, round(pay / 5) * 5), "expected_payment_unit": pay_unit,
                    "digital_handover_acceptance": P11[int(np.clip(w + rng.integers(-1, 2), 0, 3))],
                    "pickup_delivery_acceptance": P12[int(rng.choice(4, p=_p([.35, .12, .38, .15] if w <= 1 else [.1, .02, .38, .5])))],
                    "partner_barriers": "; ".join(barriers),
                    "pilot_participation": P14[0 if w == 0 and rng.random() < .7 else 1 if w <= 2 else 2],
                })
    return pd.DataFrame(rows)
