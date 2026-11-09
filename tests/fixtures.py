"""Throwaway TEST FIXTURES for automated tests only.

These random records exist purely to exercise dashboard code paths (parsing,
filters, charts, missing-column handling). They are written to a temporary
directory during tests, are never committed to data/, and are not research
data. Values are uniformly random and carry no business meaning.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RNG_SEED = 7

CITIES = {"Goa": ["Panaji", "Calangute"], "Jaipur": ["Pink City", "Sindhi Camp"],
          "Delhi": ["Paharganj", "Aerocity"], "Mumbai": ["CSMT / Fort", "Bandra"]}
Q = {
    "age_group": ["18–24", "25–34", "35–44", "45–54", "55 or above", "Prefer not to disclose"],
    "travel_purpose": ["Leisure / holiday.", "Business / professional.", "Visiting friends or family.", "Other."],
    "trips_12m": ["0.", "1.", "2–3.", "4–6.", "7 or more."],
    "group_size": ["Alone.", "One other person.", "3–4 people.", "5–6 people.", "More than 6 people."],
    "travel_mode": ["Aeroplane.", "Train.", "Bus.", "Personal vehicle / rental car."],
    "accommodation_type": ["Hotels.", "Hostels.", "Airbnb / short-term rentals.", "Homestays."],
    "checkout_gap_hours": ["No gap.", "Less than 1 hour.", "1–2 hours.", "3–4 hours.", "5–6 hours.", "7–8 hours.",
                           "More than 8 hours.", "Not applicable."],
    "luggage_management_hours": ["I did not have to manage it.", "Less than 1 hour.", "1–2 hours.", "3–4 hours.",
                                 "5–6 hours.", "7–8 hours."],
    "bag_count": ["1.", "2.", "3.", "4 or more."],
    "problem_frequency": ["Almost every trip.", "Frequently.", "Occasionally.", "Rarely.", "Never."],
    "past_spend_inr": ["₹0 — Free storage / no spending.", "₹1–100.", "₹101–250.", "₹251–500.",
                       "I did not use a paid storage or handling service.", "I do not remember."],
    "storage_search_difficulty": ["Very easy.", "Easy.", "Neither easy nor difficult.", "Difficult.", "Very difficult.",
                                  "I was unaware of any such facility.", "I did not search for one."],
    "maximum_detour": ["No additional distance.", "Up to a 5-minute walk.", "Up to a 10-minute walk.",
                       "Up to a 15-minute walk.", "Up to a 30-minute journey by vehicle.",
                       "I would only consider a service with pickup from my accommodation."],
    "first_service_preference": ["Drop my luggage at a verified partner location and collect it later.",
                                 "Use a dedicated self-service luggage locker.",
                                 "Get my luggage picked up from my accommodation and stored at a partner location.",
                                 "None of these."],
    "preferred_pricing_model": ["Hourly pricing per bag.", "Fixed price for 3–4 hours.", "Full-day pricing.",
                                "Whichever option is cheapest for my required duration."],
    "main_nonuse_barrier": ["I would not trust a third-party business with my luggage.",
                            "The service would be too expensive.", "I would prefer free hotel storage.", "Other."],
    "pilot_interest": ["Yes, definitely. I would like to try it.",
                       "Maybe, depending on the location, price and security arrangements.", "No."],
}
MULTI = {
    "existing_storage_methods": ["Asked my hotel / accommodation to hold it.", "Used an airport storage facility.",
                                 "Used a railway station cloakroom.", "Carried the luggage with me.",
                                 "Asked friends or relatives to keep it."],
    "existing_barriers": ["No suitable storage facility nearby.", "Prices were too high.",
                          "Security arrangements were unclear.", "Free hotel storage was sufficient.",
                          "I have never experienced these difficulties."],
    "acceptable_venue_types": ["Verified hotels.", "Verified restaurants / cafés.", "Dedicated luggage-storage facilities.",
                               "Self-service smart lockers.", "None of these. I would only use my own accommodation."],
    "security_requirements": ["CCTV surveillance.", "Tamper-evident luggage seals.",
                              "Digital photographs of luggage at drop-off and collection.",
                              "Secure, individual luggage compartments.", "Identity verification during collection."],
    "preferred_features": ["Real-time luggage tracking.", "Photographic proof of luggage handover.",
                           "Scheduled luggage collection from accommodation.", "Flexible extensions if my departure is delayed."],
}


def _ms(rng, opts, n):
    out = []
    for _ in range(n):
        k = rng.integers(0, 4)
        out.append(np.nan if k == 0 and rng.random() < 0.3 else "; ".join(rng.choice(opts, size=max(k, 1), replace=False)))
    return out


def consumer(n: int = 240, seed: int = RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    city = rng.choice(list(CITIES), n)
    df = pd.DataFrame({
        "respondent_id": [f"T{i:04d}" for i in range(n)],
        "country": "India",
        "city": city,
        "microzone": [rng.choice(CITIES[c]) for c in city],
    })
    for k, opts in Q.items():
        df[k] = rng.choice(opts, n)
    for k, opts in MULTI.items():
        df[k] = _ms(rng, opts, n)
    df["inconvenience_score"] = rng.integers(1, 6, n)
    df["booking_intent_score"] = rng.integers(0, 11, n)
    df["storage_wtp_inr"] = rng.choice([0, 50, 100, 150, 200, 300, 500, 2500], n).astype(float)
    df["pickup_delivery_wtp_inr"] = rng.choice([0, 100, 200, 300, 400], n).astype(float)
    df.loc[rng.choice(n, 10, replace=False), "storage_wtp_inr"] = np.nan
    return df


def partners(n: int = 60, seed: int = RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed + 1)
    city = rng.choice(list(CITIES), n)
    return pd.DataFrame({
        "partner_id": [f"P{i:03d}" for i in range(n)],
        "establishment_type": rng.choice(["Hotel / hostel.", "Restaurant / café.", "Retail store."], n),
        "city": city, "microzone": [rng.choice(CITIES[c]) for c in city],
        "opening_time": rng.choice(["08:00", "9 AM", "10:30", "11:00"], n),
        "closing_time": rng.choice(["18:00", "21:00", "11 PM", "16:00"], n),
        "secure_storage_available": rng.choice(["Yes, a dedicated storage room.", "Yes, a restricted-access area.",
                                                "Limited space behind the counter / reception.",
                                                "No suitable space currently available."], n),
        "storage_capacity": rng.integers(0, 40, n),
        "security_features": _ms(rng, ["CCTV.", "Restricted staff-only access.", "Individual lockers / compartments.",
                                       "Staff responsible for recording handovers."], n),
        "partner_willingness": rng.choice(["Definitely yes.", "Yes, subject to security procedures.",
                                           "Maybe, depending on compensation.", "No."], n),
        "commercial_preference": rng.choice(["Commission per successful booking.", "Open to negotiation."], n),
        "expected_payment_inr": rng.integers(20, 120, n),
        "digital_handover_acceptance": rng.choice(["Yes.", "Yes, with basic staff training.", "Maybe.", "No."], n),
        "pickup_delivery_acceptance": rng.choice(["Yes, we can receive and release bags for couriers.", "Storage only.",
                                                  "No."], n),
        "partner_barriers": _ms(rng, ["Lack of space.", "Security risks.", "Liability for lost or damaged luggage."], n),
        "evidence_status": rng.choice(["Verified", "Needs verification"], n),
    })


def model_outputs(cons: pd.DataFrame, seed: int = RNG_SEED) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed + 2)
    ids = cons["respondent_id"].tolist()
    y = (cons["booking_intent_score"] >= 8).astype(int).to_numpy()
    preds, metrics = [], []
    for m in ["Logistic Regression", "Random Forest"]:
        p = np.clip(y * 0.6 + rng.random(len(y)) * 0.5, 0, 1)
        yhat = (p >= 0.5).astype(int)
        preds.append(pd.DataFrame({"respondent_id": ids, "model": m, "y_true": y, "y_pred": yhat, "y_prob": p}))
        metrics.append({"model": m, "accuracy": 0.5, "precision": 0.5, "recall": 0.5, "f1": 0.5, "roc_auc": 0.5})
    rp = []
    for tgt in ["storage_wtp_inr", "pickup_delivery_wtp_inr"]:
        for m in ["Linear", "Ridge", "Lasso"]:
            a = cons[tgt].fillna(0).to_numpy()
            rp.append(pd.DataFrame({"respondent_id": ids, "model": m, "target": tgt, "actual": a,
                                    "predicted": a + rng.normal(0, 50, len(a))}))
    return {
        "classification_predictions.csv": pd.concat(preds),
        "classification_metrics.csv": pd.DataFrame(metrics),
        "cluster_assignments.csv": pd.DataFrame({"respondent_id": ids, "cluster": rng.integers(0, 3, len(ids))}),
        "cluster_profiles.csv": pd.DataFrame({"cluster": [0, 1, 2], "size": [1, 2, 3]}),
        "association_rules.csv": pd.DataFrame({
            "antecedents": ["frozenset({'CCTV'})", "frozenset({'Real-time tracking', 'CCTV'})", "Seals"],
            "consequents": ["frozenset({'Seals'})", "frozenset({'Photos'})", "CCTV"],
            "support": [0.2, 0.1, 0.05], "confidence": [0.6, 0.4, 0.3], "lift": [1.4, 1.1, 0.9]}),
        "regression_predictions.csv": pd.concat(rp),
        "regression_metrics.csv": pd.DataFrame({"model": ["Linear", "Ridge", "Lasso"] * 2,
                                                "target": ["storage_wtp_inr"] * 3 + ["pickup_delivery_wtp_inr"] * 3,
                                                "mae": [1.0] * 6, "rmse": [2.0] * 6, "r2": [0.1] * 6}),
        "regression_coefficients.csv": pd.DataFrame({"model": "Ridge", "target": "storage_wtp_inr",
                                                     "feature": ["bag_count", "inconvenience_score"],
                                                     "coefficient": [1.0, -0.5]}),
    }


def write_all(folder: Path, include_models: bool = True, include_partners: bool = True) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    c = consumer()
    c.to_csv(folder / "consumer_survey_clean.csv", index=False)
    raw = pd.concat([c, c.head(5)])  # duplicates for the data-quality page
    raw.to_csv(folder / "consumer_survey_raw.csv", index=False)
    if include_partners:
        partners().to_csv(folder / "partner_survey.csv", index=False)
    if include_models:
        for name, df in model_outputs(c).items():
            df.to_csv(folder / name, index=False)
    return folder


def competitors(seed: int = RNG_SEED) -> pd.DataFrame:
    """Test-only records with arbitrary coordinates (exercise the map code path)."""
    rng = np.random.default_rng(seed + 3)
    rows = []
    for i, (city, zones) in enumerate(CITIES.items()):
        for z in zones:
            rows.append({"competitor_id": f"C{len(rows)}", "competitor_name": f"Test provider {len(rows)}",
                         "city": city, "microzone": z, "latitude": 20 + i + rng.random(), "longitude": 75 + i + rng.random(),
                         "service_type": rng.choice(["Railway cloakroom", "Smart locker", "Aggregator network"]),
                         "hourly_price": rng.choice([np.nan, 50, 80]), "evidence_status": rng.choice(["Verified", "Needs verification"]),
                         "source_url": "https://example.org"})
    return pd.DataFrame(rows)


def geo() -> pd.DataFrame:
    rows = []
    for i, (city, zones) in enumerate(CITIES.items()):
        for z in zones:
            rows.append({"city": city, "microzone": z, "node_name": f"Test node {z}", "node_type": "Railway station",
                         "latitude": 20.5 + i, "longitude": 75.5 + i, "nearest_storage_option": "Test provider",
                         "nearest_storage_distance_m": 400, "distance_type": "straight-line",
                         "evidence_status": "Needs verification"})
    return pd.DataFrame(rows)


def booking_funnel() -> pd.DataFrame:
    stages = ["Awareness", "Website visit", "Location search", "Availability check", "Booking initiation",
              "Paid booking", "Completed handover", "Repeat purchase"]
    return pd.DataFrame({"stage": stages, "count": [1000, 400, 200, 150, 60, 40, 38, 5]})
