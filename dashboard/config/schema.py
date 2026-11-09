"""Data contract for the dashboard.

Everything the dashboard expects from the prepared datasets lives here, so the
contract can be changed without touching page code:

* DATASETS     — expected files, accepted file-name aliases, expected columns
* PROVENANCE   — the five evidence classes shown throughout the dashboard
* ALIASES      — alternative column names (including questionnaire numbers)
* ORDINALS     — ordered answer scales from the questionnaires
* MULTI_SELECT — "select all that apply" questions and short labels
* KEYWORDS     — answer phrases used by dashboard-level definitions

The dashboard only *reads* data. Nothing in this module creates records.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# =====================================================================
# Provenance classes (Part 4.4 of the brief)
# =====================================================================
PROVENANCE = {
    "synthetic": "Synthetic research data",
    "primary": "Actual primary survey data",
    "secondary": "Verified secondary market research",
    "transaction": "Actual pilot transaction data",
    "illustrative": "Illustrative financial scenario",
    "unclassified": "Provenance not declared",
}
PROVENANCE_SHORT = {
    "synthetic": "SYNTHETIC",
    "primary": "ACTUAL SURVEY",
    "secondary": "SECONDARY RESEARCH",
    "transaction": "ACTUAL TRANSACTIONS",
    "illustrative": "ILLUSTRATIVE",
    "unclassified": "PROVENANCE NOT DECLARED",
}

# Evidence status for competitor / partner / location records (Part 4.3).
EVIDENCE_STATUSES = [
    "Verified",
    "Company-reported",
    "Third-party reported",
    "Needs verification",
    "Inactive / unavailable",
]


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    title: str
    filenames: tuple[str, ...]
    purpose: str
    group: str  # "core", "market", "model", "pilot", "support"
    expected_columns: tuple[str, ...] = ()
    required_columns: tuple[str, ...] = ()
    default_provenance: str = "unclassified"  # or "inherit:<dataset key>"
    page_hint: str = ""
    optional: bool = True
    alias_group: str = ""  # which ALIASES table maps its columns


CONSUMER_COLUMNS = (
    "respondent_id", "country", "city", "locality", "microzone", "age_group", "residence_country",
    "travel_purpose", "trips_12m", "group_size", "travel_mode", "accommodation_type",
    "checkout_gap_hours", "luggage_management_hours", "bag_count", "problem_frequency",
    "existing_storage_methods", "past_spend_inr", "inconvenience_score", "storage_search_difficulty",
    "existing_barriers", "maximum_detour", "acceptable_venue_types", "security_requirements",
    "booking_intent_score", "first_service_preference", "preferred_pricing_model", "storage_wtp_inr",
    "pickup_delivery_wtp_inr", "preferred_features", "main_nonuse_barrier", "pilot_interest",
)
PARTNER_COLUMNS = (
    "partner_id", "establishment_type", "city", "microzone", "latitude", "longitude", "opening_time",
    "closing_time", "secure_storage_available", "storage_capacity", "cctv_available", "security_features",
    "partner_willingness", "commercial_preference", "expected_payment_inr", "digital_handover_acceptance",
    "pickup_delivery_acceptance", "partner_barriers", "pilot_participation", "evidence_status",
)
COMPETITOR_COLUMNS = (
    "competitor_id", "competitor_name", "city", "microzone", "latitude", "longitude", "service_type",
    "competitor_category", "hourly_price", "daily_price", "storage_types", "operating_hours",
    "pickup_available", "delivery_available", "verification_date", "source_url", "evidence_status",
)
GEO_COLUMNS = (
    "city", "microzone", "node_name", "node_type", "latitude", "longitude",
    "nearest_storage_option", "nearest_storage_distance_m", "distance_type", "evidence_status", "source_url",
)

DATASETS: dict[str, DatasetSpec] = {s.key: s for s in [
    DatasetSpec(
        "consumer_raw", "Consumer survey — raw",
        ("consumer_survey_raw.csv", "consumer_survey_raw_synthetic.csv"),
        "Original survey responses; used only for the data-quality comparison.",
        "core", CONSUMER_COLUMNS, (), "synthetic", "Data Quality", alias_group="consumer"),
    DatasetSpec(
        "consumer_clean", "Consumer survey — cleaned",
        ("consumer_survey_clean.csv", "consumer_survey_cleaned.csv"),
        "Main consumer analysis dataset (all consumer pages).",
        "core", CONSUMER_COLUMNS, (), "synthetic", "All consumer pages", optional=False, alias_group="consumer"),
    DatasetSpec(
        "cleaning_log", "Data-cleaning log",
        ("data_cleaning_log.csv", "cleaning_log.csv"),
        "Optional record of each cleaning decision (step, column, action, rows_affected, reason).",
        "support", ("step", "column", "action", "rows_affected", "reason"), (), "inherit:consumer_clean",
        "Data Quality"),
    DatasetSpec(
        "partner_survey", "Partner survey",
        ("partner_survey.csv",),
        "Potential storage-partner survey (P1–P14).",
        "market", PARTNER_COLUMNS, (), "unclassified", "Partner Network, Geographic", alias_group="partner"),
    DatasetSpec(
        "partner_audit", "Partner audit",
        ("partner_audit.csv",),
        "Field assessment of local establishments: availability, capacity, verification status.",
        "market", PARTNER_COLUMNS, (), "unclassified", "Partner Network, Geographic", alias_group="partner"),
    DatasetSpec(
        "competitors", "Competitor locations",
        ("competitor_locations.csv",),
        "Existing storage providers with evidence status and source URL.",
        "market", COMPETITOR_COLUMNS, (), "unclassified", "Geographic", alias_group="competitor"),
    DatasetSpec(
        "geo_audit", "Geographic market audit",
        ("geographic_market_audit.csv",),
        "Travel nodes (airports, stations, tourist areas) and microzone reference points.",
        "market", GEO_COLUMNS, (), "unclassified", "Geographic", alias_group="geo"),
    DatasetSpec(
        "classification_metrics", "Classification metrics",
        ("classification_metrics.csv",),
        "One row per model: accuracy, precision, recall, F1, ROC-AUC, PR-AUC.",
        "model", ("model", "accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "target", "split"),
        ("model",), "inherit:consumer_clean", "Classification", alias_group="model"),
    DatasetSpec(
        "classification_predictions", "Classification predictions",
        ("classification_predictions.csv",),
        "Actual vs predicted High_Intent per respondent and model (probability optional).",
        "model", ("respondent_id", "model", "actual", "predicted", "probability", "target", "split"),
        ("actual", "predicted"), "inherit:consumer_clean", "Classification", alias_group="model"),
    DatasetSpec(
        "cluster_assignments", "Cluster assignments",
        ("cluster_assignments.csv",),
        "Respondent-level cluster IDs from the clustering pipeline.",
        "model", ("respondent_id", "cluster"), ("respondent_id", "cluster"), "inherit:consumer_clean",
        "Customer Segmentation", alias_group="model"),
    DatasetSpec(
        "cluster_profiles", "Cluster profiles",
        ("cluster_profiles.csv",),
        "Cluster summary table produced by the clustering pipeline (shown as supplied).",
        "model", ("cluster",), ("cluster",), "inherit:consumer_clean", "Customer Segmentation",
        alias_group="model"),
    DatasetSpec(
        "association_rules", "Association rules",
        ("association_rules.csv",),
        "Rules with antecedents, consequents, support, confidence and lift.",
        "model", ("antecedents", "consequents", "support", "confidence", "lift", "count"),
        ("antecedents", "consequents", "support", "confidence", "lift"), "inherit:consumer_clean",
        "Association Mining", alias_group="model"),
    DatasetSpec(
        "regression_metrics", "Regression metrics",
        ("regression_metrics.csv",),
        "One row per model × target: MAE, RMSE, R².",
        "model", ("model", "target", "mae", "rmse", "r2", "split"), ("model",), "inherit:consumer_clean",
        "Pricing & Regression", alias_group="model"),
    DatasetSpec(
        "regression_predictions", "Regression predictions",
        ("regression_predictions.csv",),
        "Actual vs predicted willingness to pay per respondent, model and target.",
        "model", ("respondent_id", "model", "target", "actual", "predicted", "split"), ("actual", "predicted"),
        "inherit:consumer_clean", "Pricing & Regression", alias_group="model"),
    DatasetSpec(
        "regression_coefficients", "Regression coefficients",
        ("regression_coefficients.csv",),
        "Optional: model, target, feature, coefficient (for the coefficient chart).",
        "model", ("model", "target", "feature", "coefficient"), ("feature", "coefficient"),
        "inherit:consumer_clean", "Pricing & Regression", alias_group="model"),
    DatasetSpec(
        "booking_funnel", "Booking funnel",
        ("booking_funnel.csv",),
        "Actual pilot/analytics data: either stage + count rows, or one row per event with a stage column.",
        "pilot", ("stage", "count", "city", "microzone", "date", "booking_revenue", "partner_payment",
                  "transport_cost", "payment_fee", "other_variable_cost"), (), "transaction",
        "Business Strategy", alias_group="funnel"),
]}

DATA_GROUP_LABELS = {
    "core": "Consumer survey (individual-stage core)",
    "market": "Market, partner & geographic evidence",
    "model": "Group-stage model outputs",
    "pilot": "Pilot / transaction data",
    "support": "Supporting documentation",
}

# =====================================================================
# Column aliases. Keys are canonical names; values are alternatives that
# will be recognised after normalisation (lower-case, non-alphanumerics -> _).
# =====================================================================
ALIASES: dict[str, dict[str, tuple[str, ...]]] = {
    "consumer": {
        "respondent_id": ("id", "response_id", "respondentid", "resp_id"),
        "country": ("q3_country",),
        "residence_country": ("country_of_residence", "home_country"),
        "city": ("visit_city", "destination_city", "q4", "q4_city"),
        "locality": ("q5", "q5_locality", "travel_area", "area"),
        "microzone": ("micro_zone", "zone", "research_zone"),
        "age_group": ("age", "q2", "q2_age_group"),
        "travel_purpose": ("purpose", "q6", "q6_travel_purpose"),
        "trips_12m": ("trips", "trips_last_12m", "trips_in_12_months", "travel_frequency", "q7"),
        "group_size": ("travel_group", "q8"),
        "travel_mode": ("transport_mode", "mode_of_transport", "q9"),
        "accommodation_type": ("accommodation", "q10"),
        "checkout_gap_hours": ("checkout_gap", "q11", "checkout_departure_gap"),
        "luggage_management_hours": ("luggage_management_duration", "q12", "luggage_hours"),
        "bag_count": ("bags", "number_of_bags", "q13"),
        "problem_frequency": ("gap_frequency", "q14"),
        "existing_storage_methods": ("storage_methods", "luggage_methods", "q15"),
        "past_spend_inr": ("past_spend", "q16"),
        "inconvenience_score": ("inconvenience", "q17"),
        "storage_search_difficulty": ("search_difficulty", "q18"),
        "existing_barriers": ("barriers", "storage_barriers", "q19"),
        "maximum_detour": ("max_detour", "detour", "q20"),
        "acceptable_venue_types": ("acceptable_venues", "venue_types", "q21"),
        "security_requirements": ("security_features", "q22"),
        "booking_intent_score": ("booking_intent", "intent_score", "q23"),
        "first_service_preference": ("first_service", "q24"),
        "preferred_pricing_model": ("pricing_model", "pricing_preference", "q25"),
        "storage_wtp_inr": ("storage_wtp", "wtp_storage", "wtp_storage_inr", "q26"),
        "pickup_delivery_wtp_inr": ("pickup_delivery_wtp", "delivery_wtp", "wtp_pickup_delivery_inr", "q27"),
        "preferred_features": ("features", "q28"),
        "main_nonuse_barrier": ("nonuse_barrier", "main_barrier", "q29"),
        "pilot_interest": ("pilot", "q30"),
        "high_intent": ("high_intent_flag",),
        "data_source": ("provenance", "source_type"),
    },
    "partner": {
        "partner_id": ("id", "establishment_id"),
        "establishment_type": ("type", "partner_type", "p1"),
        "city": ("p2_city",),
        "microzone": ("locality", "micro_zone", "zone", "p2_locality"),
        "latitude": ("lat",),
        "longitude": ("lon", "lng", "long"),
        "opening_time": ("open_time", "opens"),
        "closing_time": ("close_time", "closes"),
        "secure_storage_available": ("secure_storage", "storage_area", "p5"),
        "storage_capacity": ("capacity", "bag_capacity", "p6"),
        "cctv_available": ("cctv",),
        "security_features": ("security_facilities", "p7"),
        "partner_willingness": ("willingness", "store_non_customers", "p8"),
        "commercial_preference": ("preferred_commercial_model", "commercial_model", "p9"),
        "expected_payment_inr": ("expected_payment", "p10"),
        "digital_handover_acceptance": ("digital_handover", "p11"),
        "pickup_delivery_acceptance": ("pickup_delivery", "courier_support", "p12"),
        "partner_barriers": ("barriers", "concerns", "p13"),
        "pilot_participation": ("pilot_interest", "pilot", "p14"),
        "current_storage_offer": ("p4", "currently_provides_storage"),
        "evidence_status": ("verification_status", "status", "partner_status", "verified"),
    },
    "competitor": {
        "competitor_id": ("id",),
        "competitor_name": ("name", "provider", "provider_name"),
        "microzone": ("locality", "micro_zone", "zone", "area"),
        "latitude": ("lat",),
        "longitude": ("lon", "lng", "long"),
        "competitor_category": ("category", "provider_category"),
        "operating_hours": ("opening_hours", "hours"),
        "verification_date": ("verified_date", "date_verified"),
        "source_url": ("url", "source"),
        "evidence_status": ("verification_status", "status", "verified"),
    },
    "geo": {
        "microzone": ("locality", "micro_zone", "zone"),
        "node_name": ("name", "location_name", "facility"),
        "node_type": ("type", "category", "facility_type"),
        "latitude": ("lat",),
        "longitude": ("lon", "lng", "long"),
        "evidence_status": ("verification_status", "status", "verified"),
        "source_url": ("url", "source"),
    },
    "model": {
        "respondent_id": ("id", "response_id"),
        "model": ("model_name", "algorithm", "classifier", "regressor"),
        "accuracy": ("acc",),
        "precision": ("prec",),
        "recall": ("sensitivity", "tpr"),
        "f1": ("f1_score", "f1score"),
        "roc_auc": ("auc", "rocauc", "auc_roc"),
        "pr_auc": ("average_precision", "prauc", "auc_pr", "ap"),
        "actual": ("y_true", "true", "actual_label", "y_test", "true_label", "actual_value", "y_actual"),
        "predicted": ("y_pred", "prediction", "predicted_label", "pred", "predicted_value"),
        "probability": ("y_prob", "prob", "proba", "predicted_proba", "score", "y_score", "probability_1"),
        "cluster": ("cluster_id", "cluster_label", "segment", "kmeans_cluster"),
        "antecedents": ("antecedent", "lhs"),
        "consequents": ("consequent", "rhs"),
        "count": ("support_count", "n_respondents", "respondents", "supporting_respondents"),
        "mae": ("mean_absolute_error",),
        "rmse": ("root_mean_squared_error",),
        "r2": ("r_squared", "r2_score", "rsquared"),
        "target": ("target_variable", "dependent_variable", "y"),
        "feature": ("variable", "predictor", "term"),
        "coefficient": ("coef", "estimate", "weight"),
        "split": ("dataset", "data_split", "set"),
    },
    "funnel": {
        "stage": ("funnel_stage", "event", "step"),
        "count": ("n", "users", "sessions", "events"),
        "booking_revenue": ("revenue",),
        "partner_payment": ("partner_payout",),
        "transport_cost": ("courier_cost", "delivery_cost"),
        "payment_fee": ("payment_fees", "processing_fee"),
        "other_variable_cost": ("other_cost",),
    },
}

# =====================================================================
# Ordinal scales. Each level: (label, regex patterns, approximate value).
# numeric_bins (lo, hi, label) use lo <= x < hi and apply when a column holds numbers.
# Patterns are matched against normalised text (lower case, dashes -> "-").
# Special levels (code None) are kept as their own category — e.g. "Not
# applicable" is never treated as missing or as zero.
# =====================================================================
ORDINALS: dict[str, dict] = {
    "checkout_gap_hours": {
        "levels": [
            ("No gap", [r"^no gap", r"^none$"], 0.0),
            ("< 1 h", [r"less than 1", r"^<\s*1", r"under 1"], 0.5),
            ("1–2 h", [r"^1\s*-\s*2"], 1.5),
            ("3–4 h", [r"^3\s*-\s*4"], 3.5),
            ("5–6 h", [r"^5\s*-\s*6"], 5.5),
            ("7–8 h", [r"^7\s*-\s*8"], 7.5),
            ("> 8 h", [r"more than 8", r"^>\s*8", r"^8\s*\+", r"over 8"], 9.0),
        ],
        "special": [("Not applicable", [r"not applicable", r"^n/?a$"])],
        "numeric_bins": [(0.0, 1e-9, "No gap"), (1e-9, 1.0, "< 1 h"), (1.0, 3.0, "1–2 h"),
                         (3.0, 5.0, "3–4 h"), (5.0, 7.0, "5–6 h"), (7.0, 8.000001, "7–8 h"), (8.000001, 1e9, "> 8 h")],
        "unit": "hours",
    },
    "luggage_management_hours": {
        "levels": [
            ("Did not manage", [r"did not have to", r"^none$", r"^no "], 0.0),
            ("< 1 h", [r"less than 1", r"^<\s*1"], 0.5),
            ("1–2 h", [r"^1\s*-\s*2"], 1.5),
            ("3–4 h", [r"^3\s*-\s*4"], 3.5),
            ("5–6 h", [r"^5\s*-\s*6"], 5.5),
            ("7–8 h", [r"^7\s*-\s*8"], 7.5),
            ("> 8 h", [r"more than 8", r"^>\s*8", r"^8\s*\+"], 9.0),
        ],
        "special": [("Not applicable", [r"not applicable", r"^n/?a$"])],
        "numeric_bins": [(0.0, 1e-9, "Did not manage"), (1e-9, 1.0, "< 1 h"), (1.0, 3.0, "1–2 h"),
                         (3.0, 5.0, "3–4 h"), (5.0, 7.0, "5–6 h"), (7.0, 8.000001, "7–8 h"), (8.000001, 1e9, "> 8 h")],
        "unit": "hours",
    },
    "bag_count": {
        "levels": [("1", [r"^1$", r"^1 bag", r"^one"], 1.0), ("2", [r"^2$", r"^2 bag", r"^two"], 2.0),
                   ("3", [r"^3$", r"^3 bag", r"^three"], 3.0),
                   ("4+", [r"^4", r"or more", r"more than 3", r"\+$"], 4.0)],
        "special": [("Not applicable", [r"not applicable", r"^n/?a$"])],
        "numeric_bins": [(0.5, 1.5, "1"), (1.5, 2.5, "2"), (2.5, 3.5, "3"), (3.5, 1e9, "4+")],
        "unit": "bags",
    },
    "problem_frequency": {
        "levels": [("Never", [r"^never"], 0.0), ("Rarely", [r"^rarely"], 1.0),
                   ("Occasionally", [r"^occasionally"], 2.0), ("Frequently", [r"^frequently"], 3.0),
                   ("Almost every trip", [r"almost every", r"every trip", r"always"], 4.0)],
        "special": [],
    },
    "trips_12m": {
        "levels": [("0", [r"^0$", r"^0 ", r"^none"], 0.0), ("1", [r"^1$", r"^1 trip", r"^one"], 1.0),
                   ("2–3", [r"^2\s*-\s*3"], 2.5), ("4–6", [r"^4\s*-\s*6"], 5.0),
                   ("7+", [r"^7", r"or more", r"\+$"], 7.0)],
        "special": [],
        "numeric_bins": [(-0.5, 0.5, "0"), (0.5, 1.5, "1"), (1.5, 3.5, "2–3"), (3.5, 6.5, "4–6"), (6.5, 1e9, "7+")],
        "unit": "trips",
    },
    "group_size": {
        "levels": [("Alone", [r"^alone", r"^1$", r"^solo"], 1.0), ("2 people", [r"one other", r"^2$", r"^two"], 2.0),
                   ("3–4", [r"^3\s*-\s*4"], 3.5), ("5–6", [r"^5\s*-\s*6"], 5.5),
                   ("7+", [r"more than 6", r"^>\s*6", r"^7"], 7.0)],
        "special": [],
        "numeric_bins": [(0.5, 1.5, "Alone"), (1.5, 2.5, "2 people"), (2.5, 4.5, "3–4"), (4.5, 6.5, "5–6"),
                         (6.5, 1e9, "7+")],
        "unit": "people",
    },
    "age_group": {
        "levels": [("18–24", [r"^18\s*-\s*24"], 1.0), ("25–34", [r"^25\s*-\s*34"], 2.0),
                   ("35–44", [r"^35\s*-\s*44"], 3.0), ("45–54", [r"^45\s*-\s*54"], 4.0),
                   ("55+", [r"^55"], 5.0)],
        "special": [("Prefer not to disclose", [r"prefer not"])],
    },
    "storage_search_difficulty": {
        "levels": [("Very easy", [r"^very easy"], 1.0), ("Easy", [r"^easy"], 2.0),
                   ("Neutral", [r"^neither", r"^neutral"], 3.0), ("Difficult", [r"^difficult"], 4.0),
                   ("Very difficult", [r"^very difficult"], 5.0)],
        "special": [("Unaware of any facility", [r"unaware", r"not aware"]),
                    ("Did not search", [r"did not search", r"didn.t search"])],
    },
    "maximum_detour": {
        "levels": [("No extra distance", [r"^no additional", r"^no extra", r"^none"], 0.0),
                   ("≤ 5-min walk", [r"\b5[- ]?min"], 1.0), ("≤ 10-min walk", [r"\b10[- ]?min"], 2.0),
                   ("≤ 15-min walk", [r"\b15[- ]?min"], 3.0), ("≤ 30-min by vehicle", [r"\b30[- ]?min"], 4.0)],
        "special": [("Pickup from accommodation only", [r"pickup", r"pick-up", r"pick up"])],
    },
    "past_spend_inr": {
        "levels": [("₹0", [r"^(₹|rs\.?|inr)?\s*0\b", r"free storage"], 0.0),
                   ("₹1–100", [r"1\s*-\s*100\b"], 50.0), ("₹101–250", [r"101\s*-\s*250"], 175.0),
                   ("₹251–500", [r"251\s*-\s*500"], 375.0), ("₹501–1,000", [r"501\s*-\s*1,?000"], 750.0),
                   ("> ₹1,000", [r"more than", r"^>\s*"], 1250.0)],
        "special": [("Did not use paid service", [r"did not use"]),
                    ("Do not remember", [r"remember", r"don.t know"])],
        "numeric_bins": [(0.0, 1e-9, "₹0"), (1e-9, 100.5, "₹1–100"), (100.5, 250.5, "₹101–250"),
                         (250.5, 500.5, "₹251–500"), (500.5, 1000.5, "₹501–1,000"), (1000.5, 1e12, "> ₹1,000")],
        "unit": "INR",
    },
    "pilot_interest": {
        "levels": [("No", [r"^no\b"], 0.0), ("Maybe", [r"^maybe"], 1.0), ("Yes", [r"^yes"], 2.0)],
        "special": [],
    },
    # ---- partner scales
    "partner_willingness": {
        "levels": [("No", [r"^no\b"], 0.0), ("Maybe (compensation)", [r"^maybe"], 1.0),
                   ("Yes, with security procedures", [r"subject to", r"^yes,? (with|subject)"], 2.0),
                   ("Definitely yes", [r"^definitely", r"^yes$"], 3.0)],
        "special": [],
    },
    "digital_handover_acceptance": {
        "levels": [("No", [r"^no\b"], 0.0), ("Maybe", [r"^maybe"], 1.0),
                   ("Yes, with training", [r"training"], 2.0), ("Yes", [r"^yes"], 3.0)],
        "special": [],
    },
    "pickup_delivery_acceptance": {
        "levels": [("No", [r"^no\b"], 0.0), ("Storage only", [r"storage only"], 1.0),
                   ("Receive/release for couriers", [r"receive", r"courier"], 2.0),
                   ("Also arrange transport", [r"arrange", r"transport"], 3.0)],
        "special": [],
    },
    "pilot_participation": {
        "levels": [("No", [r"^no\b"], 0.0), ("Maybe", [r"^maybe"], 1.0), ("Yes", [r"^yes"], 2.0)],
        "special": [],
    },
    "secure_storage_available": {
        "levels": [("No suitable space", [r"^no\b"], 0.0), ("Behind counter (limited)", [r"limited", r"counter"], 1.0),
                   ("Restricted-access area", [r"restricted"], 2.0), ("Dedicated storage room", [r"dedicated"], 3.0)],
        "special": [],
    },
}
# Score scales (numeric with optional "4 — Significantly" style labels).
SCORE_SCALES = {
    "inconvenience_score": (1, 5),
    "booking_intent_score": (0, 10),
}
NUMERIC_FIELDS = ("storage_wtp_inr", "pickup_delivery_wtp_inr")

# Nominal (unordered) consumer fields.
NOMINAL_FIELDS = ("city", "microzone", "locality", "country", "residence_country", "travel_purpose", "travel_mode",
                  "accommodation_type", "first_service_preference", "preferred_pricing_model",
                  "main_nonuse_barrier")

# =====================================================================
# Multi-select questions: canonical short label -> patterns
# =====================================================================
MULTI_SELECT: dict[str, list[tuple[str, list[str]]]] = {
    "existing_storage_methods": [
        ("Hotel / accommodation held it", [r"hotel", r"accommodation"]),
        ("Airport storage", [r"airport"]),
        ("Railway cloakroom", [r"railway", r"cloakroom"]),
        ("Paid storage / smart locker", [r"paid", r"locker", r"storage company"]),
        ("Friends / relatives", [r"friend", r"relative"]),
        ("Carried it", [r"carried", r"^carry"]),
        ("Kept in car / taxi", [r"car\b", r"taxi"]),
        ("Changed itinerary", [r"itinerary"]),
        ("Other", [r"^other"]),
    ],
    "existing_barriers": [
        ("No suitable facility nearby", [r"no suitable"]),
        ("Facility too far from route", [r"too far"]),
        ("Hard to locate / book", [r"locate", r"to book"]),
        ("Prices too high", [r"price"]),
        ("Inconvenient hours", [r"hours"]),
        ("Uncomfortable with third party", [r"uncomfortable", r"third party"]),
        ("Security unclear", [r"security"]),
        ("Loss / damage concern", [r"loss", r"damage"]),
        ("No pickup / delivery", [r"pickup", r"delivery"]),
        ("Free hotel storage sufficient", [r"free hotel", r"sufficient"]),
        ("Never experienced difficulties", [r"never experienced"]),
        ("Other", [r"^other"]),
    ],
    "acceptable_venue_types": [
        ("Verified hotels", [r"hotel"]),
        ("Verified restaurants / cafés", [r"restaurant", r"caf"]),
        ("Verified retail stores", [r"retail"]),
        ("Dedicated storage facilities", [r"dedicated"]),
        ("Smart lockers", [r"locker"]),
        ("Airport / railway facilities", [r"airport", r"railway"]),
        ("Any registered business meeting standards", [r"any registered", r"registered business"]),
        ("None — own accommodation / official only", [r"^none"]),
    ],
    "security_requirements": [
        ("Verified partners with address", [r"verified business", r"identifiable address"]),
        ("CCTV", [r"cctv"]),
        ("Tamper-evident seals", [r"tamper", r"seal"]),
        ("Photos at drop-off / collection", [r"photo"]),
        ("QR / luggage ID", [r"qr", r"identification$", r"luggage identification"]),
        ("Loss & damage protection terms", [r"loss and damage", r"loss & damage", r"protection terms"]),
        ("Real-time status", [r"real-time", r"real time"]),
        ("Reviews & ratings", [r"review", r"rating"]),
        ("24/7 support", [r"24/7", r"customer support"]),
        ("Individual compartments", [r"compartment"]),
        ("ID check at collection", [r"identity"]),
        ("Transparent pricing / refunds", [r"transparent", r"refund"]),
        ("Complaints / dispute process", [r"complaint", r"dispute"]),
    ],
    "preferred_features": [
        ("Real-time tracking", [r"tracking"]),
        ("Digital ID & security seal", [r"digital luggage", r"security seal"]),
        ("Photo proof of handover", [r"photo"]),
        ("Scheduled collection", [r"collection from"]),
        ("Scheduled delivery to airport/station", [r"delivery to"]),
        ("Multi-bag / family packages", [r"multiple-bag", r"multiple bag", r"family"]),
        ("Group booking", [r"group booking"]),
        ("Flexible extensions", [r"extension"]),
        ("Insurance / loss protection", [r"insurance", r"loss protection"]),
        ("Priority support", [r"priority"]),
        ("Route-based recommendations", [r"recommendation"]),
        ("Repeat-customer discounts", [r"discount"]),
        ("None of these", [r"^none"]),
    ],
    # partner side
    "security_features": [
        ("CCTV", [r"cctv"]),
        ("Restricted staff-only access", [r"restricted", r"staff-only"]),
        ("Dedicated storage room", [r"dedicated"]),
        ("Individual lockers / compartments", [r"locker", r"compartment"]),
        ("Staff record handovers", [r"record"]),
        ("Digital access / ID systems", [r"digital"]),
        ("Fire & emergency safety", [r"fire", r"emergency"]),
        ("None of these", [r"^none"]),
    ],
    "partner_barriers": [
        ("Lack of space", [r"space"]),
        ("Security risks", [r"security"]),
        ("Liability for loss / damage", [r"liability"]),
        ("Additional staff workload", [r"workload", r"staff"]),
        ("Insurance / regulatory", [r"insurance", r"regulat"]),
        ("Low expected earnings", [r"earning"]),
        ("Disruption to customers", [r"disruption"]),
        ("Uncertain demand", [r"demand"]),
        ("Other", [r"^other"]),
    ],
}

# =====================================================================
# Keyword groups used by dashboard-level working definitions.
# =====================================================================
ACCESS_BARRIERS = ("No suitable facility nearby", "Facility too far from route", "Hard to locate / book",
                   "Inconvenient hours")
PARTNER_VENUES = ("Verified hotels", "Verified restaurants / cafés", "Verified retail stores",
                  "Any registered business meeting standards")
DIFFICULT_SEARCH = ("Difficult", "Very difficult", "Unaware of any facility")

# Customer security requirement -> partner security facility (for trust-vs-capability chart).
TRUST_CAPABILITY_PAIRS = [
    ("CCTV", "CCTV"),
    ("Individual compartments", "Individual lockers / compartments"),
    ("ID check at collection", "Digital access / ID systems"),
    ("Photos at drop-off / collection", "Staff record handovers"),
]

# Questionnaire items that must never be classification predictors (Q23–Q30).
LEAKAGE_FIELDS = ("booking_intent_score", "first_service_preference", "preferred_pricing_model", "storage_wtp_inr",
                  "pickup_delivery_wtp_inr", "preferred_features", "main_nonuse_barrier", "pilot_interest",
                  "high_intent", "q23", "q24", "q25", "q26", "q27", "q28", "q29", "q30")

# Human-friendly names for charts and tables.
LABELS = {
    "respondent_id": "Respondent", "city": "City", "microzone": "Microzone", "locality": "Locality",
    "country": "Country", "residence_country": "Residence country", "age_group": "Age group",
    "travel_purpose": "Travel purpose", "trips_12m": "Trips (last 12 months)", "group_size": "Travel group",
    "travel_mode": "Transport mode", "accommodation_type": "Accommodation type",
    "checkout_gap_hours": "Checkout-to-departure gap", "luggage_management_hours": "Time managing luggage",
    "bag_count": "Bags to store", "problem_frequency": "How often the gap occurs",
    "existing_storage_methods": "Luggage-management methods", "past_spend_inr": "Past spend per bag",
    "inconvenience_score": "Inconvenience (1–5)", "storage_search_difficulty": "Ease of finding storage",
    "existing_barriers": "Difficulties with existing storage", "maximum_detour": "Maximum detour accepted",
    "acceptable_venue_types": "Acceptable venue types", "security_requirements": "Essential security features",
    "booking_intent_score": "Booking intent (0–10)", "first_service_preference": "First service to try",
    "preferred_pricing_model": "Preferred pricing model", "storage_wtp_inr": "Storage WTP (₹, 3 h, 1 bag)",
    "pickup_delivery_wtp_inr": "Pickup/delivery WTP (₹, extra per bag)", "preferred_features": "Preferred features",
    "main_nonuse_barrier": "Main reason not to use", "pilot_interest": "Pilot interest",
    "high_intent": "High stated intent (8–10)", "long_gap": "Gap ≥ 3 h", "has_problem": "Experiences problem",
    "unmet_need": "Unmet storage need", "accepts_partner": "Accepts partner venues", "cluster": "Cluster",
    "establishment_type": "Establishment type", "storage_capacity": "Bag capacity",
    "partner_willingness": "Willing to store non-customers", "digital_handover_acceptance": "Digital handover",
    "pickup_delivery_acceptance": "Pickup/delivery support", "secure_storage_available": "Secure storage space",
    "commercial_preference": "Commercial preference", "expected_payment_inr": "Expected payment (₹)",
    "partner_barriers": "Partner concerns", "security_features": "Security facilities",
}


def label(col: str) -> str:
    base = col.split("__")[0]
    return LABELS.get(col, LABELS.get(base, base.replace("_", " ").capitalize()))
