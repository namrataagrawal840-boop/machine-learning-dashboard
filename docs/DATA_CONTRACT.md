# Data contract

_Generated from `dashboard/config/schema.py` by `scripts/generate_data_contract.py` — edit the schema, then regenerate._

Place files anywhere under `data/` (sub-folders are searched). The dashboard reads them; it never modifies, cleans or overwrites them. Every file is optional — pages show a pending state for anything missing.

## Provenance classes

- `synthetic` — Synthetic research data
- `primary` — Actual primary survey data
- `secondary` — Verified secondary market research
- `transaction` — Actual pilot transaction data
- `illustrative` — Illustrative financial scenario
- `unclassified` — Provenance not declared

Declare them in `data/sources.toml` (copy `data/sources.example.toml`). A `data_source` column inside a file is also recognised. Model outputs inherit the consumer dataset's provenance.

## Consumer survey (individual-stage core)

### Consumer survey — raw

- **File name(s):** `consumer_survey_raw.csv` or `consumer_survey_raw_synthetic.csv`
- **Purpose:** Original survey responses; used only for the data-quality comparison.
- **Used on:** Data Quality
- **Required columns:** none
- **Expected columns:** `respondent_id`, `country`, `city`, `locality`, `microzone`, `age_group`, `residence_country`, `travel_purpose`, `trips_12m`, `group_size`, `travel_mode`, `accommodation_type`, `checkout_gap_hours`, `luggage_management_hours`, `bag_count`, `problem_frequency`, `existing_storage_methods`, `past_spend_inr`, `inconvenience_score`, `storage_search_difficulty`, `existing_barriers`, `maximum_detour`, `acceptable_venue_types`, `security_requirements`, `booking_intent_score`, `first_service_preference`, `preferred_pricing_model`, `storage_wtp_inr`, `pickup_delivery_wtp_inr`, `preferred_features`, `main_nonuse_barrier`, `pilot_interest`
- **Default provenance:** `synthetic`

### Consumer survey — cleaned

- **File name(s):** `consumer_survey_clean.csv` or `consumer_survey_cleaned.csv`
- **Purpose:** Main consumer analysis dataset (all consumer pages).
- **Used on:** All consumer pages
- **Required columns:** none
- **Expected columns:** `respondent_id`, `country`, `city`, `locality`, `microzone`, `age_group`, `residence_country`, `travel_purpose`, `trips_12m`, `group_size`, `travel_mode`, `accommodation_type`, `checkout_gap_hours`, `luggage_management_hours`, `bag_count`, `problem_frequency`, `existing_storage_methods`, `past_spend_inr`, `inconvenience_score`, `storage_search_difficulty`, `existing_barriers`, `maximum_detour`, `acceptable_venue_types`, `security_requirements`, `booking_intent_score`, `first_service_preference`, `preferred_pricing_model`, `storage_wtp_inr`, `pickup_delivery_wtp_inr`, `preferred_features`, `main_nonuse_barrier`, `pilot_interest`
- **Default provenance:** `synthetic`

## Market, partner & geographic evidence

### Partner survey

- **File name(s):** `partner_survey.csv`
- **Purpose:** Potential storage-partner survey (P1–P14).
- **Used on:** Partner Network, Geographic
- **Required columns:** none
- **Expected columns:** `partner_id`, `establishment_type`, `city`, `microzone`, `latitude`, `longitude`, `opening_time`, `closing_time`, `secure_storage_available`, `storage_capacity`, `cctv_available`, `security_features`, `partner_willingness`, `commercial_preference`, `expected_payment_inr`, `digital_handover_acceptance`, `pickup_delivery_acceptance`, `partner_barriers`, `pilot_participation`, `evidence_status`
- **Default provenance:** `unclassified`

### Partner audit

- **File name(s):** `partner_audit.csv`
- **Purpose:** Field assessment of local establishments: availability, capacity, verification status.
- **Used on:** Partner Network, Geographic
- **Required columns:** none
- **Expected columns:** `partner_id`, `establishment_type`, `city`, `microzone`, `latitude`, `longitude`, `opening_time`, `closing_time`, `secure_storage_available`, `storage_capacity`, `cctv_available`, `security_features`, `partner_willingness`, `commercial_preference`, `expected_payment_inr`, `digital_handover_acceptance`, `pickup_delivery_acceptance`, `partner_barriers`, `pilot_participation`, `evidence_status`
- **Default provenance:** `unclassified`

### Competitor locations

- **File name(s):** `competitor_locations.csv`
- **Purpose:** Existing storage providers with evidence status and source URL.
- **Used on:** Geographic
- **Required columns:** none
- **Expected columns:** `competitor_id`, `competitor_name`, `city`, `microzone`, `latitude`, `longitude`, `service_type`, `competitor_category`, `hourly_price`, `daily_price`, `storage_types`, `operating_hours`, `pickup_available`, `delivery_available`, `verification_date`, `source_url`, `evidence_status`
- **Default provenance:** `unclassified`

### Geographic market audit

- **File name(s):** `geographic_market_audit.csv`
- **Purpose:** Travel nodes (airports, stations, tourist areas) and microzone reference points.
- **Used on:** Geographic
- **Required columns:** none
- **Expected columns:** `city`, `microzone`, `node_name`, `node_type`, `latitude`, `longitude`, `nearest_storage_option`, `nearest_storage_distance_m`, `distance_type`, `evidence_status`, `source_url`
- **Default provenance:** `unclassified`

## Group-stage model outputs

### Classification metrics

- **File name(s):** `classification_metrics.csv`
- **Purpose:** One row per model: accuracy, precision, recall, F1, ROC-AUC, PR-AUC.
- **Used on:** Classification
- **Required columns:** `model`
- **Expected columns:** `model`, `accuracy`, `precision`, `recall`, `f1`, `roc_auc`, `pr_auc`, `target`, `split`
- **Default provenance:** `inherit:consumer_clean`

### Classification predictions

- **File name(s):** `classification_predictions.csv`
- **Purpose:** Actual vs predicted High_Intent per respondent and model (probability optional).
- **Used on:** Classification
- **Required columns:** `actual`, `predicted`
- **Expected columns:** `respondent_id`, `model`, `actual`, `predicted`, `probability`, `target`, `split`
- **Default provenance:** `inherit:consumer_clean`

### Cluster assignments

- **File name(s):** `cluster_assignments.csv`
- **Purpose:** Respondent-level cluster IDs from the clustering pipeline.
- **Used on:** Customer Segmentation
- **Required columns:** `respondent_id`, `cluster`
- **Expected columns:** `respondent_id`, `cluster`
- **Default provenance:** `inherit:consumer_clean`

### Cluster profiles

- **File name(s):** `cluster_profiles.csv`
- **Purpose:** Cluster summary table produced by the clustering pipeline (shown as supplied).
- **Used on:** Customer Segmentation
- **Required columns:** `cluster`
- **Expected columns:** `cluster`
- **Default provenance:** `inherit:consumer_clean`

### Association rules

- **File name(s):** `association_rules.csv`
- **Purpose:** Rules with antecedents, consequents, support, confidence and lift.
- **Used on:** Association Mining
- **Required columns:** `antecedents`, `consequents`, `support`, `confidence`, `lift`
- **Expected columns:** `antecedents`, `consequents`, `support`, `confidence`, `lift`, `count`
- **Default provenance:** `inherit:consumer_clean`

### Regression metrics

- **File name(s):** `regression_metrics.csv`
- **Purpose:** One row per model × target: MAE, RMSE, R².
- **Used on:** Pricing & Regression
- **Required columns:** `model`
- **Expected columns:** `model`, `target`, `mae`, `rmse`, `r2`, `split`
- **Default provenance:** `inherit:consumer_clean`

### Regression predictions

- **File name(s):** `regression_predictions.csv`
- **Purpose:** Actual vs predicted willingness to pay per respondent, model and target.
- **Used on:** Pricing & Regression
- **Required columns:** `actual`, `predicted`
- **Expected columns:** `respondent_id`, `model`, `target`, `actual`, `predicted`, `split`
- **Default provenance:** `inherit:consumer_clean`

### Regression coefficients

- **File name(s):** `regression_coefficients.csv`
- **Purpose:** Optional: model, target, feature, coefficient (for the coefficient chart).
- **Used on:** Pricing & Regression
- **Required columns:** `feature`, `coefficient`
- **Expected columns:** `model`, `target`, `feature`, `coefficient`
- **Default provenance:** `inherit:consumer_clean`

## Pilot / transaction data

### Booking funnel

- **File name(s):** `booking_funnel.csv`
- **Purpose:** Actual pilot/analytics data: either stage + count rows, or one row per event with a stage column.
- **Used on:** Business Strategy
- **Required columns:** none
- **Expected columns:** `stage`, `count`, `city`, `microzone`, `date`, `booking_revenue`, `partner_payment`, `transport_cost`, `payment_fee`, `other_variable_cost`
- **Default provenance:** `transaction`

## Supporting documentation

### Data-cleaning log

- **File name(s):** `data_cleaning_log.csv` or `cleaning_log.csv`
- **Purpose:** Optional record of each cleaning decision (step, column, action, rows_affected, reason).
- **Used on:** Data Quality
- **Required columns:** none
- **Expected columns:** `step`, `column`, `action`, `rows_affected`, `reason`
- **Default provenance:** `inherit:consumer_clean`

## Accepted column-name aliases

Names are compared after lower-casing and replacing non-alphanumerics with `_`. Questionnaire numbers (e.g. `Q11`, `q11_checkout_gap`) are recognised for consumer columns.

**consumer**

| canonical | also accepted |
|---|---|
| `respondent_id` | `id`, `response_id`, `respondentid`, `resp_id` |
| `country` | `q3_country` |
| `residence_country` | `country_of_residence`, `home_country` |
| `city` | `visit_city`, `destination_city`, `q4`, `q4_city` |
| `locality` | `q5`, `q5_locality`, `travel_area`, `area` |
| `microzone` | `micro_zone`, `zone`, `research_zone` |
| `age_group` | `age`, `q2`, `q2_age_group` |
| `travel_purpose` | `purpose`, `q6`, `q6_travel_purpose` |
| `trips_12m` | `trips`, `trips_last_12m`, `trips_in_12_months`, `travel_frequency`, `q7` |
| `group_size` | `travel_group`, `q8` |
| `travel_mode` | `transport_mode`, `mode_of_transport`, `q9` |
| `accommodation_type` | `accommodation`, `q10` |
| `checkout_gap_hours` | `checkout_gap`, `q11`, `checkout_departure_gap` |
| `luggage_management_hours` | `luggage_management_duration`, `q12`, `luggage_hours` |
| `bag_count` | `bags`, `number_of_bags`, `q13` |
| `problem_frequency` | `gap_frequency`, `q14` |
| `existing_storage_methods` | `storage_methods`, `luggage_methods`, `q15` |
| `past_spend_inr` | `past_spend`, `q16` |
| `inconvenience_score` | `inconvenience`, `q17` |
| `storage_search_difficulty` | `search_difficulty`, `q18` |
| `existing_barriers` | `barriers`, `storage_barriers`, `q19` |
| `maximum_detour` | `max_detour`, `detour`, `q20` |
| `acceptable_venue_types` | `acceptable_venues`, `venue_types`, `q21` |
| `security_requirements` | `security_features`, `q22` |
| `booking_intent_score` | `booking_intent`, `intent_score`, `q23` |
| `first_service_preference` | `first_service`, `q24` |
| `preferred_pricing_model` | `pricing_model`, `pricing_preference`, `q25` |
| `storage_wtp_inr` | `storage_wtp`, `wtp_storage`, `wtp_storage_inr`, `q26` |
| `pickup_delivery_wtp_inr` | `pickup_delivery_wtp`, `delivery_wtp`, `wtp_pickup_delivery_inr`, `q27` |
| `preferred_features` | `features`, `q28` |
| `main_nonuse_barrier` | `nonuse_barrier`, `main_barrier`, `q29` |
| `pilot_interest` | `pilot`, `q30` |
| `high_intent` | `high_intent_flag` |
| `data_source` | `provenance`, `source_type` |

**partner**

| canonical | also accepted |
|---|---|
| `partner_id` | `id`, `establishment_id` |
| `establishment_type` | `type`, `partner_type`, `p1` |
| `city` | `p2_city` |
| `microzone` | `locality`, `micro_zone`, `zone`, `p2_locality` |
| `latitude` | `lat` |
| `longitude` | `lon`, `lng`, `long` |
| `opening_time` | `open_time`, `opens` |
| `closing_time` | `close_time`, `closes` |
| `secure_storage_available` | `secure_storage`, `storage_area`, `p5` |
| `storage_capacity` | `capacity`, `bag_capacity`, `p6` |
| `cctv_available` | `cctv` |
| `security_features` | `security_facilities`, `p7` |
| `partner_willingness` | `willingness`, `store_non_customers`, `p8` |
| `commercial_preference` | `preferred_commercial_model`, `commercial_model`, `p9` |
| `expected_payment_inr` | `expected_payment`, `p10` |
| `digital_handover_acceptance` | `digital_handover`, `p11` |
| `pickup_delivery_acceptance` | `pickup_delivery`, `courier_support`, `p12` |
| `partner_barriers` | `barriers`, `concerns`, `p13` |
| `pilot_participation` | `pilot_interest`, `pilot`, `p14` |
| `current_storage_offer` | `p4`, `currently_provides_storage` |
| `evidence_status` | `verification_status`, `status`, `partner_status`, `verified` |

**competitor**

| canonical | also accepted |
|---|---|
| `competitor_id` | `id` |
| `competitor_name` | `name`, `provider`, `provider_name` |
| `microzone` | `locality`, `micro_zone`, `zone`, `area` |
| `latitude` | `lat` |
| `longitude` | `lon`, `lng`, `long` |
| `competitor_category` | `category`, `provider_category` |
| `operating_hours` | `opening_hours`, `hours` |
| `verification_date` | `verified_date`, `date_verified` |
| `source_url` | `url`, `source` |
| `evidence_status` | `verification_status`, `status`, `verified` |

**geo**

| canonical | also accepted |
|---|---|
| `microzone` | `locality`, `micro_zone`, `zone` |
| `node_name` | `name`, `location_name`, `facility` |
| `node_type` | `type`, `category`, `facility_type` |
| `latitude` | `lat` |
| `longitude` | `lon`, `lng`, `long` |
| `evidence_status` | `verification_status`, `status`, `verified` |
| `source_url` | `url`, `source` |

**model**

| canonical | also accepted |
|---|---|
| `respondent_id` | `id`, `response_id` |
| `model` | `model_name`, `algorithm`, `classifier`, `regressor` |
| `accuracy` | `acc` |
| `precision` | `prec` |
| `recall` | `sensitivity`, `tpr` |
| `f1` | `f1_score`, `f1score` |
| `roc_auc` | `auc`, `rocauc`, `auc_roc` |
| `pr_auc` | `average_precision`, `prauc`, `auc_pr`, `ap` |
| `actual` | `y_true`, `true`, `actual_label`, `y_test`, `true_label`, `actual_value`, `y_actual` |
| `predicted` | `y_pred`, `prediction`, `predicted_label`, `pred`, `predicted_value` |
| `probability` | `y_prob`, `prob`, `proba`, `predicted_proba`, `score`, `y_score`, `probability_1` |
| `cluster` | `cluster_id`, `cluster_label`, `segment`, `kmeans_cluster` |
| `antecedents` | `antecedent`, `lhs` |
| `consequents` | `consequent`, `rhs` |
| `count` | `support_count`, `n_respondents`, `respondents`, `supporting_respondents` |
| `mae` | `mean_absolute_error` |
| `rmse` | `root_mean_squared_error` |
| `r2` | `r_squared`, `r2_score`, `rsquared` |
| `target` | `target_variable`, `dependent_variable`, `y` |
| `feature` | `variable`, `predictor`, `term` |
| `coefficient` | `coef`, `estimate`, `weight` |
| `split` | `dataset`, `data_split`, `set` |

**funnel**

| canonical | also accepted |
|---|---|
| `stage` | `funnel_stage`, `event`, `step` |
| `count` | `n`, `users`, `sessions`, `events` |
| `booking_revenue` | `revenue` |
| `partner_payment` | `partner_payout` |
| `transport_cost` | `courier_cost`, `delivery_cost` |
| `payment_fee` | `payment_fees`, `processing_fee` |
| `other_variable_cost` | `other_cost` |

## Answer formats

### Ordered scales (text answers or numbers)

Text answers are matched to the questionnaire wording (case, dashes and trailing full stops are ignored). Where a numeric column is supplied instead, it is banded with the bins below. Special answers such as 'Not applicable' are kept as their own category and never converted to zero or missing.

- `checkout_gap_hours`: No gap < < 1 h < 1–2 h < 3–4 h < 5–6 h < 7–8 h < > 8 h · special: Not applicable
- `luggage_management_hours`: Did not manage < < 1 h < 1–2 h < 3–4 h < 5–6 h < 7–8 h < > 8 h · special: Not applicable
- `bag_count`: 1 < 2 < 3 < 4+ · special: Not applicable
- `problem_frequency`: Never < Rarely < Occasionally < Frequently < Almost every trip
- `trips_12m`: 0 < 1 < 2–3 < 4–6 < 7+
- `group_size`: Alone < 2 people < 3–4 < 5–6 < 7+
- `age_group`: 18–24 < 25–34 < 35–44 < 45–54 < 55+ · special: Prefer not to disclose
- `storage_search_difficulty`: Very easy < Easy < Neutral < Difficult < Very difficult · special: Unaware of any facility, Did not search
- `maximum_detour`: No extra distance < ≤ 5-min walk < ≤ 10-min walk < ≤ 15-min walk < ≤ 30-min by vehicle · special: Pickup from accommodation only
- `past_spend_inr`: ₹0 < ₹1–100 < ₹101–250 < ₹251–500 < ₹501–1,000 < > ₹1,000 · special: Did not use paid service, Do not remember
- `pilot_interest`: No < Maybe < Yes
- `partner_willingness`: No < Maybe (compensation) < Yes, with security procedures < Definitely yes
- `digital_handover_acceptance`: No < Maybe < Yes, with training < Yes
- `pickup_delivery_acceptance`: No < Storage only < Receive/release for couriers < Also arrange transport
- `pilot_participation`: No < Maybe < Yes
- `secure_storage_available`: No suitable space < Behind counter (limited) < Restricted-access area < Dedicated storage room

### Numeric scores

- `inconvenience_score`: 1–5 (a leading number such as `4 — Significantly` is accepted)
- `booking_intent_score`: 0–10 (a leading number such as `4 — Significantly` is accepted)
- `storage_wtp_inr`, `pickup_delivery_wtp_inr`: numbers in ₹ (₹ signs and commas are stripped). Zero is a valid answer; blank stays missing; negative values are flagged invalid.

### Multi-select questions

Either one text column with options separated by `;` (or `|`), or one binary column per option named `<question>__<option>` (values 0/1, true/false, yes/no). Recognised options:

- `existing_storage_methods`: Hotel / accommodation held it; Airport storage; Railway cloakroom; Paid storage / smart locker; Friends / relatives; Carried it; Kept in car / taxi; Changed itinerary; Other
- `existing_barriers`: No suitable facility nearby; Facility too far from route; Hard to locate / book; Prices too high; Inconvenient hours; Uncomfortable with third party; Security unclear; Loss / damage concern; No pickup / delivery; Free hotel storage sufficient; Never experienced difficulties; Other
- `acceptable_venue_types`: Verified hotels; Verified restaurants / cafés; Verified retail stores; Dedicated storage facilities; Smart lockers; Airport / railway facilities; Any registered business meeting standards; None — own accommodation / official only
- `security_requirements`: Verified partners with address; CCTV; Tamper-evident seals; Photos at drop-off / collection; QR / luggage ID; Loss & damage protection terms; Real-time status; Reviews & ratings; 24/7 support; Individual compartments; ID check at collection; Transparent pricing / refunds; Complaints / dispute process
- `preferred_features`: Real-time tracking; Digital ID & security seal; Photo proof of handover; Scheduled collection; Scheduled delivery to airport/station; Multi-bag / family packages; Group booking; Flexible extensions; Insurance / loss protection; Priority support; Route-based recommendations; Repeat-customer discounts; None of these
- `security_features`: CCTV; Restricted staff-only access; Dedicated storage room; Individual lockers / compartments; Staff record handovers; Digital access / ID systems; Fire & emergency safety; None of these
- `partner_barriers`: Lack of space; Security risks; Liability for loss / damage; Additional staff workload; Insurance / regulatory; Low expected earnings; Disruption to customers; Uncertain demand; Other

## Model-output conventions

- Classification target: High_Intent = 1 when `booking_intent_score` is 8–10, 0 when 0–7 (stated intent). `actual`/`predicted` may be 0/1, true/false, yes/no or high/low. Add a `target` column to separate an actual-booking target from survey intent; add a `features` column to `classification_metrics.csv` to enable the leakage check (Q23–Q30 must not be predictors).
- Optional `split` column (e.g. train/test) — the dashboard defaults to `test`.
- Regression `target` values must mention storage or pickup/delivery.
- Association rules accept mlxtend `frozenset({...})` strings or `a; b` lists.
- `booking_funnel.csv`: either `stage,count` rows or one row per event with a `stage` column; optional `booking_revenue, partner_payment, transport_cost, payment_fee, other_variable_cost` enable actual contribution margins.
