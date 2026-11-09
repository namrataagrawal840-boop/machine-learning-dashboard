# Synthetic data and model outputs

**Everything in `data/` is simulated.** It is hypothetical data created to test whether the business idea can be analysed. It is not evidence about real travellers, businesses or markets. Every record carries `data_source = synthetic`, `data/sources.toml` declares the provenance, and the dashboard shows a SYNTHETIC badge on every figure.

To regenerate (the output is identical each time because the seeds are fixed):

```bash
pip install -r requirements-analysis.txt
python -m analysis.run_pipeline
```

## Files produced

| File | Rows | How it is made |
|---|---|---|
| `consumer_survey_raw_synthetic.csv` | 2,042 | `analysis/synthetic.py`: 2,000 simulated respondents, plus deliberately injected data-quality problems |
| `consumer_survey_clean.csv` | 1,975 | `analysis/cleaning.py`. Every decision is in `data_cleaning_log.csv` |
| `partner_survey.csv` | 240 | Anonymous simulated establishments: 15 per microzone, P1–P14. No names, no coordinates |
| `classification_*.csv` | — | `analysis/models.py`. Five classifiers, actually trained on the clean data |
| `cluster_*.csv` | — | K-means; k chosen by silhouette score (k = 3–6) |
| `association_rules.csv` | 400 | Apriori (mlxtend). Support ≥ 0.05, confidence ≥ 0.35, lift > 1 |
| `regression_*.csv` | — | Linear, RidgeCV and LassoCV, run separately for each WTP target |

The model outputs come from real model runs on the synthetic data. None of the numbers were typed in. Because the data is simulated, the models recover the relationships that were built into the simulation, not facts about the market.

## Consumer design (follows the brief)

- **Sample:** 500 respondents per city (Goa, Jaipur, Delhi, Mumbai). This balanced allocation is experimental and does not represent market size.
- **Microzones:** respondents are spread across the four research microzones per city in the brief, weighted 30 / 25 / 25 / 20 %.
- **Hidden traveller types:** business, leisure, backpacker, family, and visiting family or pilgrimage. These drive correlated answers. The type is not exported, so clustering has to find structure on its own.
- **City mixes are assumptions:** for example, Goa leans leisure, flights and homestays; Delhi and Mumbai lean business, flights and trains; Jaipur leans heritage leisure and trains.
- **Planning-only travellers:** about 9% have only planned trips. They answer Q11 and Q12 as "Not applicable", leave Q14–Q19 blank, and are flagged by `past_trip_experience`. "Not applicable" therefore stays distinct from missing.
- **Built-in relationships (assumptions, chosen to be plausible):**
  - Hotels and serviced apartments usually hold luggage; Airbnb and homestays rarely do.
  - Flights and trains are associated with longer checkout gaps than personal vehicles.
  - Time spent managing luggage is never longer than the gap.
  - Inconvenience rises with the time spent carrying luggage, the number of bags, and actually carrying them.
  - Storage awareness is lower for international and infrequent travellers.
  - Booking intent rises with inconvenience, difficulty finding storage, acceptance of partner venues, long gaps and trust; it is lower for the visiting-family/pilgrimage type.
  - WTP is log-normal, with medians that vary by traveller type and city (Mumbai highest, Jaipur lowest). About 11% say ₹0, more often when intent is low, and a few legitimately extreme values (₹1,500–5,000) are included.
  - Pickup/delivery WTP is correlated with storage WTP and is higher for travellers who prefer pickup.
- **Questionnaire rules respected:**
  - Q19 allows up to 3 answers and Q28 up to 4.
  - "Never experienced difficulties" and "None of these" are exclusive answers.
  - Past spend is consistent with whether a paid storage method was used.
- **Raw-file problems injected for the cleaning exercise:**
  - 30 duplicate submissions.
  - 12 screened-out respondents (Q1 = "Neither").
  - 25 abandoned responses.
  - Inconsistent capitalisation and spacing.
  - WTP typed as "₹200" or "200 rs", negative WTP values, and out-of-range intent scores (11, 12, 15, −1).
  - Scattered unanswered questions.

## Partner design

- **Establishment mix:** varies by zone. Transit zones have more dedicated storage facilities.
- **Linked answers:** storage space determines capacity and willingness; opening hours depend on establishment type (hotels are often open 24 hours); security facilities depend on the space available.
- **Expected payment:** the unit (per bag-hour, per bag-day or per booking) is recorded in `expected_payment_unit`.
- **Not verified:** no record carries a "Verified" status, because nothing has been field-verified. The dashboard therefore correctly reports zero verified operational partners.

## Deliberately NOT generated

- **`competitor_locations.csv` and `geographic_market_audit.csv`:** the brief forbids inventing competitor information or coordinates. These must come from verified research, so the map and market-gap evidence stay "pending".
- **`partner_audit.csv`:** this is a record of physical field checks.
- **`booking_funnel.csv`:** this must contain actual pilot transactions only.

## Results on this data (test split)

| Analysis | Result |
|---|---|
| Classification (High_Intent ≈ 28%) | Gradient Boosting ROC-AUC 0.79; Logistic Regression F1 0.48 |
| Clustering | k = 4, silhouette 0.14 (weak separation, as expected from overlapping traveller types) |
| Regression | Ridge R² ≈ 0.15 for storage WTP and ≈ 0.04–0.06 for pickup WTP. Stated WTP is mostly individual noise in this simulation |
