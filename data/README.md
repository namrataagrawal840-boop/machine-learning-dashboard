# data/

Put the prepared project files here (sub-folders are fine). The dashboard only **reads** them — it never cleans,
modifies or overwrites a file, and it never generates records to fill gaps.

| File | Produced by | Used on |
|---|---|---|
| `consumer_survey_raw.csv` or `consumer_survey_raw_synthetic.csv` | synthetic-data generation | Data Quality |
| `consumer_survey_clean.csv` | data-cleaning pipeline | all consumer pages |
| `data_cleaning_log.csv` (optional) | data-cleaning pipeline | Data Quality |
| `partner_survey.csv` | partner survey (P1–P14) | Partner Network, Geographic |
| `partner_audit.csv` | field audit | Partner Network, Geographic |
| `competitor_locations.csv` | verified market research | Geographic |
| `geographic_market_audit.csv` | verified market research | Geographic |
| `classification_metrics.csv`, `classification_predictions.csv` | classification pipeline | Classification |
| `cluster_assignments.csv`, `cluster_profiles.csv` | clustering pipeline | Customer Segmentation |
| `association_rules.csv` | association-mining pipeline | Association Mining |
| `regression_metrics.csv`, `regression_predictions.csv`, `regression_coefficients.csv` (optional) | regression pipeline | Pricing & Regression |
| `booking_funnel.csv` | real pilot / analytics data only | Strategy & Sales Pipeline |

Full column list and accepted aliases: [`docs/DATA_CONTRACT.md`](../docs/DATA_CONTRACT.md).

Declare where each file comes from in `sources.toml` (copy `sources.example.toml`).

To keep data outside the repository, set `LUGGAGE_DATA_DIR=/path/to/folder` before running Streamlit, or upload
files for a single session on the dashboard's **Data sources & setup** page.
