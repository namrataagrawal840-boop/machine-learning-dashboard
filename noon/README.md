# noon · Customer Experience Analytics (Group 2)

Interactive Streamlit + Plotly dashboard for the Group 2 Data Analytics project, *Data-Driven Business Solutions for noon*.
It covers three linked customer problems and the solution for each:

| Problem | Solution | Techniques shown |
|---|---|---|
| 1. Late, unreliable deliveries | Delivery Promise & Recovery Engine | XGBoost / random forest / logistic classification, XGBoost + P90 quantile regression, K-Means (sellers, zones), Apriori, two-model uplift, A/B test, demand forecast (Seasonal ARIMA + regression) |
| 2. Slow, complicated refunds | Trust-Tiered Instant Refunds | Cross-validated, calibrated trust tiers (best of logistic / random forest / XGBoost), cost-based cut-off under a 0.5% fraud budget, K-Means on RFM + returns, Apriori, linear regression (refund time, keep-it cost), A/B test with fraud guardrail |
| 3. Poor complaint resolution | Case Ownership Engine | TF-IDF + logistic regression, TF-IDF + K-Means themes, resolution-time regression, churn classification, decision tree, market response model |

It also has pages for market and competitor research, data cleaning, the shared customer models (segments, CLV, churn,
next purchase), the business case (nine pillars, KPIs, gain calculator, A/B sample size) and **11 · Model Quality &
Fairness** (how the data was made realistic, cross-validation, calibration, cost-based thresholds, Qini, bootstrap CIs,
fairness by account age and emirate, forecast back-test, and the simulator's response models).

## Simulation (professor's suggestions)
* **8 · Solution Simulator** — pick Problem 1, 2 or 3 and go through four steps: the problem in detail, the data points
  captured, descriptive analytics, then **Run the simulation**. The run replays records the models never trained on
  (e.g. 7,597 Aug–Oct orders) twice — today's process and the solution — record by record with a live feed, and ends with a
  today-vs-solution result. Numbers are fixed (same records, same random draws); assumptions can be changed in an expander.
  Customer responses (repeat buying, complaints, CSAT, churn) come from models **fitted on the recorded data**, not from the
  generator's formulas, and every gain has an 80% band from 100 bootstrap refits.
* **9 · Live Platform** — a running mini-noon: 1,000 products, 3,000 customers, 120 sellers. Orders, deliveries, returns,
  tickets and **alerts** are written to a real database as simulated time passes, in two copies (today vs solutions) built
  from the same seed. Controls: run/pause, speed, Yellow Friday surge, +1 day, reset, plus a read-only SQL box.
  On Streamlit Cloud each browser session gets its own private database.

### Live platform on Docker Desktop
```bash
cd noon
docker compose up --build      # simulator container + dashboard container + shared database volume
# open http://localhost:8501 → "9 · Live Platform"
```
The `simulator` container (`python -m live.worker`) keeps the platform running and writing to the database even with the
browser closed; the `dashboard` container only reads it and sends controls (run/pause, speed, surge, reset) back.

## Where the numbers come from
* **Synthetic** — noon's orders, returns and tickets are not public. `analysis/synthetic.py` builds a realistic stand-in
  (30,000 orders, ~4,620 returns, ~8,622 tickets, ~8,140 customers in 22 UAE neighbourhoods, Nov 2025 – Oct 2026). Fixed seed, so it is reproducible.
  The data is built so it does **not** hand the models their answers: outcomes also depend on hidden causes the models never
  see (loyalty, life events, driver skill, depot load, inspector strictness, complaining habit), there is recording noise and
  wrong labels (6% of ticket topics, 4% of address flags), drift from August 2026 (a faster courier, 40 new sellers), missing
  values that carry meaning (no photo flag on the old app; unhappy customers skip CSAT), messy ticket text (typos, Arabizi,
  two issues in one message), and churn/CLV read off a separately simulated future year. Hidden columns are dropped before
  any model runs. Deliberate data-entry errors are added for the cleaning step.
* **Secondary** — market size, competitor facts, policies and review ratings in `analysis/market_research.py`, each with its source URL.
* **Model output** — everything in `data/results/`, computed from the synthetic data.
* **Assumption** — the gain calculator defaults, taken from the group's proposal.

Each number in the dashboard carries one of these badges.

## Run it
```bash
cd noon
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Rebuild the data, models and validation (about 2 minutes):
```bash
pip install -r requirements-analysis.txt
python -m analysis.run_pipeline
python -m pytest -q tests      # pages render; no leakage; realistic scores; fraud cap; reproducible simulator
```

## Deploy on Streamlit Community Cloud
share.streamlit.io → **Create app** → repository `namrataagrawal840-boop/machine-learning-dashboard`, branch `main`,
main file path **`noon/streamlit_app.py`**. Streamlit uses `noon/requirements.txt` because it sits next to the app file.

## Simple view and full detail
The dashboard opens in a **simple view**: four headline numbers per page, one *Key takeaway* line under each chart, and
the details folded into "More detail" sections. Switch on **Show full detail** in the sidebar to see every explanation
box, the data table behind each chart and the source badges. The ⓘ icon on every chart works in both views.

## Headline results on the realistic data
| Check | Result |
|---|---|
| Late-delivery model (test months) | ROC-AUC ≈ 0.77; P90 promise kept 73% → 96% |
| Refund QC model | CV ROC-AUC ≈ 0.73, test ≈ 0.78 (95% CI 0.74–0.83); Green tier 29% of returns, money at risk 0.25% of refunds (cap 0.5%) |
| A/B tests | Credit +5.5 pts (p = 0.02); instant refund +6.6 pts (p = 0.005); bootstrap agrees |
| Ticket topics | about 82% correct on messy text |
| Honest negatives | Uplift targeting does not beat random with one small test; next-purchase timing no better than guessing; new accounts are held back more often (fairness flag) |

## Layout
```
noon/
  streamlit_app.py        entry point and navigation
  noonbi/                 dashboard code (theme, cards, charts, data access, one module per page)
  analysis/               synthetic data, cleaning, models, market research, simulator records, pipeline runner
  live/                   live platform engine (database, events, alerts) and the Docker worker
  data/                   raw/ (messy exports), clean CSVs, results/ (model outputs), research/ (secondary sources)
  assets/                 CSS and logo (same visual system as the luggage dashboard)
  tests/                  AppTest end-to-end tests + model checks (leakage, realism, fraud cap, CV choice)
  Dockerfile, docker-compose.yml   live platform on Docker Desktop
```
