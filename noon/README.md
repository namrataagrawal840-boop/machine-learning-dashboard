# noon · Customer Experience Analytics (Group 2)

Interactive Streamlit + Plotly dashboard for the Group 2 Data Analytics project, *Data-Driven Business Solutions for noon*.
It covers three linked customer problems and the solution for each:

| Problem | Solution | Techniques shown |
|---|---|---|
| 1. Late, unreliable deliveries | Delivery Promise & Recovery Engine | XGBoost / random forest / logistic classification, XGBoost + P90 quantile regression, K-Means (sellers, zones), Apriori, two-model uplift, A/B test, demand forecast |
| 2. Slow, complicated refunds | Trust-Tiered Instant Refunds | Random forest trust tiers, K-Means on RFM + returns, Apriori, linear regression (refund time, keep-it cost), A/B test with fraud guardrail |
| 3. Poor complaint resolution | Case Ownership Engine | TF-IDF + logistic regression, TF-IDF + K-Means themes, resolution-time regression, churn classification, decision tree, market response model |

It also has pages for market and competitor research, data cleaning, the shared customer models (segments, CLV, churn,
next purchase) and the business case (nine pillars, KPIs, gain calculator, A/B sample size).

## Simulation (professor's suggestions)
* **8 · Solution Simulator** — pick Problem 1, 2 or 3 and go through four steps: the problem in detail, the data points
  captured, descriptive analytics, then **Run the simulation**. The run replays records the models never trained on
  (e.g. 7,592 Aug–Oct orders) twice — today's process and the solution — record by record with a live feed, and ends with a
  today-vs-solution result. Numbers are fixed (same records, same random draws); assumptions can be changed in an expander.
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
  (30,000 orders, ~4,900 returns, ~8,500 tickets, ~8,100 customers in 22 UAE neighbourhoods, Nov 2025 – Oct 2026), using
  cause-and-effect rules plus deliberate data-entry errors for the cleaning step. Fixed seed, so it is reproducible.
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

Rebuild the data and models (about 30 seconds):
```bash
pip install -r requirements-analysis.txt
python -m analysis.run_pipeline
python -m pytest -q tests      # every page renders, with and without filters
```

## Deploy on Streamlit Community Cloud
share.streamlit.io → **Create app** → repository `namrataagrawal840-boop/machine-learning-dashboard`, branch `main`,
main file path **`noon/streamlit_app.py`**. Streamlit uses `noon/requirements.txt` because it sits next to the app file.

## Layout
```
noon/
  streamlit_app.py        entry point and navigation
  noonbi/                 dashboard code (theme, cards, charts, data access, one module per page)
  analysis/               synthetic data, cleaning, models, market research, simulator records, pipeline runner
  live/                   live platform engine (database, events, alerts) and the Docker worker
  data/                   raw/ (messy exports), clean CSVs, results/ (model outputs), research/ (secondary sources)
  assets/                 CSS and logo (same visual system as the luggage dashboard)
  tests/                  AppTest end-to-end tests (pages, simulator runs, live platform)
  Dockerfile, docker-compose.yml   live platform on Docker Desktop
```
