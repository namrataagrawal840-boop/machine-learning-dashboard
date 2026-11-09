# machine-learning-dashboard

**Luggage Marketplace BI** — an interactive Streamlit dashboard for an India-first luggage-storage and
travel-convenience marketplace (verified local storage partners, with possible scheduled pickup and delivery to
airports and stations).

The dashboard turns the project's prepared datasets and model outputs into a decision-support tool. It is
**dashboard-only**: it does not generate survey data, clean data, train models or change research conclusions.
Every figure is calculated from the files in `data/`; anything missing is shown as a clearly labelled pending
state — never filled with invented values.

## Pages

| Section | Page | What it answers |
|---|---|---|
| Individual submission — core | 1 · Overview | Size of the checkout-gap problem, unmet need, stated demand, WTP (KPI cards + profile charts) |
| | 2 · Data Quality | Duplicates, missingness map, data types, invalid answers, outlier flags, raw vs cleaned |
| | 3 · Descriptive & Diagnostic | Sections A–E, ten diagnostic questions, Pearson / Spearman / Cramér's V |
| Market & supply evidence | 4 · Geographic Opportunity | Map of real coordinates only, microzone table, adjustable opportunity score, city comparison |
| | 5 · Partner Network | Willingness, capacity, hours, security, payments, barriers; surveyed vs verified partners |
| Group-stage analytics | 6 · Customer Segmentation | Cluster sizes, profiles, intent overlay, data-generated segment interpretation |
| | 7 · Classification | Model comparison, confusion matrix, report, threshold what-if, FP/FN business meaning |
| | 8 · Association Mining | Filterable rules, strength chart, network, bundle hypotheses |
| | 9 · Pricing & Regression | Storage and pickup WTP separately; Linear / Ridge / Lasso metrics, residuals, coefficients |
| Decision | 10 · Strategy & Sales Pipeline | Evidence-graded answers, phased roadmap, survey-intent vs actual funnel, contribution calculator, validation tracker |
| Setup | Data sources & setup | Dataset status, provenance, data contract, session uploads |

Individual-assignment analysis (pages 1–3) is kept separate from group-stage analytics (pages 6–9) in the sidebar.
Every chart sits in a card with: business question · what is measured · what the data shows (computed live) ·
business implication · limitations. Every percentage shows its numerator and denominator.

## Run in GitHub Codespaces

1. On GitHub: **Code → Codespaces → Create codespace on `main`** (or on this branch).
2. The dev container installs `requirements.txt` and starts the app automatically; a preview opens on port **8501**.
3. To start it manually: `streamlit run streamlit_app.py`

## Run locally

```bash
git clone https://github.com/namrataagrawal840-boop/machine-learning-dashboard.git
cd machine-learning-dashboard
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run streamlit_app.py                           # opens http://localhost:8501
```

Python 3.11 or newer is required (3.12 recommended).

## Add the data

Copy the prepared CSVs into `data/` (see [`data/README.md`](data/README.md) and the full
[data contract](docs/DATA_CONTRACT.md)), then copy `data/sources.example.toml` to `data/sources.toml` and declare
each file's provenance (synthetic / primary survey / secondary research / pilot transactions / illustrative).
Restart is not needed — reload the page.

Alternatives: set `LUGGAGE_DATA_DIR=/path/to/folder`, or upload CSVs for one browser session on
**Data sources & setup** (held in memory only, never saved).

Column names are matched flexibly (e.g. `Q11` or `checkout_gap` → `checkout_gap_hours`). Multi-select answers can be
semicolon-separated text or binary `<question>__<option>` columns. "Not applicable" stays distinct from missing,
and zero WTP is never removed.

## Deploy to Streamlit Community Cloud

1. Merge this branch into `main` (or deploy the branch directly).
2. Go to <https://share.streamlit.io> → **Create app** → **Deploy a public app from GitHub**.
3. Repository `namrataagrawal840-boop/machine-learning-dashboard`, branch `main`, main file `streamlit_app.py`.
4. **Advanced settings → Python version 3.12** → **Deploy**.
5. Data: commit the CSVs to `data/` (public repo = public data), or keep the repo data-free and use the in-app
   upload page. Each push to the branch redeploys automatically.

## Change the look or the charts

| What | Where |
|---|---|
| Colours, chart palette, chart heights, which chart elements are editable in the browser | `dashboard/config/theme.py` |
| Card styling, spacing, typography | `assets/styles.css` |
| Streamlit theme (fonts, base colours, radius) | `.streamlit/config.toml` |
| Expected files, column aliases, answer scales, option labels | `dashboard/config/schema.py` (then run `python scripts/generate_data_contract.py`) |
| Working definitions (long gap, unmet need, partner acceptance) | `dashboard/data/transforms.py` (`_derive`) |
| Opportunity-score components and default weights | `dashboard/views/market.py` |
| Page content | `dashboard/views/pNN_*.py` |

All charts are native Plotly figures. In the browser you can hover, zoom, isolate legend items, drag/edit legends,
annotations and colour bars, draw lines/shapes on a chart (mode-bar tools), and download the chart as SVG. Every chart
has a **View, sort & download the data** panel with a CSV export.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest -q tests
```

The tests render every page with no data, partial data, renamed/missing columns, binary-encoded answers, malformed
files, tight filters and small samples, and confirm source files are never modified. They use throwaway fixtures
generated in a temporary folder (`tests/fixtures.py`) — these are not research data and are never written to `data/`.
See [`docs/TESTING_REPORT.md`](docs/TESTING_REPORT.md).

## Project structure

```
streamlit_app.py              entry point and navigation
dashboard/
  config/theme.py             colours, Plotly template, editable-chart config
  config/schema.py            data contract: files, aliases, scales, options, provenance
  data/loader.py              read-only discovery, encoding fallback, provenance resolution, uploads
  data/mapping.py             column-name normalisation and alias mapping
  data/transforms.py          answer parsing (ordinal / multi-select / numeric) and working definitions
  data/filters.py             global sidebar filters + reset
  context.py                  per-run context shared by pages
  components/cards.py         card, KPI, explanation and notice components
  components/charts.py        Plotly builders + chart/data/download wrapper
  components/stats.py         shares, Wilson CI, IQR, crosstabs, Pearson/Spearman, Cramér's V
  views/p01 … p11             one module per page; market.py = shared microzone calculations
assets/styles.css, assets/logo.svg
data/                         your CSVs (read-only) + sources.toml
docs/DATA_CONTRACT.md, docs/TESTING_REPORT.md
tests/                        unit + end-to-end tests
.devcontainer/                GitHub Codespaces configuration
```
