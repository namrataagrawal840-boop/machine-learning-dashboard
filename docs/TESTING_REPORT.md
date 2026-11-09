# Functional testing report

Date: 10 Nov 2026 · Branch: `feature/streamlit-dashboard`

## Environments

| Stack | Result |
|---|---|
| Python 3.13 · Streamlit 1.67 · pandas 3.0.5 · numpy 2.5 · plotly 7.1 · scipy 1.18 | 19 / 19 tests pass |
| Python 3.13 · Streamlit 1.67 · pandas 2.2.3 · numpy 2.0 · plotly 5.24 | 19 / 19 tests pass |

## Automated tests (`python -m pytest -q tests`)

**Unit tests (`tests/test_parsing.py`).** These cover ordinal banding and 'Not applicable' handling, numeric-to-band conversion, 5- versus 15-minute detour parsing, score ranges, multi-select parsing (where an unanswered question stays missing), column aliases including Q-numbers, zero WTP being kept while blank WTP is never turned into zero, time parsing, Cramér's V bounds and association-rule item parsing.

**End-to-end tests (`tests/test_app.py`).** Every page is rendered with Streamlit's AppTest. A test fails on any uncaught exception or any error notice the dashboard raises itself. Scenarios:

1. No data at all (the repository's current state).
2. Full fixture set: consumer data (raw and clean), partner survey and audit, competitors, geographic audit, all model outputs and a booking funnel, with all 10 diagnostic questions shown.
3. Consumer data only.
4. Missing columns plus renamed columns (`Q11`, `Booking Intent`).
5. Binary-encoded multi-select answers, numeric hours and bag counts, an out-of-range score, and a case variant.
6. An empty consumer file, a malformed rules file and unreadable labels. The app must not crash, and must show the expected error notice.
7. Tight filters (one city, one microzone, 7+ trips) with a minimum sample of 200.
8. A stale microzone filter combined with a city change.
9. Source CSVs are byte-identical before and after rendering every page.

10. The committed synthetic datasets and model outputs render on every page without errors, and every survey
    answer is recognised (`test_repository_data_renders_cleanly`). Total: 20 / 20 pass.

## Manual browser checks

All 11 pages were checked in headless Chrome at a 1440 px width with the fixture data. No exceptions occurred, cards and explanations rendered, and value labels were not clipped. The Overview, Geographic and Diagnostic pages were also checked at 420 px: the cards stack and the sidebar collapses.

## Coverage of the brief's checklist

Every item below was covered: startup, every page, missing files, empty files, missing columns, unexpected categories, missing numbers, filters and reset, small samples, multi-select answers, city and microzone filters, sortable and searchable tables with CSV export, model outputs, the confusion matrix, cluster profiles, rule filtering, regression residuals and the narrow layout. Widget keys are unique; a duplicate key would raise an exception in these tests. Upload handling only reruns the app when a file changes, so it cannot loop.

## Not verified here

- **Street-map tiles.** The test sandbox blocks the tile server and has no WebGL, so the tile map could not be seen rendering. The marker and table logic is tested. An offline "coordinate plot" base map is available for when tiles cannot load.
- **Real data.** None is available; `data/` holds synthetic data only (see `docs/SYNTHETIC_DATA.md`).
- **Streamlit Community Cloud.** Deployment has not been performed.
