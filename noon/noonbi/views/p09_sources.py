"""Page 9 — sources, method notes and how to reproduce."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from noonbi import cards, charts
from noonbi import data as D

esc = cards.esc


def render(f: D.Filters) -> None:
    cards.page_header("Sources, methods & reproducibility",
                      "Where every number comes from, and how to rebuild the data and models from scratch.",
                      "Reference", [("Research", "secondary"), ("Operational data", "synthetic")])
    links = []
    for name in ("market_size", "market_facts", "noon_facts", "competitors", "benchmarks"):
        df = D.table(name)
        col = "url" if "url" in df else "source"
        for r in df.itertuples():
            url = getattr(r, col)
            label = getattr(r, "source", None) if col == "url" else getattr(r, "competitor", "")
            if isinstance(url, str) and url.startswith("http"):
                links.append({"used_for": name.replace("_", " "), "source": label or "", "url": url})
    src = pd.DataFrame(links).drop_duplicates("url")
    with cards.card(f"Secondary sources ({len(src)})"):
        st.markdown(cards.html_table([{"Used for": esc(r.used_for), "Source": esc(r.source),
                                       "Link": f'<a href="{esc(r.url)}" target="_blank">{esc(r.url[:70])}…</a>'} for r in src.itertuples()]),
                    unsafe_allow_html=True)
        charts.table(src, "src_tbl", "sources.csv")
    with cards.card("Method notes"):
        st.markdown(
            '<div class="lc-body">'
            f'<p><b>Boosting model:</b> {esc(D.summary().get("boost_model", "XGBoost"))}. Random seed {D.summary().get("seed", "")}.</p>'
            '<p><b>Testing:</b> delivery models are trained on Nov 2025 – Jul 2026 and tested on Aug – Oct 2026. Other models use '
            'a random 70/30 or 75/25 split, stratified on the outcome. Reported scores are always on the unseen part.</p>'
            '<p><b>Leakage:</b> delivery models only use what is known at checkout or dispatch (seller history from the training '
            'months, zone, item, courier, slot, address). The uplift model scores each customer with a model trained on the other half.</p>'
            '<p><b>Clustering:</b> features are standardised; the number of groups is chosen with the silhouette score and business sense.</p>'
            '<p><b>Association rules:</b> Apriori with lift above 1; longer rules are kept only when clearly stronger than the shorter '
            'rules inside them.</p>'
            '<p><b>Forecasting:</b> Seasonal ARIMA (SARIMAX, statsmodels) on log daily orders, weekly season s = 7, yearly Fourier terms and sale-event flags as inputs; order chosen by AIC on training months; residuals checked with Ljung-Box. Compared with a time-series regression and a same-weekday-last-week benchmark.</p>'
            '<p><b>A/B tests:</b> two-proportion z-test, 95% confidence interval on the difference, confirmed with a 2,000-draw bootstrap.</p>'
            '<p><b>Realistic data:</b> outcomes depend on hidden causes the models never see (loyalty, life events, driver skill, depot '
            'load, inspector strictness, complaining habit), plus recording noise, wrong labels, drift from August 2026, missing values '
            'that carry meaning, and a separately simulated future for churn and CLV. Hidden columns are removed before any model runs.</p>'
            '<p><b>Validation (page 11):</b> 5-fold cross-validation (monthly expanding window for delivery), a small hyper-parameter '
            'search chosen on cross-validated AUC, Platt calibration kept only if it lowers the out-of-fold Brier score, thresholds from '
            'costs and the 0.5% fraud budget, Qini curve for uplift, bootstrap CIs, fairness by account age and emirate, and a '
            'four-origin forecast back-test.</p>'
            '<p><b>Simulator:</b> customer responses come from logistic/linear models fitted on the recorded data with 100 bootstrap '
            'refits (uncertainty bands); the live platform scores returns with an exported logistic model on observable fields only.</p>'
            '<p><b>Provenance badges:</b> SYNTHETIC = generated for this study; SECONDARY = published source; MODEL OUTPUT = computed '
            'from the synthetic data; ASSUMPTION = the group\'s proposal figures.</p></div>', unsafe_allow_html=True)
    with cards.card("Rebuild everything"):
        st.code("cd noon\npip install -r requirements-analysis.txt\npython -m analysis.run_pipeline   # data + models + validation (~2 min)\n"
                "streamlit run streamlit_app.py", language="bash")
