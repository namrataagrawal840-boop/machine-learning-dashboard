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
            '<p><b>A/B tests:</b> two-proportion z-test, 95% confidence interval on the difference.</p>'
            '<p><b>Provenance badges:</b> SYNTHETIC = generated for this study; SECONDARY = published source; MODEL OUTPUT = computed '
            'from the synthetic data; ASSUMPTION = the group\'s proposal figures.</p></div>', unsafe_allow_html=True)
    with cards.card("Rebuild everything"):
        st.code("cd noon\npip install -r requirements-analysis.txt\npython -m analysis.run_pipeline   # data + models (~30 s)\n"
                "streamlit run streamlit_app.py", language="bash")
