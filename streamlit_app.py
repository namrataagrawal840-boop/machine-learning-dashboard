"""Entry point: `streamlit run streamlit_app.py`.

India luggage-storage & travel-convenience marketplace — analytics dashboard.
Dashboard-only code: it reads prepared datasets from data/ and never modifies them.
"""
from __future__ import annotations

import streamlit as st

st.set_page_config(page_title="Luggage Marketplace BI", page_icon=":material/luggage:", layout="wide",
                   initial_sidebar_state="auto")

from dashboard import context  # noqa: E402  (set_page_config must run first)
from dashboard.components import cards  # noqa: E402
from dashboard.views import (p01_overview, p02_data_quality, p03_diagnostics, p04_geography,  # noqa: E402
                             p05_partners, p06_segmentation, p07_classification, p08_association, p09_pricing,
                             p10_strategy, p11_sources)

cards.inject_css()
st.logo(str(cards.CSS_PATH.parent / "logo.svg"), size="large")
ctx = context.build()


def _page(module, title: str, icon: str, slug: str, default: bool = False):
    return st.Page(lambda: module.render(ctx), title=title, icon=icon, url_path=slug, default=default)


pages = {
    "Individual submission — core": [
        _page(p01_overview, "1 · Overview", ":material/dashboard:", "overview", default=True),
        _page(p02_data_quality, "2 · Data Quality", ":material/fact_check:", "data-quality"),
        _page(p03_diagnostics, "3 · Descriptive & Diagnostic", ":material/insights:", "descriptive-diagnostic"),
    ],
    "Market & supply evidence": [
        _page(p04_geography, "4 · Geographic Opportunity", ":material/map:", "geographic-opportunity"),
        _page(p05_partners, "5 · Partner Network", ":material/storefront:", "partner-network"),
    ],
    "Group-stage analytics": [
        _page(p06_segmentation, "6 · Customer Segmentation", ":material/bubble_chart:", "segmentation"),
        _page(p07_classification, "7 · Classification", ":material/rule:", "classification"),
        _page(p08_association, "8 · Association Mining", ":material/hub:", "association-mining"),
        _page(p09_pricing, "9 · Pricing & Regression", ":material/payments:", "pricing-regression"),
    ],
    "Decision": [
        _page(p10_strategy, "10 · Strategy & Sales Pipeline", ":material/flag:", "strategy"),
    ],
    "Setup": [
        _page(p11_sources, "Data sources & setup", ":material/database:", "data-sources"),
    ],
}

nav = st.navigation(pages, position="sidebar", expanded=True)
context.sidebar(ctx)
nav.run()
