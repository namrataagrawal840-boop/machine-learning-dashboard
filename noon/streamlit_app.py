"""noon customer-experience analytics dashboard (Group 2).

Run locally:            cd noon && streamlit run streamlit_app.py
Streamlit Cloud:        main file path = noon/streamlit_app.py
The dashboard only reads the prepared files in noon/data/; it never changes them.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))  # so `noonbi` imports work from any working directory

st.set_page_config(page_title="noon · CX Analytics", page_icon=":material/local_shipping:", layout="wide",
                   initial_sidebar_state="auto")

from noonbi import cards  # noqa: E402
from noonbi import data as D  # noqa: E402
from noonbi.views import (p01_overview, p02_market, p03_data, p04_customers, p05_delivery, p06_refunds,  # noqa: E402
                          p07_service, p08_business, p09_sources)

cards.inject_css()
st.logo(str(Path(__file__).resolve().parent / "assets" / "logo.svg"), size="large")
filters = D.sidebar_filters()


def _page(module, title: str, icon: str, slug: str, default: bool = False):
    return st.Page(lambda: module.render(filters), title=title, icon=icon, url_path=slug, default=default)


pages = {
    "Start": [
        _page(p01_overview, "1 · Overview", ":material/dashboard:", "overview", default=True),
        _page(p02_market, "2 · Market & Competitors", ":material/travel_explore:", "market"),
        _page(p03_data, "3 · Data & Cleaning", ":material/fact_check:", "data"),
        _page(p04_customers, "4 · Customer Foundations", ":material/groups:", "customers"),
    ],
    "The three problems": [
        _page(p05_delivery, "5 · Problem 1 · Delivery", ":material/local_shipping:", "delivery"),
        _page(p06_refunds, "6 · Problem 2 · Refunds", ":material/currency_exchange:", "refunds"),
        _page(p07_service, "7 · Problem 3 · Service", ":material/support_agent:", "service"),
    ],
    "Decision": [
        _page(p08_business, "8 · Business Case", ":material/flag:", "business-case"),
        _page(p09_sources, "Sources & Methods", ":material/menu_book:", "sources"),
    ],
}
nav = st.navigation(pages, position="sidebar", expanded=True)
with st.sidebar:
    st.divider()
    st.caption("Group 2 · Data Analytics · Prof. Anshul Gupta. Operational data is synthetic; market data is cited.")
nav.run()
