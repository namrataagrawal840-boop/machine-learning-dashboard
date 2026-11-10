"""Read-only data access. The dashboard never writes to data/."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import streamlit as st

DATA = Path(__file__).resolve().parents[1] / "data"
RES = DATA / "results"
RESEARCH = DATA / "research"
DATE_COLS = {"orders.csv": ["order_date"], "returns.csv": ["order_date"], "tickets.csv": ["created_date"],
             "daily_demand.csv": ["date"], "customers.csv": ["last_order", "first_order"], "p1_demand.csv": ["date"]}


@st.cache_data(show_spinner=False)
def _read(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p, parse_dates=DATE_COLS.get(p.name, []))


def table(name: str) -> pd.DataFrame:
    """Load a prepared table by name, looking in data/, data/results/ and data/research/."""
    for folder in (DATA, RES, RESEARCH):
        p = folder / f"{name}.csv"
        if p.exists():
            return _read(str(p)).copy()
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def summary() -> dict:
    p = RES / "summary.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def text(name: str) -> str:
    p = RES / name
    return p.read_text(encoding="utf-8") if p.exists() else ""


@dataclass
class Filters:
    emirates: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    months: tuple[str, str] | None = None

    def apply(self, df: pd.DataFrame, date_col: str | None = None) -> pd.DataFrame:
        out = df
        if self.emirates and "emirate" in out:
            out = out[out["emirate"].isin(self.emirates)]
        if self.categories and "category" in out:
            out = out[out["category"].isin(self.categories)]
        if self.months and date_col and date_col in out:
            m = out[date_col].dt.to_period("M").astype(str)
            out = out[(m >= self.months[0]) & (m <= self.months[1])]
        return out

    @property
    def active(self) -> bool:
        return bool(self.emirates or self.categories or self.months)


def sidebar_filters() -> Filters:
    o = table("orders")
    if o.empty:
        return Filters()
    months = sorted(o["order_date"].dt.to_period("M").astype(str).unique())
    with st.sidebar:
        st.markdown('<div class="lc-side-brand">Filters</div><div class="lc-side-sub">Apply to the descriptive charts on '
                    'every page. Model results were computed on the full data.</div>', unsafe_allow_html=True)
        em = st.multiselect("Emirate", sorted(o["emirate"].unique()), key="f_emirate", placeholder="All emirates")
        cat = st.multiselect("Category", sorted(o["category"].unique()), key="f_category", placeholder="All categories")
        rng = st.select_slider("Months", options=months, value=(months[0], months[-1]), key="f_months")
    f = Filters(em, cat, None if rng == (months[0], months[-1]) else tuple(rng))
    return f
