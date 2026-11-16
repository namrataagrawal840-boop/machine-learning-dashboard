"""Builds the per-run dashboard context shared by every page."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import streamlit as st

from dashboard.data import filters, loader, transforms
from dashboard.data.loader import DatasetResult


@dataclass
class Ctx:
    results: dict[str, DatasetResult]
    consumer: pd.DataFrame | None = None  # prepared, unfiltered
    cf: pd.DataFrame | None = None  # prepared, filtered
    cmeta: dict = field(default_factory=dict)
    partner_survey: pd.DataFrame | None = None
    partner_audit: pd.DataFrame | None = None
    pmeta: dict = field(default_factory=dict)
    competitors: pd.DataFrame | None = None
    geo: pd.DataFrame | None = None
    active_filters: list[str] = field(default_factory=list)
    min_n: int = 30

    def r(self, key: str) -> DatasetResult:
        return self.results[key]

    @property
    def cprov(self) -> str:
        return self.results["consumer_clean"].provenance

    @property
    def n_total(self) -> int:
        return 0 if self.consumer is None else len(self.consumer)

    @property
    def n_filtered(self) -> int:
        return 0 if self.cf is None else len(self.cf)

    def filter_note(self) -> str:
        if not self.active_filters:
            return "No filters applied."
        return "Filters: " + "; ".join(self.active_filters) + "."

    def filter_ids(self) -> set[str] | None:
        """Respondent ids in the filtered frame (None = no filtering possible / needed)."""
        if self.cf is None or "respondent_id" not in self.cf or not self.active_filters:
            return None
        return set(self.cf["respondent_id"].astype(str).str.strip())


def _norm_id(s: pd.Series) -> pd.Series:
    out = s.astype(str).str.strip()
    return out.str.replace(r"\.0$", "", regex=True)


def build() -> Ctx:
    results = loader.load_all()
    ctx = Ctx(results=results)
    cc = results["consumer_clean"]
    if cc.ok:
        try:
            consumer, meta = transforms.prepare_consumer(cc.df)
            consumer = consumer.copy()
            ca = results["cluster_assignments"]
            if ca.ok and "respondent_id" in consumer and "respondent_id" in ca.df:
                cl = ca.df[["respondent_id", "cluster"]].copy()
                cl["respondent_id"] = _norm_id(cl["respondent_id"])
                cl["cluster"] = "Cluster " + cl["cluster"].astype(str).str.replace(r"\.0$", "", regex=True)
                cl = cl.drop_duplicates("respondent_id")
                consumer["respondent_id"] = _norm_id(consumer["respondent_id"])
                consumer = consumer.merge(cl, on="respondent_id", how="left")
                meta["cluster_join"] = int(consumer["cluster"].notna().sum())
            ctx.consumer, ctx.cmeta = consumer, meta
        except Exception as e:  # noqa: BLE001 — never crash the whole app on one dataset
            cc.status, cc.message = "error", f"Could not interpret consumer data: {e}"
    for key, attr in (("partner_survey", "partner_survey"), ("partner_audit", "partner_audit")):
        r = results[key]
        if r.ok:
            try:
                d, m = transforms.prepare_partner(r.df)
                setattr(ctx, attr, d)
                ctx.pmeta[key] = m
            except Exception as e:  # noqa: BLE001
                r.status, r.message = "error", f"Could not interpret partner data: {e}"
    if results["competitors"].ok:
        ctx.competitors = transforms.prepare_competitors(results["competitors"].df)
    if results["geo_audit"].ok:
        ctx.geo = transforms.prepare_geo(results["geo_audit"].df)
    return ctx


def sidebar(ctx: Ctx) -> None:
    with st.sidebar:
        with st.expander("Global filters", expanded=True, icon=":material/filter_alt:"):
            ctx.cf, ctx.active_filters = filters.render(ctx.consumer)
            if ctx.consumer is not None:
                st.caption(f"Showing **{ctx.n_filtered:,}** of {ctx.n_total:,} consumer records.")
        with st.expander("Analysis settings", icon=":material/tune:"):
            ctx.min_n = st.number_input("Minimum sample for subgroup comparisons", min_value=5, max_value=500,
                                        value=30, step=5, key="set_min_n",
                                        help="Groups smaller than this are flagged and excluded from rankings.")
        with st.expander("Dataset status", icon=":material/database:"):
            for r in ctx.results.values():
                icon = {"loaded": "🟢", "missing": "⚪", "empty": "🟡", "error": "🔴"}.get(r.status, "⚪")
                st.markdown(f"{icon} **{r.spec.title}** — {r.status}" + (f" · n={r.n:,}" if r.ok else ""))
