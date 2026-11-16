"""Global sidebar filters. Every consumer-based page reads the same filtered frame."""
from __future__ import annotations

import pandas as pd
import streamlit as st

FILTER_PREFIX = "flt_"
# (key, label, column in prepared consumer frame)
FILTERS = [
    ("city", "City", "city__label"),
    ("microzone", "Microzone", "microzone__label"),
    ("age_group", "Age group", "age_group__label"),
    ("travel_purpose", "Travel purpose", "travel_purpose__label"),
    ("trips_12m", "Travel frequency (trips, 12 m)", "trips_12m__label"),
    ("travel_mode", "Transport mode", "travel_mode__label"),
    ("accommodation_type", "Accommodation type", "accommodation_type__label"),
    ("checkout_gap_hours", "Checkout-gap duration", "checkout_gap_hours__label"),
    ("cluster", "Traveller segment (cluster)", "cluster"),
]


def _options(s: pd.Series) -> list[str]:
    if isinstance(s.dtype, pd.CategoricalDtype):
        present = set(s.dropna().astype(str))
        return [str(c) for c in s.cat.categories if str(c) in present]
    return sorted(s.dropna().astype(str).unique().tolist())


def reset_filters() -> None:
    for k in list(st.session_state.keys()):
        if str(k).startswith(FILTER_PREFIX):
            st.session_state[k] = []


def render(df: pd.DataFrame | None) -> tuple[pd.DataFrame | None, list[str]]:
    """Draw filter widgets in the sidebar and return (filtered frame, active-filter descriptions)."""
    if df is None or df.empty:
        st.caption("Filters appear once the cleaned consumer dataset is available.")
        return df, []
    active: list[str] = []
    mask = pd.Series(True, index=df.index)
    for key, lbl, col in FILTERS:
        if col not in df.columns or df[col].dropna().empty:
            continue
        base = df.loc[mask, col] if key == "microzone" else df[col]  # microzone options follow city choice
        opts = _options(base) if key == "microzone" else _options(df[col])
        wkey = FILTER_PREFIX + key
        if wkey in st.session_state:  # drop stale selections (e.g. microzone after a city change)
            st.session_state[wkey] = [v for v in st.session_state[wkey] if v in opts]
        sel = st.multiselect(lbl, opts, key=wkey, placeholder="All")
        if sel:
            mask &= df[col].astype(str).isin(sel)
            active.append(f"{lbl}: {', '.join(sel)}")
    st.button("Reset all filters", on_click=reset_filters, icon=":material/restart_alt:", width="stretch",
              key="btn_reset_filters")
    return df.loc[mask], active
