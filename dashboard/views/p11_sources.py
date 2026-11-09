"""Data sources & setup — dataset status, data contract and session uploads."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.components import cards, charts
from dashboard.config.schema import ALIASES, DATA_GROUP_LABELS, DATASETS, PROVENANCE
from dashboard.data import loader

esc = cards.esc


def render(ctx) -> None:
    cards.page_header("Data sources & setup",
                      "Which prepared files are connected, where they come from, and what each page still needs.",
                      "Setup", [])
    rows = []
    for r in ctx.results.values():
        rows.append({"dataset": r.spec.title, "group": DATA_GROUP_LABELS[r.spec.group], "status": r.status,
                     "records": r.n if r.ok else None, "source": r.source or "—",
                     "provenance": PROVENANCE.get(r.provenance, r.provenance) if r.status == "loaded" else "—",
                     "provenance basis": r.provenance_note if r.status == "loaded" else "—",
                     "expected file": " or ".join(r.spec.filenames),
                     "missing expected columns": ", ".join(r.missing_columns) if r.ok else "—",
                     "used on": r.spec.page_hint, "message": r.message})
    ok = sum(r.ok for r in ctx.results.values())
    cards.kpi_grid([cards.kpi("Datasets connected", f"{ok} of {len(ctx.results)}", "recognised and readable"),
                    cards.kpi("Data folder", str(loader.data_dir().name) + "/", str(loader.data_dir())),
                    cards.kpi("Uploaded this session", f"{len(st.session_state.get(loader.UPLOAD_STATE_KEY, {}))}",
                              "held in memory only")], per_row=3)
    with cards.card("Dataset status", help_text="Files are found anywhere under the data folder by name. Nothing is "
                                                "ever written back to them."):
        charts.table(pd.DataFrame(rows), key="src_status", filename="dataset_status.csv")
    _uploads(ctx)
    with cards.card("Declaring provenance"):
        st.markdown(
            "Every figure carries one of five evidence classes: **Synthetic research data**, **Actual primary survey "
            "data**, **Verified secondary market research**, **Actual pilot transaction data**, **Illustrative "
            "financial scenario**. Set them in `data/sources.toml` (see `data/sources.example.toml`):")
        st.code('[provenance]\nconsumer_clean = "synthetic"\nconsumer_raw = "synthetic"\npartner_survey = "primary"\n'
                'partner_audit = "primary"\ncompetitors = "secondary"\ngeo_audit = "secondary"\n'
                'booking_funnel = "transaction"', language="toml")
        st.caption("A data_source column inside a file is also recognised. Model outputs inherit the consumer "
                   "dataset's provenance. Consumer files default to 'synthetic' as described in the project brief; "
                   "other files default to 'not declared'.")
    with cards.card("Data contract — expected files and columns"):
        for key, spec in DATASETS.items():
            with st.expander(f"{spec.title} — {' / '.join(spec.filenames)}"):
                st.markdown(f"**Purpose:** {spec.purpose}  \n**Required columns:** "
                            f"{', '.join(spec.required_columns) or 'none strictly required'}  \n"
                            f"**Expected columns:** {', '.join(spec.expected_columns)}")
                al = ALIASES.get(spec.alias_group, {})
                if al:
                    st.caption("Accepted alternative names: " + "; ".join(
                        f"{k} ← {', '.join(v)}" for k, v in al.items() if k in spec.expected_columns or not spec.expected_columns))
        st.caption("Multi-select answers may be semicolon-separated text (e.g. 'CCTV; Tamper-evident seals') or binary "
                   "columns named <question>__<option> (e.g. security_requirements__cctv).")


def _uploads(ctx) -> None:
    with cards.card("Upload files for this session", help_text="Useful on a hosted deployment without data in the "
                                                               "repository. Uploads stay in memory for this browser "
                                                               "session and are never saved to disk or GitHub."):
        files = st.file_uploader("CSV files", type=["csv"], accept_multiple_files=True, key="src_uploader")
        by_name = {f.lower(): k for k, s in DATASETS.items() for f in s.filenames}
        if files:
            changed = False
            for f in files:
                guess = by_name.get(f.name.lower())
                keys = list(DATASETS)
                k = st.selectbox(f"Dataset for {f.name}", keys, index=keys.index(guess) if guess else 0,
                                 format_func=lambda x: DATASETS[x].title, key=f"src_map_{f.name}")
                changed |= loader.register_upload(k, f.name, f.getvalue())
            if changed:
                st.rerun()
            st.success("Uploads registered — every page now uses them for this session.")
        if st.session_state.get(loader.UPLOAD_STATE_KEY):
            st.button("Clear uploaded files", on_click=loader.clear_uploads, key="src_clear",
                      icon=":material/delete:")
