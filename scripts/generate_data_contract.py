"""Regenerate docs/DATA_CONTRACT.md from dashboard/config/schema.py.

Usage:  python scripts/generate_data_contract.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dashboard.config.schema import (ALIASES, DATA_GROUP_LABELS, DATASETS, MULTI_SELECT, ORDINALS,  # noqa: E402
                                     PROVENANCE, SCORE_SCALES)


def main() -> None:
    out = ["# Data contract", "",
           "_Generated from `dashboard/config/schema.py` by `scripts/generate_data_contract.py` — edit the schema, "
           "then regenerate._", "",
           "Place files anywhere under `data/` (sub-folders are searched). The dashboard reads them; it never "
           "modifies, cleans or overwrites them. Every file is optional — pages show a pending state for anything "
           "missing.", "",
           "## Provenance classes", ""]
    out += [f"- `{k}` — {v}" for k, v in PROVENANCE.items()]
    out += ["", "Declare them in `data/sources.toml` (copy `data/sources.example.toml`). A `data_source` column inside "
                "a file is also recognised. Model outputs inherit the consumer dataset's provenance.", ""]
    for group, title in DATA_GROUP_LABELS.items():
        specs = [s for s in DATASETS.values() if s.group == group]
        if not specs:
            continue
        out += [f"## {title}", ""]
        for s in specs:
            out += [f"### {s.title}", "",
                    f"- **File name(s):** " + " or ".join(f"`{f}`" for f in s.filenames),
                    f"- **Purpose:** {s.purpose}",
                    f"- **Used on:** {s.page_hint}",
                    f"- **Required columns:** " + (", ".join(f"`{c}`" for c in s.required_columns) or "none"),
                    f"- **Expected columns:** " + ", ".join(f"`{c}`" for c in s.expected_columns),
                    f"- **Default provenance:** `{s.default_provenance}`", ""]
    out += ["## Accepted column-name aliases", "",
            "Names are compared after lower-casing and replacing non-alphanumerics with `_`. Questionnaire numbers "
            "(e.g. `Q11`, `q11_checkout_gap`) are recognised for consumer columns.", ""]
    for g, al in ALIASES.items():
        out += [f"**{g}**", "", "| canonical | also accepted |", "|---|---|"]
        out += [f"| `{k}` | {', '.join(f'`{a}`' for a in v)} |" for k, v in al.items()]
        out += [""]
    out += ["## Answer formats", "",
            "### Ordered scales (text answers or numbers)", "",
            "Text answers are matched to the questionnaire wording (case, dashes and trailing full stops are ignored). "
            "Where a numeric column is supplied instead, it is banded with the bins below. Special answers such as "
            "'Not applicable' are kept as their own category and never converted to zero or missing.", ""]
    for f, spec in ORDINALS.items():
        lv = " < ".join(l for l, _, _ in spec["levels"])
        sp = ", ".join(s for s, _ in spec.get("special", []))
        out.append(f"- `{f}`: {lv}" + (f" · special: {sp}" if sp else ""))
    out += ["", "### Numeric scores", ""]
    out += [f"- `{f}`: {lo}–{hi} (a leading number such as `4 — Significantly` is accepted)" for f, (lo, hi) in SCORE_SCALES.items()]
    out += ["- `storage_wtp_inr`, `pickup_delivery_wtp_inr`: numbers in ₹ (₹ signs and commas are stripped). Zero is a "
            "valid answer; blank stays missing; negative values are flagged invalid.", "",
            "### Multi-select questions", "",
            "Either one text column with options separated by `;` (or `|`), or one binary column per option named "
            "`<question>__<option>` (values 0/1, true/false, yes/no). Recognised options:", ""]
    for f, opts in MULTI_SELECT.items():
        out.append(f"- `{f}`: " + "; ".join(o for o, _ in opts))
    out += ["", "## Model-output conventions", "",
            "- Classification target: High_Intent = 1 when `booking_intent_score` is 8–10, 0 when 0–7 (stated intent). "
            "`actual`/`predicted` may be 0/1, true/false, yes/no or high/low. Add a `target` column to separate an "
            "actual-booking target from survey intent; add a `features` column to `classification_metrics.csv` to "
            "enable the leakage check (Q23–Q30 must not be predictors).",
            "- Optional `split` column (e.g. train/test) — the dashboard defaults to `test`.",
            "- Regression `target` values must mention storage or pickup/delivery.",
            "- Association rules accept mlxtend `frozenset({...})` strings or `a; b` lists.",
            "- `booking_funnel.csv`: either `stage,count` rows or one row per event with a `stage` column; optional "
            "`booking_revenue, partner_payment, transport_cost, payment_fee, other_variable_cost` enable actual "
            "contribution margins.", ""]
    (ROOT / "docs" / "DATA_CONTRACT.md").write_text("\n".join(out), encoding="utf-8")
    print("wrote docs/DATA_CONTRACT.md")


if __name__ == "__main__":
    main()
