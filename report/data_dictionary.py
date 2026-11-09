"""Write data/data_dictionary.csv describing every column of every dataset in data/.

Usage:  python -m report.data_dictionary
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dashboard.config.schema import LABELS  # noqa: E402

DATA = ROOT / "data"
QNUM = {"travel_status": "Q1", "age_group": "Q2", "residence_country": "Q3", "country": "Q3", "city": "Q4",
        "microzone": "Q5", "travel_purpose": "Q6", "trips_12m": "Q7", "group_size": "Q8", "travel_mode": "Q9",
        "accommodation_type": "Q10", "checkout_gap_hours": "Q11", "luggage_management_hours": "Q12", "bag_count": "Q13",
        "problem_frequency": "Q14", "existing_storage_methods": "Q15", "past_spend_inr": "Q16",
        "inconvenience_score": "Q17", "storage_search_difficulty": "Q18", "existing_barriers": "Q19",
        "maximum_detour": "Q20", "acceptable_venue_types": "Q21", "security_requirements": "Q22",
        "booking_intent_score": "Q23", "first_service_preference": "Q24", "preferred_pricing_model": "Q25",
        "storage_wtp_inr": "Q26", "pickup_delivery_wtp_inr": "Q27", "preferred_features": "Q28",
        "main_nonuse_barrier": "Q29", "pilot_interest": "Q30", "establishment_type": "P1", "opening_time": "P3",
        "closing_time": "P3", "days_of_operation": "P3", "current_storage_offer": "P4", "secure_storage_available": "P5",
        "storage_capacity": "P6", "security_features": "P7", "partner_willingness": "P8", "commercial_preference": "P9",
        "expected_payment_inr": "P10", "expected_payment_unit": "P10", "digital_handover_acceptance": "P11",
        "pickup_delivery_acceptance": "P12", "partner_barriers": "P13", "pilot_participation": "P14"}
MULTI = {"existing_storage_methods", "existing_barriers", "acceptable_venue_types", "security_requirements",
         "preferred_features", "security_features", "partner_barriers"}


def main() -> None:
    rows = []
    for f in sorted(DATA.glob("*.csv")):
        if f.name == "data_dictionary.csv":
            continue
        df = pd.read_csv(f, low_memory=False)
        for c in df.columns:
            s = df[c]
            kind = ("multi-select (semicolon-separated)" if c in MULTI else "numeric" if pd.api.types.is_numeric_dtype(s)
                    else "categorical / text")
            ex = "; ".join(map(str, s.dropna().astype(str).unique()[:4]))[:160]
            rows.append({"file": f.name, "column": c, "question": QNUM.get(c, ""), "description": LABELS.get(c, ""),
                         "type": kind, "non_empty": int(s.notna().sum()), "rows": len(s), "example_values": ex})
    pd.DataFrame(rows).to_csv(DATA / "data_dictionary.csv", index=False)
    print(f"data_dictionary.csv: {len(rows)} columns described")


if __name__ == "__main__":
    main()
