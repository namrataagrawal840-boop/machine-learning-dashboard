"""Step 2 — cleaning and transformation. Every change is written to a log."""
from __future__ import annotations

import numpy as np
import pandas as pd

EMIRATE_FIX = {"dxb": "Dubai", "dubai": "Dubai", "abu dhabi": "Abu Dhabi", "sharjah": "Sharjah", "ajman": "Ajman",
               "rak": "Ras Al Khaimah", "ras al khaimah": "Ras Al Khaimah", "fujairah": "Fujairah",
               "uaq": "Umm Al Quwain", "umm al quwain": "Umm Al Quwain"}
REASON_FIX = {"size issue": "Size or fit issue", "defective": "Defective / not working",
              "not as described": "Not as described", "changed mind": "Changed my mind"}


class Log:
    def __init__(self):
        self.rows = []

    def add(self, dataset, step, rows_affected, action, why):
        self.rows.append({"dataset": dataset, "step": step, "rows_affected": int(rows_affected),
                          "action": action, "why": why})

    def df(self):
        return pd.DataFrame(self.rows)


def _money(x):
    if pd.isna(x):
        return np.nan
    s = str(x).lower().replace("aed", "").replace("dhs", "").replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


def clean_orders(o: pd.DataFrame, log: Log, zone_lookup: pd.DataFrame) -> pd.DataFrame:
    o = o.copy()
    n0 = len(o)
    o = o.drop_duplicates("order_id")
    log.add("orders", "Remove duplicates", n0 - len(o), "Dropped repeated order_id rows",
            "The same order exported twice would double-count late deliveries.")
    d1 = pd.to_datetime(o["order_date"], format="%Y-%m-%d", errors="coerce")
    d2 = pd.to_datetime(o["order_date"], format="%d/%m/%Y", errors="coerce")
    log.add("orders", "Standardise dates", int(d1.isna().sum()), "Converted DD/MM/YYYY dates to YYYY-MM-DD",
            "Two date formats in one column break monthly trends.")
    o["order_date"] = d1.fillna(d2)
    before = o["emirate"].copy()
    o["emirate"] = o["emirate"].astype(str).str.strip().str.lower().map(EMIRATE_FIX).fillna(o["emirate"])
    log.add("orders", "Fix emirate names", int((before != o["emirate"]).sum()), "Mapped DXB, RAK, UAQ, case variants",
            "Spelling variants would split one emirate into several.")
    raw = o["order_value_aed"].copy()
    o["order_value_aed"] = o["order_value_aed"].map(_money)
    log.add("orders", "Parse order value", int(pd.to_numeric(raw, errors="coerce").isna().sum()),
            "Removed 'AED'/'dhs' text and commas, kept the number", "Prices typed as text cannot be summed.")
    bad = o["actual_hours"] < 0
    log.add("orders", "Impossible delivery time", int(bad.sum()), "Rebuilt hours from actual_days × 24 (upper bound)",
            "A negative duration is a scanner error; the delivered day is still known, so the row is kept.")
    o.loc[bad, "actual_hours"] = o.loc[bad, "actual_days"] * 24.0
    miss = o["zone"].isna()
    log.add("orders", "Missing zone", int(miss.sum()), "Kept rows, zone set to 'Unknown'; excluded only from zone charts",
            "Dropping them would hide late orders; guessing a zone would invent data.")
    o["zone"] = o["zone"].fillna("Unknown")
    # derived fields
    o["month"] = o["order_date"].dt.to_period("M").astype(str)
    o["weekday"] = o["order_date"].dt.day_name()
    o["on_sale"] = (o["sale_event"] != "None").astype(int)
    o["is_bulky"] = (o["item_size"] == "Bulky").astype(int)
    o["evening"] = o["time_slot"].isin(["Evening", "Late evening"]).astype(int)
    o["value_band"] = pd.cut(o["order_value_aed"], [0, 100, 250, 500, 1000, 1e9],
                             labels=["<100", "100–250", "250–500", "500–1,000", "1,000+"]).astype(str)
    log.add("orders", "Derived fields", len(o), "Added month, weekday, on_sale, is_bulky, evening, value_band",
            "Plain yes/no and band columns make the charts and models easier to read.")
    return o.reset_index(drop=True)


def clean_returns(r: pd.DataFrame, log: Log) -> pd.DataFrame:
    r = r.copy()
    n0 = len(r)
    r = r.drop_duplicates("return_id")
    log.add("returns", "Remove duplicates", n0 - len(r), "Dropped repeated return_id rows", "Avoid double-counting refunds.")
    before = r["return_reason"].copy()
    r["return_reason"] = r["return_reason"].map(lambda x: REASON_FIX.get(str(x).strip().lower(), x))
    log.add("returns", "Standardise return reasons", int((before != r["return_reason"]).sum()),
            "Mapped short or upper-case reasons to the 6 official labels", "One reason must have one name.")
    r["refund_days"] = pd.to_numeric(r["refund_days"], errors="coerce")
    bad = r["refund_days"] < 0
    log.add("returns", "Impossible refund time", int(bad.sum()), "Set negative refund days to missing",
            "A refund cannot arrive before the return; the rest of the row is still valid.")
    r.loc[bad, "refund_days"] = np.nan
    r["order_date"] = pd.to_datetime(r["order_date"])
    return r.reset_index(drop=True)


def clean_tickets(t: pd.DataFrame, log: Log) -> pd.DataFrame:
    t = t.copy()
    n0 = len(t)
    t = t.drop_duplicates("ticket_id")
    log.add("tickets", "Remove duplicates", n0 - len(t), "Dropped repeated ticket_id rows", "Avoid double-counting complaints.")
    t["csat"] = pd.to_numeric(t["csat"], errors="coerce")
    bad = ~t["csat"].between(1, 5)
    log.add("tickets", "Out-of-range CSAT", int(bad.sum()), "Set CSAT outside 1–5 to missing",
            "The survey scale is 1–5; a 7 is a typing error.")
    t.loc[bad, "csat"] = np.nan
    empty = t["text"].fillna("").str.strip().eq("")
    log.add("tickets", "Empty ticket text", int(empty.sum()), "Kept for counts, excluded from text models",
            "There is nothing to read, but the contact still happened.")
    t["has_text"] = (~empty).astype(int)
    t["created_date"] = pd.to_datetime(t["created_date"])
    t["repeat_contact"] = (t["contacts"] > 1).astype(int)
    return t.reset_index(drop=True)
