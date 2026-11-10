"""Build every dataset and model output used by the noon dashboard.

    cd noon
    pip install -r requirements-analysis.txt
    python -m analysis.run_pipeline

Writes raw (messy) files, clean files, the cleaning log, model results and research tables to data/.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import cleaning, market_research, models, simulation_data
from analysis import synthetic as S

ROOT = Path(__file__).resolve().parents[1]
DATA, RES = ROOT / "data", ROOT / "data" / "results"


def _save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    t0 = time.time()
    rng = np.random.default_rng(S.SEED)
    customers = S.generate_customers(rng)
    sellers = S.generate_sellers(rng)
    orders = S.generate_orders(rng, customers, sellers)
    returns = S.generate_returns(rng, orders, customers, sellers)
    tickets = S.generate_tickets(rng, orders, returns, customers)
    demand = S.daily_demand(rng)

    # ---- raw exports with realistic problems, then cleaning
    o_raw, r_raw, t_raw = S.add_quality_issues(rng, orders, returns.drop(columns=["_returner"]), tickets)
    _save(o_raw, DATA / "raw" / "orders_raw.csv")
    _save(r_raw, DATA / "raw" / "returns_raw.csv")
    _save(t_raw, DATA / "raw" / "tickets_raw.csv")
    log = cleaning.Log()
    o = cleaning.clean_orders(pd.read_csv(DATA / "raw" / "orders_raw.csv", keep_default_na=False, na_values=[""]), log, S.ZONE_DF)
    r = cleaning.clean_returns(pd.read_csv(DATA / "raw" / "returns_raw.csv", keep_default_na=False, na_values=[""]), log)
    t = cleaning.clean_tickets(pd.read_csv(DATA / "raw" / "tickets_raw.csv", keep_default_na=False, na_values=[""]), log)
    cust = S.customer_summary(o.assign(order_date=pd.to_datetime(o["order_date"])), r, t, customers)
    cust_public = cust.drop(columns=[c for c in cust.columns if c.startswith("_")])
    _save(o, DATA / "orders.csv")
    _save(r, DATA / "returns.csv")
    _save(t, DATA / "tickets.csv")
    _save(cust_public, DATA / "customers.csv")
    _save(demand, DATA / "daily_demand.csv")
    _save(sellers.drop(columns=[c for c in sellers.columns if c.startswith("_")]), DATA / "sellers.csv")
    _save(log.df(), DATA / "cleaning_log.csv")
    quality = []
    for name, raw, clean in [("orders", o_raw, o), ("returns", r_raw, r), ("tickets", t_raw, t)]:
        quality.append({"dataset": name, "raw_rows": len(raw), "clean_rows": len(clean),
                        "raw_missing_cells": int(raw.isna().sum().sum()), "columns": clean.shape[1]})
    _save(pd.DataFrame(quality), DATA / "quality_summary.csv")
    print(f"data ready ({time.time() - t0:.0f}s)")

    # ---- models
    summary = {"boost_model": models.BOOST_NAME, "seed": S.SEED}
    o["order_date"] = pd.to_datetime(o["order_date"])
    out1, te1 = models.problem1(o, r)
    up, ab1 = models.uplift_and_ab(o)
    dem, dinfo = models.demand_forecast(demand)
    print(f"problem 1 done ({time.time() - t0:.0f}s)")
    out2, info2, ab2 = models.problem2(r, o, cust)
    print(f"problem 2 done ({time.time() - t0:.0f}s)")
    out3, info3 = models.problem3(t)
    chain = models.event_chain_rules(o, r, t, cust)
    outc, infoc = models.customer_models(cust)
    print(f"problem 3 + customers done ({time.time() - t0:.0f}s)")
    for d in (out1, up, dem, out2, out3, outc):
        for k, v in d.items():
            _save(v, RES / f"{k}.csv")
    _save(chain, RES / "chain_rules.csv")
    _save(pd.DataFrame([ab1, ab2]), RES / "ab_tests.csv")
    summary.update({"demand": dinfo, "p2": info2, "p3": {k: v for k, v in info3.items() if k != "fix_tree_rules"},
                    "customers": infoc})
    (RES / "fix_tree_rules.txt").write_text(info3["fix_tree_rules"], encoding="utf-8")
    (RES / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")

    # ---- simulator records (fixed-number replay of today vs solution)
    SIM = DATA / "sim"
    _save(simulation_data.problem1_records(o, te1, customers, cust), SIM / "p1_orders.csv")
    _save(simulation_data.problem2_records(r, out2["p2_test_scores"], customers, cust), SIM / "p2_returns.csv")
    _save(simulation_data.problem3_records(t, out3["p3_test_pred"], customers, cust), SIM / "p3_tickets.csv")
    print(f"simulator records done ({time.time() - t0:.0f}s)")

    for k, v in market_research.market_tables().items():
        _save(v, DATA / "research" / f"{k}.csv")
    print(f"all done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
