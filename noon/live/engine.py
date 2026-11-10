"""A small, live imitation of the noon platform backed by a real database (SQLite).

Two "worlds" run side by side from the same seed and see exactly the same customers, carts and orders:
  * today    — the current process (fixed 'tomorrow' promise, no warnings, slow refunds, no case owner)
  * solution — the three solutions switched on (honest promise + priority courier + early warning + credits,
               trust-tiered refunds, case ownership with deadlines)
Every simulated step writes orders, deliveries, returns, tickets and ALERTS into the database, exactly like a
production event stream would. The dashboard (or a separate worker process in Docker) calls `step()` repeatedly.

Times are stored as simulated minutes since the start (REAL); `clock_label()` turns them into dates.
"""
from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

START = pd.Timestamp("2026-11-20 08:00")  # the week of Yellow Friday
HOUR_W = np.array([1, .5, .3, .2, .2, .3, .6, 1.2, 2, 2.6, 3, 3.2, 3.4, 3.2, 3, 3, 3.3, 3.8, 4.5, 5.2, 5.6, 5, 3.6, 2])
HOUR_W = HOUR_W / HOUR_W.mean()
ZONES = [  # zone, emirate, zone_type, km_from_hub, weight
    ("Dubai Marina", "Dubai", "Dense urban", 28, 7), ("Downtown Dubai", "Dubai", "Dense urban", 22, 6),
    ("Deira", "Dubai", "Dense urban", 30, 7), ("Al Quoz", "Dubai", "Industrial", 18, 3),
    ("Al Barsha", "Dubai", "Dense urban", 20, 6), ("Jumeirah Village Circle", "Dubai", "Suburban", 16, 6),
    ("Arabian Ranches", "Dubai", "Villa community", 14, 4), ("Mirdif", "Dubai", "Villa community", 34, 5),
    ("International City", "Dubai", "Dense urban", 30, 5), ("Dubai Silicon Oasis", "Dubai", "Suburban", 26, 4),
    ("Al Reem Island", "Abu Dhabi", "Dense urban", 118, 6), ("Khalifa City", "Abu Dhabi", "Villa community", 104, 5),
    ("Mussafah", "Abu Dhabi", "Industrial", 108, 3), ("Yas Island", "Abu Dhabi", "Suburban", 98, 3),
    ("Al Ain", "Abu Dhabi", "Remote", 150, 4), ("Al Nahda (Sharjah)", "Sharjah", "Dense urban", 38, 6),
    ("Al Majaz", "Sharjah", "Dense urban", 41, 4), ("Muwaileh", "Sharjah", "Suburban", 46, 3),
    ("Al Nuaimiya", "Ajman", "Dense urban", 52, 4), ("Al Nakheel (RAK)", "Ras Al Khaimah", "Remote", 112, 3),
    ("Fujairah City", "Fujairah", "Remote", 128, 2), ("Umm Al Quwain", "Umm Al Quwain", "Remote", 72, 1),
]
ZONE_DELAY = {"Dense urban": 1.5, "Suburban": 3, "Villa community": 8, "Industrial": 5, "Remote": 16}
CATS = {  # category: (share, price median AED, bulky share, return rate, product words)
    "Electronics": (.20, 420, .07, .06, ["Wireless earbuds", "Smartwatch", "Power bank", "Bluetooth speaker", "Laptop stand", "4K monitor",
                                         "Gaming mouse", "Phone case", "Tablet", "Smart TV"]),
    "Fashion": (.19, 140, 0, .17, ["Linen shirt", "Abaya", "Running shoes", "Denim jacket", "Summer dress", "Kandura", "Sneakers",
                                   "Handbag", "Sunglasses", "Hoodie"]),
    "Beauty & Fragrance": (.11, 110, 0, .04, ["Oud perfume", "Face serum", "Hair dryer", "Lipstick set", "Sunscreen", "Body lotion",
                                              "Beard trimmer", "Shampoo", "Eye palette", "Nail kit"]),
    "Home & Kitchen": (.14, 160, .15, .07, ["Air fryer", "Coffee machine", "Blender", "Cookware set", "Bed sheets", "Vacuum cleaner",
                                            "Desk lamp", "Storage boxes", "Kettle", "Rug"]),
    "Grocery": (.16, 95, 0, .01, ["Basmati rice 5kg", "Dates box", "Olive oil", "Coffee beans", "Water pack", "Laban pack",
                                  "Chocolate box", "Tea bags", "Nuts mix", "Pasta pack"]),
    "Baby & Toys": (.08, 130, .10, .06, ["Diapers pack", "Stroller", "Building blocks", "Baby monitor", "Car seat", "Puzzle",
                                         "Plush toy", "Kids scooter", "Feeding set", "Board game"]),
    "Furniture": (.05, 900, .80, .08, ["Office chair", "Sofa bed", "Bookshelf", "Dining table", "Wardrobe", "TV unit",
                                       "Bed frame", "Coffee table", "Shoe rack", "Recliner"]),
    "Sports & Outdoors": (.07, 210, .15, .07, ["Yoga mat", "Dumbbells", "Treadmill", "Camping tent", "Cycling helmet",
                                               "Football", "Gym bag", "Water bottle", "Resistance bands", "Bike"]),
}
TOPIC_TEXT = {
    "Delivery delay": "My order is late and there is no update.",
    "Failed delivery": "Driver marked me unavailable but I was at home.",
    "Refund delay": "Still waiting for my refund.",
    "Refund rejected": "My refund was rejected without a reason.",
}
WORLDS = ("today", "solution")


def _sig(x: float) -> float:
    return 1 / (1 + math.exp(-x))


SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS sellers(seller_id TEXT PRIMARY KEY, tier TEXT, fulfilment TEXT, handover_h REAL);
CREATE TABLE IF NOT EXISTS products(sku TEXT PRIMARY KEY, name TEXT, category TEXT, price REAL, stock INTEGER,
    start_stock INTEGER, popularity REAL, seller_id TEXT, bulky INTEGER, restock_at REAL, sold INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS customers(customer_id TEXT PRIMARY KEY, zone TEXT, emirate TEXT, zone_type TEXT, km REAL,
    engagement REAL, returner TEXT, member INTEGER, orders INTEGER DEFAULT 0, bad_experiences INTEGER DEFAULT 0,
    at_risk INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS orders(order_id TEXT PRIMARY KEY, t REAL, customer_id TEXT, zone TEXT, emirate TEXT,
    n_items INTEGER, value REAL, category TEXT, sku TEXT, seller_id TEXT, courier TEXT, address TEXT, risk REAL,
    promise_t REAL, eta_t REAL, status TEXT, delivered_t REAL, late INTEGER DEFAULT 0, failed INTEGER DEFAULT 0,
    warned INTEGER DEFAULT 0, priority INTEGER DEFAULT 0, credit INTEGER DEFAULT 0, ticket INTEGER DEFAULT 0, return_at REAL);
CREATE INDEX IF NOT EXISTS ix_orders_status ON orders(status);
CREATE TABLE IF NOT EXISTS order_items(order_id TEXT, sku TEXT, qty INTEGER, price REAL);
CREATE TABLE IF NOT EXISTS returns(return_id TEXT PRIMARY KEY, order_id TEXT, customer_id TEXT, t REAL, value REAL,
    category TEXT, p_fail REAL, tier TEXT, qc_fail INTEGER, status TEXT, refund_due_t REAL, refunded_t REAL,
    chased INTEGER DEFAULT 0);
CREATE INDEX IF NOT EXISTS ix_returns_status ON returns(status);
CREATE TABLE IF NOT EXISTS tickets(ticket_id TEXT PRIMARY KEY, t REAL, customer_id TEXT, ref_id TEXT, topic TEXT,
    text TEXT, status TEXT, owner TEXT, deadline_t REAL, resolve_t REAL, resolved_t REAL, contacts INTEGER,
    next_chase_t REAL, priority INTEGER DEFAULT 0, breached INTEGER DEFAULT 0);
CREATE INDEX IF NOT EXISTS ix_tickets_status ON tickets(status);
CREATE TABLE IF NOT EXISTS alerts(alert_id INTEGER PRIMARY KEY AUTOINCREMENT, t REAL, severity TEXT, kind TEXT,
    entity TEXT, message TEXT);
CREATE TABLE IF NOT EXISTS metrics(t REAL PRIMARY KEY, orders INTEGER, gmv REAL, delivered INTEGER, late INTEGER,
    late_rate REAL, open_tickets INTEGER, repeat_contacts INTEGER, refunds_done INTEGER, refund_days_avg REAL,
    customers_at_risk INTEGER, alerts INTEGER, credits_aed REAL);
"""


@dataclass
class Config:
    seed: int = 20261120
    customers: int = 3000
    products: int = 1000
    sellers: int = 120
    base_orders_per_hour: float = 45.0


class Platform:
    """One world (today or solution) living in one SQLite file."""

    def __init__(self, path: str | Path, world: str, cfg: Config | None = None, create: bool = True):
        assert world in WORLDS
        self.path, self.world, self.cfg = Path(path), world, cfg or Config()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(self.path), check_same_thread=False, timeout=30)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.executescript(SCHEMA)
        if self.meta("t") is None and create:
            self.reset()

    # ------------------------------------------------------------ meta helpers
    def meta(self, key, default=None):
        r = self.db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return r[0] if r else default

    def set_meta(self, key, value):
        self.db.execute("INSERT OR REPLACE INTO meta VALUES(?,?)", (key, str(value)))

    @property
    def t(self) -> float:
        return float(self.meta("t", 0))

    @property
    def solution(self) -> bool:
        return self.world == "solution"

    def q(self, sql: str, params=()) -> pd.DataFrame:
        return pd.read_sql_query(sql, self.db, params=params)

    def alert(self, severity: str, kind: str, entity: str, message: str):
        self.db.execute("INSERT INTO alerts(t,severity,kind,entity,message) VALUES(?,?,?,?,?)",
                        (self.t, severity, kind, entity, message))

    # ------------------------------------------------------------ catalogue
    def reset(self):
        for tbl in ("meta", "sellers", "products", "customers", "orders", "order_items", "returns", "tickets", "alerts", "metrics"):
            self.db.execute(f"DELETE FROM {tbl}")
        rng = np.random.default_rng(self.cfg.seed)  # identical catalogue in both worlds
        tiers = rng.choice(["Good", "Average", "Poor"], self.cfg.sellers, p=[.55, .30, .15])
        sellers = []
        for i, tr in enumerate(tiers):
            fbn = rng.random() < {"Good": .7, "Average": .5, "Poor": .25}[tr]
            sellers.append((f"S{3000 + i}", tr, "Fulfilled by noon" if fbn else "Ships directly",
                            float({"Good": 4, "Average": 10, "Poor": 26}[tr] * rng.lognormal(0, .25))))
        self.db.executemany("INSERT INTO sellers VALUES(?,?,?,?)", sellers)
        names = list(CATS)
        cat_of = rng.choice(names, self.cfg.products, p=[CATS[c][0] for c in names])
        prods = []
        for i, c in enumerate(cat_of):
            share, med, bulky, _, words = CATS[c]
            nm = f"{words[i % len(words)]} · model {chr(65 + (i * 7) % 26)}{(i * 13) % 90 + 10}"
            stock = int(rng.integers(25, 260))
            prods.append((f"SKU-{i + 1:05d}", nm, c, round(float(med * rng.lognormal(0, .5)), 2), stock, stock,
                          float(rng.pareto(1.6) + 0.2), sellers[int(rng.integers(0, len(sellers)))][0], int(rng.random() < bulky), None))
        self.db.executemany("INSERT INTO products(sku,name,category,price,stock,start_stock,popularity,seller_id,bulky,restock_at) "
                            "VALUES(?,?,?,?,?,?,?,?,?,?)", prods)
        zw = np.array([z[4] for z in ZONES], float)
        zi = rng.choice(len(ZONES), self.cfg.customers, p=zw / zw.sum())
        custs = []
        for i, k in enumerate(zi):
            z = ZONES[k]
            e = float(rng.normal(0, 1))
            custs.append((f"C{200000 + i}", z[0], z[1], z[2], z[3], e,
                          str(rng.choice(["Typical", "Size-swapper", "Serial returner", "Possible abuser"], p=[.84, .10, .045, .015])),
                          int(rng.random() < _sig(-1.2 + .7 * e))))
        self.db.executemany("INSERT INTO customers(customer_id,zone,emirate,zone_type,km,engagement,returner,member) "
                            "VALUES(?,?,?,?,?,?,?,?)", custs)
        self.set_meta("t", 0)
        self.set_meta("tick", 0)
        self.set_meta("sale_mode", 0)
        self.alert("info", "System", "platform", f"Platform started ({self.world}): {self.cfg.products:,} products, "
                   f"{self.cfg.customers:,} customers, {self.cfg.sellers} sellers.")
        self.db.commit()

    def _id(self, prefix: str) -> str:
        n = int(self.meta(f"seq_{prefix}", 0)) + 1  # one counter per prefix → order ids match across worlds
        self.set_meta(f"seq_{prefix}", n)
        return f"{prefix}{n:07d}"

    # ------------------------------------------------------------ simulation step
    def step(self, minutes: float = 30.0, sale_mode: bool = False):
        t0 = self.t
        t1 = t0 + minutes
        tick = int(self.meta("tick", 0)) + 1
        self.set_meta("tick", tick)
        self.set_meta("sale_mode", int(sale_mode))
        # arrivals use a seed that depends only on the tick, so both worlds get identical customers and carts
        arr = np.random.default_rng(self.cfg.seed * 7 + tick)
        proc = np.random.default_rng(self.cfg.seed * 11 + tick + (0 if self.world == "today" else 500_000))
        hour = int(((START + pd.Timedelta(minutes=t0)).hour))
        rate = self.cfg.base_orders_per_hour * HOUR_W[hour] * (2.4 if sale_mode else 1.0) * minutes / 60
        n_new = int(arr.poisson(rate))
        self.set_meta("t", t1)
        self._new_orders(n_new, t0, t1, arr, proc, sale_mode)
        self._progress_orders(t1, proc)
        self._returns(t1, proc)
        self._tickets(t1, proc)
        self._restock(t1)
        self._watchdogs(t1, tick)
        self._snapshot(t1)
        self.db.commit()

    # orders --------------------------------------------------------------
    def _new_orders(self, n, t0, t1, arr, proc, sale_mode):
        if n <= 0:
            return
        cust = self.q("SELECT customer_id, zone, emirate, zone_type, km, engagement FROM customers")
        prod = self.q("SELECT sku, category, price, stock, popularity, seller_id, bulky FROM products")
        sel = self.q("SELECT * FROM sellers").set_index("seller_id")
        w = np.exp(.5 * cust["engagement"].to_numpy())
        ci = arr.choice(len(cust), n, p=w / w.sum())
        pw = prod["popularity"].to_numpy() * (prod["stock"].to_numpy() > 0)
        for k in range(n):
            c = cust.iloc[ci[k]]
            n_items = int(arr.choice([1, 2, 3], p=[.62, .28, .10]))
            items = arr.choice(len(prod), n_items, replace=False, p=pw / pw.sum())
            lines = prod.iloc[items]
            main = lines.sort_values("price", ascending=False).iloc[0]
            value = float(lines["price"].sum())
            s = sel.loc[main["seller_id"]]
            t = float(t0 + arr.random() * (t1 - t0))
            hour = (START + pd.Timedelta(minutes=t)).hour
            evening = hour >= 17
            courier = "In-house fleet" if arr.random() < .70 else ("3PL partner A" if arr.random() < .6 else "3PL partner B")
            if c["zone_type"] == "Remote" and arr.random() < .5:
                courier = "3PL partner B"
            addr_p = {"Dense urban": [.45, .37, .18], "Suburban": [.40, .36, .24], "Villa community": [.30, .33, .37],
                      "Industrial": [.35, .35, .30], "Remote": [.33, .35, .32]}[c["zone_type"]]
            address = str(arr.choice(["Map pin", "Full address", "Partial address"], p=addr_p))
            bulky = int(main["bulky"])
            direct = s["fulfilment"] == "Ships directly"
            third = courier != "In-house fleet"
            # expected hours (what the model can know) and the real hours (with noise)
            handover = s["handover_h"] if direct else 3.0
            exp_h = (handover + .07 * c["km"] + ZONE_DELAY[c["zone_type"]] + (10 if sale_mode else 0) + 27 * bulky
                     + 4 * evening + (0 if not third else (4 if courier == "3PL partner A" else 9)) + 3)
            real_h = exp_h * float(arr.lognormal(0, .35)) + (float(arr.gamma(2, 3)) if sale_mode else 0)
            risk = _sig(-2.6 + .07 * (exp_h - 12) + (1.0 if s["tier"] == "Poor" else 0) + (.8 if address == "Partial address" else 0))
            p_fail = _sig(-3.3 + {"Partial address": 1.7, "Full address": .6, "Map pin": 0}[address]
                          + {"Villa community": .6, "Remote": .5, "Industrial": .4}.get(c["zone_type"], 0) + .35 * evening)
            failed = arr.random() < p_fail
            gps_ok = arr.random() < .62
            promise_days = (1 if c["emirate"] in ("Dubai", "Sharjah", "Ajman") else 2) + bulky
            oid = self._id("N")
            priority = warned = 0
            if self.solution:
                promise_days = max(1, math.ceil(exp_h * 1.55 / 24))          # P90-style honest promise
                if risk >= .65:
                    priority = 1
                    real_h = max(2.0, real_h - 6)
                    self.alert("warning", "High-risk order", oid, f"Risk {risk:.0%} ({s['tier'].lower()} seller, {courier}, {c['zone']}) "
                               "→ priority courier assigned.")
                if failed and address == "Partial address" and proc.random() < .5:
                    failed = False
                    self.alert("info", "Pin requested", oid, f"WhatsApp location pin requested for {c['zone']} — first attempt saved.")
            if failed:
                if self.solution and not gps_ok:
                    self.alert("warning", "GPS lock", oid, "Driver tried to mark 'customer unavailable' away from the address — blocked, re-attempt today.")
                    real_h += 6
                else:
                    real_h += 24
            promise_t = (START + pd.Timedelta(minutes=t)).normalize() + pd.Timedelta(days=promise_days, hours=22)
            promise_t = float((promise_t - START) / pd.Timedelta(minutes=1))
            self.db.execute("INSERT INTO orders(order_id,t,customer_id,zone,emirate,n_items,value,category,sku,seller_id,courier,address,"
                            "risk,promise_t,eta_t,status,failed,priority) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                            (oid, t, c["customer_id"], c["zone"], c["emirate"], n_items, value, main["category"], main["sku"],
                             main["seller_id"], courier, address, risk, promise_t, t + real_h * 60, "Placed", int(failed), priority))
            for _, ln in lines.iterrows():
                self.db.execute("INSERT INTO order_items VALUES(?,?,?,?)", (oid, ln["sku"], 1, ln["price"]))
                self.db.execute("UPDATE products SET stock = stock - 1, sold = sold + 1 WHERE sku=?", (ln["sku"],))
            self.db.execute("UPDATE customers SET orders = orders + 1 WHERE customer_id=?", (c["customer_id"],))
            low = self.db.execute("SELECT name, stock FROM products WHERE sku=? AND stock IN (10, 0)", (main["sku"],)).fetchone()
            if low:
                self.alert("warning" if low[1] else "critical", "Stock", main["sku"],
                           f"{low[0]}: {'only 10 left' if low[1] else 'OUT OF STOCK'} — restock ordered.")
                self.db.execute("UPDATE products SET restock_at=? WHERE sku=? AND restock_at IS NULL", (t + 8 * 60, main["sku"]))

    def _progress_orders(self, now, proc):
        self.db.execute("UPDATE orders SET status='Out for delivery' WHERE status='Placed' AND eta_t - ? < 360", (now,))
        # solution: warn before the promise is missed
        if self.solution:
            for oid, cid in self.db.execute("SELECT order_id, customer_id FROM orders WHERE status!='Delivered' AND warned=0 "
                                            "AND eta_t > promise_t AND promise_t - ? < 720", (now,)).fetchall():
                self.db.execute("UPDATE orders SET warned=1 WHERE order_id=?", (oid,))
                self.alert("info", "Early warning", oid, "Will miss its date — customer messaged with a new date before chasing.")
        # promise missed
        rows = self.db.execute("SELECT order_id, customer_id, warned, value FROM orders WHERE status!='Delivered' AND late=0 "
                               "AND promise_t < ? AND eta_t > promise_t", (now,)).fetchall()
        for oid, cid, warned, value in rows:
            self.db.execute("UPDATE orders SET late=1 WHERE order_id=?", (oid,))
            self.db.execute("UPDATE customers SET bad_experiences = bad_experiences + 1 WHERE customer_id=?", (cid,))
            if proc.random() < (.17 if warned else .42):
                self._open_ticket(cid, oid, "Delivery delay", now, proc)
            if self.solution:
                eng = self.db.execute("SELECT engagement, member FROM customers WHERE customer_id=?", (cid,)).fetchone()
                if abs(eng[0] - .1) < .75 and not eng[1]:  # persuadable → credit
                    self.db.execute("UPDATE orders SET credit=15 WHERE order_id=?", (oid,))
                    self.alert("info", "Recovery credit", oid, "Late order, persuadable customer → AED 15 credit + apology.")
        # failed attempts create tickets in 'today' (and fewer in solution, because fake 'unavailable' is blocked)
        for oid, cid in self.db.execute("SELECT order_id, customer_id FROM orders WHERE failed=1 AND ticket=0 AND eta_t < ?", (now,)).fetchall():
            self.db.execute("UPDATE orders SET ticket=1 WHERE order_id=?", (oid,))
            if proc.random() < (.30 if self.solution else .55):
                self._open_ticket(cid, oid, "Failed delivery", now, proc)
        # delivery: decide once whether the customer will ask to return it (uses the arrival seed → same in both worlds)
        for oid, eta, cat, returner, late in self.db.execute(
                "SELECT o.order_id, o.eta_t, o.category, c.returner, o.late FROM orders o JOIN customers c USING(customer_id) "
                "WHERE o.status!='Delivered' AND o.eta_t <= ?", (now,)).fetchall():
            r = np.random.default_rng(self.cfg.seed + int(oid[1:]))
            rr = CATS[cat][3] * {"Typical": 1, "Size-swapper": 2.4, "Serial returner": 4.2, "Possible abuser": 5}[returner] + .03 * late
            ret_at = eta + float(r.uniform(1, 5)) * 1440 if r.random() < rr else None
            self.db.execute("UPDATE orders SET status='Delivered', delivered_t=eta_t, return_at=? WHERE order_id=?", (ret_at, oid))

    # returns -------------------------------------------------------------
    def _returns(self, now, proc):
        # new return requests from orders delivered 1–5 days ago
        rows = self.db.execute("SELECT o.order_id, o.customer_id, o.value, o.category, c.returner, o.late FROM orders o JOIN customers c "
                               "USING(customer_id) WHERE o.return_at IS NOT NULL AND o.return_at <= ? "
                               "AND o.order_id NOT IN (SELECT order_id FROM returns)", (now,)).fetchall()
        for oid, cid, value, cat, returner, late in rows:
            abuse = returner == "Possible abuser"
            qc_fail = proc.random() < (.55 if abuse else (.11 if returner == "Serial returner" else .04))
            p_fail = min(.95, max(.01, (.62 if abuse else .18 if returner == "Serial returner" else .05) * float(proc.lognormal(0, .35))))
            rid = self._id("R")
            if self.solution:
                tier = "Green" if p_fail < .15 else ("Amber" if p_fail < .5 else "Red")
                due = now + {"Green": 1, "Amber": 3, "Red": 12}[tier] * 1440
                if tier == "Red":
                    self.alert("warning", "Red-tier return", rid, f"AED {value:,.0f} {cat} — full inspection before refund.")
            else:
                tier = "—"
                due = now + float(proc.integers(9, 17)) * 1440
            self.db.execute("INSERT INTO returns VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                            (rid, oid, cid, now, value, cat, p_fail, tier, int(qc_fail), "Requested", due, None, 0))
        # refunds
        for rid, cid, qc, tier, value in self.db.execute("SELECT return_id, customer_id, qc_fail, tier, value FROM returns "
                                                          "WHERE status='Requested' AND refund_due_t <= ?", (now,)).fetchall():
            if qc and tier != "Green":
                self.db.execute("UPDATE returns SET status='Rejected', refunded_t=? WHERE return_id=?", (now, rid))
                self.db.execute("UPDATE customers SET bad_experiences = bad_experiences + 1 WHERE customer_id=?", (cid,))
                if proc.random() < (.5 if self.solution else .78):
                    self._open_ticket(cid, rid, "Refund rejected", now, proc)
            else:
                self.db.execute("UPDATE returns SET status='Refunded', refunded_t=? WHERE return_id=?", (now, rid))
                if qc and tier == "Green":
                    self.alert("critical", "Refund risk", rid, f"Green refund of AED {value:,.0f} failed the later check — recovery started.")
        # customers chasing slow refunds
        for rid, cid in self.db.execute("SELECT return_id, customer_id FROM returns WHERE status='Requested' AND chased=0 "
                                        "AND ? - t > 8 * 1440", (now,)).fetchall():
            self.db.execute("UPDATE returns SET chased=1 WHERE return_id=?", (rid,))
            if proc.random() < .35:
                self._open_ticket(cid, rid, "Refund delay", now, proc)

    # tickets -------------------------------------------------------------
    def _open_ticket(self, cid, ref, topic, now, proc):
        tid = self._id("T")
        hard = topic == "Refund rejected"
        base_h = {"Delivery delay": 20, "Failed delivery": 18, "Refund delay": 50, "Refund rejected": 96}[topic]
        if self.solution:
            eng = self.db.execute("SELECT engagement, bad_experiences FROM customers WHERE customer_id=?", (cid,)).fetchone()
            priority = int(eng[0] < 0 or eng[1] >= 2)
            res_h = base_h * float(proc.lognormal(0, .35)) * (.7 if priority else 1)
            owner = f"Agent {chr(65 + int(proc.integers(0, 12)))}"
            deadline = now + base_h * 60 * (.8 if priority else 1.1)
            self.alert("warning" if priority else "info", "Case owner", tid,
                       f"{topic} → {owner}{' (senior, churn risk)' if priority else ''}, deadline "
                       f"{(START + pd.Timedelta(minutes=deadline)):%a %H:%M}.")
        else:
            priority = 0
            res_h = base_h * float(proc.lognormal(0, .6)) * (1.8 if hard else 1.3)
            owner, deadline = None, None
        self.db.execute("INSERT INTO tickets VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (tid, now, cid, ref, topic, TOPIC_TEXT[topic], "Open", owner, deadline, now + res_h * 60, None, 1,
                         now + 24 * 60, priority, 0))

    def _tickets(self, now, proc):
        # customers chase every 24 h while unresolved; without an owner each chase starts from zero
        for tid, contacts in self.db.execute("SELECT ticket_id, contacts FROM tickets WHERE status='Open' AND next_chase_t <= ? "
                                             "AND resolve_t > ?", (now, now)).fetchall():
            extra = 0 if (self.solution and proc.random() < .6) else 1
            self.db.execute("UPDATE tickets SET contacts = contacts + ?, next_chase_t = next_chase_t + 1440 WHERE ticket_id=?", (extra, tid))
            if extra and not self.solution and contacts + 1 >= 3:
                self.alert("warning", "Repeat contact", tid, f"Customer contacted support {contacts + 1} times on the same issue.")
            if extra and self.solution and contacts + 1 == 2:
                self.alert("info", "Resolution desk", tid, "Second contact → moved to the Resolution Desk with authority to fix.")
                self.db.execute("UPDATE tickets SET resolve_t = MIN(resolve_t, ? + 360) WHERE ticket_id=?", (now, tid))
        if self.solution:
            for (tid,) in self.db.execute("SELECT ticket_id FROM tickets WHERE status='Open' AND breached=0 AND deadline_t < ? "
                                          "AND resolve_t > ?", (now, now)).fetchall():
                self.db.execute("UPDATE tickets SET breached=1 WHERE ticket_id=?", (tid,))
                self.alert("critical", "SLA breach", tid, "Past the promised deadline — escalated to team lead.")
        for tid, cid, contacts in self.db.execute("SELECT ticket_id, customer_id, contacts FROM tickets WHERE status='Open' AND resolve_t <= ?",
                                                  (now,)).fetchall():
            self.db.execute("UPDATE tickets SET status='Resolved', resolved_t=resolve_t WHERE ticket_id=?", (tid,))
            eng = self.db.execute("SELECT engagement FROM customers WHERE customer_id=?", (cid,)).fetchone()[0]
            if proc.random() < _sig(-1.6 - .75 * eng + .35 * (contacts - 1)):
                self.db.execute("UPDATE customers SET at_risk=1 WHERE customer_id=?", (cid,))

    # housekeeping --------------------------------------------------------
    def _restock(self, now):
        for sku, name, start in self.db.execute("SELECT sku, name, start_stock FROM products WHERE restock_at IS NOT NULL AND restock_at <= ?",
                                                (now,)).fetchall():
            self.db.execute("UPDATE products SET stock = stock + ?, restock_at=NULL WHERE sku=?", (start, sku))

    def _watchdogs(self, now, tick):
        if tick % 6:
            return
        z = self.db.execute("SELECT zone, AVG(late), COUNT(*) FROM orders WHERE promise_t BETWEEN ? AND ? GROUP BY zone "
                            "HAVING COUNT(*) >= 8 ORDER BY AVG(late) DESC LIMIT 1", (now - 1440, now)).fetchone()
        if z and z[1] >= .35:
            self.alert("warning", "Zone spike", z[0], f"{z[1]:.0%} of orders due in the last 24 h were late ({z[2]} orders).")
        s = self.db.execute("SELECT o.seller_id, AVG(o.late), COUNT(*) FROM orders o WHERE o.promise_t BETWEEN ? AND ? "
                            "GROUP BY o.seller_id HAVING COUNT(*) >= 5 ORDER BY AVG(o.late) DESC LIMIT 1", (now - 2880, now)).fetchone()
        if s and s[1] >= .5:
            self.alert("warning", "Slow seller", s[0], f"{s[1]:.0%} late in the last 48 h — express badge "
                       f"{'removed' if self.solution else 'still shown (no rule today)'}.")

    def _snapshot(self, now):
        r = self.db.execute("""SELECT COUNT(*), COALESCE(SUM(value),0),
              SUM(CASE WHEN status='Delivered' THEN 1 ELSE 0 END), SUM(late), COALESCE(SUM(credit),0) FROM orders""").fetchone()
        due = self.db.execute("SELECT COUNT(*), COALESCE(SUM(late),0) FROM orders WHERE promise_t <= ?", (now,)).fetchone()
        tk = self.db.execute("SELECT SUM(CASE WHEN status='Open' THEN 1 ELSE 0 END), SUM(CASE WHEN contacts>1 THEN 1 ELSE 0 END) FROM tickets").fetchone()
        rf = self.db.execute("SELECT COUNT(*), AVG((refunded_t - t)/1440.0) FROM returns WHERE status='Refunded'").fetchone()
        risk = self.db.execute("SELECT SUM(at_risk) FROM customers").fetchone()[0] or 0
        al = self.db.execute("SELECT COUNT(*) FROM alerts").fetchone()[0]
        self.db.execute("INSERT OR REPLACE INTO metrics VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (now, r[0], r[1], r[2] or 0, r[3] or 0, (due[1] / due[0]) if due[0] else None, tk[0] or 0, tk[1] or 0,
                         rf[0], rf[1], risk, al, r[4]))

    # ------------------------------------------------------------ read helpers for the dashboard
    def clock_label(self) -> str:
        return f"{START + pd.Timedelta(minutes=self.t):%a %d %b %Y · %H:%M}"

    def kpis(self) -> dict:
        m = self.q("SELECT * FROM metrics ORDER BY t DESC LIMIT 1")
        if m.empty:
            return {}
        m = m.iloc[0]
        return {"orders": int(m["orders"]), "gmv": float(m["gmv"]), "late_rate": m["late_rate"], "open_tickets": int(m["open_tickets"]),
                "repeat_contacts": int(m["repeat_contacts"]), "refunds_done": int(m["refunds_done"] or 0),
                "refund_days": m["refund_days_avg"], "at_risk": int(m["customers_at_risk"]), "alerts": int(m["alerts"]),
                "credits": float(m["credits_aed"] or 0)}


def open_worlds(folder: str | Path, cfg: Config | None = None, create: bool = True) -> dict[str, Platform]:
    """create=False is used by a read-only dashboard when a separate worker owns the database."""
    folder = Path(folder)
    return {w: Platform(folder / f"noon_live_{w}.db", w, cfg, create) for w in WORLDS}


def step_all(worlds: dict[str, Platform], minutes: float, sale_mode: bool) -> None:
    for p in worlds.values():
        p.step(minutes, sale_mode)


def reset_all(worlds: dict[str, Platform]) -> None:
    for p in worlds.values():
        p.reset()
