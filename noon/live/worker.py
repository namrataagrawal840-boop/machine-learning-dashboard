"""Background simulator process (used by docker-compose): advances both worlds forever.

    NOON_LIVE_DIR=/data python -m live.worker

The dashboard writes control values (running, speed, sale mode, reset) into the 'today' database's meta table;
this worker reads them every loop, so the platform keeps running even when nobody has the dashboard open.
"""
from __future__ import annotations

import os
import time

from live.engine import open_worlds, reset_all, step_all

DIR = os.environ.get("NOON_LIVE_DIR", "/data")
INTERVAL = float(os.environ.get("NOON_LIVE_INTERVAL", "1.5"))


def main() -> None:
    worlds = open_worlds(DIR)
    ctl = worlds["today"]
    if ctl.meta("ctl_running") is None:
        ctl.set_meta("ctl_running", 1)
        ctl.set_meta("ctl_speed", 30)
        ctl.set_meta("ctl_sale", 0)
        ctl.db.commit()
    print(f"noon live worker: writing to {DIR} every {INTERVAL}s", flush=True)
    while True:
        if ctl.meta("ctl_reset") == "1":
            reset_all(worlds)
            ctl.set_meta("ctl_reset", 0)
            ctl.set_meta("ctl_running", 1)
            ctl.db.commit()
            print("reset", flush=True)
        if ctl.meta("ctl_running", "1") == "1":
            step_all(worlds, float(ctl.meta("ctl_speed", 30)), ctl.meta("ctl_sale", "0") == "1")
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
