"""End-to-end: every page renders without an exception on the committed data, also with filters on."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "streamlit_app.py")


def _all_pages(state: dict | None = None) -> dict[str, list[str]]:
    at = AppTest.from_file(APP, default_timeout=240)
    for k, v in (state or {}).items():
        at.session_state[k] = v
    at.run()
    problems, seen = {}, []
    for h, meta in at._registered_pages.items():
        at._page_hash = h
        at.run()
        seen.append(meta["page_name"])
        errs = [str(e.value) for e in at.exception]
        if errs:
            problems[meta["page_name"]] = errs
    assert len(seen) == 11, seen
    return problems


def test_every_page_renders():
    assert _all_pages() == {}


def test_every_page_renders_with_filters():
    assert _all_pages({"f_emirate": ["Abu Dhabi"], "f_category": ["Fashion", "Electronics"]}) == {}


def test_simulator_runs_for_each_problem():
    at = AppTest.from_file(APP, default_timeout=240)
    at.run()
    page = [h for h, m in at._registered_pages.items() if m["page_name"].endswith("Solution Simulator")][0]
    at._page_hash = page
    for i, prob in enumerate(["Problem 1 · Delivery", "Problem 2 · Refunds", "Problem 3 · Service"], start=1):
        at.session_state["sim_problem"] = prob
        at.session_state[f"sim{i}_speed"] = "Instant"
        at.run()
        assert not at.exception, [e.value for e in at.exception]
        [b for b in at.button if b.key == f"sim{i}_run"][0].click().run()
        assert not at.exception, [e.value for e in at.exception]
        assert any("Simulation result" in m.value for m in at.markdown), prob


def test_live_platform_steps(tmp_path, monkeypatch):
    monkeypatch.setenv("NOON_LIVE_DIR", str(tmp_path))
    at = AppTest.from_file(APP, default_timeout=240)
    at.run()
    page = [h for h, m in at._registered_pages.items() if m["page_name"].endswith("Live Platform")][0]
    at._page_hash = page
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    [b for b in at.button if b.key == "live_ff"][0].click().run()
    assert not at.exception, [e.value for e in at.exception]
    from live.engine import open_worlds
    w = open_worlds(tmp_path)
    assert w["today"].kpis()["orders"] == w["solution"].kpis()["orders"] > 500
