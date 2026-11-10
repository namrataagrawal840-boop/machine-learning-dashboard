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
    assert len(seen) == 9, seen
    return problems


def test_every_page_renders():
    assert _all_pages() == {}


def test_every_page_renders_with_filters():
    assert _all_pages({"f_emirate": ["Abu Dhabi"], "f_category": ["Fashion", "Electronics"]}) == {}
