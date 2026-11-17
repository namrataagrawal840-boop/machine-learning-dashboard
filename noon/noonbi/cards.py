"""Card-based layout primitives (same visual system as the luggage dashboard)."""
from __future__ import annotations

import html
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

import streamlit as st

from noonbi import theme

PROVENANCE = {
    "synthetic": "Synthetic data generated for this study (not noon's real data)",
    "secondary": "Published secondary source (linked)",
    "derived": "Model output computed from the synthetic data",
    "illustrative": "Assumption from the group's proposal, used to show the method",
}
PROVENANCE_SHORT = {"synthetic": "SYNTHETIC", "secondary": "SECONDARY", "derived": "MODEL OUTPUT",
                    "illustrative": "ASSUMPTION"}

CSS_PATH = Path(__file__).resolve().parents[1] / "assets" / "styles.css"
_COUNTER_KEY = "_card_counter"


def esc(x: object) -> str:
    return html.escape(str(x), quote=True)


def inject_css() -> None:
    try:
        css = CSS_PATH.read_text(encoding="utf-8")
    except OSError:
        css = ""
    st.html(f"<style>{css}</style>")
    st.session_state[_COUNTER_KEY] = 0


def _next_key(prefix: str = "card") -> str:
    n = st.session_state.get(_COUNTER_KEY, 0) + 1
    st.session_state[_COUNTER_KEY] = n
    return f"{prefix}-{n}"


@contextmanager
def card(title: str | None = None, question: str | None = None, help_text: str | None = None):
    """A white bordered card that can hold native Streamlit widgets and charts."""
    with st.container(border=True, key=_next_key()) as c:
        if title or question:
            parts = []
            if title:
                parts.append(f'<div class="lc-h">{esc(title)}</div>')
            if question:
                parts.append(f'<div class="lc-q"><b>Business question:</b> {esc(question)}</div>')
            if help_text:
                parts.append(f'<div class="lc-muted">{esc(help_text)}</div>')
            st.markdown("".join(parts), unsafe_allow_html=True)
        yield c


def badge(prov: str, note: str = "") -> str:
    bg, fg = theme.PROVENANCE_COLOURS.get(prov, theme.PROVENANCE_COLOURS["unclassified"])
    title = esc(PROVENANCE.get(prov, prov) + (f" — {note}" if note else ""))
    return (f'<span class="lc-badge" title="{title}" style="background:{bg};color:{fg}">'
            f'{esc(PROVENANCE_SHORT.get(prov, prov.upper()))}</span>')


def chip(text: str) -> str:
    return f'<span class="lc-chip">{esc(text)}</span>'


def page_header(title: str, purpose: str, stage: str, provs: Iterable[tuple[str, str]] = ()) -> None:
    """provs: (label, provenance) pairs shown as chips + badges."""
    metas = [chip(stage)] + [f"{chip(lbl)} {badge(p)}" for lbl, p in provs]
    if not detailed():
        metas = []
    st.markdown(
        f'<div class="lc-card lc-page"><div class="lc-eyebrow">noon · customer experience analytics · Group 2</div>'
        f'<div class="lc-title">{esc(title)}</div><p class="lc-sub">{esc(purpose)}</p>'
        + (f'<div class="lc-meta">{" ".join(metas)}</div>' if metas else "") + '</div>',
        unsafe_allow_html=True,
    )


def text_card(title: str | None, body_html: str, tint: str = "") -> None:
    head = f'<div class="lc-h">{esc(title)}</div>' if title else ""
    st.markdown(f'<div class="lc-card {("tint-" + tint) if tint else ""}">{head}<div class="lc-body">{body_html}</div></div>',
                unsafe_allow_html=True)


def notice(kind: str, title: str, body_html: str = "") -> None:
    """kind: info | warn | pending | error | ok"""
    st.markdown(f'<div class="lc-notice {esc(kind)}"><b>{esc(title)}</b>{(" " + body_html) if body_html else ""}</div>',
                unsafe_allow_html=True)


def kpi(label: str, value: str | None, sub: str = "", prov: str | None = None, accent: str | None = None) -> str:
    v = (f'<div class="lc-kpi-value">{esc(value)}</div>' if value not in (None, "")
         else '<div class="lc-kpi-value na">Not available</div>')
    b = f"<div>{badge(prov)}</div>" if (prov and detailed()) else ""
    style = f' style="--kpi-accent:{accent}"' if accent else ""
    return (f'<div class="lc-kpi"{style}><div class="lc-kpi-label">{esc(label)}</div>{v}'
            f'<div class="lc-kpi-sub">{esc(sub)}</div>{b}</div>')


def kpi_grid(items: list[str], per_row: int = 4) -> None:
    if not detailed():
        items = items[:per_row]  # simple view: the four headline numbers only
    for i in range(0, len(items), per_row):
        cols = st.columns(per_row, gap="small")
        for c, html_ in zip(cols, items[i:i + per_row]):
            c.markdown(html_, unsafe_allow_html=True)


def detailed() -> bool:
    """True when the viewer switched on 'Show full detail' in the sidebar (default: simple, uncluttered view)."""
    return bool(st.session_state.get("detail_view", False))


def _first_sentence(html_text: str) -> str:
    import re
    m = re.search(r"^(.+?[.!?])(\s|$)", str(html_text).strip(), flags=re.S)
    return m.group(1) if m else str(html_text)


def explanation(measures: str, observation: str, implication: str, limitation: str) -> None:
    """Simple view: one key-takeaway line + a collapsed 'details' section. Full view: the four-box note."""
    boxes = ('<div class="lc-explain">'
             f'<div class="x-observe"><span class="lc-xl">What the data shows</span>{observation}</div>'
             f'<div class="x-measure"><span class="lc-xl">What is measured</span>{measures}</div>'
             f'<div class="x-implic"><span class="lc-xl">Business implication</span>{implication}</div>'
             f'<div class="x-limit" style="grid-column:1 / -1"><span class="lc-xl">Limitations &amp; caveats</span>{limitation}</div>'
             '</div>')
    if detailed():
        st.markdown(boxes, unsafe_allow_html=True)
        return
    st.markdown(f'<div class="lc-takeaway"><span class="lc-xl">Key takeaway</span>{_first_sentence(observation)} '
                f'<span class="lc-so">→ {_first_sentence(implication)}</span></div>', unsafe_allow_html=True)
    with st.expander("More detail", icon=":material/expand_more:"):
        st.markdown(boxes, unsafe_allow_html=True)


def html_table(rows: list[dict], columns: list[str] | None = None) -> str:
    if not rows:
        return ""
    columns = columns or list(rows[0].keys())
    head = "".join(f"<th>{esc(c)}</th>" for c in columns)
    body = "".join("<tr>" + "".join(f"<td>{r.get(c, '')}</td>" for c in columns) + "</tr>" for r in rows)
    return f'<table class="lc-table"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'
