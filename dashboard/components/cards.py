"""Card-based layout primitives. Every block of text on a page goes through here."""
from __future__ import annotations

import html
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

import streamlit as st

from dashboard.config import theme
from dashboard.config.schema import PROVENANCE, PROVENANCE_SHORT

CSS_PATH = Path(__file__).resolve().parents[2] / "assets" / "styles.css"
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


def page_header(title: str, purpose: str, stage: str, datasets: Iterable = ()) -> None:
    metas = [chip(stage)]
    for r in datasets:
        if r is None:
            continue
        if r.ok:
            metas.append(f'{chip(r.spec.title + f" · n={r.n:,}")} {badge(r.provenance, r.provenance_note)}')
        else:
            metas.append(chip(f"{r.spec.title}: {r.status}"))
    st.markdown(
        f'<div class="lc-card lc-page"><div class="lc-eyebrow">India luggage-storage marketplace · decision support</div>'
        f'<div class="lc-title">{esc(title)}</div><p class="lc-sub">{esc(purpose)}</p>'
        f'<div class="lc-meta">{" ".join(metas)}</div></div>',
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
    b = f"<div>{badge(prov)}</div>" if prov else ""
    style = f' style="--kpi-accent:{accent}"' if accent else ""
    return (f'<div class="lc-kpi"{style}><div class="lc-kpi-label">{esc(label)}</div>{v}'
            f'<div class="lc-kpi-sub">{esc(sub)}</div>{b}</div>')


def kpi_grid(items: list[str], per_row: int = 4) -> None:
    for i in range(0, len(items), per_row):
        cols = st.columns(per_row, gap="small")
        for c, html_ in zip(cols, items[i:i + per_row]):
            c.markdown(html_, unsafe_allow_html=True)


def explanation(measures: str, observation: str, implication: str, limitation: str) -> None:
    st.markdown(
        '<div class="lc-explain">'
        f'<div class="x-observe"><span class="lc-xl">What the data shows</span>{observation}</div>'
        f'<div class="x-measure"><span class="lc-xl">What is measured</span>{measures}</div>'
        f'<div class="x-implic"><span class="lc-xl">Business implication</span>{implication}</div>'
        f'<div class="x-limit" style="grid-column:1 / -1"><span class="lc-xl">Limitations &amp; caveats</span>{limitation}</div>'
        '</div>',
        unsafe_allow_html=True,
    )


def missing_dataset(result, what_for: str = "") -> None:
    """Explain exactly which file / columns are needed."""
    if result is None:
        return
    spec = result.spec
    files = " or ".join(f"<code>{esc(f)}</code>" for f in spec.filenames)
    cols = ", ".join(f"<code>{esc(c)}</code>" for c in spec.expected_columns)
    if result.status == "missing":
        notice("pending", f"{spec.title} not available yet.",
               f"{esc(what_for)} Add {files} to the <code>data/</code> folder (or upload it on the "
               f"<i>Data sources &amp; setup</i> page). Expected columns: {cols}.")
    elif result.status == "empty":
        notice("warn", f"{spec.title} is empty.", f"The file {esc(result.path)} contains no records. {esc(what_for)}")
    elif result.status == "error":
        notice("error", f"{spec.title} could not be used.", f"{esc(result.message)}. Expected columns: {cols}.")


def missing_columns(cols: Iterable[str], purpose: str) -> None:
    cols = list(cols)
    notice("pending", "Required fields not found in the dataset.",
           f"{esc(purpose)} Missing: " + ", ".join(f"<code>{esc(c)}</code>" for c in cols)
           + ". See <code>docs/DATA_CONTRACT.md</code> for the expected schema and accepted aliases.")


def html_table(rows: list[dict], columns: list[str] | None = None) -> str:
    if not rows:
        return ""
    columns = columns or list(rows[0].keys())
    head = "".join(f"<th>{esc(c)}</th>" for c in columns)
    body = "".join("<tr>" + "".join(f"<td>{r.get(c, '')}</td>" for c in columns) + "</tr>" for r in rows)
    return f'<table class="lc-table"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'
