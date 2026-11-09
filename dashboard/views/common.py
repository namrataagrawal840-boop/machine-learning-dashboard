"""Helpers shared by page modules."""
from __future__ import annotations

import pandas as pd

from dashboard.components import cards
from dashboard.config.schema import PROVENANCE


def require_consumer(ctx, purpose: str = "") -> bool:
    r = ctx.r("consumer_clean")
    if ctx.consumer is None:
        cards.missing_dataset(r, purpose or "This page analyses the cleaned consumer survey.")
        return False
    if ctx.cf is None or ctx.cf.empty:
        cards.notice("warn", "No records match the current filters.",
                     "Widen or reset the filters in the sidebar to see results.")
        return False
    return True


def missing_cols(df: pd.DataFrame | None, *cols: str) -> list[str]:
    if df is None:
        return list(cols)
    return [c for c in cols if c not in df.columns or df[c].dropna().empty]


def limitation(ctx, n: int, extra: str = "") -> str:
    prov = ctx.cprov
    src = PROVENANCE.get(prov, prov)
    base = f"n = {n:,} respondents with a valid answer after filters. Source: {src}."
    if prov == "synthetic":
        base += (" These are simulated responses used to test the analysis; they describe the simulation, "
                 "not the Indian travel market.")
    elif prov == "unclassified":
        base += " Provenance has not been declared — confirm whether the data is synthetic or real before citing it."
    base += " Stated survey answers are not observed behaviour."
    if extra:
        base += " " + extra
    return cards.esc(base)


def small_sample_note(n: int, min_n: int) -> str:
    return (f" Only {n:,} records — below the minimum of {min_n} set in Analysis settings, so treat this as "
            "indicative at most." if n < min_n else "")


def filter_caption(ctx) -> str:
    return cards.esc(ctx.filter_note())


# ------------------------------------------------------------------ reusable chart cards
def dist_card(ctx, title: str, question: str, col: str, measures: str, implication, horizontal: bool = False,
              order=None, sort: bool = False, colour: str | None = None, key: str | None = None) -> None:
    """Distribution of one single-choice / ordinal column with a generated explanation."""
    from dashboard.components import charts
    from dashboard.config import theme

    d = ctx.cf
    with cards.card(title, question):
        if col not in d or d[col].dropna().empty:
            cards.missing_columns([col.split("__")[0]], "This chart needs the answer column.")
            return
        s = d[col]
        if order is None and isinstance(s.dtype, pd.CategoricalDtype):
            order = list(s.cat.categories)
        fig, data = charts.count_bar(s.astype(object) if order is None else s, horizontal=horizontal, order=order,
                                     sort=sort, colour=colour or theme.BLUE)
        charts.show(fig, key or col, data)
        n = int(data["respondents"].sum())
        if n == 0:
            cards.notice("info", "No answers after filters.")
            return
        top = data.sort_values("respondents", ascending=False).iloc[0]
        obs = (f"Most common answer: <b>{cards.esc(top['category'])}</b> — {top['share_pct']:.1f}% "
               f"({int(top['respondents']):,} of {n:,}).")
        cards.explanation(cards.esc(measures), obs, implication(data, n),
                          limitation(ctx, n, small_sample_note(n, ctx.min_n).strip()))


def multi_table(d: pd.DataFrame, field: str) -> tuple[pd.DataFrame, int]:
    from dashboard.data.transforms import ms_columns, ms_option

    cols = ms_columns(d, field)
    if not cols:
        return pd.DataFrame(), 0
    answered = d[cols].notna().any(axis=1)
    n = int(answered.sum())
    sub = d.loc[answered, cols].fillna(False).astype(bool)
    data = pd.DataFrame({"option": [ms_option(c) for c in cols], "respondents": sub.sum().values.astype(int),
                         "share_pct": (sub.mean() * 100).values.round(2) if n else 0.0})
    return data.sort_values("share_pct", ascending=False).reset_index(drop=True), n


def multi_card(ctx, title: str, question: str, field: str, measures: str, implication, colour: str | None = None,
               key: str | None = None) -> None:
    from dashboard.components import charts
    from dashboard.config import theme
    from dashboard.config.schema import label as _label

    with cards.card(title, question):
        data, n = multi_table(ctx.cf, field)
        if data.empty or n == 0:
            cards.missing_columns([field], f"Needs {_label(field)} as semicolon-separated text or binary columns.")
            return
        fig = charts.bar(data["option"], data["share_pct"], data["respondents"], [n] * len(data), horizontal=True,
                         colour=colour or theme.BLUE, axis_title="% of respondents answering (multi-select)")
        charts.show(fig, key or f"ms_{field}", data)
        top = data.iloc[0]
        cards.explanation(cards.esc(measures + " Multi-select: shares do not add to 100%."),
                          f"Most selected: <b>{cards.esc(top['option'])}</b> ({top['share_pct']:.1f}%, "
                          f"{int(top['respondents']):,} of {n:,}). Average options selected: "
                          f"{data['respondents'].sum() / n:.1f}.",
                          implication(data, n), limitation(ctx, n))
