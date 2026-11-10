"""Plotly chart builders and the shared render wrapper.

All charts are native, interactive Plotly figures (never images). Each one is
rendered with the editable config from theme.py, and its underlying data can be
viewed, sorted, searched and downloaded as CSV.
"""
from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from noonbi import theme

_DL_COUNTER = "_dl_counter"


def _safe_name(s: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", s).strip("_").lower() or "chart"


def show(fig: go.Figure, key: str, data: pd.DataFrame | None = None, height: int | None = None,
         data_label: str = "View, sort & download the data behind this chart") -> None:
    fig.update_layout(height=height or fig.layout.height or theme.CHART_HEIGHT)
    st.plotly_chart(fig, key=f"fig_{key}", config=theme.plotly_config(_safe_name(key)), width="stretch")
    if data is not None and not data.empty:
        with st.expander(data_label):
            table(data, key=f"tbl_{key}", filename=f"{_safe_name(key)}.csv")


def table(df: pd.DataFrame, key: str, filename: str = "table.csv", height: int | None = None,
          column_config: dict | None = None, hide_index: bool = True) -> None:
    """Sortable, searchable table with a CSV download (built-in Streamlit dataframe toolbar)."""
    clean = df.copy()
    for c in clean.columns:
        if isinstance(clean[c].dtype, pd.CategoricalDtype):
            clean[c] = clean[c].astype(object)
    num = clean.select_dtypes(include="number").columns
    if len(num):
        clean[num] = clean[num].mask(np.isinf(clean[num].astype(float)))
    kwargs = {"hide_index": hide_index, "width": "stretch"}
    if height:
        kwargs["height"] = height
    if column_config:
        kwargs["column_config"] = column_config
    st.dataframe(clean, key=f"df_{key}", **kwargs)
    st.download_button("Download CSV", clean.to_csv(index=not hide_index).encode("utf-8"), file_name=filename,
                       mime="text/csv", key=f"dl_{key}", icon=":material/download:")


# ------------------------------------------------------------------ builders
def _wrap(s: str, width: int = 28) -> str:
    words, lines, cur = str(s).split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width and cur:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    lines.append(cur)
    return "<br>".join(lines)


def bar(labels, values, counts=None, denom=None, horizontal: bool = False, colour: str = theme.BLUE,
        value_fmt: str = "{:.1f}%", axis_title: str = "", hover_label: str = "Share") -> go.Figure:
    labels = [str(x) for x in labels]
    txt = [value_fmt.format(v) if v is not None and not (isinstance(v, float) and math.isnan(v)) else "" for v in values]
    cd = None
    hover = f"%{{{'y' if horizontal else 'x'}}}<br>{hover_label}: %{{{'x' if horizontal else 'y'}:.1f}}"
    if counts is not None:
        cd = np.column_stack([counts, denom if denom is not None else counts])
        hover += "<br>n = %{customdata[0]:,} of %{customdata[1]:,}"
    hover += "<extra></extra>"
    if horizontal:
        fig = go.Figure(go.Bar(y=[_wrap(x, 34) for x in labels], x=values, orientation="h", marker_color=colour,
                               text=txt, textposition="outside", cliponaxis=False, customdata=cd,
                               hovertemplate=hover))
        fig.update_layout(yaxis=dict(autorange="reversed"), xaxis_title=axis_title,
                          height=max(theme.CHART_HEIGHT_SMALL, 34 * len(labels) + 80))
    else:
        fig = go.Figure(go.Bar(x=labels, y=values, marker_color=colour, text=txt, textposition="outside",
                               cliponaxis=False, customdata=cd, hovertemplate=hover))
        fig.update_layout(yaxis_title=axis_title)
    fig.update_traces(marker_line_width=0)
    _headroom(fig, values, horizontal)
    return fig


def _headroom(fig: go.Figure, values, horizontal: bool, factor: float = 1.16) -> None:
    """Leave space for outside value labels so they are never clipped."""
    v = pd.to_numeric(pd.Series(np.asarray(values, dtype=object).ravel()), errors="coerce").dropna()
    if v.empty or (v < 0).any():
        return
    top = float(v.max()) * factor or 1.0
    (fig.update_xaxes if horizontal else fig.update_yaxes)(range=[0, top])


def count_bar(series: pd.Series, horizontal: bool = False, colour: str = theme.BLUE, order=None,
              sort: bool = False) -> tuple[go.Figure, pd.DataFrame]:
    s = series.dropna()
    vc = s.value_counts(sort=False)
    if order is not None:
        vc = vc.reindex([o for o in order if o in vc.index])
    if sort:
        vc = vc.sort_values(ascending=False)
    vc = vc[vc > 0] if order is None else vc.fillna(0)
    n = int(len(s))
    pct = vc / n * 100 if n else vc * np.nan
    data = pd.DataFrame({"category": vc.index.astype(str), "respondents": vc.values.astype(int),
                         "share_pct": pct.values.round(2)})
    fig = bar(data["category"], data["share_pct"], data["respondents"], [n] * len(data), horizontal=horizontal,
              colour=colour, axis_title="% of respondents answering")
    return fig, data


def stacked_share(pct: pd.DataFrame, counts: pd.DataFrame, x_title: str = "", legend_title: str = "",
                  horizontal: bool = False, colours: list | None = None) -> go.Figure:
    fig = go.Figure()
    cols = list(pct.columns)
    pal = colours or theme.CATEGORICAL
    for i, c in enumerate(cols):
        kw = dict(name=str(c), marker_color=pal[i % len(pal)],
                  customdata=np.column_stack([counts[c].values, counts.sum(axis=1).values]),
                  hovertemplate=f"{c}<br>%{{{'y' if horizontal else 'x'}}}: %{{{'x' if horizontal else 'y'}:.1f}}%"
                                "<br>n = %{customdata[0]:,} of %{customdata[1]:,}<extra></extra>")
        if horizontal:
            fig.add_bar(y=[_wrap(x, 30) for x in pct.index.astype(str)], x=pct[c].values, orientation="h", **kw)
        else:
            fig.add_bar(x=pct.index.astype(str), y=pct[c].values, **kw)
    fig.update_layout(barmode="stack", legend_title_text=legend_title)
    if horizontal:
        fig.update_layout(xaxis_title="% within row", yaxis=dict(autorange="reversed"),
                          height=max(theme.CHART_HEIGHT_SMALL, 38 * len(pct) + 110))
    else:
        fig.update_layout(yaxis_title="% within group", xaxis_title=x_title)
    return fig


def heatmap(mat: pd.DataFrame, text_fmt: str = "{:.0f}", zmin=None, zmax=None, diverging: bool = False,
            counts: pd.DataFrame | None = None, x_title: str = "", y_title: str = "", value_name: str = "Value",
            colorbar_title: str = "") -> go.Figure:
    z = mat.values.astype(float)
    text = [[("" if (v is None or (isinstance(v, float) and math.isnan(v))) else text_fmt.format(v)) for v in row]
            for row in z]
    scale = theme.DIVERGING if diverging else theme.SEQUENTIAL
    hover = f"%{{y}} × %{{x}}<br>{value_name}: %{{z:.2f}}"
    cd = None
    if counts is not None:
        cd = counts.reindex(index=mat.index, columns=mat.columns).values
        hover += "<br>n = %{customdata:,}"
    fig = go.Figure(go.Heatmap(
        z=z, x=[_wrap(c, 18) for c in mat.columns.astype(str)], y=[_wrap(i, 30) for i in mat.index.astype(str)],
        colorscale=[[i / (len(scale) - 1), c] for i, c in enumerate(scale)], zmin=zmin, zmax=zmax,
        text=text, texttemplate="%{text}", textfont=dict(size=11), customdata=cd, hovertemplate=hover + "<extra></extra>",
        colorbar=dict(title=dict(text=colorbar_title, side="right"), thickness=12), xgap=2, ygap=2))
    fig.update_layout(xaxis=dict(side="bottom", title=x_title, showgrid=False),
                      yaxis=dict(autorange="reversed", title=y_title, showgrid=False),
                      height=max(theme.CHART_HEIGHT_SMALL, 34 * len(mat.index) + 140))
    return fig


def histogram(values: pd.Series, nbins: int = 30, colour: str = theme.BLUE, x_title: str = "",
              marker_lines: dict | None = None) -> go.Figure:
    v = pd.to_numeric(values, errors="coerce").dropna()
    fig = go.Figure(go.Histogram(x=v, nbinsx=nbins, marker_color=colour, marker_line_width=0,
                                 hovertemplate="%{x}<br>Respondents: %{y}<extra></extra>"))
    fig.update_layout(xaxis_title=x_title, yaxis_title="Respondents", bargap=0.06)
    for name, x in (marker_lines or {}).items():
        if x is not None and not math.isnan(x):
            fig.add_vline(x=x, line_dash="dot", line_color=theme.MUTED, line_width=1,
                          annotation_text=f"{name}: {x:,.0f}", annotation_position="top right",
                          annotation_font_color=theme.MUTED)
    return fig


def box_by(df: pd.DataFrame, group: str, value: str, order=None, y_title: str = "", points: str = "outliers",
           horizontal: bool = False) -> go.Figure:
    d = df[[group, value]].dropna()
    groups = order if order is not None else sorted(d[group].astype(str).unique())
    fig = go.Figure()
    for i, g in enumerate(groups):
        v = d.loc[d[group].astype(str) == str(g), value]
        if v.empty:
            continue
        kw = dict(name=f"{_wrap(g, 22)} (n={len(v)})", marker_color=theme.CATEGORICAL[i % len(theme.CATEGORICAL)],
                  boxpoints=points, boxmean=True, line_width=1.4)
        fig.add_trace(go.Box(x=v, **kw) if horizontal else go.Box(y=v, **kw))
    fig.update_layout(showlegend=False, **({"xaxis_title": y_title} if horizontal else {"yaxis_title": y_title}))
    return fig


def scatter(df: pd.DataFrame, x: str, y: str, color: str | None = None, hover: list[str] | None = None,
            x_title: str = "", y_title: str = "", jitter: float = 0.0, opacity: float = 0.55,
            diagonal: bool = False) -> go.Figure:
    d = df.dropna(subset=[x, y]).copy()
    rng = np.random.default_rng(0)
    xs = d[x].astype(float) + (rng.uniform(-jitter, jitter, len(d)) if jitter else 0)
    ys = d[y].astype(float) + (rng.uniform(-jitter, jitter, len(d)) if jitter else 0)
    fig = go.Figure()
    hover_cols = [h for h in (hover or []) if h in d.columns]
    cd = d[hover_cols].astype(str).values if hover_cols else None
    ht = f"{x_title or x}: %{{x:.2f}}<br>{y_title or y}: %{{y:.2f}}" + "".join(
        f"<br>{h}: %{{customdata[{i}]}}" for i, h in enumerate(hover_cols)) + "<extra></extra>"
    if color and color in d.columns:
        for i, (g, sub) in enumerate(d.groupby(color, observed=True)):
            idx = sub.index
            fig.add_scattergl(x=xs.loc[idx], y=ys.loc[idx], mode="markers", name=str(g),
                              marker=dict(size=6, opacity=opacity, color=theme.CATEGORICAL[i % len(theme.CATEGORICAL)]),
                              customdata=d.loc[idx, hover_cols].astype(str).values if hover_cols else None,
                              hovertemplate=ht)
    else:
        fig.add_scattergl(x=xs, y=ys, mode="markers", marker=dict(size=6, opacity=opacity, color=theme.BLUE),
                          customdata=cd, hovertemplate=ht, showlegend=False)
    if diagonal and len(d):
        lo = float(min(d[x].min(), d[y].min()))
        hi = float(max(d[x].max(), d[y].max()))
        fig.add_shape(type="line", x0=lo, y0=lo, x1=hi, y1=hi, line=dict(color=theme.MUTED, dash="dot", width=1))
    fig.update_layout(xaxis_title=x_title or x, yaxis_title=y_title or y)
    return fig


def grouped_bar(df: pd.DataFrame, x: str, y_cols: list[str], names: list[str] | None = None, y_title: str = "",
                value_fmt: str = "{:.1f}", horizontal: bool = False, colours: list | None = None) -> go.Figure:
    fig = go.Figure()
    pal = colours or theme.CATEGORICAL
    for i, c in enumerate(y_cols):
        nm = (names or y_cols)[i]
        vals = df[c].astype(float)
        txt = [value_fmt.format(v) if pd.notna(v) else "" for v in vals]
        if horizontal:
            fig.add_bar(y=[_wrap(v, 30) for v in df[x].astype(str)], x=vals, name=nm, orientation="h",
                        marker_color=pal[i % len(pal)], text=txt, textposition="outside", cliponaxis=False)
        else:
            fig.add_bar(x=df[x].astype(str), y=vals, name=nm, marker_color=pal[i % len(pal)], text=txt,
                        textposition="outside", cliponaxis=False)
    fig.update_layout(barmode="group")
    _headroom(fig, df[y_cols].to_numpy(), horizontal)
    if horizontal:
        fig.update_layout(xaxis_title=y_title, yaxis=dict(autorange="reversed"),
                          height=max(theme.CHART_HEIGHT_SMALL, 30 * len(df) * max(1, len(y_cols)) // 2 + 120))
    else:
        fig.update_layout(yaxis_title=y_title)
    return fig


def radar(df: pd.DataFrame, group_col: str, value_cols: list[str], labels: list[str]) -> go.Figure:
    fig = go.Figure()
    for i, (_, row) in enumerate(df.iterrows()):
        vals = [row[c] for c in value_cols]
        fig.add_trace(go.Scatterpolar(r=vals + vals[:1], theta=labels + labels[:1], name=str(row[group_col]),
                                      line=dict(color=theme.CATEGORICAL[i % len(theme.CATEGORICAL)], width=2),
                                      fill="toself", opacity=0.55))
    fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100], gridcolor="#EEF2F6", tickfont=dict(size=9)),
                                 angularaxis=dict(gridcolor="#EEF2F6")), height=theme.CHART_HEIGHT_TALL)
    return fig


def funnel(stages: list[str], values: list[float], colour: str = theme.BLUE) -> go.Figure:
    fig = go.Figure(go.Funnel(y=stages, x=values, textinfo="value+percent initial",
                              marker=dict(color=colour), connector=dict(line=dict(color=theme.BORDER))))
    return fig


def empty_figure(message: str) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(color=theme.MUTED, size=13), x=0.5, y=0.5,
                       xref="paper", yref="paper")
    fig.update_layout(xaxis=dict(visible=False), yaxis=dict(visible=False), height=200)
    return fig
