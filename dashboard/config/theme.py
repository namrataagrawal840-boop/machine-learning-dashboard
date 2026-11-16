"""Visual system: colours, Plotly template and chart interaction config.

Edit this file to change colours, fonts, chart dimensions or which chart elements
users can edit in the browser. Nothing here is data — it is presentation only.
"""
from __future__ import annotations

import copy

import plotly.graph_objects as go
import plotly.io as pio

# ---------------------------------------------------------------- palette
WHITE = "#FFFFFF"
SIDEBAR = "#F8FAFC"
BORDER = "#E6EBF1"
TEXT = "#263548"
MUTED = "#6B788A"

SOFT_BLUE = "#EAF2FC"
SOFT_SAGE = "#EAF4EF"
SOFT_LAVENDER = "#F1EDFA"
SOFT_PEACH = "#FFF2E8"
SOFT_YELLOW = "#FFF8E8"

BLUE = "#668DBA"
GREEN = "#78A48D"
LAVENDER = "#9C88C9"
PEACH = "#E3A27A"
GOLD = "#D9B85C"
TEAL = "#7FB3C8"
ROSE = "#C98B9F"
SLATE = "#8E9AAB"
RED = "#C46A6A"

# Categorical order used for every multi-series chart.
CATEGORICAL = [BLUE, GREEN, LAVENDER, PEACH, GOLD, TEAL, ROSE, SLATE,
               "#5E7FA6", "#6A947D", "#8A77B5", "#C98D66"]
# Sequential (low -> high) for heatmaps of shares / counts.
SEQUENTIAL = ["#F4F8FC", "#D9E6F4", "#B3CBE6", "#8DB0D6", "#668DBA", "#4A6F9A", "#33557A"]
# Diverging (negative -> 0 -> positive) for correlations / residuals.
DIVERGING = ["#B8645E", "#DDA59C", "#F3DDD8", "#F7F7F7", "#D6E4F2", "#93B3D6", "#4F78A8"]

# Provenance badge colours (background, text).
PROVENANCE_COLOURS = {
    "synthetic": (SOFT_LAVENDER, "#5E4C8F"),
    "primary": (SOFT_SAGE, "#3F6B54"),
    "secondary": (SOFT_BLUE, "#3D5F87"),
    "transaction": (SOFT_YELLOW, "#7A5F12"),
    "illustrative": (SOFT_PEACH, "#8A4E26"),
    "derived": (SOFT_BLUE, "#3D5F87"),
    "unclassified": ("#F1F3F6", MUTED),
}

FONT_FAMILY = "Inter, 'Segoe UI', 'Helvetica Neue', Arial, sans-serif"

# ---------------------------------------------------------------- chart sizes
CHART_HEIGHT = 380
CHART_HEIGHT_SMALL = 300
CHART_HEIGHT_TALL = 520

# ---------------------------------------------------------------- Plotly template
_template = go.layout.Template()
_template.layout = go.Layout(
    font=dict(family=FONT_FAMILY, size=12, color=TEXT),
    title=dict(font=dict(size=14, color=TEXT), x=0.0, xanchor="left"),
    paper_bgcolor=WHITE,
    plot_bgcolor=WHITE,
    colorway=CATEGORICAL,
    margin=dict(l=10, r=10, t=40, b=10),
    hoverlabel=dict(bgcolor=WHITE, bordercolor=BORDER, font=dict(color=TEXT, family=FONT_FAMILY)),
    legend=dict(bgcolor="rgba(255,255,255,0)", font=dict(color=MUTED, size=11),
                orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    xaxis=dict(gridcolor="#EEF2F6", linecolor=BORDER, zerolinecolor=BORDER,
               tickfont=dict(color=MUTED), title=dict(font=dict(color=MUTED, size=12)), automargin=True),
    yaxis=dict(gridcolor="#EEF2F6", linecolor=BORDER, zerolinecolor=BORDER,
               tickfont=dict(color=MUTED), title=dict(font=dict(color=MUTED, size=12)), automargin=True),
    coloraxis=dict(colorbar=dict(outlinewidth=0, tickfont=dict(color=MUTED))),
    bargap=0.28,
)
pio.templates["luggage_bi"] = _template
pio.templates.default = "plotly_white+luggage_bi"

# ---------------------------------------------------------------- interaction config
# `editable` lets users drag/edit legends, annotations, colour bars and drawn shapes in the browser,
# and the mode bar adds drawing tools (line, path, rectangle, circle, eraser) for marking up a chart.
# Shapes and annotation positions can also be moved. Edits live in the browser
# session only; they never touch the source data.
PLOTLY_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "scrollZoom": False,
    "editable": True,
    "edits": {
        "annotationPosition": True,
        "annotationTail": True,
        "annotationText": True,
        "axisTitleText": False,  # True shows "Click to enter…" placeholders on untitled axes
        "colorbarPosition": True,
        "colorbarTitleText": True,
        "legendPosition": True,
        "legendText": True,
        "shapePosition": True,
        "titleText": False,  # card headings carry the title; True shows an empty-title placeholder
    },
    "modeBarButtonsToAdd": ["drawline", "drawopenpath", "drawrect", "drawcircle", "eraseshape"],
    "toImageButtonOptions": {"format": "svg", "scale": 2},
}


def plotly_config(filename: str = "chart") -> dict:
    cfg = copy.deepcopy(PLOTLY_CONFIG)
    cfg["toImageButtonOptions"]["filename"] = filename
    return cfg
