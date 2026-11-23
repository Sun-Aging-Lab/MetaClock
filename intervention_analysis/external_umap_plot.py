"""Render the single-panel intervention UMAP and its sample-level hover view."""

from __future__ import annotations

from pathlib import Path

from intervention_analysis import matplotlib_runtime  # noqa: F401
from generalization.fonts import register_fonts
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import plotly.graph_objects as go


STEM = "intervention_with_controls_umap"
COLORS = {"control": "#3C5488", "intervention": "#00A087"}


def render(samples: pd.DataFrame, dest: Path) -> None:
    counts = samples.plot_group.value_counts()
    labels = {group: f"{group.capitalize()} (n = {counts[group]:,})" for group in COLORS}
    colors = samples.plot_group.map(COLORS)
    limits = []
    for column in ("umap_1", "umap_2"):
        low, high = samples[column].min(), samples[column].max()
        margin = .045 * (high - low)
        limits.append((low - margin, high + margin))
    register_fonts()
    with plt.rc_context({"font.family": ["Liberation Sans"], "font.sans-serif": ["Liberation Sans"],
                         "axes.linewidth": .2, "xtick.major.width": .2, "ytick.major.width": .2,
                         "patch.linewidth": .2, "pdf.fonttype": 42, "svg.fonttype": "none"}):
        fig, ax = plt.subplots(figsize=(90 / 25.4, 100 / 25.4))
        try:
            points = ax.scatter(samples.umap_1, samples.umap_2, s=samples.marker_area_all,
                                c=colors, alpha=.85, linewidths=0)
            ax.set(xlim=limits[0], ylim=limits[1], xticks=[], yticks=[], aspect="equal")
            ax.spines[["top", "right"]].set_visible(False)
            fig.subplots_adjust(left=.13, right=.98, top=.982, bottom=.208)
            handles = [Line2D([], [], linestyle="none", marker="o", markersize=6, markerfacecolor=color,
                              markeredgewidth=0, label=labels[group]) for group, color in COLORS.items()]
            fig.canvas.draw()
            anchor = fig.transFigure.inverted().transform(((ax.bbox.x0 + ax.bbox.x1) / 2,
                                                           ax.bbox.y0 - 4 * fig.dpi / 72))
            fig.legend(handles=handles, loc="upper center", bbox_to_anchor=anchor, frameon=False, fontsize=12,
                       handlelength=.7, handletextpad=.25, columnspacing=.6, labelspacing=.4,
                       borderaxespad=0, borderpad=0)
            fig.savefig(dest / f"{STEM}.png", dpi=240)
            points.set_rasterized(True)
            for ext in ("pdf", "svg"):
                fig.savefig(dest / f"{STEM}.{ext}", dpi=600)
        finally:
            plt.close(fig)
    plot = go.Figure(go.Scattergl(
        x=samples.umap_1, y=samples.umap_2, mode="markers", showlegend=False,
        marker=dict(size=3, color=colors.tolist(), opacity=.85, symbol="circle", line=dict(width=0)),
        customdata=samples[["sample_id", "plot_group", "role", "hover_labels"]].to_numpy(),
        hovertemplate="Sample: %{customdata[0]}<br>Plotted group: %{customdata[1]}<br>"
                      "Original role: %{customdata[2]}<br>%{customdata[3]}<extra></extra>",
    ))
    for group, color in COLORS.items():
        plot.add_scatter(x=[None], y=[None], mode="markers", marker=dict(size=10, color=color),
                         name=labels[group], showlegend=True, hoverinfo="skip")
    axes = dict(showticklabels=False, ticks="", zeroline=False, showgrid=False,
                showline=True, linecolor="black", linewidth=1, mirror=False, constrain="domain")
    plot.update_xaxes(range=limits[0], **axes)
    plot.update_yaxes(range=limits[1], scaleanchor="x", scaleratio=1, **axes)
    plot.update_layout(template="plotly_white", height=680, showlegend=True, font=dict(family="Liberation Sans, Arial"),
                       margin=dict(l=55, r=15, t=15, b=95),
                       legend=dict(orientation="v", x=.5, xanchor="center", y=-.01, yanchor="top",
                                   font=dict(size=16), itemclick=False, itemdoubleclick=False))
    graph = plot.to_html(full_html=False, include_plotlyjs=True, div_id="umap_graph",
                         config=dict(displaylogo=False, responsive=True))
    (dest / f"{STEM}.html").write_text(
        '<!doctype html><html><head><meta charset="utf-8"><title>All interventions and controls UMAP</title>'
        '<style>body{margin:0;background:white}#umap_graph{min-width:440px;max-width:700px;margin:auto}'
        f'</style></head><body>{graph}</body></html>', encoding="utf-8")
