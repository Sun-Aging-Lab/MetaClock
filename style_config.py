from __future__ import annotations

FIGURE_TITLE_FONT_SIZE = 24
SUBPLOT_TITLE_FONT_SIZE = 22
AXIS_TITLE_FONT_SIZE = 20
TICK_FONT_SIZE = 18
LEGEND_FONT_SIZE = TICK_FONT_SIZE
ANNOTATION_FONT_SIZE = TICK_FONT_SIZE
FONT_FAMILY = "Arial"

FIGURE2_3_FONT_SCALE = 1.5
FIGURE2_3_FIGURE_TITLE_FONT_SIZE = FIGURE_TITLE_FONT_SIZE * FIGURE2_3_FONT_SCALE
FIGURE2_3_SUBPLOT_TITLE_FONT_SIZE = SUBPLOT_TITLE_FONT_SIZE * FIGURE2_3_FONT_SCALE
FIGURE2_3_AXIS_TITLE_FONT_SIZE = AXIS_TITLE_FONT_SIZE * FIGURE2_3_FONT_SCALE
FIGURE2_3_TICK_FONT_SIZE = TICK_FONT_SIZE * FIGURE2_3_FONT_SCALE
FIGURE2_3_LEGEND_FONT_SIZE = LEGEND_FONT_SIZE * FIGURE2_3_FONT_SCALE
FIGURE2_3_ANNOTATION_FONT_SIZE = ANNOTATION_FONT_SIZE * FIGURE2_3_FONT_SCALE

FIG3_PANEL_FIGSIZE = (12.0, 9.0)
BIOAGE_FINAL_FIGSIZE = (12.0, 9.0)

FIG3_SHADE_ALPHA = 0.30
FIG3_BASELINE_SHADE_ALPHA = FIG3_SHADE_ALPHA
FIG3_META_SHADE_ALPHA = FIG3_SHADE_ALPHA
FIG3_EVOLUTION_SHADE_ALPHA = FIG3_SHADE_ALPHA


def apply_plot_style(plt, font_scale: float = 1.0) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [FONT_FAMILY, "Liberation Sans", "DejaVu Sans"],
            "font.size": TICK_FONT_SIZE * font_scale,
            "figure.titlesize": FIGURE_TITLE_FONT_SIZE * font_scale,
            "axes.titlesize": SUBPLOT_TITLE_FONT_SIZE * font_scale,
            "axes.labelsize": AXIS_TITLE_FONT_SIZE * font_scale,
            "xtick.labelsize": TICK_FONT_SIZE * font_scale,
            "ytick.labelsize": TICK_FONT_SIZE * font_scale,
            "legend.fontsize": LEGEND_FONT_SIZE * font_scale,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
