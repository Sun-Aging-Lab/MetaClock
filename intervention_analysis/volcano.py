"""Render intervention effects and their BH-adjusted significance."""

from __future__ import annotations

import re
import textwrap

from generalization.fonts import register_fonts
from intervention_analysis import matplotlib_runtime as _matplotlib_runtime  # Set the cache before pyplot.

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.text import Text
from matplotlib.ticker import MultipleLocator

CUTOFF = 0.05
TOP_N = 5
FIGURE_MM = (80, 80)
LABEL_PT, AXIS_TITLE_PT, TICK_PT = 8, 9, 8
BLUE, RED = "#2166AC", "#B2182B"
LIMITS = {
    "fhs": ((-3.5, 3.0), (-0.22, 9.1)),
    "whi": ((-3.7, 3.2), (-0.17, 7.0)),
    "external": ((-10, 10), (-0.40, 16.2)),
}
# Text positions are in data coordinates. Each leader ends at the original estimate.
LABELS = {
    "fhs": {
        "ex1_8s::H518": ("Jogging", -2.65, 4.65),
        "vr_ffreq_ex08_1_0615s::NUT_PRO710": ("Proanthocyanidin\n(7–10-mers)", -2.35, 3.25),
        "ex1_8s::PAI": ("High activity index", -2.03, 7.23),
        "ex1_8s::H513": ("Hiking", -0.30, 5.83),
        "vr_ffreq_ex08_1_0615s::NUT_PRO46": ("Proanthocyanidin\n(4–6-mers)", 0.30, 4.58),
        "vr_wkthru_ex10_1b_1488s::currsmk8": ("Current cigarette\nsmoking", 1.85, 5.50),
        "vr_packyr_2022_a_1563s::PKYRS8": ("Cigarette pack-years", 1.34, 8.57),
        "vr_smokst_2022_m_1564s::STATUS8::current_vs_never": ("Current smoker", 1.74, 7.05),
        "vr_ffreq_ex08_1_0615s::FFD78": ("Processed\nmeats", 2.17, 3.68),
        "ex1_8s::H084": ("5+ drinks\nper day", 2.02, 2.24),
    },
    "whi": {
        "f60_mpeds_hei_rel1::HEI2005": ("High diet quality", -2.25, 5.12),
        "f34_rel2::TEXPWK": ("Recreation\nactivity", -2.50, 4.00),
        "f60_mpeds_hei_rel1::HEI12": ("Fewer empty\ncalories", -2.56, 3.06),
        "f34_rel2::TMINWK": ("Recreation\nactivity duration", -2.49, 2.08),
        "f34_rel2::MSMINWK": ("Moderate–strenuous\nactivity", -0.65, 5.88),
        "f60_nutr_rel1::F60MF141": ("MUFA 14:1", 1.92, 4.08),
        "f34_rel2::PACKYRS": ("Cigarette pack-years", 1.09, 6.46),
        "f34_rel2::SMOKING::current_vs_never": ("Current smoker", 2.27, 5.48),
        "f60_mpeds_hei_rel1::DISCFAT_SOL": ("Discretionary\nsolid fat", 2.40, 3.13),
        "f60_mpeds_hei_rel1::M_MEAT": ("Red meat", 2.19, 2.38),
    },
    "external": {
        "translage::GSE217633": ("Antiretroviral therapy", -4.00, 15.31),
        "geo::curated_1fc35799c70a1f70": ("Abdominal radiotherapy", 4.50, 14.76),
        "geo::curated_fe95adeead1b848b": ("Dexamethasone stimulation", 4.60, 10.44),
        "geo::curated_a5aa77c6d7ffdccc": ("Cigarette smoking", 6.00, 4.64),
    },
}


def configure_style():
    register_fonts()
    plt.rcParams.update({
        "font.family": ["Liberation Sans"], "font.sans-serif": ["Liberation Sans"],
        "axes.labelsize": AXIS_TITLE_PT, "xtick.labelsize": TICK_PT, "ytick.labelsize": TICK_PT,
        "axes.linewidth": 0.2, "xtick.major.width": 0.2, "ytick.major.width": 0.2,
        "xtick.minor.width": 0.2, "ytick.minor.width": 0.2, "patch.linewidth": 0.2,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    })


def select_labels(data, panel):
    significant = data.loc[data.fdr.le(CUTOFF)]
    if panel == "external":
        return significant
    anti = significant.loc[significant.effect.lt(0)].sort_values(["effect", "id"]).head(TOP_N)
    pro = significant.loc[significant.effect.gt(0)].sort_values(
        ["effect", "id"], ascending=[False, True]
    ).head(TOP_N)
    return pd.concat([anti, pro])


def short_name(name):
    text = re.sub(r"\s*\([^()]*\bvs\.?\s+[^()]*\)\s*$", "", str(name), flags=re.I)
    text = re.sub(r"^(?:class|treatment|activity|nutrient|past year):\s*", "", text, flags=re.I)
    return re.split(r"\s+vs\.?\s+", text, maxsplit=1, flags=re.I)[0].strip()


def panel_limits(data, panel, reference):
    extent = data if reference is None else pd.concat([data, reference], ignore_index=True)
    low, high = extent.effect.min(), extent.effect.max()
    peak = max(-np.log10(CUTOFF), extent.height.max())
    padding = max(0.4, (high - low) * 0.08)
    if panel not in LIMITS:
        return (min(0, low) - padding, max(0, high) + padding), (-0.08, peak * 1.25)
    (left, right), (bottom, top) = LIMITS[panel]
    return (
        (low - padding if low <= left else left,
         high + padding if high >= right else right),
        (bottom, peak * 1.10 if peak >= top else top),
    )


def render_volcano(data, output_stem, panel, *, reference=None, offsets=None, formats=("pdf", "svg")):
    """Draw a panel from id, effect, fdr and name columns."""
    if data.empty or data.id.isna().any() or not data.id.is_unique:
        raise ValueError("Expected one row per comparison ID.")
    if not np.isfinite(data.effect).all() or not data.fdr.between(0, 1).all():
        raise ValueError("Effects must be finite and FDR values must be in [0, 1].")
    data = data.assign(height=-np.log10(data.fdr.clip(lower=np.finfo(float).tiny)))
    selected = select_labels(data, panel)
    xlim, ylim = panel_limits(data, panel, reference)

    configure_style()
    fig, ax = plt.subplots(figsize=np.asarray(FIGURE_MM) / 25.4)
    fig.subplots_adjust(left=0.145, right=0.976, bottom=0.15, top=0.973)
    ax.set(xlim=xlim, ylim=ylim, xlabel="Δ MetaClock-biological prediction", ylabel="-log10(FDR)")
    if panel == "external" and xlim[1] - xlim[0] <= 30:
        ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(length=2.5, pad=3)
    ax.axvline(0, color="#BABCC0", linewidth=0.45, zorder=0)
    threshold = -np.log10(CUTOFF)
    ax.axhline(threshold, color="#62676C", linestyle=(0, (4, 3)), linewidth=0.6, zorder=1)
    threshold_label = ax.annotate(
        "FDR = 0.05", (0.99, threshold), xytext=(0, 3), textcoords="offset points",
        xycoords=ax.get_yaxis_transform(), ha="right", va="bottom", color="#505459", fontsize=7,
    )
    for direction, color in [(-1, BLUE), (0, "#888888"), (1, RED)]:
        side = data.loc[np.sign(data.effect).eq(direction)]
        for passed in (False, True):
            group = side.loc[side.fdr.le(CUTOFF).eq(passed)]
            if not group.empty:
                ax.scatter(group.effect, group.height, s=14 if passed else 9,
                           facecolors=color if passed else "none", edgecolors="white" if passed else color,
                           linewidths=0.32 if passed else 0.45, alpha=1 if passed else 0.55,
                           zorder=3 if passed else 2)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    blocked = [Text.get_window_extent(threshold_label, renderer)]
    markers = ax.transData.transform(data[["effect", "height"]].to_numpy())
    records = []
    layout = LABELS.get(panel, {})
    for row in selected.itertuples(index=False):
        key = str(row.id)
        preset = layout.get(key)
        name = preset[0] if preset else textwrap.fill(short_name(row.name), 22, break_long_words=False)
        color = BLUE if row.effect < 0 else RED if row.effect > 0 else "#888888"
        target = (row.effect, row.height)
        note = ax.annotate(name, target, ha="center", va="center", fontsize=LABEL_PT, linespacing=1.21,
                           color=color, zorder=4, bbox=dict(facecolor="white", edgecolor="none", pad=0.4),
                           arrowprops=dict(arrowstyle="-", color=color, lw=0.45, shrinkA=1.2, shrinkB=1.8))
        if offsets and key in offsets:
            dx, dy, ha, va = offsets[key]
            note.set_anncoords("offset points")
            note.set(position=(dx, dy), ha=ha, va=va)
        elif preset and (xlim, ylim) == LIMITS[panel]:
            note.set_position(preset[1:])
        else:
            # Place other labels nearest their points, clearing existing text and markers.
            note.set_anncoords("axes fraction")
            xs, ys = np.meshgrid(np.linspace(0.04, 0.96, 24), np.linspace(0.05, 0.95, 21))
            candidates = np.column_stack((xs.ravel(), ys.ravel()))
            distances = np.linalg.norm(ax.transAxes.transform(candidates) - ax.transData.transform(target), axis=1)
            fewest_hits = np.inf
            best = None
            for i in np.argsort(distances, kind="stable"):
                note.set_position(candidates[i])
                note.update_positions(renderer)
                box = Text.get_window_extent(note, renderer).padded(2.5)
                if not (ax.bbox.contains(*box.min) and ax.bbox.contains(*box.max)):
                    continue
                if any(box.overlaps(other) for other in blocked):
                    continue
                hits = np.sum(np.all((markers >= box.min) & (markers <= box.max), axis=1))
                if hits < fewest_hits:
                    best, fewest_hits = candidates[i], hits
                    if not hits:
                        break
            if best is None:
                plt.close(fig)
                raise ValueError(f"Insufficient space for intervention label {name!r}.")
            note.set_position(best)
        note.update_positions(renderer)
        box = Text.get_window_extent(note, renderer)
        blocked.append(box.padded(1))
        records.append({
            "comparison_id": key, "plot_label": name.replace("\n", " "), "bh_q_value": row.fdr,
            "side": int(np.sign(row.effect)),
            "text_y_axes_fraction": float(ax.transAxes.inverted().transform((0, (box.y0 + box.y1) / 2))[1]),
        })
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in formats:
        fig.savefig(output_stem.with_suffix(f".{extension}"), dpi=300 if extension == "png" else "figure")
    ticks = ax.get_yticks().tolist()
    plt.close(fig)
    return {
        "canvas_mm": list(FIGURE_MM), "xlim": list(xlim), "ylim": list(ylim), "yticks": ticks,
        "n_tests": len(data), "n_labels": len(records), "label_q_threshold": CUTOFF,
        "top_per_direction": None if panel == "external" else TOP_N, "font_family": "Liberation Sans",
        "axis_title_pt": AXIS_TITLE_PT, "intervention_label_pt": LABEL_PT,
        "shared_style_parameters": {"axes_label_size": AXIS_TITLE_PT, "tick_label_size": TICK_PT, "outline_width_pt": 0.2},
        "directional_palette": {"anti": BLUE, "pro": RED}, "labels": records,
    }
