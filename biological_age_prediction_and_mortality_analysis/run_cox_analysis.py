"""Plot the primary FHS and WHI Cox estimates from the group-level source table."""

from pathlib import Path

from .plotting import clean_axis, configure_style, ordered_rows, save_vector_pair
from .config import MORTALITY_MODELS

import matplotlib.pyplot as plt
import pandas as pd

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "cox"


def plot_external_cox(coefficients: pd.DataFrame) -> None:
    """Draw the separate FHS and WHI primary-model log hazard ratios per score SD."""
    configure_style()
    for cohort, models in MORTALITY_MODELS.items():
        rows = ordered_rows(coefficients.loc[coefficients.cohort.eq(cohort)], models)
        figure, axis = plt.subplots(figsize=(50 / 25.4, 50 / 25.4))
        figure.subplots_adjust(left=0.50, right=0.96, bottom=0.28, top=0.84)
        for y, row in enumerate(rows.itertuples(index=False)):
            beta = row.log_hr_per_sd
            axis.errorbar(
                beta, y, xerr=[[beta - row.log_hr_ci95_low], [row.log_hr_ci95_high - beta]],
                fmt="o", color=row.color, markersize=2.8, linewidth=0.8, capsize=1.5,
                markeredgecolor="#222222", markeredgewidth=0.2,
            )
            axis.text(
                3 / 14, y + (0.24 if y < 2 else -0.24), f"β={beta:.2e}",
                transform=axis.get_yaxis_transform(), ha="left", va="center", fontsize=6,
            )
        axis.axvline(0, color="#777777", linestyle="--", linewidth=0.5)
        axis.set(xlim=(-0.12, 0.88), xticks=[0, 0.4, 0.8], title=cohort.upper())
        axis.set_yticks(range(len(rows)), rows.plot_label.str.replace("-", "\n-", n=1, regex=False))
        axis.invert_yaxis()
        clean_axis(axis, grid=False)
        figure.supxlabel("Log Hazard Ratio", y=0.07, fontsize=9.75)
        save_vector_pair(figure, OUTPUT_DIR, f"04_{cohort}_cox_coefficients_standardized")


def main() -> None:
    coefficients = pd.read_csv(OUTPUT_DIR / "fhs_whi_cox_coefficients_standardized_source_data.csv")
    plot_external_cox(coefficients)


if __name__ == "__main__":
    main()
