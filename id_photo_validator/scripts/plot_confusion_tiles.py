"""
Render fairness confusion-style tiles for each demographic subgroup using
`metadata/fairness_metrics.csv`.

The output image shows TP/FN/FP/TN counts in a 2x2 grid for every
attribute/group combination.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _load_metrics(metrics_path: Path) -> pd.DataFrame:
    if not metrics_path.exists():
        raise SystemExit(f"fairness metrics not found: {metrics_path}")
    df = pd.read_csv(metrics_path)
    required = {"attribute", "group", "TP", "TN", "FP", "FN"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns in {metrics_path}: {sorted(missing)}")
    return df


def _draw_confusion_tile(ax: plt.Axes, tp: int, fn: int, fp: int, tn: int) -> None:
    """
    Draw a 2x2 confusion-style grid on the provided axes.
    """
    correct_color = "#336D92"
    error_color = "#9BD0F5"

    cells = {
        (0, 0): ("TP", tp, correct_color),
        (0, 1): ("FN", fn, error_color),
        (1, 0): ("FP", fp, error_color),
        (1, 1): ("TN", tn, correct_color),
    }

    for (row, col), (label, value, color) in cells.items():
        rect = plt.Rectangle((col, row), 1, 1, facecolor=color, edgecolor="white", linewidth=2)
        ax.add_patch(rect)
        ax.text(col + 0.5, row + 0.6, label, ha="center", va="center", fontsize=16, color="white", fontweight="bold")
        ax.text(col + 0.5, row + 0.25, f"{value}", ha="center", va="center", fontsize=11, color="white")

    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(["Positive", "Negative"])
    ax.tick_params(bottom=False, labelbottom=True)
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(["Positive", "Negative"])
    ax.tick_params(left=False, labelleft=True)
    ax.invert_yaxis()
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_tiles(df: pd.DataFrame, out_path: Path) -> None:
    # Group by attribute so rows share the same attribute label.
    grouped = list(df.groupby("attribute"))
    max_per_row = max(len(g) for _, g in grouped)
    n_rows = len(grouped)
    fig, axes = plt.subplots(n_rows, max_per_row, figsize=(4 * max_per_row, 4 * n_rows), squeeze=False)

    for row_idx, (attr, group_df) in enumerate(grouped):
        for col_idx in range(max_per_row):
            ax = axes[row_idx, col_idx]
            if col_idx >= len(group_df):
                ax.axis("off")
                continue
            row = group_df.iloc[col_idx]
            _draw_confusion_tile(ax, int(row["TP"]), int(row["FN"]), int(row["FP"]), int(row["TN"]))
            title = f"{attr}: {row['group']}"
            ax.set_title(title, fontsize=12, pad=8)

        # Label the left-most plot with the attribute name.
        axes[row_idx, 0].text(
            -0.8,
            1,
            attr,
            ha="right",
            va="center",
            rotation=90,
            fontsize=12,
            fontweight="bold",
            transform=axes[row_idx, 0].transData,
        )

    fig.suptitle("Confusion-style view per subgroup (counts)", fontsize=16, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    print(f"Saved confusion tiles to {out_path}")


def main() -> None:
    script_dir = Path(__file__).parent
    metrics_path = script_dir / "metadata" / "fairness_metrics.csv"
    output_path = script_dir / "metadata" / "confusion_tiles.png"

    df = _load_metrics(metrics_path)
    plot_tiles(df, output_path)


if __name__ == "__main__":
    main()
