"""
Generate quick EDA visuals (histograms, heatmap) from metadata CSVs.

Outputs:
- metadata/eda_plots/label_distribution.png
- metadata/eda_plots/skin_group_vs_label_heatmap.png
- metadata/eda_plots/correlation_heatmap.png
"""

from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def ensure_outdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_metadata(root: Path, fname: str) -> pd.DataFrame:
    file_path = root / fname
    if not file_path.exists():
        raise SystemExit(f"Metadata file missing: {file_path}")
    return pd.read_csv(file_path)


def save_plot(out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")


def plot_label_hist(df: pd.DataFrame, out_path: Path) -> None:
    sns.countplot(data=df, x="label_id")
    plt.title("Label distribution (0=bad, 1=good)")
    save_plot(out_path)


def plot_skin_group_bar(df: pd.DataFrame, out_path: Path) -> None:
    """
    Representation/fairness view: counts of skin tone groups I–VI, broken out by label.
    """
    order = [g for g in ["I", "II", "III", "IV", "V", "VI"] if g in df["skin_group"].unique()]
    sns.countplot(data=df, x="skin_group", hue="label_id", order=order)
    plt.title("Skin tone distribution by label (0=bad, 1=good)")
    save_plot(out_path)


def plot_categorical_heatmap(df: pd.DataFrame, column: str, out_path: Path) -> None:
    ct = pd.crosstab(df[column], df["label_id"], normalize="index")
    sns.heatmap(ct, annot=True, fmt=".2f", cmap="Blues")
    plt.title(f"{column} vs label (row-normalized)")
    save_plot(out_path)


def plot_numeric_corr(df: pd.DataFrame, out_path: Path, numeric_cols: List[str]) -> None:
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Numeric feature correlations")
    save_plot(out_path)


def plot_quality_scatter(df: pd.DataFrame, out_path: Path) -> None:
    """
    Scatter of blur vs brightness, colored by true label, sized by det_score.
    Shows patterns between image quality indicators and labels.
    """
    required = {"blur_metric", "brightness", "true_label", "det_score"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        print(f"Skipping scatter plot; missing columns: {missing}")
        return
    sns.scatterplot(
        data=df,
        x="brightness",
        y="blur_metric",
        hue="true_label",
        size="det_score",
        sizes=(20, 150),
        alpha=0.6,
    )
    plt.title("Blur vs Brightness by label (size = det_score)")
    save_plot(out_path)


def main() -> None:
    meta_root = Path(__file__).parent / "metadata"
    out_root = meta_root / "eda_plots"
    ensure_outdir(out_root)

    df = load_metadata(meta_root, "train_set.csv")
    # Optional: predictions file for feature-label patterns
    pred_path = meta_root / "test_predictions.csv"
    pred_df = None
    if pred_path.exists():
        pred_df = pd.read_csv(pred_path)
    else:
        print(f"Prediction file not found: {pred_path} (skipping scatter analysis)")

    # Plots
    plot_label_hist(df, out_root / "label_distribution.png")
    if "skin_group" in df.columns:
        plot_skin_group_bar(df, out_root / "skin_group_distribution_by_label.png")
        plot_categorical_heatmap(df, "skin_group", out_root / "skin_group_vs_label_heatmap.png")

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if numeric_cols:
        plot_numeric_corr(df, out_root / "correlation_heatmap.png", numeric_cols)
    else:
        print("No numeric columns found for correlation heatmap.")

    # Patterns between image quality indicators and labels (requires predictions CSV)
    if pred_df is not None:
        # Correlation among numeric quality indicators and predicted/true labels
        num_cols_pred = [
            c
            for c in pred_df.columns
            if pd.api.types.is_numeric_dtype(pred_df[c])
        ]
        if num_cols_pred:
            plot_numeric_corr(pred_df[num_cols_pred], out_root / "prediction_quality_correlations.png", num_cols_pred)
        plot_quality_scatter(pred_df, out_root / "blur_vs_brightness_scatter.png")


if __name__ == "__main__":
    main()
