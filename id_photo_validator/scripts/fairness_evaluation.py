from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - environment dependency
    plt = None

MIN_GROUP_SIZE = 5


def _normalize_stem(path_like: str) -> str:
    """Return lowercase file stem from a path-like string."""
    stem = str(path_like).replace("\\", "/").split("/")[-1]
    if "." in stem:
        stem = stem.rsplit(".", 1)[0]
    return stem.lower()


def _extract_numeric_key(stem: str) -> str | None:
    """Pick the longest numeric run from the stem (helps match ids like 1185)."""
    matches = re.findall(r"\d+", stem)
    if not matches:
        return None
    matches.sort(key=lambda x: (-len(x), x))
    return matches[0]


def _attach_keys(df: pd.DataFrame, file_col: str) -> pd.DataFrame:
    """Add matching keys used to align demographics with predictions."""
    df = df.copy()
    df["file_key"] = df[file_col].apply(_normalize_stem)
    df["numeric_key"] = df["file_key"].apply(_extract_numeric_key)
    return df


def _unique_merge(left: pd.DataFrame, right: pd.DataFrame, key: str) -> Tuple[pd.DataFrame, str]:
    """
    Merge on a key while discarding ambiguous duplicates.

    We keep only rows where the key appears exactly once on each side to avoid
    creating incorrect cartesian matches when filenames repeat.
    """
    left_clean = left.dropna(subset=[key])
    right_clean = right.dropna(subset=[key])

    if left_clean.empty or right_clean.empty:
        return pd.DataFrame(), key

    left_unique = left_clean[~left_clean.duplicated(key, keep=False)]
    right_unique = right_clean[~right_clean.duplicated(key, keep=False)]

    merged = pd.merge(
        left_unique,
        right_unique,
        on=key,
        how="inner",
        suffixes=("_truth", "_pred"),
    )
    return merged, key


def _align_rows(test_df: pd.DataFrame, pred_df: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """Find the best overlap between demographics and predictions."""
    attempts: List[Tuple[int, str, pd.DataFrame]] = []
    for key in ("file_key", "numeric_key"):
        merged, used_key = _unique_merge(test_df, pred_df, key)
        if not merged.empty:
            attempts.append((len(merged), used_key, merged))

    if not attempts:
        raise ValueError("Could not align predictions with demographics; no overlapping file ids found.")

    attempts.sort(key=lambda x: x[0], reverse=True)
    _, key_used, merged_df = attempts[0]
    return merged_df, key_used


def _ensure_prediction_column(pred_df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Create a binary `pred_label` column from whichever prediction data is available."""
    pred_df = pred_df.copy()
    if "pred_label" in pred_df.columns:
        pred_df["pred_label"] = pred_df["pred_label"].astype(int)
        return pred_df

    if "prediction" in pred_df.columns:
        pred_df["pred_label"] = (pred_df["prediction"] >= threshold).astype(int)
        return pred_df

    if "accept" in pred_df.columns:
        pred_df["pred_label"] = pred_df["accept"].astype(int)
        return pred_df

    raise ValueError("Prediction file must include `pred_label`, `prediction`, or `accept`.")


def _confusion_from_series(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    """Compute confusion-matrix-derived metrics for a set of labels."""
    # Drop any rows where either value is missing to avoid sklearn NaN errors.
    valid = pd.concat([y_true, y_pred], axis=1).dropna()
    if valid.empty:
        return {
            "TPR": 0.0,
            "FPR": 0.0,
            "PPV": 0.0,
            "NPV": 0.0,
            "Accuracy": 0.0,
            "TP": 0,
            "TN": 0,
            "FP": 0,
            "FN": 0,
            "count": 0,
        }

    cm = confusion_matrix(valid.iloc[:, 0], valid.iloc[:, 1], labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    tpr = tp / (tp + fn) if tp + fn else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    ppv = tp / (tp + fp) if tp + fp else 0.0
    npv = tn / (tn + fn) if tn + fn else 0.0
    acc = (tp + tn) / max(tn + fp + fn + tp, 1)

    return {
        "TPR": tpr,
        "FPR": fpr,
        "PPV": ppv,
        "NPV": npv,
        "Accuracy": acc,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "count": tn + fp + fn + tp,
    }


def calculate_fairness_metrics(
    merged_df: pd.DataFrame,
    demographic_cols: Iterable[str],
    min_group_size: int = MIN_GROUP_SIZE,
) -> pd.DataFrame:
    """Calculate TPR/FPR and related metrics for each demographic subgroup."""
    rows: List[Dict[str, object]] = []

    for attr in demographic_cols:
        for group_value, group_df in merged_df.groupby(attr):
            if len(group_df) < min_group_size:
                continue
            metrics = _confusion_from_series(group_df["true_label"], group_df["pred_label"])
            rows.append(
                {
                    "attribute": attr,
                    "group": group_value,
                    **metrics,
                }
            )

    return pd.DataFrame(rows)


def plot_fairness_metrics(metrics_df: pd.DataFrame, save_path: Path) -> None:
    """Create a bar plot of TPR/FPR per demographic group."""
    if plt is None:
        print("matplotlib not installed; skipping plot.")
        return
    if metrics_df.empty:
        print("No metrics to plot (no matching data or all groups below minimum size).")
        return

    attributes = metrics_df["attribute"].unique()
    fig, axes = plt.subplots(len(attributes), 1, figsize=(10, 4 * len(attributes)), squeeze=False)
    axes = axes.flatten()

    for ax, attr in zip(axes, attributes):
        subset = metrics_df[metrics_df["attribute"] == attr]
        idx = np.arange(len(subset))

        ax.bar(idx - 0.2, subset["TPR"], width=0.4, color="steelblue", label="TPR")
        ax.bar(idx + 0.2, subset["FPR"], width=0.4, color="coral", label="FPR")
        ax.set_xticks(idx)
        ax.set_xticklabels(subset["group"])
        ax.set_title(f"{attr} groups")
        ax.set_ylim(0, 1)
        ax.legend()

        for x, (_, row) in zip(idx, subset.iterrows()):
            ax.text(x - 0.2, row["TPR"] + 0.02, f"n={int(row['count'])}", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved fairness plot to {save_path}")


def main() -> None:
    test_path = Path("test_set.csv")
    preds_path = Path("test_predictions.csv")

    test_df = pd.read_csv(test_path)
    preds_df = pd.read_csv(preds_path)

    label_col = "label_id" if "label_id" in test_df.columns else "label"
    if label_col not in test_df.columns:
        raise ValueError("test_set.csv must include a label column (label_id or label).")

    demographics = [col for col in ("gender_norm", "glasses_norm", "skin_group") if col in test_df.columns]
    if not demographics:
        raise ValueError("No demographic columns found in test_set.csv.")

    test_df = test_df.rename(columns={label_col: "true_label"})
    preds_df = _ensure_prediction_column(preds_df)

    pred_file_col = None
    for candidate in ("file", "file_name", "filepath", "path"):
        if candidate in preds_df.columns:
            pred_file_col = candidate
            break
    if pred_file_col is None:
        raise ValueError("Prediction file must include a column with file paths (e.g., `file`).")

    test_file_col = "file_name" if "file_name" in test_df.columns else "split_path"

    test_df = _attach_keys(test_df, test_file_col)
    preds_df = _attach_keys(preds_df, pred_file_col)

    merged_df, key_used = _align_rows(test_df, preds_df)

    # Keep the ground-truth label from the demographics file
    if "true_label_truth" in merged_df.columns:
        merged_df = merged_df.rename(columns={"true_label_truth": "true_label"})
    if "true_label_pred" in merged_df.columns:
        merged_df = merged_df.drop(columns=["true_label_pred"])

    coverage_ratio = len(merged_df) / max(len(test_df), 1)
    print(f"Matched {len(merged_df)} of {len(test_df)} demographic rows using `{key_used}` ({coverage_ratio:.1%}).")

    if merged_df.empty:
        raise SystemExit("No overlapping rows between demographics and predictions; cannot compute fairness metrics.")

    metrics_df = calculate_fairness_metrics(merged_df, demographics)
    metrics_path = Path("fairness_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Wrote subgroup metrics to {metrics_path}")

    overall_metrics = _confusion_from_series(merged_df["true_label"], merged_df["pred_label"])
    overall_line = ", ".join([f"{k}={v:.3f}" for k, v in overall_metrics.items() if k in {"TPR", "FPR", "Accuracy"}])
    print(f"Overall: {overall_line}")

    plot_fairness_metrics(metrics_df, Path("fairness_metrics.png"))


if __name__ == "__main__":
    main()
