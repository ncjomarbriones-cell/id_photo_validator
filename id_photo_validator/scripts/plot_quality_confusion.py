"""
Generate a confusion-style 2x2 tile for the quality classifier on the test split.

Usage:
  python -m python.id_photo_validator.scripts.plot_quality_confusion --dataset-root data/id_photo_dataset --model models/quality_head.joblib

Output:
  Saves PNG to python/id_photo_validator/scripts/metadata/quality_confusion_test.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

from validator_service import ArcFacePipeline, ValidatorConfig
from validator_service.ofiq_adapter import OFIQScorer

# Reuse the data collection routine from training to stay consistent.
from .train_quality import collect_split  # type: ignore


def draw_confusion_tile(
    ax: plt.Axes,
    tp: int,
    fn: int,
    fp: int,
    tn: int,
    positive_label: str,
    negative_label: str,
) -> None:
    """
    Plot a 2x2 confusion grid with predicted classes on rows and actual on columns.
    Layout matches the reference image:
        columns: Actual [Positive, Negative]
        rows:    Predicted [Positive, Negative]
    """
    correct_color = "#B7E283"
    error_color = "#F9A48B"
    header_color = "#F7F3A3"

    # Background headers (top and left) for actual/predicted cues.
    ax.add_patch(plt.Rectangle((-0.02, -0.02), 2.04, 0.24, facecolor=header_color, edgecolor="none", zorder=0))
    ax.add_patch(plt.Rectangle((-0.24, -0.02), 0.24, 2.04, facecolor=header_color, edgecolor="none", zorder=0))

    cells = {
        (0, 0): ("TP", tp, correct_color),
        (0, 1): ("FP", fp, error_color),
        (1, 0): ("FN", fn, error_color),
        (1, 1): ("TN", tn, correct_color),
    }

    for (row, col), (label, value, color) in cells.items():
        rect = plt.Rectangle((col, row), 1, 1, facecolor=color, edgecolor="white", linewidth=2)
        ax.add_patch(rect)
        ax.text(col + 0.5, row + 0.52, label, ha="center", va="center", fontsize=15, color="black", fontweight="bold")
        ax.text(col + 0.5, row + 0.2, f"({value})", ha="center", va="center", fontsize=11, color="black")

    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels([positive_label, negative_label], fontsize=11, fontweight="bold")
    ax.tick_params(bottom=False, labelbottom=True)
    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels([positive_label, negative_label], fontsize=11, fontweight="bold")
    ax.tick_params(left=False, labelleft=True)
    ax.invert_yaxis()

    ax.set_xlabel("ACTUAL VALUES", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("PREDICTED VALUES", fontsize=12, fontweight="bold", labelpad=12)
    for spine in ax.spines.values():
        spine.set_visible(False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("data/id_photo_dataset"))
    parser.add_argument("--model", type=Path, default=ValidatorConfig().quality_model_path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("python/id_photo_validator/scripts/metadata/quality_confusion_test.png"),
        help="Where to save the confusion tile PNG.",
    )
    parser.add_argument(
        "--with-ofiq",
        action="store_true",
        help="Force OFIQ usage when scoring test embeddings (overrides model metadata).",
    )
    parser.add_argument(
        "--det-score-thresh",
        type=float,
        default=0.45,
        help="Detection score threshold to keep faces, matches training default.",
    )
    args = parser.parse_args()

    test_dir = args.dataset_root / "test"
    if not test_dir.exists():
        raise SystemExit(f"Test split not found at {test_dir}")

    if not args.model.exists():
        raise SystemExit(f"Trained quality model not found at {args.model}")

    payload = joblib.load(args.model)
    clf = payload["classifier"]
    meta = payload.get("feature_meta", {})
    use_ofiq = True if args.with_ofiq else bool(meta.get("use_ofiq", False))

    ofiq = None
    if use_ofiq:
        try:
            ofiq = OFIQScorer()
        except Exception:
            # Fall back silently; we do not want plotting to fail if OFIQ is unavailable.
            use_ofiq = False

    pipeline = ArcFacePipeline(ValidatorConfig())

    test_embeddings, test_labels = collect_split(
        test_dir,
        pipeline,
        "test",
        use_ofiq=use_ofiq,
        ofiq=ofiq,
        augment_bad=False,
        augment_good=False,
        max_aug_bad=0,
        max_aug_good=0,
        det_score_thresh=args.det_score_thresh,
    )

    if not test_embeddings:
        raise SystemExit("No samples gathered from test split; cannot plot confusion matrix.")

    y_true = np.array(test_labels)
    y_pred = clf.predict(np.stack(test_embeddings))
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    pos_label = meta.get("positive_label", meta.get("label_names", ["good", "bad"])[-1] if meta.get("label_names") else "positive")
    neg_label = meta.get("negative_label", meta.get("label_names", ["bad", "good"])[0] if meta.get("label_names") else "negative")

    fig, ax = plt.subplots(figsize=(5, 5))
    draw_confusion_tile(
        ax,
        tp=int(tp),
        fn=int(fn),
        fp=int(fp),
        tn=int(tn),
        positive_label=pos_label,
        negative_label=neg_label,
    )
    fig.suptitle("Quality classifier – Test confusion", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=300)
    plt.close(fig)
    print(f"Saved confusion plot to {args.output}")


if __name__ == "__main__":
    main()
