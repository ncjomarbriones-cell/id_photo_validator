"""
Batch quality evaluation script.

- Scans a test folder containing class subfolders (e.g., high_quality/, low_quality/).
- Uses the trained quality head (models/quality_head.joblib) with the InsightFace
  ArcFace pipeline to generate predictions.
- Writes a CSV with per-image results and prints a quick accuracy summary.

Usage:
  python testing.py --dataset-root data/id_photo_dataset/test --output test_predictions.csv
"""

import argparse
import logging
import csv
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError

from validator_service import ArcFacePipeline, QualityClassifier, ValidatorConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_images(directory: Path) -> Iterable[Path]:
    for path in directory.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def label_from_dir(dirname: str) -> int:
    name = dirname.lower()
    if "high" in name:
        return 1
    if "low" in name:
        return 0
    raise ValueError(f"Unknown label folder: {dirname}")


def evaluate_split(
    split_dir: Path,
    pipeline: ArcFacePipeline,
    classifier: QualityClassifier,
) -> Tuple[List[dict], float]:
    rows: List[dict] = []
    total = 0
    correct = 0

    label_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
    if not label_dirs:
        raise SystemExit(f"No class folders found in {split_dir}")

    for class_dir in sorted(label_dirs, key=lambda p: p.name):
        true_label = label_from_dir(class_dir.name)
        for img_path in iter_images(class_dir):
            try:
                img = load_image(img_path)
            except UnidentifiedImageError:
                logging.warning("Skipping non-image file: %s", img_path)
                continue
            try:
                result = pipeline.extract(np.asarray(img))
            except Exception as exc:
                logging.warning("Skipping %s (%s)", img_path, exc)
                continue
            try:
                q = classifier.score(
                    embedding=result.embedding,
                    aligned_face=result.aligned_face,
                    face_count=result.face_count,
                    det_score=result.det_score,
                )
            except Exception as exc:
                logging.warning("Quality scoring failed for %s (%s)", img_path, exc)
                continue

            pred_label = 1 if q.accept else 0
            total += 1
            correct += int(pred_label == true_label)

            rows.append(
                {
                    "file": str(img_path),
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "quality_score": q.quality_score,
                    "borderline": q.borderline,
                    "accept": q.accept,
                    "face_count": q.face_count,
                    "det_score": q.det_score,
                    "blur_metric": q.blur_metric,
                    "brightness": q.brightness,
                    "ofiq_score": q.ofiq_score,
                    "reasons": "; ".join(q.reasons),
                }
            )

    accuracy = correct / total if total else 0.0
    return rows, accuracy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/id_photo_dataset/test"),
        help="Root folder with class subfolders (e.g., high_quality/ and low_quality/).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("test_predictions.csv"),
        help="Where to save the CSV of predictions.",
    )
    args = parser.parse_args()

    if not args.dataset_root.exists():
        raise SystemExit(f"Dataset root does not exist: {args.dataset_root}")

    config = ValidatorConfig()
    pipeline = ArcFacePipeline(config)
    classifier = QualityClassifier(config)
    if not classifier.is_ready():
        raise SystemExit(f"Quality model not found at {config.quality_model_path}")

    rows, acc = evaluate_split(args.dataset_root, pipeline, classifier)
    if not rows:
        raise SystemExit("No predictions were generated. Check dataset and logs.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    logging.info("Wrote predictions to %s", args.output)
    logging.info("Accuracy on %s: %.3f (%d samples)", args.dataset_root, acc, len(rows))


if __name__ == "__main__":
    main()
