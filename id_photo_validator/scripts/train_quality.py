"""
Train a quality-only classifier on ArcFace embeddings, optionally appending OFIQ scores.
Dataset layout (binary good/bad):

data/id_photo_dataset/
  train/good/...
  train/bad/...
  val/good/...
  val/bad/...
  test/good/...   # optional

Usage:
  python -m scripts.train_quality --dataset-root data/id_photo_dataset --output models/quality_head.joblib
"""

import argparse
import logging
import random
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from validator_service import ArcFacePipeline, ValidatorConfig
from validator_service.augmentations import generate_bad_variants, generate_good_variants
from validator_service.ofiq_adapter import OFIQScorer

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_images(directory: Path) -> Iterable[Path]:
    for path in sorted(directory.iterdir(), key=lambda p: p.name):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


def load_image(path: Path) -> Optional[Image.Image]:
    """
    Open an image; return None if unreadable/corrupt instead of raising.
    """
    try:
        return Image.open(path).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        logging.warning("Skipping unreadable file: %s (%s)", path, exc)
        return None


def label_from_dirname(name: str) -> Optional[int]:
    name = name.lower()
    if name in {"good", "high-quality", "high_quality", "highquality"}:
        return 1
    if name in {"bad", "low-quality", "low_quality", "lowquality"}:
        return 0
    return None


def embed_image(img: Image.Image, pipeline: ArcFacePipeline) -> Tuple[np.ndarray, int, np.ndarray, float]:
    arr = np.asarray(img)
    result = pipeline.extract(arr)
    return result.embedding, result.face_count, result.aligned_face, result.det_score


def collect_split(
    split_dir: Path,
    pipeline: ArcFacePipeline,
    split_name: str,
    use_ofiq: bool,
    ofiq: Optional[OFIQScorer],
    augment_bad: bool,
    augment_good: bool,
    max_aug_bad: int,
    max_aug_good: int,
    det_score_thresh: float,
) -> Tuple[List[np.ndarray], List[int]]:
    embeddings: List[np.ndarray] = []
    labels: List[int] = []
    label_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
    if not label_dirs:
        raise SystemExit(f"No class folders found in {split_dir}; expected 'good' and 'bad'.")

    for class_dir in sorted(label_dirs, key=lambda p: p.name):
        label = label_from_dirname(class_dir.name)
        if label is None:
            logging.warning("Skipping unknown label folder %s", class_dir)
            continue
        for img_path in iter_images(class_dir):
            img = load_image(img_path)
            if img is None:
                continue
            try:
                emb, face_count, aligned, det_score = embed_image(img, pipeline)
            except Exception as exc:
                logging.warning("Skipping %s (%s)", img_path, exc)
                continue
            if face_count != 1:
                logging.warning("Skipping %s because face_count=%s", img_path, face_count)
                continue
            if det_score < det_score_thresh:
                logging.warning("Skipping %s because det_score=%.3f < %.2f", img_path, det_score, det_score_thresh)
                continue

            def add_sample(e: np.ndarray, a: np.ndarray) -> None:
                feature = e
                if use_ofiq and ofiq is not None:
                    score = ofiq.score(a)
                    score_val = score if score is not None else 0.0
                    feature = np.concatenate([e, np.array([score_val], dtype=e.dtype)])
                embeddings.append(feature)
                labels.append(label)

            add_sample(emb, aligned)

            # Augmentations only for train split.
            if split_name == "train":
                if label == 0 and augment_bad:
                    for aug in generate_bad_variants(img, max_variants=max_aug_bad):
                        try:
                            emb_bad, fc_bad, aligned_bad, det_bad = embed_image(aug, pipeline)
                        except Exception:
                            continue
                        if fc_bad != 1 or det_bad < det_score_thresh:
                            continue
                        add_sample(emb_bad, aligned_bad)
                if label == 1 and augment_good:
                    for aug in generate_good_variants(img, max_variants=max_aug_good):
                        try:
                            emb_good, fc_good, aligned_good, det_good = embed_image(aug, pipeline)
                        except Exception:
                            continue
                        if fc_good != 1 or det_good < det_score_thresh:
                            continue
                        add_sample(emb_good, aligned_good)

    return embeddings, labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("data/id_photo_dataset"),
        help="Root directory containing train/ val/ test/ with good/ and bad/ folders.",
    )
    parser.add_argument("--train-dir", type=Path, help="Override training dir (defaults to dataset-root/train).")
    parser.add_argument("--val-dir", type=Path, help="Override validation dir (defaults to dataset-root/val).")
    parser.add_argument("--test-dir", type=Path, help="Override test dir (defaults to dataset-root/test).")
    parser.add_argument(
        "--output",
        type=Path,
        default=ValidatorConfig().quality_model_path,
        help="Where to save the trained classifier.",
    )
    parser.add_argument(
        "--with-ofiq",
        action="store_true",
        help="Append OFIQ score as an extra feature (requires ofiq package).",
    )
    parser.add_argument("--augment-bad", action="store_true", help="Augment bad samples with degradations.")
    parser.add_argument("--augment-good", action="store_true", help="Mild jitter for good samples.")
    parser.add_argument("--max-aug-bad", type=int, default=6, help="Cap per-image bad augmentations.")
    parser.add_argument("--max-aug-good", type=int, default=2, help="Cap per-image good augmentations.")
    parser.add_argument("--max-bad", type=int, default=None, help="Optional cap on total bad samples after augment.")
    parser.add_argument(
        "--det-score-thresh",
        type=float,
        default=0.45,
        help="Drop detections with confidence below this threshold.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()

    train_dir = args.train_dir or args.dataset_root / "train"
    val_dir = args.val_dir or args.dataset_root / "val"
    test_dir = args.test_dir or args.dataset_root / "test"

    for required_dir in (train_dir, val_dir):
        if not required_dir.exists():
            raise SystemExit(f"Missing split directory: {required_dir}")

    # Seed everything for reproducibility.
    random.seed(args.seed)
    np.random.seed(args.seed)

    use_ofiq = args.with_ofiq
    ofiq: Optional[OFIQScorer] = None
    if use_ofiq:
        try:
            ofiq = OFIQScorer()
            logging.info("OFIQ enabled for training features")
        except Exception as exc:
            logging.warning("OFIQ requested but unavailable (%s); continuing without it", exc)
            use_ofiq = False

    config = ValidatorConfig()
    pipeline = ArcFacePipeline(config)

    logging.info("Collecting train split from %s", train_dir)
    train_embeddings, train_labels = collect_split(
        train_dir,
        pipeline,
        "train",
        use_ofiq,
        ofiq,
        augment_bad=args.augment_bad,
        augment_good=args.augment_good,
        max_aug_bad=args.max_aug_bad,
        max_aug_good=args.max_aug_good,
        det_score_thresh=args.det_score_thresh,
    )
    logging.info("Collecting val split from %s", val_dir)
    val_embeddings, val_labels = collect_split(
        val_dir,
        pipeline,
        "val",
        use_ofiq,
        ofiq,
        augment_bad=False,
        augment_good=False,
        max_aug_bad=0,
        max_aug_good=0,
        det_score_thresh=args.det_score_thresh,
    )

    test_embeddings: List[np.ndarray] = []
    test_labels: List[int] = []
    if test_dir.exists():
        logging.info("Collecting test split from %s", test_dir)
        test_embeddings, test_labels = collect_split(
            test_dir,
            pipeline,
            "test",
            use_ofiq,
            ofiq,
            augment_bad=False,
            augment_good=False,
            max_aug_bad=0,
            max_aug_good=0,
            det_score_thresh=args.det_score_thresh,
        )

    if not train_embeddings:
        raise SystemExit("No embeddings collected from training split; check your data paths.")

    X_train = np.stack(train_embeddings)
    y_train = np.array(train_labels)

    # Optional cap on bad samples to avoid flooding when augmenting.
    if args.max_bad is not None:
        bad_idx = np.where(y_train == 0)[0]
        if len(bad_idx) > args.max_bad:
            rng = np.random.default_rng(args.seed)
            keep_bad = rng.choice(bad_idx, size=args.max_bad, replace=False)
            good_idx = np.where(y_train == 1)[0]
            keep = np.concatenate([good_idx, keep_bad])
            X_train = X_train[keep]
            y_train = y_train[keep]
            logging.info(
                "Downsampled bad samples from %d to %d; total samples now %d",
                len(bad_idx),
                args.max_bad,
                len(y_train),
            )

    clf = LogisticRegression(max_iter=400, class_weight="balanced", solver="lbfgs")
    clf.fit(X_train, y_train)

    def report_split(name: str, X: List[np.ndarray], y: List[int]) -> None:
        if not X:
            logging.info("No samples found in %s split; skipping report.", name)
            return
        y_pred = clf.predict(np.stack(X))
        report = classification_report(y, y_pred, target_names=["bad", "good"])
        logging.info("%s report:\n%s", name.capitalize(), report)

    report_split("val", val_embeddings, val_labels)
    report_split("test", test_embeddings, test_labels)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    import joblib

    payload = {
        "classifier": clf,
        "label_names": ["bad", "good"],
        "feature_meta": {"use_ofiq": use_ofiq},
    }
    joblib.dump(payload, args.output)
    logging.info("Saved quality classifier to %s (use_ofiq=%s)", args.output, use_ofiq)


if __name__ == "__main__":
    main()
