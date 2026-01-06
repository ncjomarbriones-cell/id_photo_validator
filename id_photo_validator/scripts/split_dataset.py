"""
Split raw ID photo data into train/val/test folders with a fixed ratio.

Usage:
  python scripts/split_dataset.py --src data/raw_id_photo_dataset --dst data/id_photo_dataset --train 0.7 --val 0.15
"""

import argparse
import random
import shutil
from pathlib import Path
from typing import Dict, Iterable, Tuple

# Accepted image extensions (lowercase)
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jfif"}


def iter_images(directory: Path) -> Iterable[Path]:
    return (p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in EXTS)


def clear_destination(dst: Path, classes: Tuple[str, str]) -> None:
    for split in ("train", "val", "test"):
        for cls in classes:
            d = dst / split / cls
            d.mkdir(parents=True, exist_ok=True)
            for f in d.glob("*"):
                if f.is_file():
                    f.unlink()


def split_and_copy(src: Path, dst: Path, ratios: Dict[str, float], seed: int = 42) -> None:
    classes = ("high_quality", "low_quality")
    clear_destination(dst, classes)
    rng = random.Random(seed)

    for cls in classes:
        src_dir = src / cls
        if not src_dir.exists():
            print(f"Warning: missing class dir {src_dir}")
            continue

        files = list(iter_images(src_dir))
        rng.shuffle(files)
        n = len(files)
        n_train = int(n * ratios["train"])
        n_val = int(n * ratios["val"])
        n_test = n - n_train - n_val

        splits = (
            ("train", files[:n_train]),
            ("val", files[n_train : n_train + n_val]),
            ("test", files[n_train + n_val :]),
        )

        for split, items in splits:
            tgt_dir = dst / split / cls
            for f in items:
                shutil.copy2(f, tgt_dir / f.name)

        print(f"Class {cls}: total {n}, train {n_train}, val {n_val}, test {n_test}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, default=Path("data/raw_id_photo_dataset"), help="Source raw data dir")
    parser.add_argument("--dst", type=Path, default=Path("data/id_photo_dataset"), help="Destination split dir")
    parser.add_argument("--train", type=float, default=0.7, help="Train ratio")
    parser.add_argument("--val", type=float, default=0.15, help="Val ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    ratios = {"train": args.train, "val": args.val}
    ratios["test"] = 1.0 - args.train - args.val
    if ratios["test"] <= 0:
        raise SystemExit("Invalid ratios; train + val must be < 1.0")

    if not args.src.exists():
        raise SystemExit(f"Source directory missing: {args.src}")

    split_and_copy(args.src, args.dst, ratios, seed=args.seed)
    print("Done splitting.")


if __name__ == "__main__":
    main()
