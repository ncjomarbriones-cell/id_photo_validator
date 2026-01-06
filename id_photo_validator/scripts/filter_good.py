"""
Filter good training images by ensuring exactly one detected face.

Moves unusable files from data/good to data/good_rejects.
"""

import shutil
from pathlib import Path

import cv2

from validator_service import ArcFacePipeline, ValidatorConfig


def main() -> None:
    base = Path(__file__).resolve().parent.parent
    good_dir = base / "data" / "good"
    reject_dir = base / "data" / "good_rejects"
    reject_dir.mkdir(parents=True, exist_ok=True)

    cfg = ValidatorConfig()
    pipeline = ArcFacePipeline(cfg)

    total = 0
    moved = 0

    for path in good_dir.iterdir():
        if not path.is_file():
            continue
        total += 1
        try:
            bgr = cv2.imread(str(path))
            if bgr is None:
                raise ValueError("Could not read image")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            result = pipeline.extract(rgb)
            if result.face_count != 1:
                raise ValueError(f"face_count={result.face_count}")
            # keep
        except Exception:
            dest = reject_dir / path.name
            shutil.move(str(path), dest)
            moved += 1

    print(f"Checked: {total}, moved to rejects: {moved}, kept: {total - moved}")
    print(f"Rejects folder: {reject_dir}")


if __name__ == "__main__":
    main()
