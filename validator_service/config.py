from dataclasses import dataclass
from pathlib import Path


@dataclass
class ValidatorConfig:
    """
    Central place for tunable knobs used by the validator.
    Quality-only: no identity recognition paths are referenced.
    """

    # InsightFace model name; buffalo_l ships with ArcFace + RetinaFace.
    arcface_model: str = "buffalo_l"
    # Detection canvas size (w, h).
    det_size: tuple[int, int] = (640, 640)

    # Where the trained quality head is stored.
    quality_model_path: Path = Path(__file__).resolve().parent.parent / "models" / "quality_head.joblib"

    # Thresholds for decisioning on the quality score (quality head only).
    accept_threshold: float = 0.5
    borderline_threshold: float = 0.35

    # Detection handling: if True, reject when more than one face is detected.
    # If False, keep the highest-score face and continue (face_count still reported).
    strict_multi_face_reject: bool = False
    # When True, collapse multiple detections to the best one and report face_count=1.
    collapse_multi_face: bool = True

    # Toggle verbose logging for debugging.
    verbose: bool = False
