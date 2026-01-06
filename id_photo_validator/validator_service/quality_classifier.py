from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import logging

import numpy as np

try:
    import joblib
except Exception as exc:  # pragma: no cover - environment guard
    raise ImportError(
        "joblib (from scikit-learn) is required to load the quality classifier. "
        "Install via requirements.txt inside the Docker image if local wheels are missing."
    ) from exc

try:
    import cv2
except Exception:  # pragma: no cover - optional metric
    cv2 = None

from .config import ValidatorConfig
from .ofiq_adapter import OFIQScorer


@dataclass
class QualityResult:
    quality_score: float
    accept: bool
    borderline: bool
    reasons: List[str] = field(default_factory=list)
    face_count: int = 0
    det_score: float = 0.0
    blur_metric: Optional[float] = None
    brightness: Optional[float] = None
    ofiq_score: Optional[float] = None


class QualityClassifier:
    """
    Wraps the trained sklearn classifier + heuristic checks.
    """

    def __init__(self, config: ValidatorConfig):
        self.config = config
        self.model_path = Path(config.quality_model_path)
        self.model = None
        self.label_names: Optional[List[str]] = None
        self.feature_meta: dict = {}
        self._proba_positive_index: int = 1  # default to second column
        self.ofiq: Optional[OFIQScorer] = None

        # Attempt to load OFIQ (optional)
        try:
            self.ofiq = OFIQScorer()
            logging.info("OFIQ scorer initialized")
        except Exception as exc:
            logging.info("OFIQ scorer not available (%s)", exc)

        if self.model_path.exists():
            loaded = joblib.load(self.model_path)
            if isinstance(loaded, dict) and "classifier" in loaded:
                self.model = loaded["classifier"]
                self.label_names = loaded.get("label_names")
                self.feature_meta = loaded.get("feature_meta", {})
            else:
                self.model = loaded
            try:
                classes = getattr(self.model, "classes_", None)
                if classes is not None:
                    if 1 in classes:
                        self._proba_positive_index = int(list(classes).index(1))
                    elif "good" in classes:
                        self._proba_positive_index = int(list(classes).index("good"))
                    else:
                        self._proba_positive_index = len(classes) - 1
                logging.info(
                    "Loaded quality classifier from %s (%s classes: %s)",
                    self.model_path,
                    len(classes) if classes is not None else "unknown",
                    classes.tolist() if classes is not None else "n/a",
                )
            except Exception:
                logging.info("Loaded quality classifier from %s", self.model_path)

    def is_ready(self) -> bool:
        return self.model is not None

    def _blur_metric(self, aligned_face: np.ndarray) -> Optional[float]:
        if cv2 is None:
            return None
        gray = cv2.cvtColor(aligned_face, cv2.COLOR_RGB2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _brightness_metric(self, aligned_face: np.ndarray) -> float:
        return float(np.mean(aligned_face) / 255.0)

    def score(
        self,
        embedding: np.ndarray,
        aligned_face: np.ndarray,
        face_count: int,
        det_score: float,
    ) -> QualityResult:
        if self.model is None:
            raise FileNotFoundError(
                f"Quality model not found. Expected at {self.model_path}. "
                "Train with scripts/train_quality.py."
            )

        ofiq_score: Optional[float] = None
        if self.feature_meta.get("use_ofiq") and self.ofiq is not None:
            try:
                ofiq_score = self.ofiq.score(aligned_face)
            except Exception as exc:
                logging.warning("OFIQ scoring failed: %s", exc)

        features = embedding
        if self.feature_meta.get("use_ofiq"):
            if ofiq_score is None:
                # If OFIQ was expected but unavailable, fall back to zero and flag reason.
                features = np.concatenate([embedding, np.array([0.0], dtype=embedding.dtype)])
            else:
                features = np.concatenate([embedding, np.array([ofiq_score], dtype=embedding.dtype)])

        proba = self.model.predict_proba(features.reshape(1, -1))[0]
        prob_good = float(proba[self._proba_positive_index])
        accept = prob_good >= self.config.accept_threshold and face_count == 1
        borderline = self.config.borderline_threshold <= prob_good < self.config.accept_threshold

        reasons: List[str] = []
        if face_count == 0:
            reasons.append("No face detected")
        elif face_count > 1:
            if self.config.strict_multi_face_reject:
                reasons.append("Multiple faces detected")
            else:
                reasons.append("Multiple faces detected (using best match)")

        if det_score < 0.45:
            reasons.append("Low detection confidence")

        blur_metric = self._blur_metric(aligned_face)
        if blur_metric is not None and blur_metric < 120:
            reasons.append("Image appears blurry")

        brightness = self._brightness_metric(aligned_face)
        if brightness < 0.20:
            reasons.append("Image is too dark")
        elif brightness > 0.85:
            reasons.append("Image is over-exposed")

        if prob_good < self.config.borderline_threshold:
            reasons.append("Photo standard not satisfied. Please try again")
        if self.feature_meta.get("use_ofiq") and ofiq_score is None:
            reasons.append("OFIQ score unavailable; falling back to baseline")

        face_fail = (
            face_count == 0
            or (self.config.strict_multi_face_reject and face_count != 1)
        )
        hard_fail = (
            face_fail
            or det_score < 0.45
            or (blur_metric is not None and blur_metric < 120)
            or brightness < 0.20
            or brightness > 0.85
        )
        if hard_fail:
            accept = False
            borderline = False

        return QualityResult(
            quality_score=prob_good,
            accept=accept,
            borderline=borderline,
            reasons=reasons,
            face_count=face_count,
            det_score=det_score,
            blur_metric=blur_metric,
            brightness=brightness,
            ofiq_score=ofiq_score,
        )
