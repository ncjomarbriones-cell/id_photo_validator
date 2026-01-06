from dataclasses import dataclass
from typing import Optional

import numpy as np

try:  # Optional dependency: full InsightFace pipeline
    from insightface.app import FaceAnalysis
    from insightface.utils.face_align import norm_crop

    _INSIGHT_AVAILABLE = True
except Exception:  # pragma: no cover - optional
    _INSIGHT_AVAILABLE = False

import cv2

from .config import ValidatorConfig


@dataclass
class ArcFaceResult:
    embedding: np.ndarray
    aligned_face: np.ndarray
    face_count: int
    det_score: float


class ArcFacePipeline:
    """
    Wraps InsightFace's FaceAnalysis to provide detection + embedding + aligned crop.
    """

    def __init__(self, config: ValidatorConfig):
        self.config = config
        self.use_insight = _INSIGHT_AVAILABLE
        if self.use_insight:
            self.app = FaceAnalysis(name=config.arcface_model, providers=None)
            # ctx_id = 0 means CPU if no GPU is found.
            self.app.prepare(ctx_id=0, det_size=config.det_size)
        else:
            # Fallback: simple Haar cascade detection.
            self.app = None
            self.detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def extract(self, image_rgb: np.ndarray) -> ArcFaceResult:
        """
        Run detection and embedding extraction on an RGB image.
        Raises ValueError if no suitable face is found.
        """
        if self.use_insight:
            faces = self.app.get(image_rgb)
            if faces is None or len(faces) == 0:
                raise ValueError("No face detected")

            faces = sorted(faces, key=lambda f: float(f.det_score), reverse=True)
            face = faces[0]

            # Prefer the 5-point landmarks for alignment; fall back if missing.
            keypoints = None
            if hasattr(face, "kps") and face.kps is not None:
                keypoints = face.kps
            elif hasattr(face, "landmark_2d_106") and face.landmark_2d_106 is not None:
                keypoints = face.landmark_2d_106[:5]

            if keypoints is None:
                raise ValueError("Detected face has no landmarks for alignment")

            aligned = norm_crop(image_rgb, keypoints, image_size=112)

            if not hasattr(face, "embedding") or face.embedding is None:
                # Compute embedding on the aligned crop if not populated.
                face.embedding = self.app.models["recognition"].get_feat(aligned)

            embedding = np.asarray(face.embedding, dtype=np.float32)
            det_score = float(face.det_score)
            reported_face_count = 1 if self.config.collapse_multi_face else len(faces)
            face_count = reported_face_count
        else:
            # Fallback detection: Haar cascade, largest box
            gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
            detections = self.detector.detectMultiScale(
                gray, scaleFactor=1.05, minNeighbors=8, minSize=(120, 120)
            )
            if len(detections) == 0:
                raise ValueError("No face detected (fallback detector)")

            # Merge highly-overlapping boxes to reduce duplicate counts from background noise.
            detections = self._nms_boxes(detections, iou_thresh=0.3)
            detections = self._filter_boxes_by_area(detections, min_area_ratio=0.3)
            detections = sorted(detections, key=lambda b: b[2] * b[3], reverse=True)
            x, y, w, h = detections[0]
            face_count = len(detections)
            det_score = float(min(1.0, (w * h) / (image_rgb.shape[0] * image_rgb.shape[1] + 1e-6)))

            # Crop and resize to 112x112
            crop = image_rgb[max(y, 0) : y + h, max(x, 0) : x + w]
            aligned = cv2.resize(crop, (112, 112), interpolation=cv2.INTER_LINEAR)
            # Simple embedding: normalized flattened pixels downsampled to 512 dims.
            flat = aligned.astype(np.float32).flatten() / 255.0
            if flat.size >= 512:
                embedding = flat[:512]
            else:
                # pad with zeros to 512
                embedding = np.zeros(512, dtype=np.float32)
                embedding[: flat.size] = flat

        return ArcFaceResult(
            embedding=np.asarray(embedding, dtype=np.float32),
            aligned_face=aligned,
            face_count=face_count,
            det_score=det_score,
        )

    @staticmethod
    def _nms_boxes(boxes, iou_thresh: float = 0.3):
        """
        Simple IoU-based non-max suppression to merge duplicate Haar detections.
        boxes: iterable of (x, y, w, h)
        """
        boxes = list(boxes) if boxes is not None else []
        if len(boxes) <= 1:
            return boxes
        boxes = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)
        kept = []
        while boxes:
            cur = boxes.pop(0)
            kept.append(cur)
            boxes = [b for b in boxes if ArcFacePipeline._iou(cur, b) < iou_thresh]
        return kept

    @staticmethod
    def _iou(b1, b2) -> float:
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2
        xa1, ya1, xa2, ya2 = x1, y1, x1 + w1, y1 + h1
        xb1, yb1, xb2, yb2 = x2, y2, x2 + w2, y2 + h2
        inter_x1, inter_y1 = max(xa1, xb1), max(ya1, yb1)
        inter_x2, inter_y2 = min(xa2, xb2), min(ya2, yb2)
        inter_w, inter_h = max(0, inter_x2 - inter_x1), max(0, inter_y2 - inter_y1)
        inter = inter_w * inter_h
        area_a = w1 * h1
        area_b = w2 * h2
        union = area_a + area_b - inter + 1e-6
        return inter / union

    @staticmethod
    def _filter_boxes_by_area(boxes, min_area_ratio: float = 0.3):
        """
        Keep boxes that are not tiny relative to the largest detection.
        Helps drop small false positives (e.g., background edges).
        """
        boxes = list(boxes) if boxes is not None else []
        if len(boxes) == 0:
            return boxes
        areas = [b[2] * b[3] for b in boxes]
        max_area = max(areas)
        filtered = [b for b, a in zip(boxes, areas) if a >= min_area_ratio * max_area]
        return filtered
