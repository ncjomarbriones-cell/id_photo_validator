from io import BytesIO
from typing import Optional, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError

from .config import ValidatorConfig
from .pipeline import ArcFacePipeline
from .quality_classifier import QualityClassifier, QualityResult


def _load_image(image_bytes: bytes) -> Tuple[np.ndarray, Tuple[int, int]]:
    try:
        image = Image.open(BytesIO(image_bytes))
    except UnidentifiedImageError as exc:
        raise ValueError("Uploaded file is not a valid image") from exc

    rgb_image = image.convert("RGB")
    np_img = np.asarray(rgb_image)
    return np_img, rgb_image.size


def validate_image_bytes(
    image_bytes: bytes,
    pipeline: ArcFacePipeline,
    classifier: QualityClassifier,
    config: Optional[ValidatorConfig] = None,
) -> QualityResult:
    _ = config  # reserved for future overrides
    image_rgb, size = _load_image(image_bytes)

    # Simple guard for tiny images.
    if min(size) < 112:
        raise ValueError("Image is too small; minimum edge must be >=112px.")

    arcface_result = pipeline.extract(image_rgb)
    result = classifier.score(
        embedding=arcface_result.embedding,
        aligned_face=arcface_result.aligned_face,
        face_count=arcface_result.face_count,
        det_score=arcface_result.det_score,
    )
    return result
