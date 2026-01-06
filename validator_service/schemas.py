from typing import List, Optional

from pydantic import BaseModel, Field


class ValidationResponse(BaseModel):
    quality_score: float = Field(..., description="Probability that the photo is a valid ID-quality face.")
    accept: bool = Field(..., description="Whether the photo is accepted outright.")
    borderline: bool = Field(..., description="True if the score is near the threshold.")
    reasons: List[str] = Field(default_factory=list)
    face_count: int = 0
    det_score: float = 0.0
    blur_metric: Optional[float] = None
    brightness: Optional[float] = None
    ofiq_score: Optional[float] = Field(
        default=None, description="Optional OFIQ quality score if the library is available."
    )
