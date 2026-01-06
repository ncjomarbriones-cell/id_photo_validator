from pathlib import Path
from io import BytesIO

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from validator_service import ArcFacePipeline, QualityClassifier, ValidatorConfig
from validator_service.schemas import ValidationResponse
from validator_service.service import validate_image_bytes

app = FastAPI(title="ID Photo Validator", version="0.1.0")

# Allow calls from the Laravel app (127.0.0.1:8000) and localhost frontends.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:8000",
        "http://localhost:8000",
        "http://127.0.0.1:8080",
        "http://localhost:8080",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
BASE_DIR = Path(__file__).resolve().parent.parent
WEB_DIR = BASE_DIR / "web"


@app.on_event("startup")
def _startup() -> None:
    config = ValidatorConfig()
    app.state.config = config
    app.state.pipeline = ArcFacePipeline(config)
    app.state.classifier = QualityClassifier(config)

    if WEB_DIR.exists():
        app.mount("/web", StaticFiles(directory=WEB_DIR), name="web")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/validate", response_model=ValidationResponse)
async def validate(file: UploadFile = File(...)):
    if not getattr(app.state, "classifier", None) or not app.state.classifier.is_ready():
        raise HTTPException(
            status_code=503,
            detail="Quality model not loaded. Train and place models/quality_head.joblib first.",
        )

    image_bytes = await file.read()
    try:
        result = validate_image_bytes(
            image_bytes=image_bytes,
            pipeline=app.state.pipeline,
            classifier=app.state.classifier,
            config=app.state.config,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    return ValidationResponse(
        quality_score=result.quality_score,
        accept=result.accept,
        borderline=result.borderline,
        reasons=result.reasons,
        face_count=result.face_count,
        det_score=result.det_score,
        blur_metric=result.blur_metric,
        brightness=result.brightness,
        ofiq_score=result.ofiq_score,
    )


@app.post("/embed")
async def embed(file: UploadFile = File(...)):
    """
    Return ArcFace embedding + detection meta for a single face.
    """
    image_bytes = await file.read()
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        arcface_result = app.state.pipeline.extract(np.asarray(image))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {exc}")

    if arcface_result.face_count != 1:
        raise HTTPException(
            status_code=400,
            detail=f"Expected 1 face, found {arcface_result.face_count}",
        )

    return {
        "embedding": arcface_result.embedding.tolist(),
        "det_score": arcface_result.det_score,
        "face_count": arcface_result.face_count,
    }


@app.get("/", include_in_schema=False)
def index():
    index_file = WEB_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="GUI not found; ensure web/index.html exists.")
    return FileResponse(index_file)


def run() -> None:
    """
    Entrypoint for `python -m api.main`.
    """
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    run()
