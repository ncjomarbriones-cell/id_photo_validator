# ID Photo Validator (ArcFace + Quality Head)

FastAPI microservice that checks whether an uploaded ID photo looks valid. It uses InsightFace (ArcFace) to extract a 512-D embedding and a lightweight logistic head to decide accept / borderline / reject, with heuristic messages (blur, brightness, detection confidence).

## Quick start (local, Python 3.11 shown)
This repo is set up to run without InsightFace (fallback face detector + dummy embedding). For the full ArcFace pipeline, install InsightFace/ONNXRuntime on Python 3.10-3.12 and uncomment them in `requirements.txt`.

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# (Auto-created dummy model: already saved at models/quality_head.joblib)
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Docker alternative (Python 3.10, recommended for full InsightFace):
```bash
docker build -t id-photo-validator .
docker run --rm -p 8000:8000 -v %cd%:/app id-photo-validator
```

## GUI (quick check)
The FastAPI app serves a simple GUI at `/` (file `web/index.html`). After the container or uvicorn is running, open:

- http://localhost:8000/  (form to upload a photo and view the JSON result)

## API
- `GET /health` -> `{ "status": "ok" }`
- `POST /validate` (multipart form, field `file` = image) -> JSON:
  ```json
  {
    "quality_score": 0.87,
    "accept": true,
    "borderline": false,
    "reasons": [],
    "face_count": 1,
    "det_score": 0.99,
    "blur_metric": 123.4,
    "brightness": 0.52
  }
  ```

Example call (PowerShell):
```bash
curl -X POST http://localhost:8000/validate `
  -F "file=@C:/path/to/photo.jpg"
```

## Training (quality-only, no identity recognition)
Use the updated dataset with `good/` and `bad/` folders inside each split:
```
data/id_photo_dataset/
  train/good/...
  train/bad/...
  val/good/...
  val/bad/...
  test/good/...  # optional
```
Train the quality head (optionally with OFIQ as an extra feature):
```bash
# baseline quality head
python -m scripts.train_quality --dataset-root data/id_photo_dataset --output models/quality_head.joblib
# enable OFIQ feature (install ofiq first)
python -m scripts.train_quality --dataset-root data/id_photo_dataset --with-ofiq --output models/quality_head.joblib
```
The service uses only `models/quality_head.joblib`; identity recognition is not used.

## Project layout
- `api/main.py` - FastAPI app exposing `/validate` and serving the GUI.
- `validator_service/config.py` - thresholds and model paths.
- `validator_service/pipeline.py` - detection/alignment/ArcFace embedding via InsightFace (with OpenCV fallback if InsightFace is missing).
- `validator_service/quality_classifier.py` - loads the quality head and applies heuristics + optional OFIQ score.
- `validator_service/augmentations.py` - degradations for training negatives.
- `scripts/train_quality.py` - CLI trainer for the quality head (good/bad).
- `web/index.html` - simple GUI to test the API.
- `models/quality_head.joblib` - saved quality classifier (you create this).
- `Dockerfile` - Python 3.10 container with all dependencies.

## Notes and troubleshooting
- If you run outside Docker, use Python 3.10-3.12 so ONNX/OpenCV/InsightFace wheels install cleanly; Python 3.14 currently lacks them.
- The validator rejects when no face or multiple faces are detected. Detection confidence, blur variance, and brightness contribute to the `reasons` array.
- To add duplicate detection, persist the ArcFace embedding returned during validation and compare via cosine similarity against stored embeddings (threshold around 0.45-0.6).
- A lightweight dummy classifier is pre-generated in `models/quality_head.joblib` so the API runs immediately; retrain with your data for real use.
