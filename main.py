"""
FastAPI backend for HAM10000 skin lesion classification PoC.

Loads model artifacts once at startup and exposes POST /predict.
"""

import io
import json
import os
import pickle
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import tensorflow as tf

from database import init_db, save_prediction

IMG_SIZE = (224, 224)
TOP_K = 3
HIGH_RISK = {"mel", "bcc", "akiec"}

# Filled during lifespan startup
artifacts: dict[str, Any] = {}


def _models_dir() -> str:
    return os.getenv("MODELS_DIR", "models")


def _load_artifacts() -> dict[str, Any]:
    models_dir = _models_dir()
    model_path = os.path.join(models_dir, "model.keras")
    le_path = os.path.join(models_dir, "label_encoder.pkl")
    scaler_path = os.path.join(models_dir, "scaler.pkl")
    meta_path = os.path.join(models_dir, "class_meta.json")

    missing = [p for p in (model_path, le_path) if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            "Missing model artifacts: "
            + ", ".join(missing)
            + ". Run train.py first."
        )

    model = tf.keras.models.load_model(model_path)

    with open(le_path, "rb") as f:
        label_encoder = pickle.load(f)

    scaler = {"rescale": 1.0 / 255, "img_size": list(IMG_SIZE)}
    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    else:
        # Fallback metadata aligned with alphabetical LabelEncoder order
        classes = list(label_encoder.classes_)
        meta = {
            "class_indices": {c: i for i, c in enumerate(classes)},
            "dx_full_name": {
                "akiec": "Actinic keratoses / intraepithelial carcinoma",
                "bcc": "Basal cell carcinoma",
                "bkl": "Benign keratosis-like lesions",
                "df": "Dermatofibroma",
                "mel": "Melanoma",
                "nv": "Melanocytic nevi",
                "vasc": "Vascular lesions",
            },
            "dx_characteristics": {
                "akiec": "precancerous",
                "bcc": "malignant",
                "bkl": "benign",
                "df": "benign",
                "mel": "malignant",
                "nv": "benign",
                "vasc": "benign",
            },
            "high_risk_classes": sorted(HIGH_RISK),
            "img_size": list(IMG_SIZE),
        }

    # Tuned thresholds: prefer models/ copy so serve does not need training_outputs/
    thresholds = {}
    for thr_path in (
        os.path.join(models_dir, "high_risk_thresholds.json"),
        os.path.join("training_outputs", "high_risk_thresholds.json"),
    ):
        if os.path.exists(thr_path):
            with open(thr_path, "r", encoding="utf-8") as f:
                thresholds = json.load(f)
            break

    return {
        "model": model,
        "label_encoder": label_encoder,
        "scaler": scaler,
        "meta": meta,
        "thresholds": thresholds,
    }


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """RGB convert -> resize 224x224 -> rescale 1/255 -> add batch dim."""
    rescale = artifacts["scaler"].get("rescale", 1.0 / 255)
    size = tuple(artifacts["scaler"].get("img_size", list(IMG_SIZE)))

    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(size, Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) * rescale
    return np.expand_dims(arr, axis=0)


def _threshold_flagged_risks(probs: np.ndarray) -> list[dict]:
    """
    Flag high-risk classes whose probability meets the tuned decision threshold,
    even if they are not the argmax top-1 prediction (higher recall, higher FPR).
    """
    le = artifacts["label_encoder"]
    thresholds = artifacts.get("thresholds") or {}
    meta = artifacts["meta"]
    full_names = meta.get("dx_full_name", {})
    characteristics = meta.get("dx_characteristics", {})

    flagged = []
    for dx in sorted(HIGH_RISK):
        try:
            idx = int(le.transform([dx])[0])
        except (ValueError, AttributeError):
            continue
        thr_info = thresholds.get(dx, {})
        thr = float(thr_info.get("threshold", 0.5))
        prob = float(probs[idx])
        if prob >= thr:
            flagged.append(
                {
                    "dx": dx,
                    "label": full_names.get(dx, dx),
                    "probability": prob,
                    "threshold": thr,
                    "characteristic": characteristics.get(dx, "unknown"),
                }
            )
    return flagged


def build_predictions(probs: np.ndarray) -> list[dict]:
    """Rank class probabilities and return top-k with clinical metadata."""
    le = artifacts["label_encoder"]
    meta = artifacts["meta"]
    full_names = meta.get("dx_full_name", {})
    characteristics = meta.get("dx_characteristics", {})

    order = np.argsort(probs)[::-1]
    results = []
    for idx in order[:TOP_K]:
        dx = le.inverse_transform([int(idx)])[0]
        results.append(
            {
                "dx": dx,
                "label": full_names.get(dx, dx),
                "probability": float(probs[idx]),
                "characteristic": characteristics.get(dx, "unknown"),
                "high_risk": dx in HIGH_RISK,
            }
        )
    return results


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    try:
        loaded = _load_artifacts()
        artifacts.update(loaded)
        app.state.model_ready = True
        print("[Startup] Model artifacts loaded successfully.")
    except FileNotFoundError as exc:
        app.state.model_ready = False
        print(f"[Startup] WARNING: {exc}")
    yield
    artifacts.clear()


app = FastAPI(
    title="Skin Lesion Classification API",
    description=(
        "Proof of Concept API for HAM10000-based skin lesion classification. "
        "Not a diagnostic tool."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "message": "Skin Lesion Classification API (PoC)",
        "model_ready": getattr(app.state, "model_ready", False),
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {
        "status": "ok" if getattr(app.state, "model_ready", False) else "model_missing",
        "model_ready": getattr(app.state, "model_ready", False),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not getattr(app.state, "model_ready", False):
        raise HTTPException(
            status_code=503,
            detail="Model artifacts not loaded. Run train.py and restart the API.",
        )

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file.")

    try:
        batch = preprocess_image(raw)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Cannot process image: {exc}") from exc

    probs = artifacts["model"].predict(batch, verbose=0)[0]
    top_predictions = build_predictions(probs)
    top = top_predictions[0]

    high_risk_flags = [p["dx"] for p in top_predictions if p["high_risk"]]
    threshold_flags = _threshold_flagged_risks(probs)
    for item in threshold_flags:
        if item["dx"] not in high_risk_flags:
            high_risk_flags.append(item["dx"])

    payload = {
        "filename": file.filename or "upload.jpg",
        "top_prediction": top,
        "predictions": top_predictions,
        "high_risk_detected": len(high_risk_flags) > 0,
        "high_risk_classes": high_risk_flags,
        "threshold_alerts": threshold_flags,
        "disclaimer": (
            "Proof of Concept only. Results do not replace a physician's diagnosis. "
            "Consult a healthcare professional for any concerning lesion."
        ),
    }

    try:
        save_prediction(
            filename=payload["filename"],
            top_dx=top["dx"],
            top_label=top["label"],
            top_probability=top["probability"],
            characteristic=top["characteristic"],
            predictions_json=json.dumps(top_predictions),
        )
    except Exception as exc:
        # Prediction still succeeds even if DB write fails
        print(f"[DB] Failed to save history: {exc}")

    return payload
