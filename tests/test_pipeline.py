"""API / preprocessing tests that require TensorFlow."""

import io
import os
import sys

import numpy as np
import pytest
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

pytest.importorskip("tensorflow", reason="TensorFlow required for API/model tests")


def test_health_endpoint_without_model(tmp_path, monkeypatch):
    monkeypatch.setenv("MODELS_DIR", str(tmp_path / "empty_models"))
    from fastapi.testclient import TestClient
    import importlib
    import main as api_main

    importlib.reload(api_main)

    with TestClient(api_main.app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "model_missing"
        assert body["model_ready"] is False


def test_predict_rejects_when_model_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("MODELS_DIR", str(tmp_path / "empty_models"))
    from fastapi.testclient import TestClient
    import importlib
    import main as api_main

    importlib.reload(api_main)

    with TestClient(api_main.app) as client:
        img = Image.new("RGB", (64, 64), color=(10, 20, 30))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)
        resp = client.post(
            "/predict",
            files={"file": ("sample.jpg", buf, "image/jpeg")},
        )
        assert resp.status_code == 503


def test_preprocess_image_shape():
    import main as api_main

    api_main.artifacts.update(
        {"scaler": {"rescale": 1.0 / 255, "img_size": [224, 224]}}
    )
    img = Image.new("RGB", (512, 384), color=(120, 40, 40))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    arr = api_main.preprocess_image(buf.getvalue())
    assert arr.shape == (1, 224, 224, 3)
    assert arr.dtype == np.float32
    assert 0.0 <= float(arr.min()) <= float(arr.max()) <= 1.0


def test_threshold_flagged_risks():
    """High-risk classes above tuned thresholds are flagged even if not top-1."""
    from sklearn.preprocessing import LabelEncoder
    import main as api_main

    classes = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
    le = LabelEncoder()
    le.fit(classes)

    api_main.artifacts.update(
        {
            "label_encoder": le,
            "thresholds": {
                "mel": {"threshold": 0.25},
                "bcc": {"threshold": 0.30},
                "akiec": {"threshold": 0.20},
            },
            "meta": {
                "dx_full_name": {c: c for c in classes},
                "dx_characteristics": {
                    "akiec": "precancerous",
                    "bcc": "malignant",
                    "mel": "malignant",
                    "nv": "benign",
                    "bkl": "benign",
                    "df": "benign",
                    "vasc": "benign",
                },
            },
        }
    )

    # Dominant nv, but mel probability exceeds its lowered threshold
    probs = np.zeros(7, dtype=np.float32)
    probs[le.transform(["nv"])[0]] = 0.60
    probs[le.transform(["mel"])[0]] = 0.28
    probs[le.transform(["bkl"])[0]] = 0.12

    flagged = api_main._threshold_flagged_risks(probs)
    assert any(item["dx"] == "mel" for item in flagged)
    assert all(item["dx"] in {"mel", "bcc", "akiec"} for item in flagged)
