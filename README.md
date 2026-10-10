# Skin Lesion Classification (HAM10000 PoC)

Proof-of-concept CNN + FastAPI + Streamlit pipeline for 7-class pigmented skin lesion classification using the public [HAM10000](https://doi.org/10.7910/DVN/DBW86T) dataset.

**Not a diagnostic tool.** Results do not replace a physician's diagnosis.

## Project layout

```text
project/
  EDA.py                 # Metadata load, preprocessing, group split, generators, EDA
  train.py               # MobileNetV2 training, evaluation, artifact export
  main.py                # FastAPI backend (POST /predict)
  database.py            # SQLite prediction history
  app.py                 # Streamlit frontend
  models/                # model.keras, label_encoder.pkl, scaler.pkl, …
  tests/
  .github/workflows/ci.yml
  requirements.txt
```

## Dataset layout

Place HAM10000 under `data/` (ignored by git):

```text
data/
  HAM10000_metadata.csv
  HAM10000_images_part_1/   # *.jpg
  HAM10000_images_part_2/   # *.jpg
```

## Setup

Requires **Python 3.10–3.12** (this project was verified on 3.12). TensorFlow needs `>=2.16` on 3.12.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## Preprocessing & EDA

Uses `EDA.py` as the shared preprocessing reference:

1. Keep missing `age` as NaN (not imputed / not dropped for CNN training)
2. Encode `dx` with `LabelEncoder` (0–6); retain `"unknown"` for sex/localization
3. Resolve image paths; drop rows without a readable image file
4. Stratified group-aware train/val/test split by `lesion_id`
5. Resize 224×224, rescale `1/255`; augment **training** only

```bash
python EDA.py
```

Plots are written to `eda_outputs/`.

## Train

```bash
python train.py
# optional
python train.py --no-eda --phase1-epochs 5 --phase2-epochs 5
```

Artifacts written to `models/`:

| File | Purpose |
|------|---------|
| `model.keras` | Trained CNN |
| `label_encoder.pkl` | dx code ↔ index |
| `scaler.pkl` | Image rescale / size params |
| `class_meta.json` | Class names & clinical tags |
| `high_risk_thresholds.json` | Tuned mel/bcc/akiec thresholds |

Training also writes metrics and plots under `training_outputs/`.

Class imbalance is handled with inverse-frequency class weights. High-risk recall for `mel`, `bcc`, and `akiec` is improved via threshold tuning (higher false-positive rate accepted).

## Backend (FastAPI)

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

- `GET /health` — model readiness
- `POST /predict` — multipart image upload; returns top-3 predictions + high-risk / threshold alerts

Optional env vars: `MODELS_DIR`, `DATABASE_URL`.

## Frontend (Streamlit)

```bash
streamlit run app.py
```

Optional: `API_URL=http://127.0.0.1:8000` (default).

## Tests / CI

```bash
pytest -q
```

GitHub Actions (`.github/workflows/ci.yml`) runs syntax compile + pytest on Python 3.10/3.11.
