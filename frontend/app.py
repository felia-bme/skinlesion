"""
Streamlit frontend for Skin Lesion Classification (HAM10000 PoC).

Sends uploaded images to the FastAPI /predict endpoint and displays
top predictions with high-risk warnings.

Palette: background #f3f3f3, accents #7cbaca and #f08b33, text #000000.
"""

import os
from html import escape

import pandas as pd
import requests
import streamlit as st
from PIL import Image

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
PREDICT_ENDPOINT = f"{API_URL.rstrip('/')}/predict"

COLOR_BG = "#f3f3f3"
COLOR_BLUE = "#7cbaca"
COLOR_ORANGE = "#f08b33"
COLOR_TEXT = "#000000"

st.set_page_config(
    page_title="Skin Lesion Classification",
    page_icon="🩺",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Styling (display only; no effect on the prediction pipeline)
# ---------------------------------------------------------------------------
CSS = """
<style>
.stApp { background-color: __BG__; }
.stApp, .stApp p, .stApp li, .stApp label, .stApp span,
.stApp h1, .stApp h2, .stApp h3, [data-testid="stCaptionContainer"] { color: __TEXT__; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 860px; padding-top: 2.5rem; }
hr { border-color: rgba(0,0,0,0.12); margin: 1.5rem 0; }
[data-testid="stImage"] img { border-radius: 14px; }

/* Header */
.app-header { display: flex; align-items: center; justify-content: center;
  gap: 1rem; flex-wrap: wrap; margin-bottom: 1.5rem; }
.app-icon { width: 60px; height: 60px; border-radius: 16px; font-size: 1.9rem;
  background: rgba(124,186,204,0.45); display: flex; align-items: center;
  justify-content: center; }
.app-title { margin: 0; padding: 0; font-size: 2.2rem; font-weight: 700;
  color: __TEXT__; line-height: 1.1; }
.pill { background: rgba(124,186,204,0.45); border-radius: 999px;
  padding: 0.35rem 0.9rem; font-size: 0.9rem; font-weight: 600; color: __TEXT__; }

/* Info and warning boxes */
.box { display: flex; align-items: center; gap: 1rem; border-radius: 12px;
  padding: 1rem 1.25rem; margin: 0.9rem 0; color: __TEXT__; }
.box-info { background: rgba(124,186,204,0.3); border: 1px solid rgba(124,186,204,0.8); }
.box-warn { background: rgba(240,139,51,0.18); border: 1px solid rgba(240,139,51,0.7);
  border-left: 6px solid __ORANGE__; }
.box-icon { flex: 0 0 34px; width: 34px; height: 34px; border-radius: 50%;
  display: flex; align-items: center; justify-content: center; font-weight: 700;
  color: __TEXT__; }
.box-info .box-icon { background: __BLUE__; }
.box-warn .box-icon { background: __ORANGE__; }
.box-msg { flex: 1; line-height: 1.5; }
.box-msg ul { margin: 0.4rem 0 0 1.1rem; padding: 0; }

/* Upload card */
[data-testid="stFileUploader"] { background: #ffffff; border-radius: 16px;
  padding: 1.5rem; box-shadow: 0 4px 18px rgba(0,0,0,0.07); }
[data-testid="stFileUploader"] label p { font-size: 1.35rem; font-weight: 700; }
[data-testid="stFileUploaderDropzone"], [data-testid="stFileUploadDropzone"] {
  flex-direction: column; justify-content: center; text-align: center; gap: 0.75rem;
  padding: 2rem 1rem; background: rgba(124,186,204,0.18);
  border: 2px dashed __BLUE__; border-radius: 12px; }
[data-testid="stFileUploaderDropzoneInstructions"] * { display: none; }
[data-testid="stFileUploaderDropzoneInstructions"] { display: flex;
  flex-direction: column; align-items: center; }
[data-testid="stFileUploaderDropzoneInstructions"]::before { content: "";
  display: block; width: 56px; height: 56px; margin-bottom: 0.5rem;
  background: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 64 64' fill='none' stroke='%23000000' stroke-width='3' stroke-linecap='round' stroke-linejoin='round'><path d='M20 44h-2a10 10 0 0 1-1-20 14 14 0 0 1 27-3 12 12 0 0 1 4 23h-3'/><path d='M32 52V30M24 38l8-8 8 8'/></svg>") center / contain no-repeat; }
[data-testid="stFileUploaderDropzoneInstructions"]::after {
  content: "JPG or PNG  ·  Maximum file size: 200 MB"; font-size: 0.9rem;
  margin-top: 0.75rem; color: __TEXT__; }
[data-testid="stFileUploaderDropzone"] button,
[data-testid="stFileUploadDropzone"] button { font-size: 0; background: __ORANGE__;
  border: 1px solid __TEXT__; border-radius: 10px; padding: 0.6rem 1.6rem; }
[data-testid="stFileUploaderDropzone"] button::after,
[data-testid="stFileUploadDropzone"] button::after { content: "Choose image";
  font-size: 1rem; font-weight: 600; color: __TEXT__; }

/* Buttons */
.stButton > button, [data-testid="stBaseButton-primary"] { background: __ORANGE__;
  color: __TEXT__; border: 1px solid __TEXT__; border-radius: 10px;
  font-weight: 600; padding: 0.55rem 1.8rem; }
.stButton > button:hover, [data-testid="stBaseButton-primary"]:hover {
  background: __BLUE__; color: __TEXT__; border: 1px solid __TEXT__; }

/* Prediction cards */
.pred { background: #ffffff; border-radius: 14px; padding: 1rem 1.25rem;
  margin: 0.6rem 0; box-shadow: 0 2px 10px rgba(0,0,0,0.06); }
.pred-top { display: flex; justify-content: space-between; align-items: center;
  gap: 0.5rem; flex-wrap: wrap; }
.pred-name { font-weight: 700; }
.pred-code { background: rgba(0,0,0,0.07); border-radius: 6px; padding: 0.05rem 0.4rem;
  font-family: monospace; font-size: 0.85rem; margin-left: 0.3rem; }
.badge { border-radius: 999px; padding: 0.15rem 0.7rem; font-size: 0.8rem; font-weight: 600; }
.badge-low { background: rgba(124,186,204,0.5); }
.badge-high { background: rgba(240,139,51,0.45); }
.bar { background: rgba(0,0,0,0.08); border-radius: 999px; height: 10px;
  margin: 0.7rem 0 0.35rem; overflow: hidden; }
.bar-fill { background: __ORANGE__; height: 100%; border-radius: 999px; }
.pred-pct { text-align: right; font-size: 0.9rem; font-weight: 600; }
</style>
"""
for token, value in {
    "__BG__": COLOR_BG,
    "__BLUE__": COLOR_BLUE,
    "__ORANGE__": COLOR_ORANGE,
    "__TEXT__": COLOR_TEXT,
}.items():
    CSS = CSS.replace(token, value)
st.markdown(CSS, unsafe_allow_html=True)


def info_box(html_text: str) -> None:
    st.markdown(
        f'<div class="box box-info"><div class="box-icon">i</div>'
        f'<div class="box-msg">{html_text}</div></div>',
        unsafe_allow_html=True,
    )


def warn_box(html_text: str) -> None:
    st.markdown(
        f'<div class="box box-warn"><div class="box-icon">!</div>'
        f'<div class="box-msg">{html_text}</div></div>',
        unsafe_allow_html=True,
    )


def prediction_card(p: dict) -> None:
    prob = min(max(p["probability"], 0.0), 1.0)
    characteristic = str(p["characteristic"])
    badge = "badge-high" if characteristic.lower() in ("malignant", "precancerous") else "badge-low"
    st.markdown(
        f'<div class="pred"><div class="pred-top"><div>'
        f'<span class="pred-name">{escape(str(p["label"]))}</span>'
        f'<span class="pred-code">{escape(str(p["dx"]))}</span></div>'
        f'<span class="badge {badge}">{escape(characteristic)}</span></div>'
        f'<div class="bar"><div class="bar-fill" style="width:{prob * 100:.1f}%"></div></div>'
        f'<div class="pred-pct">{prob * 100:.1f}%</div></div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="app-header"><div class="app-icon">🩺</div>'
    '<h1 class="app-title">Skin Lesion Classification</h1>'
    '<span class="pill">Proof of Concept</span></div>',
    unsafe_allow_html=True,
)
info_box(
    "This application is a <b>Proof of Concept</b> trained on the public HAM10000 "
    "dataset.<br>It is <b>not</b> a diagnostic or medical device."
)

uploaded = st.file_uploader(
    "Upload a dermatoscopic image",
    type=["jpg", "jpeg", "png"],
)

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption=uploaded.name, width="stretch")

    if st.button("Predict", type="primary"):
        with st.spinner("Running inference..."):
            try:
                files = {
                    "file": (
                        uploaded.name,
                        uploaded.getvalue(),
                        uploaded.type or "image/jpeg",
                    )
                }
                response = requests.post(PREDICT_ENDPOINT, files=files, timeout=60)
            except requests.exceptions.ConnectionError:
                st.error(
                    f"Cannot reach the API at `{API_URL}`. "
                    "Start the backend with: `uvicorn main:app --reload`"
                )
                st.stop()
            except requests.exceptions.Timeout:
                st.error("The API request timed out. Please try again.")
                st.stop()

        if response.status_code != 200:
            st.error(f"Prediction failed ({response.status_code}): {response.text}")
            st.stop()

        result = response.json()
        predictions = result.get("predictions", [])
        top = result.get("top_prediction", predictions[0] if predictions else None)

        st.subheader("Top Predictions")
        if not predictions:
            warn_box("No predictions returned.")
        else:
            rows = []
            for p in predictions:
                rows.append(
                    {
                        "dx": p["dx"],
                        "Lesion": p["label"],
                        "Probability (%)": round(p["probability"] * 100, 2),
                        "Characteristic": p["characteristic"],
                    }
                )
                prediction_card(p)

            chart_df = pd.DataFrame(rows).set_index("Lesion")[["Probability (%)"]]
            st.bar_chart(chart_df, color=COLOR_BLUE)

        if result.get("high_risk_detected"):
            risk_codes = ", ".join(result.get("high_risk_classes", []))
            warn_box(
                f"High-risk class detected among top predictions: <b>{escape(risk_codes)}</b> "
                "(melanoma / basal cell carcinoma / actinic keratosis). "
                "Please consult a healthcare professional for clinical evaluation."
            )
        elif top and top.get("high_risk"):
            warn_box(
                f"Top prediction <code>{escape(str(top['dx']))}</code> is a high-risk class. "
                "Please consult a healthcare professional."
            )

        threshold_alerts = result.get("threshold_alerts") or []
        if threshold_alerts:
            items = "".join(
                f"<li><b>{escape(str(a['label']))}</b> (<code>{escape(str(a['dx']))}</code>): "
                f"{a['probability'] * 100:.1f}% "
                f"(threshold {a['threshold'] * 100:.0f}%) — "
                f"{escape(str(a['characteristic']))}</li>"
                for a in threshold_alerts
            )
            warn_box(
                "Sensitivity-tuned alert: one or more high-risk classes exceeded "
                "their decision threshold (even if not the top prediction)."
                f"<ul>{items}</ul>"
            )

        st.caption(result.get("disclaimer", ""))

st.divider()
warn_box(
    "<b>Disclaimer:</b> These results do not replace a physician's diagnosis. "
    "Always seek professional medical advice for skin concerns."
)