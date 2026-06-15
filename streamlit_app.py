from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

from src.predict import DEFAULT_IMAGE_SIZE, load_labels, load_model, predict_frame


MODEL_PATH = ROOT / "models" / "asl_mobilenetv2.keras"
LABELS_PATH = ROOT / "models" / "labels.json"


@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)


@st.cache_data
def get_labels():
    return load_labels(LABELS_PATH)


def pil_to_bgr(image: Image.Image) -> np.ndarray:
    rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def show_prediction(frame: np.ndarray) -> None:
    model = get_model()
    labels = get_labels()
    label, confidence, probabilities = predict_frame(model, labels, frame, image_size=DEFAULT_IMAGE_SIZE)

    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
    st.metric("Prediction", label)
    st.progress(min(confidence, 1.0), text=f"Confidence: {confidence:.2%}")

    top_indices = np.argsort(probabilities)[-5:][::-1]
    st.caption("Top predictions")
    for index in top_indices:
        st.write(f"{labels[int(index)]}: {float(probabilities[int(index)]):.2%}")


def main() -> None:
    st.set_page_config(page_title="ASL Sign Language Translator", layout="centered")
    st.title("ASL Sign Language Translator")
    st.caption("Transfer learning with TensorFlow, OpenCV preprocessing, and Streamlit inference.")

    if not MODEL_PATH.exists() or not LABELS_PATH.exists():
        st.error(
            "Model files were not found. Run the notebook first to create "
            "`models/asl_mobilenetv2.keras` and `models/labels.json`."
        )
        st.stop()

    mode = st.tabs(["Camera", "Upload"])

    with mode[0]:
        camera_image = st.camera_input("Capture a hand gesture")
        if camera_image is not None:
            image = Image.open(camera_image)
            show_prediction(pil_to_bgr(image))

    with mode[1]:
        uploaded = st.file_uploader("Upload an ASL hand gesture image", type=["jpg", "jpeg", "png"])
        if uploaded is not None:
            image = Image.open(uploaded)
            show_prediction(pil_to_bgr(image))


if __name__ == "__main__":
    main()
