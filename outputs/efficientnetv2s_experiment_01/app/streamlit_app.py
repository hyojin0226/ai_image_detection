
from pathlib import Path
import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR.parent / "models" / "efficientnetv2s_fake_face_final.keras"
CONFIG_PATH = APP_DIR.parent / "models" / "inference_config.json"

st.set_page_config(page_title="AI Face Detector", page_icon="🔍", layout="centered")

@st.cache_resource
def load_resources():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = json.load(file)
    return model, config

model, config = load_resources()
image_size = tuple(config["image_size"])
threshold = float(config["decision_threshold_real"])

st.title("AI-Generated Face Detection")
st.caption("EfficientNetV2S 기반 AI 생성 얼굴 탐지 시연")

uploaded_file = st.file_uploader(
    "얼굴 이미지를 업로드하세요.",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    resized = image.resize(image_size)
    image_array = np.asarray(resized, dtype=np.float32)
    image_batch = np.expand_dims(image_array, axis=0)

    # 모델 내부에 EfficientNetV2S preprocessing이 포함되어 있으므로 /255 하지 않습니다.
    probability_real = float(model.predict(image_batch, verbose=0)[0][0])
    probability_fake = 1.0 - probability_real
    prediction = "REAL" if probability_real >= threshold else "AI / FAKE"

    st.subheader(f"Prediction: {prediction}")
    st.metric("AI / FAKE probability", f"{probability_fake * 100:.2f}%")
    st.metric("REAL probability", f"{probability_real * 100:.2f}%")
    st.caption(f"Decision threshold for REAL: {threshold:.3f}")

    st.progress(int(round(probability_fake * 100)), text="AI / FAKE probability")

st.divider()
st.caption("This model provides a probability-based estimate and may fail on unseen generation methods or edited images.")
