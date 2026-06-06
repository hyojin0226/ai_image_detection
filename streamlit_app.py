from pathlib import Path
import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "outputs/efficientnetv2s_experiment_01/models" / "efficientnetv2s_fake_face_final.keras"
CONFIG_PATH = APP_DIR / "outputs/efficientnetv2s_experiment_01/models" / "inference_config.json"

st.set_page_config(
    page_title="AI Face Detector",
    page_icon="🔍",
    layout="centered"
)

@st.cache_resource
def load_resources():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. "
            "Make sure the Git LFS model file is included in the deployment."
        )
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Inference config not found: {CONFIG_PATH}")

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        config = json.load(f)
    return model, config

try:
    model, config = load_resources()
except Exception as exc:
    st.error("모델을 불러오지 못했습니다. 배포 로그와 Git LFS 설정을 확인하세요.")
    st.exception(exc)
    st.stop()

image_size = tuple(config["image_size"])
threshold = float(config["decision_threshold_real"])

st.title("AI Face Detector")
st.caption("EfficientNetV2S · Accuracy 97.8% · AUC 99.8%")

uploaded_file = st.file_uploader(
    "Upload a face image (JPG, PNG, WEBP)",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="visible"
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception as exc:
        st.error("이미지 파일을 읽을 수 없습니다. JPG, PNG 또는 WEBP 파일을 사용하세요.")
        st.exception(exc)
        st.stop()

    st.image(image, caption="Uploaded image", use_container_width=True)

    with st.spinner("Analyzing..."):
        resized = image.resize(image_size)
        image_array = np.asarray(resized, dtype=np.float32)
        image_batch = np.expand_dims(image_array, axis=0)
        probability_real = float(model.predict(image_batch, verbose=0)[0][0])

    probability_fake = 1.0 - probability_real
    is_real = probability_real >= threshold
    prediction = "REAL" if is_real else "AI GENERATED / FAKE"

    st.subheader(f"Prediction: {prediction}")

    st.write("AI / FAKE probability")
    st.metric(label="AI / FAKE probability", value=f"{probability_fake * 100:.2f}%", label_visibility="collapsed")

    st.write("REAL probability")
    st.metric(label="REAL probability", value=f"{probability_real * 100:.2f}%", label_visibility="collapsed")

    st.caption(f"Decision threshold for REAL: {threshold:.3f}")

    st.write("AI / FAKE probability")
    st.progress(probability_fake)

    st.write("REAL probability")
    st.progress(probability_real)
