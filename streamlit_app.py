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

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;500&display=swap');

* { font-family: 'IBM Plex Sans', sans-serif; }

.block-container { max-width: 720px; padding-top: 2rem; }

.hero {
    text-align: center;
    padding: 2.5rem 0 1.5rem;
    border-bottom: 1px solid #e8e8e8;
    margin-bottom: 2rem;
}
.hero h1 {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.8rem;
    font-weight: 600;
    letter-spacing: -0.03em;
    color: #0f0f0f;
    margin: 0 0 0.4rem;
}
.hero p {
    color: #888;
    font-size: 0.9rem;
    margin: 0;
}

.upload-area {
    border: 1.5px dashed #d0d0d0;
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
    background: #fafafa;
    margin-bottom: 1.5rem;
}

.result-card {
    border-radius: 14px;
    padding: 1.8rem 2rem;
    margin: 1.5rem 0;
    border: 1px solid;
}
.result-fake {
    background: #fff5f5;
    border-color: #fcc;
}
.result-real {
    background: #f0faf5;
    border-color: #b2dfcc;
}
.result-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.6rem;
    font-weight: 600;
    margin-bottom: 0.3rem;
}
.result-fake .result-label { color: #c0392b; }
.result-real .result-label { color: #1a7a4a; }
.result-conf {
    font-size: 0.85rem;
    color: #999;
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.05em;
}

.prob-row {
    display: flex;
    gap: 1rem;
    margin-top: 1.2rem;
}
.prob-item {
    flex: 1;
    background: white;
    border: 1px solid #eee;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.prob-item .val {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.6rem;
    font-weight: 600;
    color: #0f0f0f;
}
.prob-item .lbl {
    font-size: 0.78rem;
    color: #999;
    margin-top: 0.2rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}

.bar-wrap {
    margin-top: 1.5rem;
}
.bar-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.82rem;
    color: #888;
    margin-bottom: 0.3rem;
}
.bar-bg {
    background: #f0f0f0;
    border-radius: 99px;
    height: 8px;
    overflow: hidden;
}
.bar-fill-fake {
    height: 100%;
    border-radius: 99px;
    background: #e74c3c;
    transition: width 0.4s ease;
}
.bar-fill-real {
    height: 100%;
    border-radius: 99px;
    background: #2ecc71;
    transition: width 0.4s ease;
}

.footer {
    text-align: center;
    font-size: 0.78rem;
    color: #ccc;
    font-family: 'IBM Plex Mono', monospace;
    padding: 2.5rem 0 1rem;
    border-top: 1px solid #f0f0f0;
    margin-top: 3rem;
}

[data-testid="stFileUploader"] {
    border: none !important;
    background: transparent !important;
}
</style>
""", unsafe_allow_html=True)

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

st.markdown("""
<div class="hero">
    <h1>AI Face Detector</h1>
    <p>EfficientNetV2S · Accuracy 97.8% · AUC 99.8%</p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "얼굴 이미지를 업로드하세요 (JPG, PNG)",
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

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, use_container_width=True)

    with st.spinner("분석 중..."):
        resized = image.resize(image_size)
        image_array = np.asarray(resized, dtype=np.float32)
        image_batch = np.expand_dims(image_array, axis=0)
        probability_real = float(model.predict(image_batch, verbose=0)[0][0])

    probability_fake = 1.0 - probability_real
    is_real = probability_real >= threshold
    label = "✓ REAL" if is_real else "✗ AI GENERATED"
    box_class = "result-real" if is_real else "result-fake"
    conf = probability_real if is_real else probability_fake

    st.markdown(f"""
    <div class="result-card {box_class}">
        <div class="result-label">{label}</div>
        <div class="result-conf">CONFIDENCE {conf*100:.1f}%</div>
        <div class="prob-row">
            <div class="prob-item">
                <div class="val">{probability_fake*100:.1f}%</div>
                <div class="lbl">AI 생성 확률</div>
            </div>
            <div class="prob-item">
                <div class="val">{probability_real*100:.1f}%</div>
                <div class="lbl">실제 얼굴 확률</div>
            </div>
        </div>
        <div class="bar-wrap">
            <div class="bar-label"><span>AI 생성</span><span>{probability_fake*100:.1f}%</span></div>
            <div class="bar-bg"><div class="bar-fill-fake" style="width:{probability_fake*100:.1f}%"></div></div>
        </div>
        <div class="bar-wrap">
            <div class="bar-label"><span>실제 얼굴</span><span>{probability_real*100:.1f}%</span></div>
            <div class="bar-bg"><div class="bar-fill-real" style="width:{probability_real*100:.1f}%"></div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("상세 정보"):
        st.markdown(f"""
        | 항목 | 값 |
        |------|-----|
        | 판별 결과 | {'실제 얼굴' if is_real else 'AI 생성'} |
        | AI 생성 확률 | {probability_fake*100:.2f}% |
        | 실제 얼굴 확률 | {probability_real*100:.2f}% |
        | 결정 임계값 | {threshold:.3f} |
        | 모델 | EfficientNetV2S |
        """)

st.markdown("""
<div class="footer">
    AI Face Detector &nbsp;·&nbsp; EfficientNetV2S &nbsp;·&nbsp; Accuracy 97.8%
</div>
""", unsafe_allow_html=True)
