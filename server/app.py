"""로컬 추론 서버: 이미지 → 얼굴 검출 → 얼굴별 REAL/FAKE 확률.

실행: uvicorn app:app --host 127.0.0.1 --port 8000  (server 폴더에서)
"""
from __future__ import annotations

import base64
import binascii
import io
import threading
import time
from contextlib import asynccontextmanager

import anyio
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps, UnidentifiedImageError

from ela import ela_map, error_levels
from face import FaceDetector, crop_and_resize, crop_box_for
from model import IMAGE_SIZE, FakeFaceModel

MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_IMAGE_SIDE = 2048  # 큰 이미지는 검출 전에 줄여서 노트북 CPU 부담을 줄인다

state: dict = {}
# MediaPipe 검출기는 스레드 안전하지 않으므로 요청 하나씩 처리
inference_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    started = time.perf_counter()
    state["model"] = FakeFaceModel()
    state["detector"] = FaceDetector()
    model = state["model"]
    print(f"[server] model: {model.model_path} ({model.source})")
    print(f"[server] threshold(real): {model.thresholds} ({model.config_path})")
    print(f"[server] ready in {time.perf_counter() - started:.1f}s")
    yield
    state["detector"].close()


app = FastAPI(title="AI Fake Face Detector", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"^(chrome-extension://.*|https?://(localhost|127\.0\.0\.1)(:\d+)?)$",
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def _decode_base64(text: str) -> bytes:
    if text.startswith("data:"):
        text = text.split(",", 1)[-1]
    try:
        return base64.b64decode(text, validate=False)
    except (binascii.Error, ValueError) as error:
        raise HTTPException(status_code=400, detail=f"base64 디코딩 실패: {error}")


async def _read_image_bytes(request: Request) -> bytes:
    """multipart(file), JSON({"image_base64": ...}), 또는 raw 바이트 본문을 받는다."""
    content_type = request.headers.get("content-type", "").lower()
    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        upload = form.get("file") or form.get("image")
        if upload is None or isinstance(upload, str):
            raise HTTPException(status_code=400, detail="multipart 에 'file' 필드가 필요합니다.")
        data = await upload.read()
    elif content_type.startswith("application/json"):
        body = await request.json()
        text = body.get("image_base64") or body.get("image") if isinstance(body, dict) else None
        if not text:
            raise HTTPException(status_code=400, detail="JSON 에 'image_base64' 필드가 필요합니다.")
        data = _decode_base64(text)
    else:
        data = await request.body()

    if not data:
        raise HTTPException(status_code=400, detail="이미지 데이터가 비어 있습니다.")
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="이미지가 너무 큽니다 (20MB 초과).")
    return data


def _open_image(data: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(data))
        image = ImageOps.exif_transpose(image)
        return image.convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        raise HTTPException(status_code=400, detail=f"이미지를 열 수 없습니다: {error}")


def _round(value: float, digits: int = 4) -> float:
    return round(float(value), digits)


def _data_url(image: Image.Image, fmt: str = "JPEG") -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt, quality=92)
    mime = "image/jpeg" if fmt == "JPEG" else "image/png"
    return f"data:{mime};base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


_CAM_ANCHORS = np.array(
    [[0, 0, 0.5], [0, 0.3, 1], [0, 0.9, 0.9], [0.5, 1, 0.3], [1, 0.85, 0], [1, 0.3, 0], [0.6, 0, 0]], dtype=np.float32
)
_CAM_LUT = np.stack(
    [np.interp(np.linspace(0, 1, 256), np.linspace(0, 1, len(_CAM_ANCHORS)), _CAM_ANCHORS[:, c]) for c in range(3)], axis=1
)


def gradcam_overlay(image: Image.Image, cam: np.ndarray, alpha: float = 0.5) -> Image.Image:
    """Grad-CAM 히트맵(0~1)을 모델 입력 이미지 위에 겹친다. 약한 영역은 원본이 더 보이게 alpha 를 cam 에 비례시킨다."""
    colors = _CAM_LUT[(cam * 255).astype(np.uint8)] * 255
    weight = (alpha * np.clip(cam * 1.5, 0, 1))[..., None]
    blended = np.asarray(image, dtype=np.float32) * (1 - weight) + colors * weight
    return Image.fromarray(blended.round().clip(0, 255).astype(np.uint8))


def predict_image(image: Image.Image, mode: str, explain: bool = False) -> dict:
    """explain=True 이면 얼굴마다 모델에 들어간 crop, Grad-CAM, ELA 맵(data URL)을 함께 돌려준다."""
    model: FakeFaceModel = state["model"]
    detector: FaceDetector = state["detector"]

    width, height = image.size
    scale = 1.0
    if max(width, height) > MAX_IMAGE_SIDE:
        scale = MAX_IMAGE_SIDE / max(width, height)
        image = image.resize((round(width * scale), round(height * scale)), Image.BILINEAR)
    work_w, work_h = image.size

    if mode == "full":
        # 얼굴 검출 없이 이미지 전체를 256x256 으로 (노트북 predict_single_image 와 동일)
        faces = [None]
        crop_boxes = [(0, 0, work_w, work_h)]
    else:
        faces = detector.detect(np.asarray(image))
        crop_boxes = [crop_box_for(face, work_w, work_h) for face in faces]

    if not faces:
        return {"status": "no_face", "label": "no_face", "message": "판단 불가(얼굴 없음)", "faces": []}

    if mode == "full":
        batch = np.asarray(image.resize(IMAGE_SIZE, Image.BILINEAR), dtype=np.float32)[None]
    else:
        batch = np.stack([crop_and_resize(image, box, IMAGE_SIZE) for box in crop_boxes])
    probs_real = model.predict_prob_real(batch)

    def to_original(box):
        x, y, w, h = box
        return [_round(x / scale, 1), _round(y / scale, 1), _round(w / scale, 1), _round(h / scale, 1)]

    def to_norm(box):
        x, y, w, h = box
        return [_round(x / work_w), _round(y / work_h), _round(w / work_w), _round(h / work_h)]

    levels = error_levels(image) if explain else None

    results = []
    for index, (face, crop_box, prob_real) in enumerate(zip(faces, crop_boxes, probs_real)):
        bbox = (face.x, face.y, face.w, face.h) if face is not None else crop_box
        results.append(
            {
                "bbox": to_original(bbox),  # [x, y, w, h] 원본 이미지 픽셀
                "bbox_norm": to_norm(bbox),  # [x, y, w, h] 0~1
                "crop_box": to_original(crop_box),  # 모델에 들어간 영역
                "detection_score": _round(face.score) if face is not None else None,
                "prob_fake": _round(1.0 - prob_real),
                "prob_real": _round(prob_real),
                "label": model.label_for(prob_real, mode),
            }
        )
        if explain:
            # 모델이 실제로 본 입력 (crop -> JPEG 90 재인코딩)
            crop_image = model.model_inputs(batch[index : index + 1])[0]
            target = results[-1]["label"]
            cam = model.gradcam(crop_image, target)
            ela_image, ela_stats = ela_map(levels, crop_box, IMAGE_SIZE)
            results[-1].update(
                {
                    "crop_image": _data_url(crop_image),
                    "gradcam_image": _data_url(gradcam_overlay(crop_image, cam)),
                    "gradcam_target": target,
                    "ela_image": _data_url(ela_image),
                    "ela_stats": ela_stats,
                }
            )
    results.sort(key=lambda r: r["bbox"][2] * r["bbox"][3], reverse=True)
    return {"status": "ok", "label": results[0]["label"], "faces": results}


async def _run(request: Request, mode: str, explain: bool) -> dict:
    if mode not in ("face", "full"):
        raise HTTPException(status_code=400, detail="mode 는 'face' 또는 'full' 입니다.")
    data = await _read_image_bytes(request)
    image = _open_image(data)

    started = time.perf_counter()

    def run():
        with inference_lock:
            return predict_image(image, mode, explain)

    result = await anyio.to_thread.run_sync(run)
    model: FakeFaceModel = state["model"]
    result.update(
        {
            "image_size": list(image.size),
            "threshold_real": model.threshold_for(mode),
            "mode": mode,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
        }
    )
    return result


@app.get("/health")
def health():
    model: FakeFaceModel | None = state.get("model")
    return {
        "status": "ok" if model else "loading",
        "model": model.source if model else None,
        "model_path": str(model.model_path) if model else None,
        "threshold_real": model.thresholds if model else None,
    }


@app.post("/predict")
async def predict(request: Request, mode: str = "face"):
    """mode=face(기본): 얼굴 검출 후 얼굴별 판정 / mode=full: 이미지 전체를 한 장으로 판정."""
    return await _run(request, mode, explain=False)


@app.post("/explain")
async def explain(request: Request, mode: str = "face", face_index: int | None = None):
    """/predict 결과 + 얼굴별 모델 입력 crop·Grad-CAM·ELA 맵. face_index 를 주면 그 얼굴만 돌려준다 (/predict 와 같은 순서)."""
    result = await _run(request, mode, explain=True)
    if face_index is not None and result["faces"]:
        if not 0 <= face_index < len(result["faces"]):
            raise HTTPException(status_code=400, detail=f"face_index 범위 밖: 0~{len(result['faces']) - 1}")
        result["faces"] = [result["faces"][face_index]]
        result["face_index"] = face_index
    return result
