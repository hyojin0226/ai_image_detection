"""MediaPipe 얼굴 검출 + 학습 데이터와 비슷한 여백을 둔 정사각형 crop."""
from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

SERVER_DIR = Path(__file__).resolve().parent
DETECTOR_PATH = SERVER_DIR / "models" / "blaze_face_short_range.tflite"
DETECTOR_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
)

MIN_DETECTION_CONFIDENCE = float(os.environ.get("FACE_MIN_CONFIDENCE", "0.5"))
# 얼굴 bbox 한 변 대비 crop 한 변의 배율. FFHQ 처럼 머리·턱·배경이 조금 들어가게 한다.
CROP_SCALE = float(os.environ.get("FACE_CROP_SCALE", "2.0"))
# crop 중심을 얼굴 중심보다 위로 올리는 비율(crop 한 변 기준). 이마·머리카락을 더 포함.
CROP_SHIFT_UP = float(os.environ.get("FACE_CROP_SHIFT_UP", "0.05"))
# 이 비율보다 작은 얼굴은 무시 (이미지 짧은 변 기준)
MIN_FACE_RATIO = float(os.environ.get("FACE_MIN_RATIO", "0.04"))
MAX_FACES = int(os.environ.get("FACE_MAX_FACES", "8"))
# 타일에서만 찾은 작은 얼굴은 오검출(나뭇잎·질감)이 많아 더 엄격하게 거른다.
TILE_MIN_CONFIDENCE = float(os.environ.get("FACE_TILE_MIN_CONFIDENCE", "0.75"))
VERIFY_MIN_CONFIDENCE = float(os.environ.get("FACE_VERIFY_MIN_CONFIDENCE", "0.6"))


@dataclass
class FaceBox:
    x: float
    y: float
    w: float
    h: float
    score: float


def _ensure_detector_file() -> bytes:
    if not DETECTOR_PATH.exists():
        DETECTOR_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"[face] 얼굴 검출 모델 다운로드: {DETECTOR_URL}")
        urllib.request.urlretrieve(DETECTOR_URL, DETECTOR_PATH)
    # 경로에 한글이 있어도 되도록 파일 경로 대신 바이트로 넘긴다.
    return DETECTOR_PATH.read_bytes()


def _center_inside(a: FaceBox, b: FaceBox) -> bool:
    cx, cy = a.x + a.w / 2, a.y + a.h / 2
    return b.x <= cx <= b.x + b.w and b.y <= cy <= b.y + b.h


def _iou(a: FaceBox, b: FaceBox) -> float:
    x1, y1 = max(a.x, b.x), max(a.y, b.y)
    x2, y2 = min(a.x + a.w, b.x + b.w), min(a.y + a.h, b.y + b.h)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = a.w * a.h + b.w * b.h - inter
    return inter / union if union > 0 else 0.0


class FaceDetector:
    def __init__(self):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        options = vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_buffer=_ensure_detector_file()),
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        )
        self._detector = vision.FaceDetector.create_from_options(options)

    def _detect_region(self, rgb: np.ndarray, offset_x: int = 0, offset_y: int = 0) -> list[FaceBox]:
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        result = self._detector.detect(mp_image)
        boxes = []
        for detection in result.detections:
            bb = detection.bounding_box
            score = detection.categories[0].score if detection.categories else 0.0
            boxes.append(FaceBox(bb.origin_x + offset_x, bb.origin_y + offset_y, bb.width, bb.height, float(score)))
        return boxes

    def _verify(self, rgb: np.ndarray, box: FaceBox) -> bool:
        """후보 주변을 얼굴이 크게 보이도록 잘라 다시 검출해, 같은 위치에서 또 얼굴이 나오면 인정."""
        height, width = rgb.shape[:2]
        side = int(max(box.w, box.h) * 2.5)
        cx, cy = box.x + box.w / 2, box.y + box.h / 2
        left = int(min(max(cx - side / 2, 0), max(0, width - side)))
        top = int(min(max(cy - side / 2, 0), max(0, height - side)))
        crop = rgb[top:top + side, left:left + side]
        for found in self._detect_region(crop, left, top):
            if found.score >= VERIFY_MIN_CONFIDENCE and _center_inside(found, box):
                return True
        return False

    def detect(self, rgb: np.ndarray) -> list[FaceBox]:
        """전체 이미지 + 큰 이미지는 겹치는 타일에서도 찾는다.
        (short-range 모델은 이미지 안에서 얼굴이 작으면 놓치기 때문)"""
        height, width = rgb.shape[:2]
        boxes = self._detect_region(rgb)

        if min(width, height) >= 400:
            tile = int(min(width, height) * 0.35)
            step = max(1, tile // 2)
            for top in sorted({*range(0, max(1, height - tile + 1), step), max(0, height - tile)}):
                for left in sorted({*range(0, max(1, width - tile + 1), step), max(0, width - tile)}):
                    for box in self._detect_region(rgb[top:top + tile, left:left + tile], left, top):
                        if box.score >= TILE_MIN_CONFIDENCE and self._verify(rgb, box):
                            boxes.append(box)

        min_side = MIN_FACE_RATIO * min(width, height)
        boxes = [b for b in boxes if max(b.w, b.h) >= min_side]

        # 타일 간 중복 제거 (NMS). 타일 경계에 잘린 큰 얼굴의 일부 검출도 같은 얼굴로 본다.
        boxes.sort(key=lambda b: b.score, reverse=True)
        kept: list[FaceBox] = []
        for box in boxes:
            if all(_iou(box, k) < 0.3 and not _center_inside(box, k) and not _center_inside(k, box) for k in kept):
                kept.append(box)
        return kept[:MAX_FACES]

    def close(self):
        try:
            self._detector.close()
        except Exception:
            pass


def crop_box_for(face: FaceBox, width: int, height: int) -> tuple[int, int, int, int]:
    """얼굴 주변에 여백을 둔 정사각형 crop 영역 (left, top, side, side).
    이미지 밖으로 나가면 패딩 대신 안쪽으로 밀고, 그래도 크면 짧은 변에 맞춘다."""
    side = max(face.w, face.h) * CROP_SCALE
    side = min(side, width, height)
    cx = face.x + face.w / 2
    cy = face.y + face.h / 2 - CROP_SHIFT_UP * side
    left = min(max(cx - side / 2, 0), width - side)
    top = min(max(cy - side / 2, 0), height - side)
    return int(round(left)), int(round(top)), int(round(side)), int(round(side))


def crop_and_resize(image: Image.Image, box: tuple[int, int, int, int], size: tuple[int, int]) -> np.ndarray:
    left, top, w, h = box
    crop = image.crop((left, top, left + w, top + h)).resize(size, Image.BILINEAR)
    return np.asarray(crop, dtype=np.float32)
