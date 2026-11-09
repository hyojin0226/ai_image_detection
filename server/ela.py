"""ELA(Error Level Analysis): 같은 품질로 다시 JPEG 저장했을 때의 차이로 편집 의심 영역을 보여준다.

한 번에 저장된 사진은 전체 오차가 비슷하고, 붙여넣기·부분 보정한 영역은 오차 수준이 달라 밝게 튀는 경향이 있다.
SNS 가 이미지를 통째로 재압축하면 신호가 약해지므로 참고용으로만 쓴다. 모델 판정과는 무관하다.
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image

ELA_QUALITY = 90

# 어두운 보라 → 빨강 → 노랑 (inferno 비슷한 색상표)
_ANCHORS = np.array(
    [[0, 0, 4], [40, 11, 84], [101, 21, 110], [159, 42, 99], [212, 72, 66], [245, 125, 21], [250, 193, 39], [252, 255, 164]],
    dtype=np.float32,
)
_LUT = np.stack(
    [np.interp(np.linspace(0, 1, 256), np.linspace(0, 1, len(_ANCHORS)), _ANCHORS[:, c]) for c in range(3)], axis=1
).astype(np.uint8)


def error_levels(image: Image.Image, quality: int = ELA_QUALITY) -> np.ndarray:
    """픽셀별 재압축 오차 (H, W), 0~255 채널 평균."""
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    resaved = np.asarray(Image.open(buffer).convert("RGB"), dtype=np.int16)
    original = np.asarray(image, dtype=np.int16)
    return np.abs(original - resaved).mean(axis=2).astype(np.float32)


def ela_map(levels: np.ndarray, box: tuple[int, int, int, int], size: tuple[int, int]) -> tuple[Image.Image, dict]:
    """box 영역의 오차를 색으로 표시한 이미지와 요약 수치.

    밝기는 영역 안 99 백분위로 정규화한다 (한두 픽셀 튀는 값 때문에 전체가 어두워지지 않도록).
    """
    left, top, width, height = box
    region = levels[top:top + height, left:left + width]
    scale = max(float(np.percentile(region, 99)), 1.0)
    normalized = np.clip(region / scale, 0, 1)
    colored = Image.fromarray(_LUT[(normalized * 255).astype(np.uint8)]).resize(size, Image.BILINEAR)
    stats = {
        "mean": round(float(region.mean()), 3),
        "p99": round(float(np.percentile(region, 99)), 3),
        "image_mean": round(float(levels.mean()), 3),
        "quality": ELA_QUALITY,
    }
    return colored, stats
