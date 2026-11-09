"""Phase 5 준비: LOGO 3종에 쓰이는 모든 이미지의 얼굴 박스를 한 번 계산해 둡니다.

- 데모 서버(server/face.py)와 같은 MediaPipe BlazeFace(short range)를 사용합니다.
- 학습 중 얼굴 crop 증강, 평가 때 서버와 같은 얼굴 crop 보기(x2.0)에 사용합니다.
- 박스는 이미지 크기에 대한 비율(0~1)로 저장하고, 얼굴이 여러 개면 가장 큰 얼굴만 씁니다.

실행 (저장소 루트):
    python scripts/50_detect_faces.py
WSL 의 MediaPipe 는 시스템 GL 라이브러리(libGLESv2)가 없으면 실행되지 않으므로,
그 경우 mediapipe 가 설치된 Windows Python 으로 실행합니다 (예: python \\\\wsl.localhost\\ubuntu\\...\\scripts\\50_detect_faces.py).
파일은 \\\\wsl.localhost 경로로 읽고, CSV 에는 manifest 와 같은 리눅스 경로를 저장합니다.
출력: data/manifests/face_boxes.csv (path, n_faces, x, y, w, h, score)
"""

import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DATA_DIR, ROOT

DETECTOR_PATH = ROOT / "models_aux" / "blaze_face_short_range.tflite"
MIN_CONFIDENCE = 0.5
WSL_PREFIX = "\\\\wsl.localhost\\ubuntu"

_detector = None


def _local(path: str) -> str:
    """Windows 에서 실행할 때 리눅스 경로를 \\\\wsl.localhost 경로로 바꿉니다."""
    if os.name == "nt" and path.startswith("/"):
        return WSL_PREFIX + path.replace("/", "\\")
    return path


def _init():
    global _detector
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    options = vision.FaceDetectorOptions(
        base_options=mp_python.BaseOptions(model_asset_buffer=DETECTOR_PATH.read_bytes()),
        min_detection_confidence=MIN_CONFIDENCE,
    )
    _detector = vision.FaceDetector.create_from_options(options)


def _detect(path: str) -> dict:
    import mediapipe as mp

    with Image.open(_local(path)) as image:
        rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]
    result = _detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
    row = {"path": path, "n_faces": len(result.detections)}
    if result.detections:
        best = max(result.detections, key=lambda d: d.bounding_box.width * d.bounding_box.height)
        box = best.bounding_box
        row.update({
            "x": box.origin_x / width, "y": box.origin_y / height,
            "w": box.width / width, "h": box.height / height,
            "score": best.categories[0].score if best.categories else np.nan,
        })
    return row


def main():
    paths = set()
    for holdout in ["gan", "diffusion", "manip"]:
        paths |= set(pd.read_csv(DATA_DIR / "manifests" / f"logo_{holdout}.csv")["path"])
    paths = sorted(paths)
    print(f"이미지 {len(paths):,}장 얼굴 검출", flush=True)

    rows = []
    with Pool(8, initializer=_init) as pool:
        for i, row in enumerate(pool.imap(_detect, paths, chunksize=256), 1):
            rows.append(row)
            if i % 20000 == 0:
                print(f"  {i:,} / {len(paths):,}", flush=True)

    boxes = pd.DataFrame(rows)
    out_path = DATA_DIR / "manifests" / "face_boxes.csv"
    boxes.to_csv(out_path, index=False)
    print(f"저장: {out_path}  (얼굴 못 찾음 {(boxes['n_faces'] == 0).mean():.2%})")


if __name__ == "__main__":
    main()
