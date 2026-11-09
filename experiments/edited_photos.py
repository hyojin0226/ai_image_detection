"""실험 B: 보정 사진 오탐 분석 + Grad-CAM / ELA 와 실제 보정 영역의 겹침 (PROJECT_PLAN 3-3).

질문
1. 진짜 사진을 보정(피부 보정·눈 키우기·얼굴 슬림·뷰티 필터 조합)하면 최종 모델이 FAKE 로 오판하는가? 강도별로 얼마나?
2. 그때 Grad-CAM(모델이 본 곳)과 ELA(재압축 오차)는 실제로 보정된 영역을 가리키는가?

데이터 (새로 수집하지 않음)
- 학습에 쓰지 않은 FFHQ real (WSL data/manifests/logo_all.csv 의 test_id / test_ood 역할, png) 에서 무작위 N 장
- 실제 흐름을 흉내: 카메라 JPEG(q90) -> 보정 앱에서 편집 -> JPEG(q95) 저장. 대조군은 편집 없이 q95 로 다시 저장.
- 보정은 스크립트로 만듭니다 (원본-보정본 쌍이 공개된 데이터가 거의 없음, PROJECT_PLAN 5절).
  "AI 뷰티 필터"는 생성 모델 기반 실제 앱 필터를 재현할 수 없어서, 피부 보정 + 눈 + 슬림 + 피부 톤 보정 조합으로 근사합니다.

판정·설명은 데모 서버 코드(server/)를 그대로 씁니다: 얼굴 검출 -> crop x2.0 -> JPEG 90 -> ConvNeXt-T, threshold 0.30.

겹침 지표 (모두 모델 입력 256x256 crop 좌표에서 계산)
- 보정 마스크: 같은 crop 영역에서 |원본 - 보정본| 채널 평균 > MASK_DIFF 인 픽셀 (작은 점 제거 후)
- IoU@0.5: Grad-CAM >= 0.5 영역과 마스크의 IoU
- IoU@area: 맵 상위 (마스크 넓이) 픽셀과 마스크의 IoU — 임계값 선택에 덜 민감
- Pointing Game: 맵 최댓값 위치가 (조금 넓힌) 마스크 안에 있는가
- Energy ratio: 맵 값 합 중 마스크 안 비율. 마스크 넓이 비율(무작위 기대값)과 비교
- 기준선: crop 중심점이 (넓힌) 마스크 안에 있는 비율(얼굴이 가운데라 생기는 우연 적중),
  무작위 점이 들어갈 확률(= 넓힌 마스크 넓이). Pointing Game 은 이 두 값과 비교해야 의미가 있습니다.

실행 (Windows, 저장소 루트): python experiments/edited_photos.py [--n 300] [--smoke]
  다른 모델로 오탐률만 비교: FAKE_FACE_MODEL_CONFIG=<설정 json> python experiments/edited_photos.py --n 150 --no-explain --out <이름>
출력: results/experiment_b/{per_image.csv, summary.csv, overlap_summary.csv, examples_*.jpg}
"""
from __future__ import annotations

import argparse
import io
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "server"))

from ela import error_levels  # noqa: E402
from face import FaceDetector, crop_and_resize, crop_box_for  # noqa: E402
from model import IMAGE_SIZE, FakeFaceModel  # noqa: E402

WSL_REPO = r"\\wsl.localhost\ubuntu\home\hyojin\ai_image_detection"
OUT_DIR = REPO / "results" / "experiment_b"
SEED = 42
CAMERA_QUALITY = 90
APP_QUALITY = 95
MASK_DIFF = 2.0  # 0~255 채널 평균 차이. 저장(JPEG) 전 원본·보정본끼리 비교하므로 0 이 아닌 차이는 모두 보정 때문
POINT_DILATE = 15  # Pointing Game 허용 오차 (마스크를 이만큼 넓혀서 판정)

EDITS = {
    "skin_smooth": [1, 2, 3],
    "eye_enlarge": [1, 2, 3],
    "face_slim": [1, 2, 3],
    "beauty_combo": [1, 2],
}


# ---------- 입출력 ----------
def jpeg_roundtrip(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def local_path(path: str) -> str:
    return WSL_REPO.rsplit("\\home", 1)[0] + path.replace("/", "\\")


# ---------- 얼굴 기하 (MediaPipe keypoints) ----------
class Geometry:
    """keypoints: 0 오른쪽 눈, 1 왼쪽 눈, 2 코끝, 3 입, 4 오른쪽 귀(이주), 5 왼쪽 귀. 픽셀 좌표."""

    def __init__(self, box, keypoints):
        self.x, self.y, self.w, self.h = box
        self.kp = np.asarray(keypoints, dtype=np.float32)
        self.eye_r, self.eye_l, self.nose, self.mouth, self.ear_r, self.ear_l = self.kp
        self.iod = float(np.linalg.norm(self.eye_l - self.eye_r))
        self.cx = float((self.ear_r[0] + self.ear_l[0]) / 2)


def detect_geometry(detector: FaceDetector, rgb: np.ndarray) -> Geometry | None:
    mp = detector._mp
    result = detector._detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)))
    if not result.detections:
        return None
    detection = max(result.detections, key=lambda d: d.bounding_box.width * d.bounding_box.height)
    bb = detection.bounding_box
    h, w = rgb.shape[:2]
    keypoints = [(k.x * w, k.y * h) for k in detection.keypoints]
    return Geometry((bb.origin_x, bb.origin_y, bb.width, bb.height), keypoints)


# ---------- 보정 ----------
def _face_mask(shape, g: Geometry, exclude_features: bool) -> np.ndarray:
    """얼굴 피부 영역 타원 (눈·입 제외 가능), 경계를 부드럽게."""
    h, w = shape[:2]
    mask = np.zeros((h, w), np.float32)
    center = (int(g.x + g.w / 2), int(g.y + g.h * 0.55))
    axes = (int(g.w * 0.55), int(g.h * 0.72))
    cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
    if exclude_features:
        for point, radius in ((g.eye_r, 0.22), (g.eye_l, 0.22), (g.mouth, 0.3)):
            cv2.circle(mask, (int(point[0]), int(point[1])), int(g.iod * radius), 0.0, -1)
    return cv2.GaussianBlur(mask, (0, 0), max(1.0, g.iod * 0.08))


def skin_smooth(rgb: np.ndarray, g: Geometry, level: int) -> np.ndarray:
    sigma_color, passes, alpha = [(30, 1, 0.6), (55, 2, 0.85), (90, 3, 1.0)][level - 1]
    smooth = rgb.copy()
    for _ in range(passes):
        smooth = cv2.bilateralFilter(smooth, d=9, sigmaColor=sigma_color, sigmaSpace=7)
    mask = (_face_mask(rgb.shape, g, exclude_features=True) * alpha)[..., None]
    return (rgb * (1 - mask) + smooth * mask).round().clip(0, 255).astype(np.uint8)


def _remap(rgb: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
    return cv2.remap(rgb, map_x.astype(np.float32), map_y.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def _grid(rgb):
    h, w = rgb.shape[:2]
    return np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))


def eye_enlarge(rgb: np.ndarray, g: Geometry, level: int) -> np.ndarray:
    """눈 주변 국소 확대 (보정 앱의 '눈 키우기'). 반경 안쪽을 중심 쪽에서 샘플링."""
    strength = [0.15, 0.28, 0.4][level - 1]
    radius = g.iod * 0.42
    xx, yy = _grid(rgb)
    map_x, map_y = xx.copy(), yy.copy()
    for eye in (g.eye_r, g.eye_l):
        dx, dy = xx - eye[0], yy - eye[1]
        d2 = (dx * dx + dy * dy) / (radius * radius)
        factor = np.where(d2 < 1, 1 - strength * (1 - d2) ** 2, 1.0)
        map_x = np.where(d2 < 1, eye[0] + dx * factor, map_x)
        map_y = np.where(d2 < 1, eye[1] + dy * factor, map_y)
    return _remap(rgb, map_x, map_y)


def face_slim(rgb: np.ndarray, g: Geometry, level: int) -> np.ndarray:
    """볼·턱선을 얼굴 중심 쪽으로 미는 리퀴파이 (보정 앱의 '얼굴 슬림')."""
    push = [0.035, 0.07, 0.105][level - 1] * g.w
    radius = g.iod * 0.75
    xx, yy = _grid(rgb)
    map_x = xx.copy()
    jaw_y = (g.nose[1] + g.mouth[1]) / 2 + g.iod * 0.15
    for ear, sign in ((g.ear_r, -1.0), (g.ear_l, 1.0)):
        # 귀와 입 사이, 턱선 높이의 점을 중심 쪽(sign 반대)으로 민다 -> 바깥쪽에서 샘플링
        cx = ear[0] * 0.65 + g.mouth[0] * 0.35
        d2 = ((xx - cx) ** 2 + (yy - jaw_y) ** 2) / (radius * radius)
        weight = np.where(d2 < 1, (1 - d2) ** 2, 0.0)
        map_x = map_x + sign * push * weight
    return _remap(rgb, map_x, yy)


def tone_whiten(rgb: np.ndarray, g: Geometry, amount: float) -> np.ndarray:
    """피부 톤 밝게·화사하게 (얼굴 영역만)."""
    lifted = 255 * (rgb / 255.0) ** (1 - amount)
    lifted = lifted * (1 - amount * 0.5) + 255 * amount * 0.5 * (rgb / 255.0)
    mask = _face_mask(rgb.shape, g, exclude_features=False)[..., None]
    return (rgb * (1 - mask) + lifted * mask).round().clip(0, 255).astype(np.uint8)


def beauty_combo(rgb: np.ndarray, g: Geometry, level: int) -> np.ndarray:
    base = level + 1  # 1 -> 각 보정 중간(2), 2 -> 강함(3)
    out = skin_smooth(rgb, g, base)
    out = eye_enlarge(out, g, base)
    out = face_slim(out, g, base)
    return tone_whiten(out, g, [0.12, 0.2][level - 1])


EDIT_FUNCS = {"skin_smooth": skin_smooth, "eye_enlarge": eye_enlarge, "face_slim": face_slim, "beauty_combo": beauty_combo}


# ---------- 겹침 지표 ----------
def _resize_float(array: np.ndarray, box, size) -> np.ndarray:
    left, top, w, h = box
    region = np.ascontiguousarray(array[top:top + h, left:left + w], dtype=np.float32)
    return cv2.resize(region, size, interpolation=cv2.INTER_LINEAR)


def edit_mask(original: np.ndarray, edited: np.ndarray, box) -> np.ndarray:
    diff = np.abs(original.astype(np.float32) - edited.astype(np.float32)).mean(axis=2)
    diff = _resize_float(diff, box, IMAGE_SIZE)
    mask = (diff > MASK_DIFF).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    return mask.astype(bool)


def _dilate(mask: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (POINT_DILATE, POINT_DILATE))
    return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)


def overlap(saliency: np.ndarray, mask: np.ndarray, prefix: str) -> dict:
    area = int(mask.sum())
    if area == 0:
        return {}
    smap = saliency / saliency.max() if saliency.max() > 0 else saliency
    top = smap >= np.partition(smap.ravel(), -area)[-area]
    above = smap >= 0.5
    dilated = _dilate(mask)
    peak = np.unravel_index(int(np.argmax(smap)), smap.shape)
    return {
        f"{prefix}_iou_05": float((above & mask).sum() / max(1, (above | mask).sum())),
        f"{prefix}_iou_area": float((top & mask).sum() / max(1, (top | mask).sum())),
        f"{prefix}_pointing": float(dilated[peak]),
        f"{prefix}_energy_in_mask": float(smap[mask].sum() / max(1e-6, smap.sum())),
    }


# ---------- 메인 ----------
def server_view(detector, image: Image.Image):
    """서버와 같은 얼굴 검출 + crop. 가장 큰 얼굴만."""
    faces = detector.detect(np.asarray(image))
    if not faces:
        return None
    face = max(faces, key=lambda f: f.w * f.h)
    box = crop_box_for(face, *image.size)
    return box, crop_and_resize(image, box, IMAGE_SIZE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--smoke", action="store_true", help="8장, 예시 그림만")
    parser.add_argument("--no-explain", action="store_true", help="오탐률만 (Grad-CAM·ELA 생략, 모델 비교용)")
    parser.add_argument("--out", default="", help="results/experiment_b/<out> 에 저장 (다른 모델 비교 시)")
    args = parser.parse_args()
    n = 8 if args.smoke else args.n
    random.seed(SEED)
    np.random.seed(SEED)

    split = pd.read_csv(local_path("/home/hyojin/ai_image_detection/data/manifests/logo_all.csv"))
    pool = split[(split["y"] == 1) & (split["source"] == "ffhq_1mff") & (split["ext"] == "png")
                 & split["role"].isin(["test_id", "test_ood"])]
    pool = pool.sample(frac=1, random_state=SEED)
    print(f"후보 {len(pool)}장 중 {n}장 사용", flush=True)

    model = FakeFaceModel()
    detector = FaceDetector()
    threshold = model.threshold_for("face")
    out_dir = OUT_DIR / args.out / ("smoke" if args.smoke else "")
    print(f"모델 {model.source} ({model.config_path}), threshold {threshold}", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, examples = [], []
    started, used = time.time(), 0
    for path in pool["path"]:
        if used >= n:
            break
        source = Image.open(local_path(path)).convert("RGB")
        camera = jpeg_roundtrip(source, CAMERA_QUALITY)  # 카메라가 저장한 원본 사진
        camera_rgb = np.asarray(camera)
        geometry = detect_geometry(detector, camera_rgb)
        if geometry is None:
            continue
        used += 1

        variants = [("original", 0, camera_rgb)]
        for edit, levels in EDITS.items():
            for level in levels:
                variants.append((edit, level, EDIT_FUNCS[edit](camera_rgb, geometry, level)))

        for edit, level, raw in variants:
            saved = jpeg_roundtrip(Image.fromarray(raw), APP_QUALITY)  # 보정 앱 저장 (원본은 편집 없이 재저장)
            view = server_view(detector, saved)
            row = {"path": path, "edit": edit, "level": level}
            if view is None:
                rows.append({**row, "detected": False})
                continue
            box, crop = view
            prob_real = float(model.predict_prob_real(crop[None])[0])
            row.update({"detected": True, "prob_real": prob_real, "pred_fake": prob_real < threshold})
            if edit != "original" and not args.no_explain:
                mask = edit_mask(camera_rgb, raw, box)
                row["mask_area"] = float(mask.mean())
                if mask.any():
                    model_input = model.model_inputs(crop[None])[0]
                    cam = model.gradcam(model_input, "fake")
                    ela = _resize_float(error_levels(saved), box, IMAGE_SIZE)
                    ela = cv2.GaussianBlur(ela, (0, 0), 3)
                    left, top, side, _ = box
                    center = (IMAGE_SIZE[1] // 2, IMAGE_SIZE[0] // 2)
                    row.update(overlap(cam, mask, "cam"))
                    row.update(overlap(ela, mask, "ela"))
                    # 기준선: 가운데 점 / 무작위 점이 (넓힌) 마스크에 들어갈 확률
                    row["center_pointing"] = float(_dilate(mask)[center])
                    row["chance_pointing"] = float(_dilate(mask).mean())
                    if len(examples) < 12 and (level == max(EDITS[edit]) or args.smoke):
                        examples.append((edit, level, prob_real, model_input, mask, cam, ela))
            rows.append(row)
        if used % 25 == 0:
            print(f"{used}/{n} ({time.time() - started:.0f}s)", flush=True)

    per_image = pd.DataFrame(rows)
    per_image.to_csv(out_dir / "per_image.csv", index=False)
    summarize(per_image, threshold, out_dir)
    save_examples(examples, out_dir / "examples.jpg")
    print(f"saved {out_dir} ({used}장, {time.time() - started:.0f}s)")


def summarize(per_image: pd.DataFrame, threshold: float, out_dir: Path):
    data = per_image[per_image["detected"] == True].copy()  # noqa: E712
    original = data[data["edit"] == "original"].set_index("path")["prob_real"]
    data["delta_prob_real"] = data["prob_real"] - data["path"].map(original)
    summary = data.groupby(["edit", "level"], sort=False).agg(
        n=("prob_real", "size"),
        fake_rate=("pred_fake", "mean"),
        mean_prob_real=("prob_real", "mean"),
        median_prob_real=("prob_real", "median"),
        mean_delta_prob_real=("delta_prob_real", "mean"),
    ).reset_index()
    detected = per_image.groupby(["edit", "level"], sort=False)["detected"].mean().rename("detect_rate").reset_index()
    summary = summary.merge(detected, on=["edit", "level"])
    summary["threshold_real"] = threshold
    summary.to_csv(out_dir / "summary.csv", index=False)

    metric_cols = [c for c in data.columns if c.startswith(("cam_", "ela_")) or c in ("mask_area", "center_pointing", "chance_pointing")]
    if not metric_cols:
        print(summary.round(3).to_string(index=False))
        return
    edited = data[data["edit"] != "original"]
    overlap_summary = edited.groupby(["edit", "level"], sort=False)[metric_cols].mean().reset_index()
    overlap_summary.to_csv(out_dir / "overlap_summary.csv", index=False)
    # FAKE 로 오판된 경우만 따로 (모델이 '왜 FAKE 라고 했나'를 볼 때 의미가 큼)
    wrong = edited[edited["pred_fake"] == True]  # noqa: E712
    if len(wrong):
        wrong.groupby("edit")[metric_cols].mean().assign(n=wrong.groupby("edit").size()).to_csv(out_dir / "overlap_when_fake.csv")
    with pd.option_context("display.width", 200):
        print(summary.round(3).to_string(index=False))
        print(overlap_summary.round(3).to_string(index=False))


def save_examples(examples, path: Path):
    if not examples:
        return
    tile = 192
    sheet = Image.new("RGB", (tile * 4, tile * len(examples)), "white")
    for row, (edit, level, prob_real, model_input, mask, cam, ela) in enumerate(examples):
        mask_img = Image.fromarray((mask * 255).astype(np.uint8)).convert("RGB")
        cam_img = Image.fromarray(cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)[..., ::-1])
        cam_img = Image.blend(model_input, cam_img, 0.45)
        ela_norm = (np.clip(ela / max(np.percentile(ela, 99), 1e-6), 0, 1) * 255).astype(np.uint8)
        ela_img = Image.fromarray(cv2.applyColorMap(ela_norm, cv2.COLORMAP_INFERNO)[..., ::-1])
        for col, image in enumerate([model_input, mask_img, cam_img, ela_img]):
            sheet.paste(image.resize((tile, tile)), (col * tile, row * tile))
        from PIL import ImageDraw

        ImageDraw.Draw(sheet).text((4, row * tile + 4), f"{edit} L{level} P(real)={prob_real:.2f}", fill=(255, 255, 0))
    sheet.save(path, quality=90)


if __name__ == "__main__":
    main()
