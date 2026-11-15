"""제이 사이트의 기존 EfficientNetV2S(1차) 와 최종 EfficientNetV2S(Phase 9) 를 같은 이미지로 비교할 세트를 만든다.

공정성: 두 모델 모두 학습·검증에 쓰지 않은 이미지만 쓴다.
- 1차 EfficientNetV2S: data/train, data/valid 로 학습·threshold 선정
- 최종 EfficientNetV2S(Phase 9): logo_all.csv 의 train / valid 역할로 학습·threshold 선정
- 남은 것: data/test 에 있으면서, 같은 사진(group_key: 소스 간 중복 묶음)이 위 어느 쪽에도 없는 이미지
  (1차 test 는 73% 가 train 과 같은 사진이라 그대로 쓰면 1차 모델이 유리하다. EXPERIMENT_PLAN 2.4)

조건
- none / jpeg_q50 / resize_0.5 / blur_1.0  (src/perturb.py 와 같은 PIL 변환)
- 보정 사진: 깨끗한 real 에 카메라 JPEG90 -> 보정(피부 보정 중간, 뷰티 조합 중간) -> 앱 JPEG95 (실험 B 와 같은 흐름)

최종 모델 점수는 여기서 서버 predict_image(face 모드) 로 계산한다 (브라우저 모듈과 판정 불일치 0 으로 확인됨).
EfficientNetV2S 점수는 compare.html 에서 제이 사이트의 TF.js 모델·전처리를 그대로 써서 계산한다.

사용 (저장소 루트에서):
    server\\.venv\\Scripts\\python.exe web_model\\compare_models.py --fakes 700
출력: web_model/reference/compare/{images/, manifest.json}  (커밋 안 함)
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))
sys.path.insert(0, str(ROOT / "experiments"))
import app as server_app  # noqa: E402
from face import FaceDetector  # noqa: E402
from model import FakeFaceModel  # noqa: E402

WSL = Path(r"\\wsl.localhost\ubuntu\home\hyojin\ai_image_detection")
OUT = Path(__file__).resolve().parent / "reference" / "compare"
SEED = 42


def jpeg(image: Image.Image, quality: int) -> Image.Image:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def resize(image: Image.Image, scale: float) -> Image.Image:
    w, h = image.size
    return image.resize((round(w * scale), round(h * scale)), Image.BILINEAR).resize((w, h), Image.BILINEAR)


PERTURB = {
    "none": None,
    "jpeg_q50": lambda im: jpeg(im, 50),
    "resize_0.5": lambda im: resize(im, 0.5),
    "blur_1.0": lambda im: im.filter(ImageFilter.GaussianBlur(radius=1.0)),
}


def load_manifest(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def folder_of(path: str) -> str:
    return path.split("/data/")[1].split("/")[0]


def local(path: str) -> Path:
    return WSL / path.removeprefix("/home/hyojin/ai_image_detection/").replace("/", "\\")


def clean_pool(manifest_dir: Path) -> list[dict]:
    full = {}
    for family in ("gan", "diffusion", "manip"):
        for row in load_manifest(manifest_dir / f"logo_{family}.csv"):
            full[row["path"]] = row
    exp01_groups = {r["group_key"] for p, r in full.items() if folder_of(p) in ("train", "valid")}
    final_groups = {r["group_key"] for r in load_manifest(manifest_dir / "logo_all.csv") if r["role"] in ("train", "valid")}
    seen, pool = set(), []
    for path, row in sorted(full.items()):
        group = row["group_key"]
        if folder_of(path) != "test" or group in exp01_groups or group in final_groups or group in seen:
            continue
        seen.add(group)
        pool.append(row)
    return pool


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fakes", type=int, default=700, help="fake 표본 수 (real 은 전부 사용)")
    parser.add_argument("--manifests", type=Path, default=WSL / "data" / "manifests")
    args = parser.parse_args()
    rng = random.Random(SEED)

    pool = clean_pool(args.manifests)
    reals = [r for r in pool if r["label"] == "real"]
    fakes_by_gen = defaultdict(list)
    for r in pool:
        if r["label"] == "fake":
            fakes_by_gen[r["generator"]].append(r)
    n_fake_pool = sum(len(v) for v in fakes_by_gen.values())
    fakes = []
    for gen, rows in sorted(fakes_by_gen.items()):
        k = max(1, round(args.fakes * len(rows) / n_fake_pool))
        fakes += rng.sample(rows, min(k, len(rows)))
    print(f"[pool] 깨끗한 이미지 {len(pool)}장 -> real {len(reals)} / fake {len(fakes)} 사용")

    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "images").mkdir(parents=True)

    server_app.state["model"] = FakeFaceModel()
    server_app.state["detector"] = FaceDetector()
    threshold = server_app.state["model"].threshold_for("face")
    threshold_full = server_app.state["model"].threshold_for("full")

    from edited_photos import EDIT_FUNCS, detect_geometry, jpeg_roundtrip  # noqa: E402

    items = []

    def add(name: str, image: Image.Image, meta: dict, fmt: str = "PNG", quality: int | None = None):
        path = OUT / "images" / name
        if fmt == "PNG":
            image.save(path, format="PNG")
        else:
            image.save(path, format="JPEG", quality=quality)
        decoded = server_app._open_image(path.read_bytes())
        result = server_app.predict_image(decoded, "face")
        face_prob = result["faces"][0]["prob_real"] if result["status"] == "ok" else None
        full = server_app.predict_image(decoded, "full")["faces"][0]["prob_real"]
        items.append({**meta, "file": name, "final_face": face_prob, "final_full": full})

    rows = reals + fakes
    for index, row in enumerate(rows):
        source = Image.open(local(row["path"])).convert("RGB")
        base = {"label": 1 if row["label"] == "real" else 0, "family": row["family"], "generator": row["generator"],
                "source": row["source"], "path": row["path"]}
        stem = f"{index:04d}"
        for cond, fn in PERTURB.items():
            image = source if fn is None else fn(source)
            add(f"{stem}_{cond}.png", image, {**base, "condition": cond})
        if row["label"] == "real" and row["generator"] == "ffhq":
            camera = jpeg_roundtrip(source, 90)
            rgb = np.asarray(camera)
            geometry = detect_geometry(server_app.state["detector"], rgb)
            if geometry is not None:
                for edit, level, cond in (("original", 0, "edit_control"), ("skin_smooth", 2, "edit_skin_mid"),
                                          ("beauty_combo", 1, "edit_beauty_mid")):
                    raw = rgb if edit == "original" else EDIT_FUNCS[edit](rgb, geometry, level)
                    add(f"{stem}_{cond}.jpg", Image.fromarray(raw), {**base, "condition": cond}, fmt="JPEG", quality=95)
        if (index + 1) % 100 == 0:
            print(f"  {index + 1}/{len(rows)}", flush=True)

    meta = {"final_threshold": threshold, "final_threshold_full": threshold_full, "count": len(items), "pool": len(pool), "items": items}
    (OUT / "manifest.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    print(f"[out] {len(items)}개 -> {OUT}")


if __name__ == "__main__":
    main()
