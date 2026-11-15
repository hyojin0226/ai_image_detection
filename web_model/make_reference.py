"""브라우저 비교용 서버(PyTorch) 기준 결과를 만든다.

server/app.py 의 predict_image 를 그대로 호출해 face / full 두 모드 결과를 저장한다.
단계별 비교를 위해 full 모드의 256 리사이즈 직후(JPEG 전)와 JPEG 90 이후 이미지도 PNG 로 남긴다.

사용 (저장소 루트에서):
    server\\.venv\\Scripts\\python.exe web_model\\make_reference.py --per-folder 12
출력: web_model/reference/{images/, stages/, reference.json}  (커밋 안 함)
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))
import app as server_app  # noqa: E402
from face import FaceDetector  # noqa: E402
from model import IMAGE_SIZE, FakeFaceModel  # noqa: E402

OUT = Path(__file__).resolve().parent / "reference"
DATA = Path(r"\\wsl.localhost\ubuntu\home\hyojin\ai_image_detection\data")
FOLDERS = [
    "test/real", "test/fake",
    "test_by_source/flux_sdxl/real", "test_by_source/flux_sdxl/fake",
    "test_by_source/gravex/real", "test_by_source/gravex/fake",
    "test_by_source/stylegan/real", "test_by_source/stylegan/fake",
]
EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def pick(per_folder: int, seed: int) -> list[Path]:
    rng = random.Random(seed)
    chosen = []
    for folder in FOLDERS:
        files = sorted(p for p in (DATA / folder).iterdir() if p.suffix.lower() in EXTS)
        chosen += rng.sample(files, min(per_folder, len(files)))
    return chosen


def composites(files: list[Path], seed: int) -> list[tuple[str, Image.Image]]:
    """타일 재검출(짧은 변 400 이상)과 2048 제한 리사이즈가 실제로 도는 입력."""
    rng = random.Random(seed + 1)
    out = []
    for index in range(4):
        grid = rng.sample(files, 9)
        canvas = Image.new("RGB", (768, 768), (235, 235, 235))
        for k, file in enumerate(grid):
            tile = Image.open(file).convert("RGB").resize((256, 256), Image.BILINEAR)
            canvas.paste(tile, ((k % 3) * 256, (k // 3) * 256))
        out.append((f"grid3x3_{index}.png", canvas))
    # 작은 얼굴 하나가 넓은 배경 안에 있는 경우 (타일에서만 잡히는 얼굴)
    for index in range(2):
        canvas = Image.new("RGB", (1400, 1000), (200 + index * 20, 210, 220))
        face = Image.open(rng.choice(files)).convert("RGB").resize((180, 180), Image.BILINEAR)
        canvas.paste(face, (900 - index * 600, 600))
        out.append((f"small_face_{index}.png", canvas))
    # 긴 변 2048 초과 -> 서버와 같은 축소 경로
    big = Image.open(rng.choice(files)).convert("RGB").resize((2600, 2600), Image.BICUBIC)
    out.append(("large_2600.png", big))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-folder", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    # reference/ 아래 compare/, compare_scene/, jei_model/ 은 다른 스크립트 몫이라 지우지 않는다
    for folder in ("images", "stages"):
        if (OUT / folder).exists():
            shutil.rmtree(OUT / folder)
    (OUT / "images").mkdir(parents=True)
    (OUT / "stages").mkdir()

    server_app.state["model"] = FakeFaceModel()
    server_app.state["detector"] = FaceDetector()
    model: FakeFaceModel = server_app.state["model"]

    entries = []
    files = pick(args.per_folder, args.seed)
    for file in files:
        name = f"{file.parent.parent.name}_{file.parent.name}_{file.name}".replace(" ", "_")
        shutil.copyfile(file, OUT / "images" / name)
        entries.append({"name": name, "source": str(file.relative_to(DATA)).replace("\\", "/"),
                        "label": 1 if file.parent.name == "real" else 0})
    for name, image in composites(files, args.seed):
        image.save(OUT / "images" / name)
        entries.append({"name": name, "source": "composite", "label": None})

    for entry in entries:
        data = (OUT / "images" / entry["name"]).read_bytes()
        image = server_app._open_image(data)
        entry["image_size"] = list(image.size)
        for mode in ("face", "full"):
            result = server_app.predict_image(image, mode)
            # 브라우저와 비교할 때 반올림 전 값이 필요하므로 다시 계산
            entry[mode] = result
        # 단계별 비교 (full 모드): 256 리사이즈 직후 / JPEG 90 이후
        work = image
        if max(work.size) > server_app.MAX_IMAGE_SIDE:
            s = server_app.MAX_IMAGE_SIDE / max(work.size)
            work = work.resize((round(work.size[0] * s), round(work.size[1] * s)), Image.BILINEAR)
        resized = work.resize(IMAGE_SIZE, Image.BILINEAR)
        jpeg = model.model_inputs(np.asarray(resized, dtype=np.float32)[None])[0]
        stem = Path(entry["name"]).stem
        resized.save(OUT / "stages" / f"{stem}.resized.png")
        jpeg.save(OUT / "stages" / f"{stem}.jpeg90.png")
        entry["full_prob_real_raw"] = float(model.predict_prob_real(np.asarray(resized, dtype=np.float32)[None])[0])

    meta = {"thresholds": model.thresholds, "count": len(entries), "entries": entries}
    (OUT / "reference.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    faces = sum(len(e["face"]["faces"]) for e in entries)
    print(f"[ref] 이미지 {len(entries)}장, face 모드 얼굴 {faces}개 -> {OUT}")


if __name__ == "__main__":
    main()
