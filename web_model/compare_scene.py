"""얼굴이 작게 나온 일반 사진 상황 비교 세트.

compare_models.py 가 만든 깨끗한 세트(두 모델 모두 학습에 안 쓴 이미지)의 none 원본을
넓은 배경(900x700) 안에 얼굴 크기 200px 로 붙인다. SNS 단체·전신 사진처럼 얼굴이 사진 일부인 경우.

사용 (compare_models.py 실행 후, 저장소 루트에서):
    server\\.venv\\Scripts\\python.exe web_model\\compare_scene.py --n 400
출력: web_model/reference/compare_scene/{images/, manifest.json}
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))
import app as server_app  # noqa: E402
from face import FaceDetector  # noqa: E402
from model import FakeFaceModel  # noqa: E402

BASE = Path(__file__).resolve().parent / "reference"
SRC = BASE / "compare"
OUT = BASE / "compare_scene"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=400)
    args = parser.parse_args()
    rng = random.Random(42)
    meta = json.loads((SRC / "manifest.json").read_text(encoding="utf-8"))
    originals = [it for it in meta["items"] if it["condition"] == "none"]
    reals = [it for it in originals if it["label"] == 1]
    fakes = [it for it in originals if it["label"] == 0]
    half = args.n // 2
    chosen = rng.sample(reals, min(half, len(reals))) + rng.sample(fakes, min(args.n - min(half, len(reals)), len(fakes)))

    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "images").mkdir(parents=True)
    server_app.state["model"] = FakeFaceModel()
    server_app.state["detector"] = FaceDetector()

    items = []
    for index, item in enumerate(chosen):
        face = Image.open(SRC / "images" / item["file"]).convert("RGB").resize((200, 200), Image.BILINEAR)
        color = tuple(rng.randint(90, 230) for _ in range(3))
        scene = Image.new("RGB", (900, 700), color)
        scene.paste(face, (rng.randint(0, 700), rng.randint(0, 500)))
        name = f"{index:04d}_scene.jpg"
        scene.save(OUT / "images" / name, format="JPEG", quality=92)
        decoded = server_app._open_image((OUT / "images" / name).read_bytes())
        result = server_app.predict_image(decoded, "face")
        face_prob = result["faces"][0]["prob_real"] if result["status"] == "ok" else None
        full = server_app.predict_image(decoded, "full")["faces"][0]["prob_real"]
        items.append({**item, "file": name, "condition": "scene_small_face", "final_face": face_prob, "final_full": full})
        if (index + 1) % 100 == 0:
            print(f"  {index + 1}/{len(chosen)}", flush=True)
    out = {"final_threshold": meta["final_threshold"], "final_threshold_full": meta.get("final_threshold_full", 0.5), "count": len(items), "pool": meta["pool"], "items": items}
    (OUT / "manifest.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"[out] {len(items)}개 -> {OUT}")


if __name__ == "__main__":
    main()
