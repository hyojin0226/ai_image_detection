"""구도(얼굴 크기·위치) 지름길 검증 실험.

같은 test 이미지를
  (1) 전체 이미지를 256x256 으로 (학습·노트북 방식)
  (2) 서버와 같은 얼굴 검출 + 정사각형 crop (배율 여러 개)
으로 넣어 source 별 정확도를 비교하고,
(3) 얼굴 bbox 크기·위치 숫자만으로 real/fake 를 맞힐 수 있는지(로지스틱 회귀) 확인한다.

실행 (저장소 루트에서):
  python experiments/framing_shortcut.py --data //wsl.localhost/ubuntu/home/hyojin/ai_image_detection/data --per-class 400
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR / "server"))

import face as F  # noqa: E402
from model import IMAGE_SIZE, FakeFaceModel  # noqa: E402

CROP_SCALES = [1.6, 2.0, 2.5, 3.0]
SOURCES = ["stylegan", "gravex", "flux_sdxl"]


def list_images(directory: Path) -> list[Path]:
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="data 폴더 (test_by_source/<source>/<fake|real>)")
    parser.add_argument("--per-class", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=str(REPO_DIR / "results" / "framing_shortcut"))
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    model = FakeFaceModel()
    detector = F.FaceDetector()
    threshold = model.threshold
    print(f"model={model.model_path} threshold_real={threshold}", flush=True)

    rows = []
    started = time.perf_counter()
    for source in SOURCES:
        for label_name, label in [("fake", 0), ("real", 1)]:
            files = list_images(Path(args.data) / "test_by_source" / source / label_name)
            files = rng.sample(files, min(args.per_class, len(files)))
            for start in range(0, len(files), 32):
                chunk = files[start:start + 32]
                images = [Image.open(p).convert("RGB") for p in chunk]
                full_batch = np.stack([np.asarray(im.resize(IMAGE_SIZE, Image.BILINEAR), np.float32) for im in images])
                full_probs = model.predict_prob_real(full_batch)

                faces = [detector.detect(np.asarray(im)) for im in images]
                mains = [max(fs, key=lambda f: f.w * f.h) if fs else None for fs in faces]
                crop_probs = {}
                for scale in CROP_SCALES:
                    F.CROP_SCALE = scale
                    batch = [
                        F.crop_and_resize(im, F.crop_box_for(m, *im.size), IMAGE_SIZE)
                        for im, m in zip(images, mains) if m is not None
                    ]
                    probs = iter(model.predict_prob_real(np.stack(batch)) if batch else [])
                    crop_probs[scale] = [next(probs) if m is not None else np.nan for m in mains]

                for i, (path, im, m) in enumerate(zip(chunk, images, mains)):
                    w, h = im.size
                    row = {
                        "source": source, "label": label, "file": path.name, "width": w, "height": h,
                        "n_faces": len(faces[i]), "p_real_full": float(full_probs[i]),
                    }
                    if m is not None:
                        row.update({
                            "face_size": max(m.w, m.h) / min(w, h),  # 얼굴 한 변 / 이미지 짧은 변
                            "face_cx": (m.x + m.w / 2) / w,
                            "face_cy": (m.y + m.h / 2) / h,
                            "det_score": m.score,
                        })
                    for scale in CROP_SCALES:
                        row[f"p_real_crop{scale}"] = float(crop_probs[scale][i])
                    rows.append(row)
            print(f"[{time.perf_counter() - started:6.0f}s] {source}/{label_name} {len(files)}장 완료", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "predictions.csv", index=False)

    # ---- 1) 정확도 비교 (얼굴이 검출된 이미지만, 같은 이미지 집합에서 비교) ----
    cols = ["p_real_full"] + [f"p_real_crop{s}" for s in CROP_SCALES]
    detected = df.dropna(subset=["face_size"])
    summary = []
    for source, group in [*detected.groupby("source"), ("ALL", detected)]:
        entry = {"source": source, "n": len(group), "no_face_rate": 1 - len(group) / (df["source"].eq(source).sum() if source != "ALL" else len(df))}
        for col in cols:
            pred = (group[col] >= threshold).astype(int)
            name = col.replace("p_real_", "")
            entry[f"acc_{name}"] = (pred == group["label"]).mean()
            entry[f"fake_acc_{name}"] = (pred[group["label"] == 0] == 0).mean()
            entry[f"real_acc_{name}"] = (pred[group["label"] == 1] == 1).mean()
        summary.append(entry)
    summary = pd.DataFrame(summary)
    summary.to_csv(out_dir / "accuracy_summary.csv", index=False)

    # ---- 2) 얼굴 크기 분포 ----
    size_stats = detected.groupby(["source", "label"])["face_size"].describe(percentiles=[0.25, 0.5, 0.75])
    size_stats.to_csv(out_dir / "face_size_stats.csv")

    # ---- 3) 구도 숫자만으로 real/fake 맞히기 ----
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    feats = ["face_size", "face_cx", "face_cy"]
    framing = []
    for source, group in [*detected.groupby("source"), ("ALL", detected)]:
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        acc = cross_val_score(clf, group[feats], group["label"], cv=5, scoring="accuracy").mean()
        auc = cross_val_score(clf, group[feats], group["label"], cv=5, scoring="roc_auc").mean()
        framing.append({"source": source, "n": len(group), "framing_only_acc": acc, "framing_only_auc": auc})
    framing = pd.DataFrame(framing)
    framing.to_csv(out_dir / "framing_only_classifier.csv", index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)
    print("\n=== 정확도: full vs face crop (얼굴 검출된 이미지) ===")
    print(summary[["source", "n", "no_face_rate"] + [c for c in summary.columns if c.startswith("acc_")]].round(3).to_string(index=False))
    print("\n=== fake / real 별 정확도 ===")
    print(summary[["source"] + [c for c in summary.columns if c.startswith(("fake_acc_", "real_acc_"))]].round(3).to_string(index=False))
    print("\n=== 얼굴 크기(얼굴 한 변 / 이미지 짧은 변) ===")
    print(size_stats[["count", "25%", "50%", "75%"]].round(3).to_string())
    print("\n=== 구도 숫자(얼굴 크기·중심)만으로 분류 (5-fold) ===")
    print(framing.round(3).to_string(index=False))
    detector.close()


if __name__ == "__main__":
    main()
