"""PyTorch 서버 모델(server/model.py FakeFaceModel) 과 ONNX 모델의 점수·Grad-CAM 이 같은지 확인한다.

같은 전처리(256 리사이즈 -> JPEG 90 -> 정규화) 를 거친 같은 텐서를 넣어 모델 자체의 동등성만 본다.
브라우저 전처리(얼굴 검출·JPEG 인코더 차이) 검증은 별도 단계.

사용 (저장소 루트에서):
    server\\.venv\\Scripts\\python.exe web_model\\verify_onnx.py --per-folder 150
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))
from model import IMAGE_SIZE, FakeFaceModel  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "out"
DATA = Path(r"\\wsl.localhost\ubuntu\home\hyojin\ai_image_detection\data")
FOLDERS = {
    "test/real": 1, "test/fake": 0,
    "test_by_source/flux_sdxl/real": 1, "test_by_source/flux_sdxl/fake": 0,
    "test_by_source/gravex/real": 1, "test_by_source/gravex/fake": 0,
    "test_by_source/stylegan/real": 1, "test_by_source/stylegan/fake": 0,
}
EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def sample_files(per_folder: int, seed: int) -> list[tuple[Path, int, str]]:
    rng = random.Random(seed)
    items = []
    for folder, label in FOLDERS.items():
        path = DATA / folder
        if not path.exists():
            print(f"[skip] {folder} 없음")
            continue
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in EXTS)
        for file in rng.sample(files, min(per_folder, len(files))):
            items.append((file, label, folder))
    return items


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def cam_from_map(cam_map: np.ndarray, target: str) -> np.ndarray:
    """서버 FakeFaceModel.gradcam 과 같은 후처리: ReLU -> /max -> uint8 -> bicubic 256."""
    cam = np.maximum(-cam_map if target == "fake" else cam_map, 0)
    if cam.max() <= 0:
        return np.zeros(IMAGE_SIZE[::-1], dtype=np.float32)
    cam = cam / cam.max()
    image = Image.fromarray((cam * 255).astype(np.uint8)).resize(IMAGE_SIZE, Image.BICUBIC)
    return np.asarray(image, dtype=np.float32) / 255.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-folder", type=int, default=150)
    parser.add_argument("--cam-samples", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch_model = FakeFaceModel()
    threshold = torch_model.threshold_for("full")
    sessions = {
        name: ort.InferenceSession(str(OUT_DIR / file), providers=["CPUExecutionProvider"])
        for name, file in (("fp32", "efficientnetv2s_fake_face.fp32.onnx"), ("fp16w", "efficientnetv2s_fake_face.onnx"))
    }

    items = sample_files(args.per_folder, args.seed)
    print(f"[data] {len(items)}장")
    rows = []
    cam_corr = {name: [] for name in sessions}
    for index, (file, label, folder) in enumerate(items):
        image = Image.open(file).convert("RGB").resize(IMAGE_SIZE, Image.BILINEAR)
        batch = np.asarray(image, dtype=np.float32)[None]
        model_input = torch_model.model_inputs(batch)[0]
        # ONNX 입력은 256 crop 그대로(확대·TTA 는 그래프 안). 서버 _tensor 는 input_size 로 확대하므로 직접 정규화한다.
        array = np.asarray(model_input, dtype=np.float32)[None] / 255.0
        tensor = ((array - np.array(torch_model.config["mean"], dtype=np.float32)) / np.array(torch_model.config["std"], dtype=np.float32))
        tensor = tensor.transpose(0, 3, 1, 2).astype(np.float32)
        prob_torch = float(torch_model.predict_prob_real(batch)[0])
        row = {"file": file.name, "folder": folder, "label": label, "torch": prob_torch}
        maps = {}
        for name, session in sessions.items():
            logit, cam_map = session.run(None, {"input": tensor})
            row[name] = float(1 / (1 + np.exp(-logit[0, 0])))
            maps[name] = cam_map[0]
        rows.append(row)
        if index < args.cam_samples or index % max(1, len(items) // args.cam_samples) == 0:
            target = torch_model.label_for(prob_torch, "full")
            ref = torch_model.gradcam(model_input, target).ravel()
            for name, cam_map in maps.items():
                got = cam_from_map(cam_map, target).ravel()
                cam_corr[name].append(float(np.corrcoef(ref, got)[0, 1]) if ref.std() > 0 and got.std() > 0 else 1.0)
        if (index + 1) % 100 == 0:
            print(f"  {index + 1}/{len(items)}")

    labels = np.array([r["label"] for r in rows])
    report = {"n": len(rows), "threshold_real_full": threshold, "per_folder": args.per_folder, "models": {}}
    ref = np.array([r["torch"] for r in rows])
    report["torch_auc"] = auc(labels, ref)
    for name in sessions:
        got = np.array([r[name] for r in rows])
        diff = np.abs(got - ref)
        report["models"][name] = {
            "max_abs_prob_diff": float(diff.max()),
            "mean_abs_prob_diff": float(diff.mean()),
            "label_disagreements": int(((ref >= threshold) != (got >= threshold)).sum()),
            "auc": auc(labels, got),
            "gradcam_corr_min": float(np.min(cam_corr[name])),
            "gradcam_corr_mean": float(np.mean(cam_corr[name])),
            "gradcam_samples": len(cam_corr[name]),
        }
    (OUT_DIR / "verify_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with open(OUT_DIR / "verify_scores.csv", "w", encoding="utf-8") as file:
        file.write("file,folder,label,torch,fp32,fp16w\n")
        for r in rows:
            file.write(f"{r['file']},{r['folder']},{r['label']},{r['torch']:.6f},{r['fp32']:.6f},{r['fp16w']:.6f}\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
