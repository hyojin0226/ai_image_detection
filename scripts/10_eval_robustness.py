"""Phase 1: 1차 EfficientNetV2S 모델의 강건성 평가 (재학습 없음).

현재 data/test 폴더(1차 ID Test와 같은 26,876장)에 JPEG 재압축·리사이즈·블러를 적용하고
1차 threshold(valid에서 정한 0.24)를 그대로 써서 성능 변화를 측정합니다.

- 변환은 모델 입력 크기(256x256)로 맞춘 뒤 적용합니다. 원본 해상도가 소스마다 달라서
  (예: FLUX/SDXL 1024px) 원본에 적용하면 같은 강도라도 실제 효과가 소스마다 달라지기 때문입니다.
- 소스별 fake/real 수가 다르므로 소스별 비교는 balanced_accuracy와 AUC를 기준으로 봅니다.

실행 (WSL, 저장소 루트):
    source scripts/tf_gpu_env.sh
    python scripts/10_eval_robustness.py            # 전체
    python scripts/10_eval_robustness.py --limit 200  # 동작 확인용
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import BASELINE_MODEL_PATH, BASELINE_CONFIG_PATH, EXP2_DIR, SEED
from src.data import scan_data_dir, load_image
from src.metrics import compute_metrics
from src.perturb import PERTURBATIONS

OUT_DIR = EXP2_DIR / "robustness"


def load_batch(paths, perturb, workers):
    def load(path):
        image = load_image(path)
        if perturb is not None:
            image = perturb(image)
        return np.asarray(image, dtype=np.float32)

    with ThreadPoolExecutor(workers) as pool:
        return np.stack(list(pool.map(load, paths)))


def predict(model, paths, perturb, batch_size, workers):
    probs = []
    for start in range(0, len(paths), batch_size):
        # EfficientNetV2S 내부 전처리를 사용하므로 0~255 그대로 입력 (/255 하지 않음)
        batch = load_batch(paths[start:start + batch_size], perturb, workers)
        probs.append(model.predict_on_batch(batch).reshape(-1))
        if (start // batch_size) % 50 == 0:
            print(f"  {start + len(batch)} / {len(paths)}", flush=True)
    return np.concatenate(probs)


def image_size_table(rows):
    """소스·클래스별 원본 해상도 분포 (해상도 차이가 지름길 단서가 될 수 있어 함께 기록)."""
    def size(path):
        with Image.open(path) as image:
            return f"{image.width}x{image.height}"

    with ThreadPoolExecutor(16) as pool:
        sizes = list(pool.map(size, rows["path"]))
    return (
        rows.assign(size=sizes)
        .groupby(["source", "label", "size"]).size()
        .rename("images").reset_index()
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="소스·클래스별 최대 이미지 수 (동작 확인용)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--perturbations", nargs="+", default=list(PERTURBATIONS))
    args = parser.parse_args()

    import tensorflow as tf

    rows = scan_data_dir()
    rows = rows[rows["split"] == "test"].reset_index(drop=True)
    if args.limit:
        rows = (
            rows.sample(frac=1, random_state=SEED)
            .groupby(["source", "label"]).head(args.limit)
            .reset_index(drop=True)
        )

    out_dir = OUT_DIR if not args.limit else OUT_DIR / f"smoke_limit{args.limit}"
    (out_dir / "predictions").mkdir(parents=True, exist_ok=True)

    threshold = json.loads(BASELINE_CONFIG_PATH.read_text())["decision_threshold_real"]
    model = tf.keras.models.load_model(BASELINE_MODEL_PATH)
    print("GPU:", tf.config.list_physical_devices("GPU"))
    print(f"test images: {len(rows)} | threshold: {threshold:.2f}")

    image_size_table(rows).to_csv(out_dir / "image_sizes.csv", index=False)

    summary = []
    for name in args.perturbations:
        print(f"[{name}]", flush=True)
        prob_real = predict(model, rows["path"].tolist(), PERTURBATIONS[name], args.batch_size, args.workers)
        rows.assign(prob_real=prob_real)[["path", "source", "label", "y", "prob_real"]].to_csv(
            out_dir / "predictions" / f"{name}.csv", index=False
        )

        subsets = {"all": rows.index} | {s: rows.index[rows["source"] == s] for s in sorted(rows["source"].unique())}
        for subset, index in subsets.items():
            metrics = compute_metrics(rows.loc[index, "y"], prob_real[index], threshold)
            summary.append({"perturbation": name, "subset": subset, **metrics})

        pd.DataFrame(summary).to_csv(out_dir / "robustness_summary.csv", index=False)

    summary = pd.DataFrame(summary)
    baseline = summary[summary["perturbation"] == "none"].set_index("subset")
    if not baseline.empty:
        for column in ["balanced_accuracy", "roc_auc_real_positive", "fake_recall"]:
            summary[f"delta_{column}"] = summary[column] - summary["subset"].map(baseline[column])
    summary.to_csv(out_dir / "robustness_summary.csv", index=False)

    view = summary[summary["subset"] == "all"][["perturbation", "accuracy", "balanced_accuracy", "roc_auc_real_positive", "fake_recall", "real_recall"]]
    print(view.to_string(index=False))
    print("저장:", out_dir)


if __name__ == "__main__":
    main()
