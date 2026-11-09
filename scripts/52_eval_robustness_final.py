"""실험 A 축소판: 서버용 최종 모델(ConvNeXt-T + VAE·SBI 의사 fake)의 SNS 열화 강건성 (재학습 없음).

Phase 1(1차 Keras 모델)과 같은 변환 목록(src/perturb.PERTURBATIONS)을 최종 모델에 적용합니다.
- 서버와 같은 순서: 이미지(256x256) -> 열화 -> [얼굴 crop x2.0] -> JPEG 90(서버 전처리) -> 모델
- threshold 는 그 run 이 valid 에서 정한 값(서버 배포값과 같음)을 고정 사용: all/crop_view/config.yaml(얼굴 crop),
  all/config.yaml(전체 이미지). 변환마다 다시 정하지 않습니다.
- 평가 세트: logo_all.csv 의 test_id(6천) + 학습에 안 쓴 추가 테스트(test_ood 역할, 2만)
- real 의 P(real) 이 threshold 바로 위(기준~0.15)에 몰린 비율도 기록: 조금만 흔들려도 FAKE 로 넘어갈 수 있는 real

실행 (.venv-exp2): python scripts/52_eval_robustness_final.py [--run convnext_tiny_pseudo_vae_sbi_deg] [--limit 200]
      지표만 다시 계산: ... --from-predictions (저장된 predictions.csv 사용, GPU 불필요)
출력: outputs/exp2/robustness_final/{summary.csv, by_family.csv, predictions.csv}
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.augment import EVAL_CROP_SCALE, face_crop, load_face_boxes
from src.config import DATA_DIR, EXP2_DIR, SEED
from src.data import load_image
from src.metrics import compute_metrics
from src.models import build_cnn, cnn_transform
from src.perturb import EVAL_JPEG, PERTURBATIONS

DEFAULT_RUN = "convnext_tiny_pseudo_vae_sbi"
NEAR_MARGIN = 0.10


def load_thresholds(run: str) -> dict:
    """run 의 valid threshold (32_train_cnn.py 가 config.yaml 에 기록)."""
    import yaml

    base = EXP2_DIR / run / "all"
    read = lambda path: float(yaml.safe_load(open(path, encoding="utf-8"))["threshold"])
    return {"face_crop": read(base / "crop_view" / "config.yaml"), "full_image": read(base / "config.yaml")}


def summarize(predictions: pd.DataFrame, thresholds: dict):
    summary, by_family = [], []
    for (view, name), group in predictions.groupby(["view", "perturbation"], sort=False):
        threshold = thresholds[view]
        for eval_set, part in group.groupby("eval_set"):
            real_prob = part.loc[part["y"] == 1, "prob_real"]
            summary.append({
                "view": view, "perturbation": name, "eval_set": eval_set, "threshold": threshold,
                **compute_metrics(part["y"], part["prob_real"].to_numpy(), threshold),
                "real_near_threshold": float(((real_prob >= threshold) & (real_prob < threshold + NEAR_MARGIN)).mean()),
            })
            for family, fake in part[part["y"] == 0].groupby("family"):
                by_family.append({"view": view, "perturbation": name, "eval_set": eval_set, "family": family,
                                  "n": len(fake), "fake_recall": float((fake["prob_real"] < threshold).mean())})
        row = [s for s in summary if s["view"] == view and s["perturbation"] == name]
        print(view, name, " | ".join(f"{s['eval_set']} auc={s['roc_auc_real_positive']:.4f} acc={s['accuracy']:.4f} "
                                     f"fake_rec={s['fake_recall']:.4f} real_rec={s['real_recall']:.4f}" for s in row), flush=True)
    return pd.DataFrame(summary), pd.DataFrame(by_family)


class RobustnessDataset:
    def __init__(self, rows, boxes, view, perturb, transform):
        self.paths, self.labels = rows["path"].tolist(), rows["y"].tolist()
        self.boxes, self.view, self.perturb, self.transform = boxes, view, perturb, transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        image = load_image(path)
        if self.perturb is not None:
            image = self.perturb(image)
        if self.view == "face_crop":
            image = face_crop(image, self.boxes.get(path), EVAL_CROP_SCALE)
        return self.transform(EVAL_JPEG(image)), self.labels[index]


@torch.no_grad()
def predict(model, dataset):
    probs = []
    for images, _ in DataLoader(dataset, batch_size=128, num_workers=10, pin_memory=True):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(images.cuda(non_blocking=True))
        probs.append(torch.sigmoid(logits.float()).squeeze(1).cpu().numpy())
    return np.concatenate(probs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--run", default=DEFAULT_RUN, help="outputs/exp2/<run>/all/best.pt")
    parser.add_argument("--from-predictions", action="store_true", help="저장된 predictions.csv 로 지표만 다시 계산")
    args = parser.parse_args()
    thresholds = load_thresholds(args.run)
    print("thresholds", thresholds)
    out_dir = EXP2_DIR / "robustness_final" / ("" if args.run == DEFAULT_RUN else args.run)
    out_dir = out_dir / "smoke" if args.limit else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.from_predictions:
        summary, by_family = summarize(pd.read_csv(out_dir / "predictions.csv"), thresholds)
        summary.to_csv(out_dir / "summary.csv", index=False)
        by_family.to_csv(out_dir / "by_family.csv", index=False)
        print(f"saved {out_dir}")
        return

    split = pd.read_csv(DATA_DIR / "manifests" / "logo_all.csv")
    rows = split[split["role"].isin(["test_id", "test_ood"])].copy()
    rows["eval_set"] = rows["role"].map({"test_id": "test_id", "test_ood": "held_out"})
    if args.limit:
        rows = rows.sample(frac=1, random_state=SEED).groupby(["eval_set", "y"]).head(args.limit)
    rows = rows.reset_index(drop=True)

    boxes = load_face_boxes()
    model = build_cnn("convnext_tiny")
    model.load_state_dict(torch.load(EXP2_DIR / args.run / "all" / "best.pt"))
    model = model.cuda().eval()
    transform = cnn_transform("convnext_tiny", train=False)

    predictions = []
    for view in thresholds:
        for name, perturb in PERTURBATIONS.items():
            prob = predict(model, RobustnessDataset(rows, boxes, view, perturb, transform))
            predictions.append(rows[["eval_set", "path", "y", "family", "generator"]].assign(view=view, perturbation=name, prob_real=prob))
            print(f"predicted {view} {name}", flush=True)
    predictions = pd.concat(predictions)
    summary, by_family = summarize(predictions, thresholds)

    summary.to_csv(out_dir / "summary.csv", index=False)
    by_family.to_csv(out_dir / "by_family.csv", index=False)
    predictions.to_csv(out_dir / "predictions.csv", index=False)
    print(f"saved {out_dir}")


if __name__ == "__main__":
    main()
