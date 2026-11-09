"""Phase 3: CNN 전체 fine-tuning (EfficientNetV2S / ConvNeXt-T), LOGO 계열 하나.

- 입력 256x256. 학습은 real·fake 모두 무작위 품질 JPEG 재인코딩(60~95) + 약한 증강, 평가는 품질 90 고정
- 매 epoch valid AUC가 가장 좋은 가중치를 사용하고, threshold는 valid Macro F1로 정합니다.
- Phase 5 (`--aug strong`): 얼굴 crop 무작위화 + SNS 열화 + 생성기 균형 샘플링 (src/augment.py).
  결과는 outputs/exp2/<model>_aug/<holdout>/ 에 따로 저장해 Phase 3 기준선과 비교합니다.
- 모든 실행은 기존 전체 이미지 보기(metrics.csv)와 함께, 데모 서버와 같은 얼굴 crop x2.0 보기(crop_view/metrics.csv)도 평가합니다.
  `--eval-only` 는 이미 학습된 best.pt 로 crop_view 만 다시 평가합니다 (Phase 3 기준선 비교용).

실행 (.venv-exp2):
    python scripts/32_train_cnn.py --model efficientnetv2s --holdout diffusion
    python scripts/32_train_cnn.py --model convnext_tiny --holdout diffusion --aug strong
    python scripts/32_train_cnn.py --model convnext_tiny --holdout diffusion --eval-only
    python scripts/32_train_cnn.py --model convnext_tiny --holdout gan --limit 500 --epochs 1   # 동작 확인용
출력: outputs/exp2/<model>/<holdout>/ (best.pt는 git 제외)
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.augment import FaceCropDataset, balanced_weights, load_face_boxes
from src.config import DATA_DIR, EXP2_DIR, SEED, set_seed
from src.data import ManifestDataset
from src.metrics import save_run_results
from src.models import build_cnn, cnn_transform
from src.perturb import TRAIN_JPEG, EVAL_JPEG, jpeg, random_degrade
from src.pseudo import PseudoDataset, add_pseudo_rows


def loader(rows, model_name, train, perturb, batch_size, boxes=None, sampler_weights=None):
    """boxes 가 있으면 얼굴 crop 보기(FaceCropDataset), 없으면 기존 전체 이미지 보기."""
    if boxes is None:
        dataset = ManifestDataset(rows, transform=cnn_transform(model_name, train=train), perturb=perturb)
    else:
        dataset = FaceCropDataset(rows, boxes, transform=cnn_transform(model_name, train=train), train=train, perturb=perturb)
    sampler = None
    if sampler_weights is not None:
        sampler = WeightedRandomSampler(torch.as_tensor(sampler_weights, dtype=torch.double), len(rows), replacement=True)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=train and sampler is None, sampler=sampler, num_workers=10,
        pin_memory=True, drop_last=train, persistent_workers=False,
    )


@torch.no_grad()
def predict(model, data_loader):
    model.eval()
    probs = []
    for images, _ in data_loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(images.cuda(non_blocking=True))
        probs.append(torch.sigmoid(logits.float()).squeeze(1).cpu().numpy())
    return np.concatenate(probs)


def evaluate(model, args, role, out_dir, config, boxes=None):
    """전체 이미지 보기(boxes=None) 또는 얼굴 crop 보기로 valid threshold 를 정하고 테스트합니다."""
    def view(rows, perturb):
        return predict(model, loader(rows, args.model, False, perturb, 64, boxes=boxes))

    return save_run_results(
        out_dir, config, role["valid"], view(role["valid"], EVAL_JPEG),
        {
            "test_id": (role["test_id"], view(role["test_id"], EVAL_JPEG)),
            "test_id_jpeg50": (role["test_id"], view(role["test_id"], jpeg(50))),
            "test_ood": (role["test_ood"], view(role["test_ood"], EVAL_JPEG)),
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["efficientnetv2s", "convnext_tiny"])
    parser.add_argument("--holdout", required=True, choices=["gan", "diffusion", "manip", "all"],
                        help="all: 서버용 최종 모델 (scripts/23_build_final_split.py, 계열을 빼지 않음)")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--aug", default="none", choices=["none", "strong"], help="strong: Phase 5 증강")
    parser.add_argument("--pseudo", default="none", choices=["none", "vae", "sbi", "vae+sbi"],
                        help="Phase 6 의사 fake 추가 (src/pseudo.py). --aug none 과 함께 사용")
    parser.add_argument("--degrade", action="store_true",
                        help="--pseudo 와 함께: 축소·블러 증강 추가 (src/perturb.random_degrade). 결과 폴더에 _deg")
    parser.add_argument("--eval-only", action="store_true", help="학습 없이 best.pt 로 crop_view 만 평가")
    parser.add_argument("--limit", type=int, default=None, help="role·클래스별 최대 장수 (동작 확인용)")
    args = parser.parse_args()
    set_seed()

    split = pd.read_csv(DATA_DIR / "manifests" / f"logo_{args.holdout}.csv")
    if args.limit:
        split = split.sample(frac=1, random_state=SEED).groupby(["role", "y"]).head(args.limit)
    role = {r: split[split["role"] == r].reset_index(drop=True) for r in ["train", "valid", "test_id", "test_ood"]}

    if args.pseudo != "none" and args.aug != "none":
        parser.error("--pseudo 는 Phase 3 기본 증강(--aug none) 위에서만 비교합니다")
    run_name = args.model if args.aug == "none" else f"{args.model}_aug"
    if args.pseudo != "none":
        run_name = f"{args.model}_pseudo_{args.pseudo.replace('+', '_')}" + ("_deg" if args.degrade else "")
    elif args.degrade:
        parser.error("--degrade 는 --pseudo 와 함께 씁니다")
    out_dir = EXP2_DIR / run_name / args.holdout
    if args.limit:
        out_dir = EXP2_DIR / "smoke" / run_name / args.holdout
    out_dir.mkdir(parents=True, exist_ok=True)

    boxes = load_face_boxes()
    config = {"model": args.model, "holdout": args.holdout, "aug": args.aug, "pseudo": args.pseudo, "degrade": args.degrade, "epochs": args.epochs,
              "batch_size": args.batch_size, "lr": args.lr, "limit": args.limit}
    model = build_cnn(args.model).cuda()

    if args.eval_only:
        model.load_state_dict(torch.load(out_dir / "best.pt"))
        metrics = evaluate(model, args, role, out_dir / "crop_view", {**config, "view": "face_crop_x2.0"}, boxes)
        print(metrics[metrics["subset"] == "all"].round(4).to_string(index=False))
        return

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    if args.aug == "strong":
        train_loader = loader(role["train"], args.model, True, None, args.batch_size,
                              boxes=boxes, sampler_weights=balanced_weights(role["train"]))
    elif args.pseudo != "none":
        train_rows = add_pseudo_rows(role["train"], args.pseudo.split("+"), boxes, SEED)
        print(train_rows.groupby(["y", "generator"]).size().to_string(), flush=True)
        train_perturb = random_degrade() if args.degrade else TRAIN_JPEG
        dataset = PseudoDataset(train_rows, boxes, cnn_transform(args.model, train=True), perturb=train_perturb)
        train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=10,
                                  pin_memory=True, drop_last=True)
    else:
        train_loader = loader(role["train"], args.model, True, TRAIN_JPEG, args.batch_size)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=args.epochs * len(train_loader), pct_start=0.1
    )
    # 모델 선택(valid AUC)은 학습 때 쓰는 보기와 같은 보기로 합니다.
    valid_loader = loader(role["valid"], args.model, False, EVAL_JPEG, 64, boxes=boxes if args.aug == "strong" else None)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    best_auc, history = -1.0, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        start, losses = time.time(), []
        for step, (images, labels) in enumerate(train_loader):
            images, labels = images.cuda(non_blocking=True), labels.float().cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = loss_fn(model(images).squeeze(1).float(), labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())
            if step % 200 == 0:
                print(f"epoch {epoch} step {step}/{len(train_loader)} loss {np.mean(losses[-200:]):.4f}", flush=True)

        valid_auc = roc_auc_score(role["valid"]["y"], predict(model, valid_loader))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "valid_auc": valid_auc, "seconds": time.time() - start})
        print(history[-1], flush=True)
        if valid_auc > best_auc:
            best_auc = valid_auc
            torch.save(model.state_dict(), out_dir / "best.pt")
    pd.DataFrame(history).to_csv(out_dir / "training_history.csv", index=False)

    model.load_state_dict(torch.load(out_dir / "best.pt"))
    config["best_valid_auc"] = float(best_auc)
    metrics = evaluate(model, args, role, out_dir, {**config, "view": "full_image"})
    crop_metrics = evaluate(model, args, role, out_dir / "crop_view", {**config, "view": "face_crop_x2.0"}, boxes)
    columns = ["eval_set", "accuracy", "macro_f1", "roc_auc_real_positive", "fake_recall", "real_recall"]
    print("[full image]\n" + metrics[metrics["subset"] == "all"][columns].round(4).to_string(index=False))
    print("[face crop x2.0]\n" + crop_metrics[crop_metrics["subset"] == "all"][columns].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
