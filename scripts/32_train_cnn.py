"""Phase 3: CNN 전체 fine-tuning (EfficientNetV2S / ConvNeXt-T), LOGO 계열 하나.

- 입력 256x256. 학습은 real·fake 모두 무작위 품질 JPEG 재인코딩(60~95) + 약한 증강, 평가는 품질 90 고정
- 매 epoch valid AUC가 가장 좋은 가중치를 사용하고, threshold는 valid Macro F1로 정합니다.

실행 (.venv-exp2):
    python scripts/32_train_cnn.py --model efficientnetv2s --holdout diffusion
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
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DATA_DIR, EXP2_DIR, SEED, set_seed
from src.data import ManifestDataset
from src.metrics import save_run_results
from src.models import build_cnn, cnn_transform
from src.perturb import TRAIN_JPEG, EVAL_JPEG, jpeg


def loader(rows, model_name, train, perturb, batch_size):
    dataset = ManifestDataset(rows, transform=cnn_transform(model_name, train=train), perturb=perturb)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=train, num_workers=10,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["efficientnetv2s", "convnext_tiny"])
    parser.add_argument("--holdout", required=True, choices=["gan", "diffusion", "manip"])
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--limit", type=int, default=None, help="role·클래스별 최대 장수 (동작 확인용)")
    args = parser.parse_args()
    set_seed()

    split = pd.read_csv(DATA_DIR / "manifests" / f"logo_{args.holdout}.csv")
    if args.limit:
        split = split.sample(frac=1, random_state=SEED).groupby(["role", "y"]).head(args.limit)
    role = {r: split[split["role"] == r].reset_index(drop=True) for r in ["train", "valid", "test_id", "test_ood"]}

    out_dir = EXP2_DIR / args.model / args.holdout
    if args.limit:
        out_dir = EXP2_DIR / "smoke" / args.model / args.holdout
    out_dir.mkdir(parents=True, exist_ok=True)

    model = build_cnn(args.model).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    train_loader = loader(role["train"], args.model, True, TRAIN_JPEG, args.batch_size)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=args.epochs * len(train_loader), pct_start=0.1
    )
    valid_loader = loader(role["valid"], args.model, False, EVAL_JPEG, 64)
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
    metrics = save_run_results(
        out_dir,
        {"model": args.model, "holdout": args.holdout, "epochs": args.epochs, "batch_size": args.batch_size,
         "lr": args.lr, "best_valid_auc": float(best_auc), "limit": args.limit},
        role["valid"], predict(model, valid_loader),
        {
            "test_id": (role["test_id"], predict(model, loader(role["test_id"], args.model, False, EVAL_JPEG, 64))),
            "test_id_jpeg50": (role["test_id"], predict(model, loader(role["test_id"], args.model, False, jpeg(50), 64))),
            "test_ood": (role["test_ood"], predict(model, loader(role["test_ood"], args.model, False, EVAL_JPEG, 64))),
        },
    )
    view = metrics[metrics["subset"] == "all"][["eval_set", "accuracy", "macro_f1", "roc_auc_real_positive", "fake_recall", "real_recall"]]
    print(view.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
