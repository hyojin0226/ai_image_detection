"""Phase 3: CLIP ViT-L/14 이미지 특징 추출 (백본 고정, 한 번만 추출).

LOGO 3종 CSV에 나오는 이미지의 특징을 JPEG 정규화 방식별로 추출해 저장합니다.
- train_jpeg: train/valid에 쓰이는 이미지. 무작위 품질(60~95) 재인코딩 (이미지마다 한 번 뽑음)
- eval_jpeg90: 모든 이미지. 품질 90 고정 재인코딩 (test_id / test_ood 평가용)
- eval_jpeg50: test_id 이미지. 품질 50 (강건성 비교용)

실행 (.venv-exp2): python scripts/30_extract_clip_features.py
출력: outputs/exp2/clip/features/{mode}.npz  (paths, features[float16])
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DATA_DIR, EXP2_DIR, set_seed
from src.data import ManifestDataset
from src.models import build_clip, clip_features
from src.perturb import TRAIN_JPEG, EVAL_JPEG, jpeg

OUT_DIR = EXP2_DIR / "clip" / "features"
MODES = {
    "train_jpeg": (["train", "valid"], TRAIN_JPEG),
    "eval_jpeg90": (["train", "valid", "test_id", "test_ood"], EVAL_JPEG),
    "eval_jpeg50": (["test_id"], jpeg(50)),
}


def main():
    set_seed()
    splits = pd.concat(pd.read_csv(DATA_DIR / "manifests" / f"logo_{h}.csv") for h in ["gan", "diffusion", "manip"])
    model, preprocess = build_clip("cuda")
    model = model.half()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for mode, (roles, perturb) in MODES.items():
        out_path = OUT_DIR / f"{mode}.npz"
        if out_path.exists():
            print(f"[{mode}] 이미 있음, 건너뜀")
            continue
        rows = splits[splits["role"].isin(roles)].drop_duplicates("path").reset_index(drop=True)
        rows["y"] = rows["y"].astype(int)
        loader = DataLoader(
            ManifestDataset(rows, transform=preprocess, perturb=perturb),
            batch_size=256, num_workers=10, pin_memory=True,
        )
        features = []
        for images, _ in tqdm(loader, desc=mode, mininterval=30):
            features.append(clip_features(model, images.cuda(non_blocking=True).half()).cpu().numpy())
        np.savez(out_path, paths=rows["path"].to_numpy(), features=np.concatenate(features).astype(np.float16))
        print(f"[{mode}] {len(rows)}장 저장: {out_path}", flush=True)


if __name__ == "__main__":
    with torch.no_grad():
        main()
