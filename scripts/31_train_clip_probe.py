"""Phase 3: CLIP ViT-L/14 특징 + Logistic Regression (UnivFD 방식), LOGO 3종.

30_extract_clip_features.py로 추출한 특징을 사용합니다.
- train: 무작위 품질 JPEG 특징 / valid·test: 품질 90 고정 특징 / test_id_jpeg50: 품질 50
- 정규화 강도 C는 valid AUC로 고르고, threshold는 valid Macro F1로 정합니다.

실행 (.venv-exp2): python scripts/31_train_clip_probe.py
출력: outputs/exp2/clip_linear/<holdout>/
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DATA_DIR, EXP2_DIR, SEED
from src.metrics import save_run_results

FEATURE_DIR = EXP2_DIR / "clip" / "features"
OUT_DIR = EXP2_DIR / "clip_linear"
C_GRID = [0.01, 0.1, 1.0, 10.0]


def load_features(mode):
    data = np.load(FEATURE_DIR / f"{mode}.npz", allow_pickle=True)
    return dict(zip(data["paths"], data["features"].astype(np.float32)))


def stack(rows, features):
    return np.stack([features[p] for p in rows["path"]])


def main():
    train_features = load_features("train_jpeg")
    eval_features = load_features("eval_jpeg90")
    jpeg50_features = load_features("eval_jpeg50")

    for holdout in ["gan", "diffusion", "manip"]:
        split = pd.read_csv(DATA_DIR / "manifests" / f"logo_{holdout}.csv")
        role = {r: split[split["role"] == r].reset_index(drop=True) for r in ["train", "valid", "test_id", "test_ood"]}
        x_train, y_train = stack(role["train"], train_features), role["train"]["y"]
        x_valid = stack(role["valid"], eval_features)

        best = None
        for c in C_GRID:
            model = LogisticRegression(C=c, max_iter=2000, random_state=SEED).fit(x_train, y_train)
            auc = roc_auc_score(role["valid"]["y"], model.predict_proba(x_valid)[:, 1])
            print(f"[{holdout}] C={c}: valid AUC {auc:.4f}", flush=True)
            if best is None or auc > best[1]:
                best = (model, auc, c)
        model, _, c = best

        def prob(rows, features):
            return model.predict_proba(stack(rows, features))[:, 1]

        metrics = save_run_results(
            OUT_DIR / holdout,
            {"model": "clip_vit_l14_linear", "holdout": holdout, "C": c},
            role["valid"], prob(role["valid"], eval_features),
            {
                "test_id": (role["test_id"], prob(role["test_id"], eval_features)),
                "test_id_jpeg50": (role["test_id"], prob(role["test_id"], jpeg50_features)),
                "test_ood": (role["test_ood"], prob(role["test_ood"], eval_features)),
            },
        )
        view = metrics[metrics["subset"] == "all"][["eval_set", "accuracy", "macro_f1", "roc_auc_real_positive", "fake_recall", "real_recall"]]
        print(view.round(4).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
