"""2차 실험 공통 설정: 경로, 클래스 매핑, 시드."""

from pathlib import Path
import random

import numpy as np

# 저장소 루트 (src/의 상위 폴더)
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
MANIFEST_PATH = DATA_DIR / "manifests" / "split_manifest.csv"

OUTPUT_DIR = ROOT / "outputs"
BASELINE_DIR = OUTPUT_DIR / "efficientnetv2s_experiment_01"
BASELINE_MODEL_PATH = BASELINE_DIR / "models" / "efficientnetv2s_fake_face_final.keras"
BASELINE_CONFIG_PATH = BASELINE_DIR / "models" / "inference_config.json"
EXP2_DIR = OUTPUT_DIR / "exp2"

# 라벨 고정: fake=0, real=1 (모델 출력 = real일 확률)
CLASS_NAMES = ["fake", "real"]
LABEL_TO_INDEX = {"fake": 0, "real": 1}

SOURCES = ["gravex", "stylegan", "flux_sdxl"]

IMAGE_SIZE = (256, 256)
SEED = 42


def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
