"""Phase 2 대조 실험: 이미지 내용 없이 "파일 형식·압축 정보"만으로 real/fake를 맞힐 수 있는지 확인합니다.

특징: PNG 여부, JPEG 양자화 테이블 평균(밝기·색), 픽셀당 파일 크기, 원본 가로·세로.
이 정보만으로 정확도가 50%보다 크게 높으면, 모델이 진짜/가짜 대신 압축 차이를 단서로 쓸 위험이 있다는 뜻입니다.
(기존 계획의 "real 출처 맞히기" 대조 실험은 real이 대부분 같은 FFHQ 사진이라 의미가 없어 이 실험으로 대체합니다.)

실행: python scripts/21_shortcut_probe.py
출력: outputs/exp2/logo_splits/shortcut_probe.csv
"""

import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DATA_DIR, EXP2_DIR, SEED
from src.data import scan_data_dir

FEATURES = ["is_png", "q_luma", "q_chroma", "bytes_per_pixel", "width", "height"]
N_PER_CLASS = 5_000


def file_features(path):
    with Image.open(path) as image:
        tables = getattr(image, "quantization", None) or {}
        width, height = image.size
        return {
            "is_png": int(image.format == "PNG"),
            "q_luma": float(np.mean(tables[0])) if 0 in tables else 0.0,
            "q_chroma": float(np.mean(tables[1])) if 1 in tables else 0.0,
            "bytes_per_pixel": os.path.getsize(path) / (width * height),
            "width": width,
            "height": height,
        }


def with_features(rows):
    with ThreadPoolExecutor(16) as pool:
        features = pd.DataFrame(list(pool.map(file_features, rows["path"])), index=rows.index)
    return rows.join(features)


def per_class_sample(rows, n):
    return rows.sample(frac=1, random_state=SEED).groupby("y").head(n)


def probe(train, tests):
    model = HistGradientBoostingClassifier(random_state=SEED).fit(train[FEATURES], train["y"])
    results = {}
    for name, test in tests.items():
        prob = model.predict_proba(test[FEATURES])[:, 1]
        results[f"{name}_accuracy"] = accuracy_score(test["y"], prob >= 0.5)
        results[f"{name}_auc"] = roc_auc_score(test["y"], prob)
    return results


def main():
    rows = []

    # 1차 실험 데이터(중복 제거 전, 실제 폴더 그대로)
    first = scan_data_dir()
    train = with_features(per_class_sample(first[first["split"] == "train"], N_PER_CLASS))
    test = with_features(per_class_sample(first[first["split"] == "test"], N_PER_CLASS))
    rows.append({"setup": "1차 데이터", **probe(train, {"test_id": test})})
    print(rows[-1], flush=True)

    for holdout in ["gan", "diffusion", "manip"]:
        split = pd.read_csv(DATA_DIR / "manifests" / f"logo_{holdout}.csv")
        train = with_features(per_class_sample(split[split["role"] == "train"], N_PER_CLASS))
        tests = {
            role: with_features(per_class_sample(split[split["role"] == role], N_PER_CLASS))
            for role in ["test_id", "test_ood"]
        }
        rows.append({"setup": f"LOGO-{holdout}", **probe(train, tests)})
        print(rows[-1], flush=True)

    out = EXP2_DIR / "logo_splits" / "shortcut_probe.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(pd.DataFrame(rows).round(4).to_string(index=False))
    print("저장:", out)


if __name__ == "__main__":
    main()
