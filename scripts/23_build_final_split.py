"""서버용 최종 모델 split: 세 계열(gan / diffusion / manip)을 모두 학습에 씁니다.

LOGO split 3개(logo_{gan,diffusion,manip}.csv)는 같은 group_key bucket 규칙으로 만들어져
한 이미지가 어느 manifest 에서든 같은 role(train/valid/test_id)을 가집니다 (test_ood 제외).
그래서 세 manifest 의 train/valid/test_id 를 합치면 누수 없는 전체 split 이 됩니다.

- train / valid / test_id: 세 manifest 합집합, 클래스별로 LOGO 실험과 같은 상한(30k/5k/3k)까지 계열 비율 유지 샘플링
- test_ood: 어느 manifest 에서도 test_ood 로만 쓰인 이미지 (test bucket, 학습과 분리). 계열을 빼지 않았으므로
  여기서는 "OOD" 가 아니라 추가 held-out 테스트입니다.

실행: python scripts/23_build_final_split.py
출력: data/manifests/logo_all.csv
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DATA_DIR, SEED

CAPS = {"train": 30_000, "valid": 5_000, "test_id": 3_000, "test_ood": 10_000}


def cap(rows: pd.DataFrame, limit: int) -> pd.DataFrame:
    parts = []
    for _, class_rows in rows.groupby("y"):
        fraction = min(1.0, limit / len(class_rows))
        for _, family_rows in class_rows.groupby("family"):
            parts.append(family_rows.sample(frac=fraction, random_state=SEED))
    return pd.concat(parts)


def main():
    manifests = pd.concat([pd.read_csv(DATA_DIR / "manifests" / f"logo_{h}.csv") for h in ["gan", "diffusion", "manip"]])
    in_dist = manifests[manifests["role"] != "test_ood"].drop_duplicates("path")
    assert in_dist.groupby("path")["role"].nunique().max() == 1
    held_out = manifests[~manifests["path"].isin(in_dist["path"])].drop_duplicates("path").assign(role="test_ood")

    rows = pd.concat([in_dist, held_out])
    split = pd.concat([cap(rows[rows["role"] == role], limit) for role, limit in CAPS.items()])
    split = split.sort_values(["role", "path"]).reset_index(drop=True)
    out = DATA_DIR / "manifests" / "logo_all.csv"
    split.to_csv(out, index=False)
    print(split.groupby(["role", "label", "family"]).size().to_string())
    print(f"saved {out} ({len(split)} rows)")


if __name__ == "__main__":
    main()
