"""Phase 2: 생성기 계열 단위 Leave-One-Generator-Out(LOGO) split 생성.

기존 소스(gravex / stylegan / flux_sdxl / ffhq_1mff)에는 같은 사진이 중복으로 들어 있어서
(EXPERIMENT_PLAN.md 2.4절), 소스 대신 생성기 계열(gan / diffusion / manip)을 빼고,
사진 단위로 중복을 제거한 뒤 split을 새로 나눕니다. 파일은 복사하지 않고 목록(CSV)만 만듭니다.

규칙
- 같은 사진(image_key)은 한 번만 사용합니다.
- 같은 묶음(group_key: 같은 FFHQ 사진, 같은 영상의 프레임 등)은 같은 bucket(train/valid/test = 70/10/20)에 들어갑니다.
  test bucket의 photo real은 test_id와 test_ood가 겹치지 않게 나눠 씁니다.
- 빼는 계열의 fake는 학습에 전혀 쓰지 않으므로 전체 bucket을 OOD 테스트 후보로 씁니다.
- manip(영상 딥페이크)을 뺄 때는 영상 프레임 real도 학습에서 빼서 "영상 프레임이면 real"이라는 지름길을 막습니다.
- 출처를 특정할 수 없는 이미지(family=unknown)는 사용하지 않습니다.

실행: python scripts/20_build_logo_splits.py
출력: data/manifests/logo_{gan,diffusion,manip}.csv, outputs/exp2/logo_splits/split_summary.csv
"""

import hashlib
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import DATA_DIR, EXP2_DIR, SEED
from src.data import scan_data_dir, annotate

FAMILIES = ["gan", "diffusion", "manip"]
MAX_TRAIN_PER_CLASS = 30_000
MAX_VALID_PER_CLASS = 5_000
MAX_TEST_ID_PER_CLASS = 3_000
MAX_TEST_OOD_PER_CLASS = 10_000

OUT_DIR = EXP2_DIR / "logo_splits"
COLUMNS = ["role", "path", "label", "y", "family", "generator", "domain", "image_key", "group_key", "ext", "source"]


def bucket(group_key: str) -> str:
    value = int(hashlib.md5(group_key.encode()).hexdigest(), 16) % 100
    return "train" if value < 70 else "valid" if value < 80 else "test"


def sample_by(rows, n, column, seed=SEED):
    """`column` 값의 원래 비율을 유지하면서 최대 n장."""
    if len(rows) <= n:
        return rows
    fraction = n / len(rows)
    return pd.concat(g.sample(frac=fraction, random_state=seed) for _, g in rows.groupby(column))


def balanced(fake, real, cap):
    """fake/real을 같은 수(둘 중 적은 쪽, 최대 cap)로 맞춥니다."""
    n = min(len(fake), len(real), cap)
    return sample_by(fake, n, "family"), sample_by(real, n, "domain")


def build(pool, holdout):
    train_fake = pool[(pool["label"] == "fake") & (pool["family"] != holdout)]
    if holdout == "manip":
        train_real = pool[(pool["label"] == "real") & (pool["domain"] == "photo")]
        ood_real = pool[(pool["label"] == "real") & (pool["domain"] == "video")]  # 학습에 전혀 쓰지 않음
    else:
        train_real = pool[pool["label"] == "real"]
        ood_real = train_real[(train_real["bucket"] == "test") & (train_real["domain"] == "photo")]
    ood_fake = pool[(pool["label"] == "fake") & (pool["family"] == holdout)]

    parts = []
    for role, source_bucket, cap in [
        ("train", "train", MAX_TRAIN_PER_CLASS),
        ("valid", "valid", MAX_VALID_PER_CLASS),
        ("test_id", "test", MAX_TEST_ID_PER_CLASS),
    ]:
        fake = train_fake[train_fake["bucket"] == source_bucket]
        real = train_real[train_real["bucket"] == source_bucket]
        if role == "train":
            fake, real = sample_by(fake, cap, "family"), sample_by(real, cap, "domain")
        else:
            fake, real = balanced(fake, real, cap)
        parts += [fake.assign(role=role), real.assign(role=role)]

    used = set(pd.concat(parts)["image_key"])
    fake, real = balanced(ood_fake, ood_real[~ood_real["image_key"].isin(used)], MAX_TEST_OOD_PER_CLASS)
    parts += [fake.assign(role="test_ood"), real.assign(role="test_ood")]
    return pd.concat(parts)[COLUMNS].reset_index(drop=True)


def check(split, holdout):
    """누수 검사: 학습(train/valid)과 테스트 사이에 같은 사진·같은 묶음이 없어야 합니다."""
    seen = split[split["role"].isin(["train", "valid"])]
    for role in ["test_id", "test_ood"]:
        test = split[split["role"] == role]
        assert not set(seen["image_key"]) & set(test["image_key"]), f"{holdout}/{role}: 같은 사진 누수"
        assert not set(seen["group_key"]) & set(test["group_key"]), f"{holdout}/{role}: 같은 묶음 누수"
    assert not (seen["family"] == holdout).any(), f"{holdout}: 빼야 할 계열이 학습에 포함됨"
    assert split["image_key"].is_unique, f"{holdout}: 중복 사진"


def main():
    rows = annotate(scan_data_dir())
    pool = rows[rows["family"] != "unknown"].drop_duplicates("image_key").copy()
    pool["bucket"] = pool["group_key"].map(bucket)
    print(f"전체 {len(rows)}장 → 중복·unknown 제외 {len(pool)}장")

    manifest_dir = DATA_DIR / "manifests"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summaries = []
    for holdout in FAMILIES:
        split = build(pool, holdout)
        check(split, holdout)
        split.to_csv(manifest_dir / f"logo_{holdout}.csv", index=False)
        summary = (
            split.groupby(["role", "label", "family"]).size().rename("images").reset_index()
            .assign(holdout=holdout)
        )
        summaries.append(summary)
        print(f"\n[LOGO-{holdout}] 누수 검사 통과")
        print(split.pivot_table(index="role", columns="label", values="path", aggfunc="count").loc[["train", "valid", "test_id", "test_ood"]])

    pd.concat(summaries)[["holdout", "role", "label", "family", "images"]].to_csv(OUT_DIR / "split_summary.csv", index=False)
    print("\n저장:", manifest_dir, OUT_DIR)


if __name__ == "__main__":
    main()
