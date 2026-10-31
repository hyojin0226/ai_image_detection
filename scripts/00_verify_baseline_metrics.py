"""Phase 0 완료 확인: 1차 predictions CSV를 src/metrics.py로 다시 계산해 evaluation_summary.csv와 비교합니다.

실행: python scripts/00_verify_baseline_metrics.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import BASELINE_DIR
from src.data import load_manifest, scan_data_dir
from src.metrics import compute_metrics

RESULT_DIR = BASELINE_DIR / "results"
METRIC_COLUMNS = [
    "accuracy", "roc_auc_real_positive", "macro_f1",
    "fake_precision", "fake_recall", "fake_f1",
    "real_precision", "real_recall", "real_f1",
]


def main():
    summary = pd.read_csv(RESULT_DIR / "evaluation_summary.csv")
    rows = []
    for _, expected in summary.iterrows():
        predictions = pd.read_csv(RESULT_DIR / f"{expected['dataset']}_predictions.csv")
        actual = compute_metrics(predictions["true_label"], predictions["prob_real"], expected["threshold"])
        max_diff = max(abs(actual[c] - expected[c]) for c in METRIC_COLUMNS)
        rows.append({
            "dataset": expected["dataset"],
            "n_images": actual["n_images"],
            "expected_n": expected["n_images"],
            "max_abs_diff": max_diff,
        })

    report = pd.DataFrame(rows)
    print(report.to_string(index=False))

    manifest = load_manifest()
    print("\nmanifest (test_by_source 제외):")
    print(manifest.groupby(["split", "source", "label"]).size().unstack("label").to_string())
    print("\nflux_sdxl fake 세부 생성기:")
    fake = manifest[(manifest["source"] == "flux_sdxl") & (manifest["label"] == "fake")]
    print(pd.crosstab(fake["split"], fake["generator"]).to_string())

    disk = scan_data_dir()
    print("\n실제 data 폴더 구성 (scan_data_dir):")
    print(disk.groupby(["split", "source", "label"]).size().unstack("label").to_string())

    ok = (report["max_abs_diff"] < 1e-9).all() and (report["n_images"] == report["expected_n"]).all()
    print("\n결과:", "일치 ✅" if ok else "불일치 ❌")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
