"""평가 지표. 1차 노트북의 evaluate_dataset()과 같은 정의를 사용합니다."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def choose_threshold(y_true, prob_real):
    """valid에서 Macro F1이 최대가 되는 threshold (1차와 같은 0.05~0.95, 0.01 간격)."""
    thresholds = np.arange(0.05, 0.951, 0.01)
    scores = [
        f1_score(y_true, (prob_real >= t).astype(int), average="macro", zero_division=0)
        for t in thresholds
    ]
    best = int(np.argmax(scores))
    return float(thresholds[best]), float(scores[best])


def save_run_results(out_dir, config: dict, valid_rows, valid_prob, eval_sets: dict):
    """Phase 3 공통 결과 저장. threshold는 valid에서만 정하고 모든 테스트에 그대로 적용합니다.

    eval_sets: {이름: (rows, prob_real)}. rows에는 y, family, generator 열이 있어야 합니다.
    저장: metrics.csv(전체·계열별), predictions.csv, config.yaml
    """
    import pandas as pd
    import yaml

    out_dir.mkdir(parents=True, exist_ok=True)
    threshold, valid_f1 = choose_threshold(valid_rows["y"].to_numpy(), valid_prob)

    metrics, predictions = [], []
    for name, (rows, prob) in eval_sets.items():
        rows = rows.assign(prob_real=prob, eval_set=name)
        predictions.append(rows)
        metrics.append({"eval_set": name, "subset": "all", **compute_metrics(rows["y"], prob, threshold)})
        # 계열별 fake recall (real은 공통이므로 fake 쪽만 나눠 봄)
        for family, fake in rows[rows["y"] == 0].groupby("family"):
            recall = float((fake["prob_real"] < threshold).mean())
            metrics.append({"eval_set": name, "subset": f"fake_recall:{family}", "n_images": len(fake), "fake_recall": recall})

    pd.DataFrame(metrics).to_csv(out_dir / "metrics.csv", index=False)
    columns = ["eval_set", "path", "y", "family", "generator", "domain", "prob_real"]
    pd.concat(predictions)[columns].to_csv(out_dir / "predictions.csv", index=False)
    with open(out_dir / "config.yaml", "w", encoding="utf-8") as file:
        yaml.safe_dump({**config, "threshold": threshold, "valid_macro_f1": valid_f1}, file, allow_unicode=True)
    return pd.DataFrame(metrics)


def compute_metrics(y_true, prob_real, threshold: float) -> dict:
    y_true = np.asarray(y_true).astype(int)
    prob_real = np.asarray(prob_real, dtype=float)
    y_pred = (prob_real >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    auc = roc_auc_score(y_true, prob_real) if len(np.unique(y_true)) == 2 else np.nan

    return {
        "n_images": len(y_true),
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        # fake/real 비율이 다른 소스별 평가용 (두 클래스 recall의 평균)
        "balanced_accuracy": (recall[0] + recall[1]) / 2,
        "roc_auc_real_positive": auc,
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "fake_precision": precision[0],
        "fake_recall": recall[0],
        "fake_f1": f1[0],
        "real_precision": precision[1],
        "real_recall": recall[1],
        "real_f1": f1[1],
    }
