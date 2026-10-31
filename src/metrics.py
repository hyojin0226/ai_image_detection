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
