"""Phase 4: Phase 3의 9개 학습 결과(3개 모델 × LOGO 3종)를 표 하나로 정리합니다.

- ID: 학습에 쓴 계열(test_id), OOD: 학습에서 뺀 계열(test_ood)
- Macro F1은 valid에서 정한 threshold 그대로, AUC는 threshold와 무관한 순위 지표입니다.
- "fake recall @ real 95%": OOD real의 95%를 REAL로 맞히도록 threshold를 다시 잡았을 때
  잡히는 OOD fake 비율. threshold 보정만으로 얼마나 회복되는지 보는 참고 지표입니다
  (test 정보를 쓰므로 모델 선택에는 쓰지 않습니다).

실행: python scripts/40_summarize_results.py
출력: outputs/exp2/summary/
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import EXP2_DIR

MODELS = {
    "efficientnetv2s": "EfficientNetV2S",
    "convnext_tiny": "ConvNeXt-T",
    "clip_linear": "CLIP + Linear",
}
HOLDOUTS = ["gan", "diffusion", "manip"]
OUT_DIR = EXP2_DIR / "summary"


def fake_recall_at_real_recall(y, prob_real, real_recall=0.95):
    threshold = np.quantile(prob_real[y == 1], 1 - real_recall)
    return float((prob_real[y == 0] < threshold).mean())


def load_runs():
    rows, generator_rows = [], []
    for model in MODELS:
        for holdout in HOLDOUTS:
            run_dir = EXP2_DIR / model / holdout
            metrics = pd.read_csv(run_dir / "metrics.csv")
            metrics = metrics[metrics["subset"] == "all"].set_index("eval_set")
            pred = pd.read_csv(run_dir / "predictions.csv")
            ood = pred[pred["eval_set"] == "test_ood"]
            y, prob = ood["y"].to_numpy(), ood["prob_real"].to_numpy()
            rows.append({
                "model": model,
                "holdout": holdout,
                "threshold": metrics.loc["test_id", "threshold"],
                "id_macro_f1": metrics.loc["test_id", "macro_f1"],
                "id_auc": metrics.loc["test_id", "roc_auc_real_positive"],
                "jpeg50_macro_f1": metrics.loc["test_id_jpeg50", "macro_f1"],
                "ood_macro_f1": metrics.loc["test_ood", "macro_f1"],
                "ood_auc": metrics.loc["test_ood", "roc_auc_real_positive"],
                "ood_fake_recall": metrics.loc["test_ood", "fake_recall"],
                "ood_real_recall": metrics.loc["test_ood", "real_recall"],
                "ood_fake_recall_at_real95": fake_recall_at_real_recall(y, prob),
            })
            # 생성기별: OOD real 전체와 비교한 AUC
            reals = ood[ood["y"] == 1]
            for generator, fakes in ood[ood["y"] == 0].groupby("generator"):
                both = pd.concat([reals, fakes])
                generator_rows.append({
                    "model": model,
                    "holdout": holdout,
                    "generator": generator,
                    "n_fake": len(fakes),
                    "auc_vs_ood_real": roc_auc_score(both["y"], both["prob_real"]),
                    "fake_recall": float((fakes["prob_real"] < rows[-1]["threshold"]).mean()),
                })
    return pd.DataFrame(rows), pd.DataFrame(generator_rows)


def final_table(runs):
    """계획서 Phase 4 형태: 모델 × (ID F1, OOD 계열별, OOD 평균, JPEG q50). ID·JPEG는 LOGO 3종 평균."""
    table = []
    for model, label in MODELS.items():
        r = runs[runs["model"] == model].set_index("holdout")
        row = {"모델": label, "ID Macro F1": r["id_macro_f1"].mean()}
        for holdout in HOLDOUTS:
            row[f"OOD {holdout} (F1 / AUC)"] = f"{r.loc[holdout, 'ood_macro_f1']:.3f} / {r.loc[holdout, 'ood_auc']:.3f}"
        row["OOD 평균 (F1 / AUC)"] = f"{r['ood_macro_f1'].mean():.3f} / {r['ood_auc'].mean():.3f}"
        row["JPEG q50 Macro F1"] = r["jpeg50_macro_f1"].mean()
        table.append(row)
    return pd.DataFrame(table)


def to_markdown(df):
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str(v)
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def plot_ood_auc(runs, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.25
    x = np.arange(len(HOLDOUTS))
    for i, (model, label) in enumerate(MODELS.items()):
        r = runs[runs["model"] == model].set_index("holdout").loc[HOLDOUTS]
        ax.bar(x + (i - 1) * width, r["ood_auc"], width, label=label)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="chance")
    ax.set_xticks(x, [f"OOD {h}" for h in HOLDOUTS])
    ax.set_ylim(0, 1)
    ax.set_ylabel("ROC AUC")
    ax.set_title("Unseen family (LOGO) ROC AUC")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    runs, generators = load_runs()
    runs.round(4).to_csv(OUT_DIR / "runs.csv", index=False)
    generators.round(4).to_csv(OUT_DIR / "ood_by_generator.csv", index=False)

    table = final_table(runs)
    table.round(4).to_csv(OUT_DIR / "final_table.csv", index=False)
    (OUT_DIR / "final_table.md").write_text(to_markdown(table), encoding="utf-8")
    plot_ood_auc(runs, OUT_DIR / "ood_auc.png")

    pd.set_option("display.width", 200)
    print(runs.round(4).to_string(index=False))
    print()
    print(generators.round(4).to_string(index=False))
    print()
    print(to_markdown(table))


if __name__ == "__main__":
    main()
