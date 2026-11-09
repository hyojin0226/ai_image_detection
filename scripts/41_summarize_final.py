"""Phase 5~7b·실험 B 결과를 리포트용 작은 표로 모읍니다 (notebooks/02_generalization_report.ipynb 가 읽음).

- 학습 run 의 metrics.csv(전체 이미지 보기)와 crop_view/metrics.csv(얼굴 crop x2.0 보기)를 한 표로
- diffusion 을 뺀 LOGO 실험의 생성기별 OOD AUC (crop_view/predictions.csv 가 있을 때만 다시 계산)
- 최종 모델 3종의 강건성(실험 A 축소판) 요약

predictions.csv 는 용량 때문에 git 에서 제외하므로, 생성기별 표는 한 번 계산한 결과 CSV 를 커밋해 둡니다.

실행: python scripts/41_summarize_final.py
출력: outputs/exp2/summary/report/
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import EXP2_DIR

OUT_DIR = EXP2_DIR / "summary" / "report"

# (run 폴더, 리포트 이름, 단계)
RUNS = [
    ("efficientnetv2s", "EfficientNetV2S", "P3 baseline"),
    ("efficientnetv2s_aug", "EfficientNetV2S + strong aug", "P5"),
    ("convnext_tiny", "ConvNeXt-T", "P3 baseline"),
    ("convnext_tiny_aug", "ConvNeXt-T + strong aug", "P5"),
    ("convnext_tiny_pseudo_vae", "ConvNeXt-T + VAE", "P6"),
    ("convnext_tiny_pseudo_sbi", "ConvNeXt-T + SBI", "P6"),
    ("convnext_tiny_pseudo_vae_sbi", "ConvNeXt-T + VAE + SBI", "P6"),
    ("convnext_tiny_pseudo_vae_sbi_deg", "ConvNeXt-T + VAE + SBI + degrade", "P7"),
    ("convnext_tiny_pseudo_vae_deg", "ConvNeXt-T + VAE + degrade", "P7b"),
]
HOLDOUTS = ["diffusion", "gan", "manip", "all"]
VIEWS = {"crop": "crop_view/metrics.csv", "full": "metrics.csv"}

# 실험 A 축소판 결과 폴더 (robustness_final/ 바로 아래가 VAE+SBI 최종 모델)
FINAL_MODELS = [
    ("", "VAE + SBI"),
    ("convnext_tiny_pseudo_vae_sbi_deg", "VAE + SBI + degrade"),
    ("convnext_tiny_pseudo_vae_deg", "VAE + degrade (deployed)"),
]


def logo_table():
    rows = []
    for run, name, phase in RUNS:
        for holdout in HOLDOUTS:
            for view, rel in VIEWS.items():
                path = EXP2_DIR / run / holdout / rel
                if not path.exists():
                    continue
                m = pd.read_csv(path)
                m = m[m["subset"] == "all"].set_index("eval_set")
                rows.append({
                    "run": run, "model": name, "phase": phase, "holdout": holdout, "view": view,
                    "threshold": m.loc["test_id", "threshold"],
                    "test_id_auc": m.loc["test_id", "roc_auc_real_positive"],
                    "test_id_macro_f1": m.loc["test_id", "macro_f1"],
                    # holdout=all 이면 test_ood 는 계열을 빼지 않은 추가 held-out 테스트
                    "ood_auc": m.loc["test_ood", "roc_auc_real_positive"],
                    "ood_fake_recall": m.loc["test_ood", "fake_recall"],
                    "ood_real_recall": m.loc["test_ood", "real_recall"],
                })
    return pd.DataFrame(rows)


def diffusion_by_generator(logo):
    rows = []
    for run, name, phase in RUNS:
        path = EXP2_DIR / run / "diffusion" / "crop_view" / "predictions.csv"
        if not path.exists():
            continue
        pred = pd.read_csv(path)
        ood = pred[pred["eval_set"] == "test_ood"]
        threshold = logo.query("run == @run and holdout == 'diffusion' and view == 'crop'")["threshold"].iloc[0]
        reals = ood[ood["y"] == 1]
        for generator, fakes in ood[ood["y"] == 0].groupby("generator"):
            both = pd.concat([reals, fakes])
            rows.append({
                "run": run, "model": name, "phase": phase, "generator": generator, "n_fake": len(fakes),
                "ood_auc": roc_auc_score(both["y"], both["prob_real"]),
                "fake_recall": float((fakes["prob_real"] < threshold).mean()),
            })
    return pd.DataFrame(rows)


def final_robustness():
    summaries, families = [], []
    for sub, name in FINAL_MODELS:
        folder = EXP2_DIR / "robustness_final" / sub
        s = pd.read_csv(folder / "summary.csv")
        f = pd.read_csv(folder / "by_family.csv")
        s.insert(0, "model", name)
        f.insert(0, "model", name)
        summaries.append(s)
        families.append(f)
    return pd.concat(summaries), pd.concat(families)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logo = logo_table()
    logo.to_csv(OUT_DIR / "runs_by_view.csv", index=False)
    by_generator = diffusion_by_generator(logo)
    if len(by_generator):
        by_generator.to_csv(OUT_DIR / "diffusion_by_generator.csv", index=False)
    summary, family = final_robustness()
    summary.to_csv(OUT_DIR / "final_robustness.csv", index=False)
    family.to_csv(OUT_DIR / "final_robustness_by_family.csv", index=False)
    print(logo.query("holdout == 'diffusion' and view == 'crop'")[["model", "ood_auc", "ood_fake_recall"]].round(3).to_string(index=False))
    print(by_generator.pivot(index="model", columns="generator", values="ood_auc").round(3).to_string())
    print(f"saved -> {OUT_DIR}")


if __name__ == "__main__":
    main()
