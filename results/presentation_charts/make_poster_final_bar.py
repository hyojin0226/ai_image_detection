"""포스터 '최종 모델 성능' 막대그래프 .

최종(배포) 모델 = EfficientNetV2S + VAE 의사 fake + 축소·블러 (Phase 9 t4), 얼굴 crop, 판정 기준 P(real) 0.34
숫자 출처: results/phase9_efficientnet/robustness_t4/summary.csv (view=face_crop, perturbation=none)

실행: server/.venv/Scripts/python.exe results/presentation_charts/make_poster_final_bar.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
FONT = "C:/Windows/Fonts/malgun.ttf"
font_manager.fontManager.addfont(FONT)
font_manager.fontManager.addfont("C:/Windows/Fonts/malgunbd.ttf")
plt.rcParams.update({
    "font.family": font_manager.FontProperties(fname=FONT).get_name(),
    "axes.unicode_minus": False,
    "savefig.dpi": 300,
})

NAVY = "#24489a"
LIGHT = "#8fa6d6"
TEXT = "#1a1d24"
TEXT2 = "#5b5f69"
GRID = "#e3e5ea"

d = pd.read_csv(ROOT / "results/phase9_efficientnet/robustness_t4/summary.csv")
d = d[(d.view == "face_crop") & (d.perturbation == "none")].set_index("eval_set")
cols = ["roc_auc_real_positive", "accuracy", "fake_recall", "real_recall"]
labels = ["ROC-AUC", "정확도", "AI 사진\n탐지율", "실제 사진\n정답률"]
test = [d.loc["test_id", c] for c in cols]
held = [d.loc["held_out", c] for c in cols]

fig, ax = plt.subplots(figsize=(7.2, 4.0), facecolor="white")
fig.subplots_adjust(left=0.08, right=0.98, top=0.84, bottom=0.17)
x = range(len(cols)); w = 0.36
b1 = ax.bar([i - w / 2 - 0.015 for i in x], test, w, color=NAVY, edgecolor="white", linewidth=1.5,
            label="테스트 6,000장")
b2 = ax.bar([i + w / 2 + 0.015 for i in x], held, w, color=LIGHT, edgecolor="white", linewidth=1.5,
            label="추가 테스트 20,000장")


def fmt(v, is_auc):
    return f"{v:.3f}" if is_auc else f"{v * 100:.1f}%"


for bars, vals in ((b1, test), (b2, held)):
    for i, (b, v) in enumerate(zip(bars, vals)):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.003, fmt(v, i == 0), ha="center", va="bottom",
                fontsize=9.5, fontweight="bold", color=TEXT)

for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(colors=TEXT2, length=0, labelsize=9)
ax.yaxis.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=10, color=TEXT)
ax.set_ylim(0.85, 1.01)
ax.legend(loc="upper right", frameon=False, fontsize=9, ncol=2, bbox_to_anchor=(1, 1.13))
ax.text(0, 1.11, "최종 모델 (EfficientNetV2S + VAE + 축소·블러)", transform=ax.transAxes,
        fontsize=11.5, fontweight="bold", color=TEXT, va="bottom")
ax.text(0, 1.02, "얼굴 crop · 판정 기준 0.34 · y축 0.85부터", transform=ax.transAxes,
        fontsize=8.5, color=TEXT2, va="bottom")

fig.savefig(OUT / "poster_final_model_bar.png", facecolor="white")
print("saved", OUT / "poster_final_model_bar.png")
print("test", [round(v, 4) for v in test]); print("held", [round(v, 4) for v in held])
