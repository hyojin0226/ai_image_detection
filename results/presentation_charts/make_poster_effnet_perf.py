"""포스터 '최종 모델 성능' 칸 그래프 (EfficientNetV2S 최종 = VAE 의사 fake + 축소·블러 증강, t4).

1. poster_effnet_generalization.png : 기존 3막대 그래프 자리 그대로 교체용
   Diffusion 계열을 학습에서 빼고 FLUX·SDXL·SD 로 테스트한 AUC (얼굴 crop)
   - results/phase9_efficientnet/logo.csv
2. poster_effnet_by_generator.png   : 생성기별로 풀어 본 버전 (기본 학습 → 최종 레시피)
   - results/phase9_efficientnet/diffusion_by_generator.csv

실행: server/.venv/Scripts/python.exe results/presentation_charts/make_poster_effnet_perf.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
P9 = ROOT / "results/phase9_efficientnet"
for f in ("C:/Windows/Fonts/malgun.ttf", "C:/Windows/Fonts/malgunbd.ttf"):
    font_manager.fontManager.addfont(f)
plt.rcParams.update({
    "font.family": font_manager.FontProperties(fname="C:/Windows/Fonts/malgun.ttf").get_name(),
    "axes.unicode_minus": False,
    "savefig.dpi": 300,
})

NAVY = "#24489a"   # 포스터 남색
GRAY = "#b9bcc4"
TEXT = "#1a1d24"
TEXT2 = "#5b5f69"
GRID = "#e3e5ea"
TILE = "#e9eef8"

BASE = "Phase 3 기준선 (튜닝 전, 의사 fake 없음)"
TUNED = "Phase 9 + 의사 fake 없음"
FINAL = "Phase 9 + 의사 fake VAE"


def clean(ax, grid_axis="y"):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT2, length=0, labelsize=9)
    getattr(ax, f"{grid_axis}axis").grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


# ---------- 1. 3막대: 기본 → 튜닝 → +VAE 의사 fake(최종) ----------
logo = pd.read_csv(P9 / "logo.csv")
logo = logo[(logo.view == "crop") & (logo.holdout == "diffusion")].set_index("model").ood_auc
vals = [logo[BASE], logo[TUNED], logo[FINAL]]
labels = ["기본 학습", "학습 설정\n튜닝", "+ VAE 의사 fake\n(최종 모델)"]

fig, ax = plt.subplots(figsize=(4.4, 4.0), facecolor="white")
fig.subplots_adjust(left=0.11, right=0.97, top=0.82, bottom=0.16)
clean(ax)
bars = ax.bar(labels, vals, width=0.58, color=[GRAY, GRAY, NAVY], edgecolor="white", linewidth=1.5)
for b, v, c in zip(bars, vals, [TEXT2, TEXT2, NAVY]):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", va="bottom",
            fontsize=13, fontweight="bold", color=c)
# 개선폭 화살표 (기본 → 최종)
gain = vals[2] - vals[0]
x0 = bars[0].get_x() + bars[0].get_width() / 2
x2 = bars[2].get_x() + bars[2].get_width() / 2
ax.annotate("", xy=(x2 - 0.2, vals[2] + 0.055), xytext=(x0 + 0.05, vals[0] + 0.055),
            arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=1.6,
                            connectionstyle="arc3,rad=-0.25", mutation_scale=14))
ax.text((x0 + x2) / 2, 0.955, f"+{gain:.3f}", ha="center", va="center", fontsize=12,
        fontweight="bold", color=NAVY,
        bbox=dict(boxstyle="round,pad=0.3", fc=TILE, ec="none"))
ax.axhline(0.5, color=TEXT2, lw=1, ls=(0, (4, 3)))
ax.text(1.5, 0.507, "무작위", ha="center", va="bottom", fontsize=7.5, color=TEXT2)
ax.set_ylim(0.45, 1.0)
ax.tick_params(axis="x", labelsize=9.5, colors=TEXT)
ax.text(0, 1.13, "처음 보는 생성기 일반화", transform=ax.transAxes, fontsize=12.5, fontweight="bold",
        color=TEXT, va="bottom")
ax.text(0, 1.04, "Diffusion 계열을 학습에서 빼고 FLUX·SDXL·SD 로 테스트 (ROC-AUC)",
        transform=ax.transAxes, fontsize=8.3, color=TEXT2, va="bottom")
fig.savefig(OUT / "poster_effnet_generalization.png", facecolor="white")
plt.close(fig)
print("saved poster_effnet_generalization.png", [round(v, 3) for v in vals])

# ---------- 2. 생성기별 덤벨: 기본 학습 → 최종 레시피 ----------
gen = pd.read_csv(P9 / "diffusion_by_generator.csv").set_index(["model", "generator"]).ood_auc
gens = [("flux_pro", "FLUX Pro"), ("flux_dev", "FLUX Dev"), ("sdxl", "SDXL"), ("sd", "Stable Diffusion")]
b = [gen[(BASE, g)] for g, _ in gens]
f = [gen[(FINAL, g)] for g, _ in gens]

fig, ax = plt.subplots(figsize=(4.4, 4.0), facecolor="white")
fig.subplots_adjust(left=0.27, right=0.80, top=0.76, bottom=0.12)
clean(ax, "x")
ax.spines["bottom"].set_visible(False)
y = list(range(len(gens)))
for yi, a, c in zip(y, b, f):
    ax.plot([a, c], [yi, yi], color=NAVY, alpha=0.25, lw=5, solid_capstyle="round", zorder=1)
ax.scatter(b, y, s=70, color=GRAY, edgecolor="white", lw=1.5, zorder=2, label="기본 학습")
ax.scatter(f, y, s=90, color=NAVY, edgecolor="white", lw=1.5, zorder=3, label="최종 모델")
for yi, a, c in zip(y, b, f):
    ax.text(a - 0.012, yi, f"{a:.2f}", ha="right", va="center", fontsize=9, color=TEXT2)
    ax.text(c + 0.012, yi, f"{c:.2f}", ha="left", va="center", fontsize=10.5, fontweight="bold", color=NAVY)
    ax.text(1.17, yi, f"+{c - a:.2f}", ha="center", va="center", fontsize=9.5, fontweight="bold",
            color=NAVY, transform=ax.get_yaxis_transform(),
            bbox=dict(boxstyle="round,pad=0.25", fc=TILE, ec="none"))
ax.text(1.17, len(gens) - 0.45, "향상폭", ha="center", va="bottom", fontsize=8, color=TEXT2,
        transform=ax.get_yaxis_transform())
ax.set_yticks(y); ax.set_yticklabels([n for _, n in gens], fontsize=10, color=TEXT)
ax.set_xlim(0.5, 1.0)
ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9])
ax.set_ylim(-0.6, len(gens) - 0.4)
ax.axvline(0.5, color=TEXT2, lw=1, ls=(0, (4, 3)))
ax.legend(loc="lower left", frameon=False, fontsize=8.5, ncol=2, bbox_to_anchor=(-0.03, 0.97),
          handletextpad=0.2, columnspacing=1.2)
ax.text(0, 1.2, "생성기별 일반화 향상", transform=ax.transAxes, fontsize=12.5, fontweight="bold",
        color=TEXT, va="bottom")
ax.text(0, 1.12, "학습에 없던 Diffusion 생성기 ROC-AUC (얼굴 crop)", transform=ax.transAxes,
        fontsize=8.3, color=TEXT2, va="bottom")
fig.savefig(OUT / "poster_effnet_by_generator.png", facecolor="white")
plt.close(fig)
print("saved poster_effnet_by_generator.png", [round(v, 3) for v in b], [round(v, 3) for v in f])
