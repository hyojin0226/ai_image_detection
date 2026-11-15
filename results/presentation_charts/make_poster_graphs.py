"""포스터 본문용 그래프 2개 (포스터 남색 톤, 세로 공간이 좁은 칸에 맞춤).

1. poster_graph_generalization.png : 처음 보는 Diffusion 생성기 일반화 (LOGO, 얼굴 crop AUC)
   - results/phase9_efficientnet/logo.csv
2. poster_graph_retouch.png        : 보정한 실제 사진을 AI 로 오판한 비율 (같은 150장)
   - results/experiment_b/model_ablation/{final_vae_sbi, deployed_t4}/summary.csv

실행: server/.venv/Scripts/python.exe results/presentation_charts/make_poster_graphs.py
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
ABL = ROOT / "results/experiment_b/model_ablation"
for f in ("C:/Windows/Fonts/malgun.ttf", "C:/Windows/Fonts/malgunbd.ttf"):
    font_manager.fontManager.addfont(f)
plt.rcParams.update({
    "font.family": font_manager.FontProperties(fname="C:/Windows/Fonts/malgun.ttf").get_name(),
    "axes.unicode_minus": False,
    "savefig.dpi": 300,
})

NAVY = "#24489a"
GRAY = "#b9bcc4"
TEXT = "#1a1d24"
TEXT2 = "#5b5f69"
GRID = "#e3e5ea"


def clean(ax, grid_axis="y"):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT2, length=0, labelsize=9)
    getattr(ax, f"{grid_axis}axis").grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def title(ax, main, sub):
    ax.text(0, 1.13, main, transform=ax.transAxes, fontsize=12, fontweight="bold", color=TEXT, va="bottom")
    ax.text(0, 1.04, sub, transform=ax.transAxes, fontsize=8.5, color=TEXT2, va="bottom")


# 1. 처음 보는 Diffusion 생성기 일반화
logo = pd.read_csv(P9 / "logo.csv")
logo = logo[(logo.view == "crop") & (logo.holdout == "diffusion")].set_index("model").ood_auc
keys = ["Phase 3 기준선 (튜닝 전, 의사 fake 없음)", "Phase 9 + 의사 fake 없음", "Phase 9 + 의사 fake VAE",
        "Phase 9 + 의사 fake VAE+SBI"]
labels = ["기본 학습", "학습 설정\n튜닝", "+ VAE\n의사 fake", "+ VAE + SBI\n의사 fake"]
vals = [logo[k] for k in keys]

fig, ax = plt.subplots(figsize=(5.2, 4.0), facecolor="white")
fig.subplots_adjust(left=0.1, right=0.98, top=0.8, bottom=0.17)
clean(ax)
bars = ax.bar(labels, vals, width=0.6, color=[GRAY, GRAY, NAVY, NAVY], edgecolor="white", linewidth=1.5)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", va="bottom",
            fontsize=11, fontweight="bold", color=TEXT)
ax.axhline(0.5, color=TEXT2, lw=1, ls=(0, (4, 3)))
ax.set_ylim(0.45, 1.0)
ax.tick_params(axis="x", labelsize=9.5, colors=TEXT)
title(ax, "처음 보는 생성기 일반화", "Diffusion 계열을 학습에서 빼고 FLUX·SDXL·SD 로 테스트 (ROC-AUC, 점선 = 무작위)")
fig.savefig(OUT / "poster_graph_generalization.png", facecolor="white")
plt.close(fig)
print("saved poster_graph_generalization.png", [round(v, 3) for v in vals])

# 2. 보정한 실제 사진 오판율
keys = [("skin_smooth_2", "피부 보정 (중)"), ("skin_smooth_3", "피부 보정 (강)"),
        ("beauty_combo_1", "뷰티 필터 (중)"), ("beauty_combo_2", "뷰티 필터 (강)")]
rate = {n: pd.read_csv(ABL / n / "summary.csv").assign(k=lambda d: d.edit + "_" + d.level.astype(str))
        .set_index("k").fake_rate * 100 for n in ("final_vae_sbi", "deployed_t4")}
sbi = [rate["final_vae_sbi"][k] for k, _ in keys]
dep = [rate["deployed_t4"][k] for k, _ in keys]

fig, ax = plt.subplots(figsize=(5.2, 4.0), facecolor="white")
fig.subplots_adjust(left=0.2, right=0.97, top=0.72, bottom=0.08)
clean(ax, "x")
y = range(len(keys)); h = 0.36
ax.barh([i - h / 2 - 0.02 for i in y], sbi, h, color=GRAY, edgecolor="white", label="SBI 포함 모델")
ax.barh([i + h / 2 + 0.02 for i in y], dep, h, color=NAVY, edgecolor="white", label="최종 모델 (SBI 제외)")
for i, (a, b) in enumerate(zip(sbi, dep)):
    ax.text(a + 1.5, i - h / 2 - 0.02, f"{a:.0f}%", va="center", fontsize=9, color=TEXT2)
    ax.text(b + 1.5, i + h / 2 + 0.02, f"{b:.1f}%", va="center", fontsize=10, fontweight="bold", color=TEXT)
ax.set_yticks(list(y)); ax.set_yticklabels([l for _, l in keys], fontsize=9.5, color=TEXT)
ax.invert_yaxis(); ax.set_xlim(0, 118)
ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
ax.legend(loc="lower left", frameon=False, fontsize=8.5, ncol=2, bbox_to_anchor=(-0.02, 0.99))
ax.text(0, 1.21, "보정한 실제 사진 오판율", transform=ax.transAxes, fontsize=12, fontweight="bold", color=TEXT, va="bottom")
ax.text(0, 1.13, "실제 사진 150장에 보정 적용 → AI 로 잘못 판정한 비율 (낮을수록 좋음)", transform=ax.transAxes, fontsize=8.5, color=TEXT2, va="bottom")
fig.savefig(OUT / "poster_graph_retouch.png", facecolor="white")
plt.close(fig)
print("saved poster_graph_retouch.png", [round(v, 1) for v in sbi], [round(v, 1) for v in dep])
