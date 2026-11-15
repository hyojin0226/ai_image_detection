"""막대 외 발표용 그래프: 데이터 정리 흐름, 덤벨, 기울기, ROC, 점수 분포.

실행: server/.venv/Scripts/python.exe results/presentation_charts/make_charts_extra.py
ROC·분포는 최종 모델(efficientnetv2s_pseudo_vae_deg_t4) 얼굴 crop 보기 예측
(final_model_crop_predictions.csv, WSL outputs/exp2/efficientnetv2s_pseudo_vae_deg_t4/all/crop_view/predictions.csv 복사본)을 씁니다.
7번은 results/phase9_efficientnet/diffusion_by_generator.csv, 8번은 results/experiment_b/model_ablation/ 을 읽습니다.
"""
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
P9 = ROOT / "results" / "phase9_efficientnet"
ABL = ROOT / "results" / "experiment_b" / "model_ablation"
THRESHOLD = 0.34   # 배포 모델 얼굴 crop 기준 (valid Macro F1)
FONT = "C:/Windows/Fonts/malgun.ttf"
font_manager.fontManager.addfont(FONT)
plt.rcParams.update({"font.family": font_manager.FontProperties(fname=FONT).get_name(),
                     "axes.unicode_minus": False, "savefig.dpi": 200})

SURFACE, TEXT, TEXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BEFORE, AFTER = "#b9b8b2", "#2a78d6"
REAL, FAKE = "#2a78d6", "#eb6834"   # 범주 1·2번 (파랑·주황)


def frame(fig, ax, title, subtitle, grid_axis="y"):
    fig.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT2, length=0, labelsize=11)
    if grid_axis:
        getattr(ax, f"{grid_axis}axis").grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    fig.text(0.04, 0.95, title, fontsize=17, fontweight="bold", color=TEXT, va="top")
    fig.text(0.04, 0.885, subtitle, fontsize=11.5, color=TEXT2, va="top")


def save(fig, name):
    fig.savefig(OUT / name, facecolor=SURFACE)
    plt.close(fig)


# 6. 데이터 정리 흐름 (가로 막대 깔때기)
fig, ax = plt.subplots(figsize=(10, 6))
fig.subplots_adjust(left=0.27, right=0.95, top=0.76, bottom=0.1)
steps = ["보유 파일 전체", "중복 제거 후 고유 이미지", "출처 확인·계열 분류 가능", "최종 모델 학습·평가에 사용"]
real = [183_700, 84_648, 84_304, 48_000]
fake = [183_697, 154_316, 151_187, 48_000]
y = np.arange(len(steps))[::-1]
ax.barh(y, real, color=REAL, height=0.55, label="실제", edgecolor=SURFACE, linewidth=2)
ax.barh(y, fake, left=real, color=FAKE, height=0.55, label="AI", edgecolor=SURFACE, linewidth=2)
for yi, r, f in zip(y, real, fake):
    ax.text(r + f + 6000, yi, f"{r + f:,}장", va="center", fontsize=13, fontweight="bold", color=TEXT)
ax.text(real[1] + fake[1] + 6000, y[1] - 0.36, "실제 사진 54% 감소 (같은 FFHQ 복사본)", va="center", fontsize=10.5, color=TEXT2)
ax.set_yticks(y); ax.set_yticklabels(steps, fontsize=12, color=TEXT)
ax.set_xlim(0, 430_000); ax.set_xticks([])
ax.spines["bottom"].set_visible(False)
ax.legend(loc="lower right", frameon=False, fontsize=11, ncol=2, bbox_to_anchor=(1, 1.0), labelcolor=TEXT)
frame(fig, ax, "데이터 정리: 367,397장 → 고유 238,964장 → 96,000장",
      "같은 사진은 한 번만, 같은 묶음은 같은 split에만 · 사용 이미지는 실제·AI 1:1 (학습 시 의사 fake 15,000장 추가)", grid_axis=None)
save(fig, "6_data_funnel.png")

# 7. 처음 보는 생성기별 AUC: 덤벨
fig, ax = plt.subplots(figsize=(10, 6))
fig.subplots_adjust(left=0.14, right=0.93, top=0.74, bottom=0.12)
gen = pd.read_csv(P9 / "diffusion_by_generator.csv")
gen["pseudo"] = gen.model.str.split("의사 fake ").str[-1]
gauc = gen.set_index(["pseudo", "generator"]).ood_auc
keys = ["sd", "sdxl", "flux_pro", "flux_dev"]
gens = ["SD", "SDXL", "FLUX-pro", "FLUX-dev"]
before = [gauc[("없음", k)] for k in keys]
after = [gauc[("VAE+SBI", k)] for k in keys]
for i, (b, a) in enumerate(zip(before, after)):
    ax.plot([b, a], [i, i], color=GRID, linewidth=4, solid_capstyle="round", zorder=1)
    ax.scatter([b], [i], s=150, color=BEFORE, zorder=2, edgecolor=SURFACE, linewidth=2)
    ax.scatter([a], [i], s=190, color=AFTER, zorder=3, edgecolor=SURFACE, linewidth=2)
    ax.text(b - 0.013, i, f"{b:.3f}", ha="right", va="center", fontsize=11, color=TEXT2)
    ax.text(a + 0.013, i, f"{a:.3f}", ha="left", va="center", fontsize=12, color=TEXT, fontweight="bold")
    ax.text(1.025, i, f"+{a - b:.3f}", ha="left", va="center", fontsize=11, color=TEXT2)
ax.set_yticks(range(len(gens))); ax.set_yticklabels(gens, fontsize=13, color=TEXT)
ax.set_xlim(0.50, 1.07); ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0]); ax.set_ylim(-0.6, len(gens) - 0.4)
ax.set_xlabel("ROC-AUC", color=TEXT2, fontsize=11)
ax.scatter([], [], s=120, color=BEFORE, label="의사 fake 없음")
ax.scatter([], [], s=120, color=AFTER, label="+ 의사 fake (VAE + SBI)")
ax.legend(loc="lower left", frameon=False, fontsize=11, ncol=2, bbox_to_anchor=(0, 1.0), labelcolor=TEXT)
frame(fig, ax, "학습에 없던 생성기, 가장 어려웠던 FLUX가 가장 크게 상승",
      "EfficientNetV2S, Diffusion 계열 제외 학습 · 생성기별 테스트 · 얼굴 crop (오른쪽 숫자 = 상승폭)", grid_axis="x")
save(fig, "7_unseen_generators_dumbbell.png")

# 8. 보정 사진 오판: 기울기 그래프 (낮을수록 좋음) — VAE+SBI 의사 fake vs 배포 모델(VAE, SBI 제외)
RETOUCH = [("beauty_combo", 2, "뷰티 조합 (강)"), ("skin_smooth", 3, "피부 보정 (강)"),
           ("beauty_combo", 1, "뷰티 조합 (중)"), ("skin_smooth", 2, "피부 보정 (중)")]
if (ABL / "deployed_t4" / "summary.csv").exists() and (ABL / "final_vae_sbi" / "summary.csv").exists():
    rate = lambda name: pd.read_csv(ABL / name / "summary.csv").set_index(["edit", "level"]).fake_rate * 100
    sbi, dep = rate("final_vae_sbi"), rate("deployed_t4")
    before = [sbi[(e, l)] for e, l, _ in RETOUCH]
    after = [dep[(e, l)] for e, l, _ in RETOUCH]
    top = max(before + after)
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.subplots_adjust(left=0.08, right=0.72, top=0.76, bottom=0.1)

    def spread(vals, gap):   # 라벨이 겹치지 않게 세로 위치만 벌림
        order = sorted(range(len(vals)), key=lambda k: -vals[k]); pos = list(vals)
        for a, b in zip(order, order[1:]):
            pos[b] = min(pos[b], pos[a] - gap)
        return pos
    by, ay = spread(before, top * 0.06), spread(after, top * 0.06)
    for (_, _, k), b, a, byy, ayy in zip(RETOUCH, before, after, by, ay):
        ax.plot([0, 1], [b, a], color=AFTER, linewidth=2, marker="o", markersize=8,
                markerfacecolor=AFTER, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.text(-0.04, byy, f"{b:.1f}%", ha="right", va="center", fontsize=11, color=TEXT2)
        ax.text(1.04, ayy, f"{a:.1f}%  {k}", ha="left", va="center", fontsize=12, color=TEXT)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["VAE + SBI\n의사 fake", "배포 모델\n(VAE, SBI 제외)"], fontsize=12, color=TEXT)
    ax.set_xlim(-0.25, 1.0); ax.set_ylim(-top * 0.08, top * 1.1); ax.set_yticks([])
    ax.spines["bottom"].set_visible(False)
    frame(fig, ax, f"보정한 실제 사진을 AI로 오판: 최대 {max(before):.0f}% → {max(after):.0f}%",
          f"실제 사진 150장에 보정 적용 · 낮을수록 좋음 · 보정 없는 사진 오판 {sbi[('original', 0)]:.1f}% / {dep[('original', 0)]:.1f}%",
          grid_axis=None)
    save(fig, "8_retouch_slope.png")
else:
    print("skip 8: experiment_b/model_ablation/{deployed_t4,final_vae_sbi} 없음")

# 9·10. 최종 모델 예측으로 ROC와 점수 분포
rows = list(csv.DictReader(open(OUT / "final_model_crop_predictions.csv", encoding="utf-8")))


def subset(name):
    s = [r for r in rows if r["eval_set"] == name]
    return np.array([int(r["y"]) for r in s]), np.array([float(r["prob_real"]) for r in s])


def roc(y, p):
    order = np.argsort(-p)
    y = y[order]
    tpr = np.concatenate([[0], np.cumsum(y) / y.sum()])
    fpr = np.concatenate([[0], np.cumsum(1 - y) / (1 - y).sum()])
    return fpr, tpr, np.trapezoid(tpr, fpr)


fig, ax = plt.subplots(figsize=(8, 7))
fig.subplots_adjust(left=0.12, right=0.95, top=0.8, bottom=0.11)
ax.plot([0, 1], [0, 1], color=TEXT2, linewidth=1, linestyle=(0, (4, 3)))
ax.text(0.62, 0.55, "무작위 (AUC 0.5)", fontsize=10, color=TEXT2, rotation=0)
for name, label, color in [("test_id", "테스트 6,000장", REAL), ("test_ood", "추가 테스트 20,000장", FAKE)]:
    yv, pv = subset(name)
    fpr, tpr, auc = roc(yv, pv)
    ax.plot(fpr, tpr, color=color, linewidth=2, label=f"{label} · AUC {auc:.3f}")
ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
ax.set_xlabel("AI 사진을 REAL로 잘못 본 비율 (FPR)", color=TEXT2, fontsize=11)
ax.set_ylabel("실제 사진을 REAL로 맞힌 비율 (TPR)", color=TEXT2, fontsize=11)
ax.legend(loc="lower right", frameon=False, fontsize=12, labelcolor=TEXT)
frame(fig, ax, "최종 모델 ROC: 곡선이 왼쪽 위에 붙어 있음",
      "EfficientNetV2S + VAE 의사 fake + 축소·블러 · 얼굴 crop 보기 · 학습 생성기 계열", grid_axis=None)
ax.grid(True, color=GRID, linewidth=0.8)
ax.spines["left"].set_visible(True); ax.spines["left"].set_color(GRID)
save(fig, "9_final_model_roc.png")

fig, ax = plt.subplots(figsize=(10, 6))
fig.subplots_adjust(left=0.08, right=0.97, top=0.74, bottom=0.12)
yv, pv = subset("test_ood")
bins = np.linspace(0, 1, 41)
ax.hist(pv[yv == 0], bins=bins, color=FAKE, alpha=0.18)
ax.hist(pv[yv == 1], bins=bins, color=REAL, alpha=0.18)
ax.hist(pv[yv == 0], bins=bins, histtype="step", color=FAKE, linewidth=2, label="AI 사진 10,000장")
ax.hist(pv[yv == 1], bins=bins, histtype="step", color=REAL, linewidth=2, label="실제 사진 10,000장")
ax.set_yscale("log")
ax.axvline(THRESHOLD, color=TEXT, linewidth=1.2, linestyle=(0, (4, 3)))
ax.text(THRESHOLD + 0.01, ax.get_ylim()[1] * 0.5, f"판정 기준 {THRESHOLD}", fontsize=11, color=TEXT, va="top")
ax.set_xlabel("모델의 REAL 점수", color=TEXT2, fontsize=11)
ax.set_ylabel("이미지 수 (로그 눈금)", color=TEXT2, fontsize=11)
ax.legend(loc="lower left", frameon=False, fontsize=11, ncol=2, bbox_to_anchor=(0, 1.0), labelcolor=TEXT)
fake_ok = (pv[yv == 0] < THRESHOLD).mean(); real_ok = (pv[yv == 1] >= THRESHOLD).mean()
frame(fig, ax, "AI와 실제 사진의 점수가 양 끝으로 갈라짐",
      f"추가 테스트 20,000장 · 기준 {THRESHOLD}에서 AI 탐지 {fake_ok:.1%} · 실제 정답 {real_ok:.1%}")
save(fig, "10_score_distribution.png")
print("saved")
