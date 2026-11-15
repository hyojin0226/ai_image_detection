"""포스터 하단 '최종 모델 성능' 스트립 (4패널, EfficientNetV2S Phase 9).

숫자 출처
- KPI: results/phase9_efficientnet/robustness_t4/summary.csv (face_crop, none)
- 일반화: results/phase9_efficientnet/logo.csv (Diffusion 계열 제외 학습 → Diffusion 테스트, 얼굴 crop)
- 강건성: phase9_efficientnet/robustness_vae_sbi(대안, VAE+SBI) vs robustness_t4(최종), face_crop, held_out
- 보정 오탐: experiment_b/model_ablation/{final_vae_sbi, deployed_t4}/summary.csv (같은 실제 사진 150장)

실행: server/.venv/Scripts/python.exe results/presentation_charts/make_poster_perf.py
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
FONT = "C:/Windows/Fonts/malgun.ttf"
FONT_B = "C:/Windows/Fonts/malgunbd.ttf"
for f in (FONT, FONT_B):
    font_manager.fontManager.addfont(f)
plt.rcParams.update({
    "font.family": font_manager.FontProperties(fname=FONT).get_name(),
    "axes.unicode_minus": False,
    "savefig.dpi": 300,
})

NAVY = "#24489a"      # 포스터 헤더 남색과 맞춤 (최종 모델)
GRAY = "#b9bcc4"      # 비교 대상
TEXT = "#1a1d24"
TEXT2 = "#5b5f69"
GRID = "#e3e5ea"
TILE = "#e9eef8"      # 포스터 연한 파랑 박스

# ---------- 데이터 ----------
prev = pd.read_csv(P9 / "robustness_vae_sbi/summary.csv")
final = pd.read_csv(P9 / "robustness_t4/summary.csv")
pick = lambda d, ev: d[(d.view == "face_crop") & (d.eval_set == ev)].set_index("perturbation")
p_ho, f_ho, f_id = pick(prev, "held_out"), pick(final, "held_out"), pick(final, "test_id")
perts = ["none", "jpeg_q90", "jpeg_q70", "jpeg_q50", "resize_0.75", "resize_0.5", "blur_0.5", "blur_1.0"]
pert_lbl = ["원본", "JPEG\n90", "JPEG\n70", "JPEG\n50", "축소\n0.75", "축소\n0.5", "블러\n0.5", "블러\n1.0"]

logo = pd.read_csv(P9 / "logo.csv")
logo = logo[(logo.view == "crop") & (logo.holdout == "diffusion")].set_index("model").ood_auc

ret = {name: pd.read_csv(ABL / name / "summary.csv").assign(key=lambda d: d.edit + "_" + d.level.astype(str))
       .set_index("key").fake_rate for name in ("final_vae_sbi", "deployed_t4")}
ret_keys = ["skin_smooth_2", "skin_smooth_3", "beauty_combo_1", "beauty_combo_2"]
ret_lbl = ["피부 보정 (중)", "피부 보정 (강)", "뷰티 조합 (중)", "뷰티 조합 (강)"]


def clean(ax, grid_axis="y"):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT2, length=0, labelsize=10)
    getattr(ax, f"{grid_axis}axis").grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def head(ax, title, sub):
    ax.set_title(title, loc="left", fontsize=14, fontweight="bold", color=TEXT, pad=26)
    ax.text(0, 1.035, sub, transform=ax.transAxes, fontsize=9.5, color=TEXT2, va="bottom")


fig = plt.figure(figsize=(20, 5.2), facecolor="white")
gs = fig.add_gridspec(1, 4, width_ratios=[0.8, 1.1, 1.5, 1.2], wspace=0.28,
                      left=0.015, right=0.985, top=0.82, bottom=0.16)

# ① KPI 타일
ax = fig.add_subplot(gs[0]); ax.axis("off")
head(ax, "① 최종 모델 한눈에", "EfficientNetV2S · 얼굴 crop · 판정 기준 0.34")
tiles = [
    (f"{f_id.loc['none', 'roc_auc_real_positive']:.3f}", "ROC-AUC", "테스트 6,000장"),
    (f"{f_ho.loc['none', 'roc_auc_real_positive']:.3f}", "ROC-AUC", "추가 테스트 20,000장"),
    (f"{f_id.loc['none', 'accuracy'] * 100:.1f}%", "정확도", "테스트 6,000장"),
    (f"{f_ho.loc['none', 'accuracy'] * 100:.1f}%", "정확도", "추가 테스트 20,000장"),
]
for i, (big, metric, note) in enumerate(tiles):
    r, c = divmod(i, 2)
    x0, y0, w, h = c * 0.51, 0.52 - r * 0.52, 0.47, 0.46
    ax.add_patch(plt.Rectangle((x0, y0), w, h, color=TILE, transform=ax.transAxes, clip_on=False))
    ax.text(x0 + w / 2, y0 + h * 0.60, big, transform=ax.transAxes, ha="center", va="center",
            fontsize=24, fontweight="bold", color=NAVY)
    ax.text(x0 + w / 2, y0 + h * 0.27, metric, transform=ax.transAxes, ha="center", fontsize=10.5, color=TEXT)
    ax.text(x0 + w / 2, y0 + h * 0.10, note, transform=ax.transAxes, ha="center", fontsize=8.5, color=TEXT2)

# ② 일반화
ax = fig.add_subplot(gs[1]); clean(ax)
head(ax, "② 처음 보는 생성기 일반화", "Diffusion 계열 빼고 학습 → Diffusion으로 테스트 (AUC, 점선 = 무작위 0.5)")
labels = ["튜닝 전", "튜닝 후", "+ VAE\n의사 fake", "+ VAE+SBI\n의사 fake"]
vals = [logo[m] for m in ["Phase 3 기준선 (튜닝 전, 의사 fake 없음)", "Phase 9 + 의사 fake 없음",
                          "Phase 9 + 의사 fake VAE", "Phase 9 + 의사 fake VAE+SBI"]]
bars = ax.bar(labels, vals, width=0.58, color=[GRAY, GRAY, NAVY, NAVY], edgecolor="white", linewidth=2)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}", ha="center", va="bottom",
            fontsize=12, fontweight="bold", color=TEXT)
ax.axhline(0.5, color=TEXT2, lw=1, ls=(0, (4, 3)))
ax.set_ylim(0.45, 1.02)
ax.tick_params(axis="x", labelsize=9.5)

# ③ 강건성
ax = fig.add_subplot(gs[2]); clean(ax)
head(ax, "③ SNS 열화에도 유지되는 성능", "추가 테스트 20,000장에 압축·축소·블러 적용 (AUC, y축 0.98부터)")
x = range(len(perts))
yp = [p_ho.loc[k, "roc_auc_real_positive"] for k in perts]
yf = [f_ho.loc[k, "roc_auc_real_positive"] for k in perts]
groups = [(0, 4), (4, 6), (6, 8)]   # 압축 / 축소 / 블러는 서로 다른 열화라 선을 끊음
for gi, (a, b) in enumerate(groups):
    xs = list(range(a, b))
    ax.plot(xs, yp[a:b], color=GRAY, lw=2.2, marker="o", ms=8, mec="white", mew=2,
            label="대안 (VAE+SBI)" if gi == 0 else None)
    ax.plot(xs, yf[a:b], color=NAVY, lw=2.2, marker="o", ms=8, mec="white", mew=2,
            label="최종 (VAE, SBI 제외)" if gi == 0 else None)
for xv in (3.5, 5.5):
    ax.axvline(xv, color=GRID, lw=1)
i_min = yf.index(min(yf))
ax.annotate(f"{yp[i_min]:.3f}", (i_min, yp[i_min]), xytext=(0, -16), textcoords="offset points",
            ha="center", fontsize=10.5, color=TEXT2)
ax.annotate(f"{yf[i_min]:.3f}", (i_min, yf[i_min]), xytext=(9, -4), textcoords="offset points",
            ha="left", fontsize=11, fontweight="bold", color=TEXT)
ax.set_xticks(list(x)); ax.set_xticklabels(pert_lbl, fontsize=9.5)
ax.set_ylim(0.98, 0.997)
ax.legend(loc="lower left", frameon=False, fontsize=9.5, ncol=2, bbox_to_anchor=(0, -0.02))

# ④ 보정 사진 오탐
ax = fig.add_subplot(gs[3]); clean(ax, "x")
head(ax, "④ 보정한 실제 사진 오탐률", "실제 사진 150장 보정 → AI로 잘못 판정한 비율 (낮을수록 좋음)")
y = range(len(ret_keys)); h = 0.36
bp = [ret["final_vae_sbi"][k] * 100 for k in ret_keys]
bf = [ret["deployed_t4"][k] * 100 for k in ret_keys]
xmax = max(bp + bf) * 1.25
ax.barh([i - h / 2 - 0.02 for i in y], bp, h, color=GRAY, edgecolor="white", label="대안 (VAE+SBI)")
ax.barh([i + h / 2 + 0.02 for i in y], bf, h, color=NAVY, edgecolor="white", label="최종")
for i, (a, b) in enumerate(zip(bp, bf)):
    ax.text(a + xmax * 0.015, i - h / 2 - 0.02, f"{a:.1f}%", va="center", fontsize=9.5, color=TEXT2)
    ax.text(b + xmax * 0.015, i + h / 2 + 0.02, f"{b:.1f}%", va="center", fontsize=10.5, fontweight="bold", color=TEXT)
ax.set_yticks(list(y)); ax.set_yticklabels(ret_lbl, fontsize=10)
ax.invert_yaxis(); ax.set_xlim(0, xmax)
ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
ax.legend(loc="lower right", frameon=False, fontsize=9.5)

fig.savefig(OUT / "poster_final_performance.png", facecolor="white")
print("saved", OUT / "poster_final_performance.png")
