"""포스터용 '전문 시각화' 4종 - 최종 모델 EfficientNetV2S (VAE + SBI 의사 fake), 얼굴 crop

  pro_1_roc_unseen.png     처음 보는 생성기(Diffusion) ROC 곡선: 기본 학습 vs 최종 모델
  pro_2_roc_test.png       테스트 6,000장 / 추가 테스트 20,000장 ROC 곡선 (+ 왼쪽 위 확대)
  pro_3_score_dist.png     추가 테스트 20,000장 판별 점수 분포 (REAL vs FAKE, 판정 기준선)
  pro_4_confusion.png      추가 테스트 20,000장 혼동 행렬

데이터 (poster_data/ 폴더, WSL outputs/exp2 의 predictions.csv 복사본):
  final_all_crop.csv             efficientnetv2s_pseudo_vae_sbi_deg_p9/all/crop_view
  final_logo_diffusion_crop.csv  efficientnetv2s_pseudo_vae_sbi_deg_p9/diffusion/crop_view
  base_logo_diffusion_crop.csv   efficientnetv2s/diffusion/crop_view   (기본 학습)
열: eval_set, y(1=real, 0=fake), prob_real

실행: python poster_effnet_pro_charts.py   (필요: numpy, pandas, matplotlib)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap

# ============================== 설정 ==============================
HERE = Path(__file__).resolve().parent
DATA = HERE / "poster_data"
OUT = HERE
THRESHOLD = 0.30          # 최종 모델 P(real) 판정 기준 (valid 에서 선택)
NAVY = "#24489a"
MID = "#6f8ac7"
LIGHT = "#c9d3ea"
GRAY = "#a9adb8"
RED = "#d1495b"
TEXT = "#1f2430"
SUB = "#6b7080"
GRID = "#e6e8ee"
DPI = 300

for path in ["C:/Windows/Fonts/malgun.ttf", "C:/Windows/Fonts/malgunbd.ttf",
             "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
             "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"]:
    if Path(path).exists():
        font_manager.fontManager.addfont(path)
names = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams.update({
    "font.family": "Malgun Gothic" if "Malgun Gothic" in names else "NanumGothic",
    "axes.unicode_minus": False,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT,
    "xtick.color": SUB, "ytick.color": SUB,
})


def roc(y, score):
    """y: 1=positive. 반환 fpr, tpr, auc (sklearn 없이)"""
    order = np.argsort(-score, kind="mergesort")
    y, s = y[order], score[order]
    distinct = np.r_[np.where(np.diff(s))[0], len(s) - 1]
    tp = np.cumsum(y)[distinct]
    fp = (distinct + 1) - tp
    tpr = np.r_[0, tp / y.sum()]
    fpr = np.r_[0, fp / (len(y) - y.sum())]
    auc = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))   # 사다리꼴 적분
    return fpr, tpr, auc


def load(name, eval_set):
    d = pd.read_csv(DATA / f"{name}.csv")
    d = d[d.eval_set == eval_set]
    # FAKE 를 양성으로: 점수 = 1 - P(real)
    return (d.y.values == 0).astype(int), 1 - d.prob_real.values, d


def clean(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GRID); ax.spines["bottom"].set_color(GRID)
    ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    ax.tick_params(length=0, labelsize=10)


# ---------------- 1. 처음 보는 생성기 ROC ----------------
yb, sb, _ = load("base_logo_diffusion_crop", "test_ood")
yf, sf, _ = load("final_logo_diffusion_crop", "test_ood")
fb, tb, ab = roc(yb, sb)
ff, tf, af = roc(yf, sf)

fig, ax = plt.subplots(figsize=(4.8, 4.6), facecolor="white")
clean(ax)
ax.fill_between(ff, tf, color=NAVY, alpha=0.10, lw=0)
ax.plot([0, 1], [0, 1], color=SUB, lw=1, ls=(0, (4, 3)))
ax.plot(fb, tb, color=GRAY, lw=2.2, label=f"기본 학습   AUC {ab:.3f}")
ax.plot(ff, tf, color=NAVY, lw=2.8, label=f"최종 모델   AUC {af:.3f}")
ax.text(0.62, 0.36, "무작위 (0.5)", rotation=45, color=SUB, fontsize=8.5, ha="center", va="center",
        rotation_mode="anchor")
ax.set_xlim(0, 1); ax.set_ylim(0, 1.005); ax.set_aspect("equal")
ax.set_xlabel("거짓 양성률 (실제 사진을 AI 로 오판)", fontsize=10.5)
ax.set_ylabel("참 양성률 (AI 이미지 탐지)", fontsize=10.5)
leg = ax.legend(loc="lower right", frameon=True, fontsize=10.5, handlelength=1.6)
leg.get_frame().set_edgecolor(GRID)
for t, c in zip(leg.get_texts(), [SUB, NAVY]):
    t.set_color(c)
leg.get_texts()[1].set_fontweight("bold")
fig.tight_layout()
fig.savefig(OUT / "pro_1_roc_unseen.png", dpi=DPI, facecolor="white")
plt.close(fig)
print(f"1. unseen diffusion ROC: base {ab:.3f} -> final {af:.3f}")

# ---------------- 2. 테스트 ROC (+ 확대) ----------------
curves = []
for es, lbl, c, ls in [("test_id", "테스트 6,000장", NAVY, "-"), ("test_ood", "추가 테스트 20,000장", MID, "-")]:
    y, s, _ = load("final_all_crop", es)
    curves.append((*roc(y, s), lbl, c, ls))

fig, ax = plt.subplots(figsize=(4.8, 4.6), facecolor="white")
clean(ax)
ax.plot([0, 1], [0, 1], color=SUB, lw=1, ls=(0, (4, 3)))
for f_, t_, a_, lbl, c, ls in curves:
    ax.plot(f_, t_, color=c, lw=2.4, ls=ls, label=f"{lbl}   AUC {a_:.3f}")
ax.set_xlim(0, 1); ax.set_ylim(0, 1.005); ax.set_aspect("equal")
ax.set_xlabel("거짓 양성률 (실제 사진을 AI 로 오판)", fontsize=10.5)
ax.set_ylabel("참 양성률 (AI 이미지 탐지)", fontsize=10.5)
leg = ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.86), frameon=True, fontsize=10, handlelength=1.6)
leg.get_frame().set_edgecolor(GRID)
ins = ax.inset_axes([0.47, 0.10, 0.48, 0.44])
for f_, t_, a_, lbl, c, ls in curves:
    ins.plot(f_, t_, color=c, lw=2.0, ls=ls)
ins.set_xlim(0, 0.10); ins.set_ylim(0.80, 1.0)
ins.set_xticks([0, 0.05, 0.10]); ins.set_yticks([0.8, 0.9, 1.0])
ins.tick_params(labelsize=7.5, length=0, colors=SUB)
ins.grid(True, color=GRID, lw=0.6)
for s in ins.spines.values():
    s.set_color(MID); s.set_linewidth(0.8)
ins.set_title("왼쪽 위 확대 (거짓 양성률 0~10%)", fontsize=8, color=SUB, pad=3)
ins.set_facecolor("white")
ax.add_patch(plt.Rectangle((0, 0.8), 0.10, 0.2, fill=False, ec=MID, lw=1.0, ls=(0, (2, 2))))
fig.tight_layout()
fig.savefig(OUT / "pro_2_roc_test.png", dpi=DPI, facecolor="white")
plt.close(fig)
print("2. test ROC:", [round(c[2], 4) for c in curves])

# ---------------- 3. 점수 분포 ----------------
_, _, d = load("final_all_crop", "test_ood")
real = d[d.y == 1].prob_real.values
fake = d[d.y == 0].prob_real.values
bins = np.linspace(0, 1, 41)

fig, ax = plt.subplots(figsize=(5.6, 4.0), facecolor="white")
clean(ax); ax.grid(False, axis="x")
for vals, c, lbl in [(fake, RED, f"AI 생성 (FAKE) {len(fake):,}장"),
                     (real, NAVY, f"실제 사진 (REAL) {len(real):,}장")]:
    ax.hist(vals, bins=bins, color=c, alpha=0.22, zorder=2)
    ax.hist(vals, bins=bins, histtype="step", color=c, lw=1.8, zorder=3, label=lbl)
ax.set_yscale("log"); ax.set_ylim(0.8, 3e4)
ax.axvline(THRESHOLD, color=TEXT, lw=1.4, ls=(0, (4, 3)), zorder=3)
ax.text(THRESHOLD + 0.015, ax.get_ylim()[1] * 0.5, f"판정 기준 {THRESHOLD:.2f}", fontsize=9.5,
        color=TEXT, va="top")
ax.text(0.02, 0.97, "◀ FAKE 판정", transform=ax.transAxes, fontsize=10, color=RED, fontweight="bold", va="top")
ax.text(0.98, 0.97, "REAL 판정 ▶", transform=ax.transAxes, fontsize=10, color=NAVY, fontweight="bold",
        va="top", ha="right")
ax.set_xlim(0, 1)
ax.set_xlabel("모델이 출력한 실제 사진 확률  P(real)", fontsize=10.5)
ax.set_ylabel("이미지 수 (로그 축)", fontsize=10.5)
ax.legend(loc="upper left", bbox_to_anchor=(0.36, 0.88), frameon=False, fontsize=9.5)
fig.tight_layout()
fig.savefig(OUT / "pro_3_score_dist.png", dpi=DPI, facecolor="white")
plt.close(fig)

# ---------------- 4. 혼동 행렬 ----------------
pred_real = d.prob_real.values >= THRESHOLD
true_real = d.y.values == 1
cm = np.array([[np.sum(~true_real & ~pred_real), np.sum(~true_real & pred_real)],
               [np.sum(true_real & ~pred_real), np.sum(true_real & pred_real)]])
acc = np.trace(cm) / cm.sum()
cmap = LinearSegmentedColormap.from_list("navy", ["#ffffff", "#e9eef8", NAVY])

fig, ax = plt.subplots(figsize=(4.4, 4.0), facecolor="white")
rate = cm / cm.sum(axis=1, keepdims=True)
ax.imshow(rate, cmap=cmap, vmin=0, vmax=1)
for i in range(2):
    for j in range(2):
        dark = rate[i, j] > 0.5
        ax.text(j, i - 0.08, f"{rate[i, j] * 100:.1f}%", ha="center", va="center", fontsize=19,
                fontweight="bold", color="white" if dark else TEXT)
        ax.text(j, i + 0.2, f"{cm[i, j]:,}장", ha="center", va="center", fontsize=10,
                color="#dfe6f5" if dark else SUB)
ax.set_xticks([0, 1]); ax.set_xticklabels(["FAKE", "REAL"], fontsize=11.5, color=TEXT)
ax.set_yticks([0, 1]); ax.set_yticklabels(["FAKE", "REAL"], fontsize=11.5, color=TEXT)
ax.set_xlabel("모델 판정", fontsize=10.5); ax.set_ylabel("실제 정답", fontsize=10.5)
ax.xaxis.set_label_position("top"); ax.xaxis.tick_top()
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_xticks([0.5], minor=True); ax.set_yticks([0.5], minor=True)
ax.grid(which="minor", color="white", lw=4)
ax.text(0.5, 1.72, f"정확도 {acc * 100:.1f}%  ·  추가 테스트 {cm.sum():,}장", ha="center", fontsize=10.5,
        color=NAVY, fontweight="bold", transform=ax.transData)
fig.tight_layout()
fig.savefig(OUT / "pro_4_confusion.png", dpi=DPI, facecolor="white")
plt.close(fig)
print(f"4. confusion: acc {acc:.4f}", cm.tolist())
