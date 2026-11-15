"""최종 모델(EfficientNetV2S, VAE + SBI 의사 fake) 평가 그래프 - matplotlib 기본 스타일 + Pretendard

  py_1_roc_unseen.png   처음 보는 생성기(Diffusion) ROC: 기본 학습 vs 최종 모델
  py_2_roc_test.png     테스트 / 추가 테스트 ROC
  py_3_score_dist.png   추가 테스트 P(real) 분포
  py_4_confusion.png    추가 테스트 혼동 행렬

데이터: poster_data/*.csv (eval_set, y(1=real), prob_real)
폰트:   Pretendard (Windows 설치 폰트 또는 이 폴더의 fonts/ 안 .otf/.ttf)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
DATA = HERE / "poster_data"
THRESHOLD = 0.30

# ---- font ----
cands = list((HERE / "fonts").glob("Pretendard*.[ot]tf"))
for d in [Path("C:/Windows/Fonts"), Path.home() / "AppData/Local/Microsoft/Windows/Fonts"]:
    cands += list(d.glob("Pretendard*.[ot]tf")) if d.exists() else []
for f in cands:
    font_manager.fontManager.addfont(str(f))
family = "Pretendard" if cands else "Malgun Gothic"
if not cands:
    print("[warn] Pretendard 없음 -> Malgun Gothic 사용")

plt.rcParams.update({
    "font.family": family,
    "axes.unicode_minus": False,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def roc_curve(y, score):
    order = np.argsort(-score, kind="mergesort")
    y, s = y[order], score[order]
    idx = np.r_[np.where(np.diff(s))[0], len(s) - 1]
    tps = np.cumsum(y)[idx]
    fps = (idx + 1) - tps
    tpr = np.r_[0, tps / tps[-1]]
    fpr = np.r_[0, fps / fps[-1]]
    return fpr, tpr, float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))


def load(name, eval_set):
    d = pd.read_csv(DATA / f"{name}.csv")
    return d[d.eval_set == eval_set]


def fake_score(d):
    # positive = FAKE
    return (d.y.values == 0).astype(int), 1.0 - d.prob_real.values


# 1. ROC - unseen generators
fig, ax = plt.subplots(figsize=(5, 5))
for name, lbl, c in [("base_logo_diffusion_crop", "Baseline", "tab:gray"),
                     ("final_logo_diffusion_crop", "Final (VAE+SBI)", "tab:blue")]:
    fpr, tpr, auc = roc_curve(*fake_score(load(name, "test_ood")))
    ax.plot(fpr, tpr, color=c, lw=2, label=f"{lbl} (AUC = {auc:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.500)")
ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve - 처음 보는 생성기 (Diffusion)")
ax.legend(loc="lower right")
ax.grid(alpha=0.3)
fig.savefig(HERE / "py_1_roc_unseen.png")
plt.close(fig)

# 2. ROC - test sets
fig, ax = plt.subplots(figsize=(5, 5))
axins = ax.inset_axes([0.38, 0.32, 0.5, 0.42])
for es, lbl, c in [("test_id", "Test (n=6,000)", "tab:blue"),
                   ("test_ood", "Additional test (n=20,000)", "tab:orange")]:
    fpr, tpr, auc = roc_curve(*fake_score(load("final_all_crop", es)))
    ax.plot(fpr, tpr, color=c, lw=2, label=f"{lbl}, AUC = {auc:.3f}")
    axins.plot(fpr, tpr, color=c, lw=1.5)
ax.plot([0, 1], [0, 1], "k--", lw=1)
axins.set_xlim(0, 0.1); axins.set_ylim(0.8, 1.0)
axins.tick_params(labelsize=8)
axins.grid(alpha=0.3)
ax.indicate_inset_zoom(axins, edgecolor="0.4")
ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve - 최종 모델")
ax.legend(loc="lower right")
ax.grid(alpha=0.3)
fig.savefig(HERE / "py_2_roc_test.png")
plt.close(fig)

# 3. score distribution
d = load("final_all_crop", "test_ood")
bins = np.linspace(0, 1, 41)
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.hist(d[d.y == 0].prob_real, bins=bins, alpha=0.6, color="tab:red", edgecolor="white", label="Fake")
ax.hist(d[d.y == 1].prob_real, bins=bins, alpha=0.6, color="tab:blue", edgecolor="white", label="Real")
ax.axvline(THRESHOLD, color="k", ls="--", lw=1.2, label=f"Threshold = {THRESHOLD:.2f}")
ax.set_yscale("log")
ax.set_xlim(0, 1)
ax.set_xlabel("P(real)")
ax.set_ylabel("Count (log scale)")
ax.set_title("예측 확률 분포 (추가 테스트, n=20,000)")
ax.legend(loc="upper center")
ax.grid(alpha=0.3, axis="y")
fig.savefig(HERE / "py_3_score_dist.png")
plt.close(fig)

# 4. confusion matrix
y_true = np.where(d.y.values == 1, 1, 0)
y_pred = (d.prob_real.values >= THRESHOLD).astype(int)
cm = np.array([[np.sum((y_true == t) & (y_pred == p)) for p in (0, 1)] for t in (0, 1)])
cmn = cm / cm.sum(axis=1, keepdims=True)
acc = np.trace(cm) / cm.sum()

fig, ax = plt.subplots(figsize=(5, 4.3))
im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=15)
plt.setp(cbar.ax.get_yticklabels(), fontweight="normal")
for i in range(2):
    for j in range(2):
        c = "white" if cmn[i, j] > 0.5 else "black"
        ax.text(j, i - 0.09, f"{cm[i, j]:,}", ha="center", va="center", fontsize=27, fontweight="semibold", color=c)
        ax.text(j, i + 0.18, f"({cmn[i, j]:.1%})", ha="center", va="center", fontsize=20,
                fontweight="normal", color=c)
ax.set_xticks([0, 1]); ax.set_xticklabels(["Fake", "Real"], fontsize=18, fontweight="normal")
ax.set_yticks([0, 1]); ax.set_yticklabels(["Fake", "Real"], fontsize=18, fontweight="normal")
ax.set_xlabel("Predicted label", fontsize=17, fontweight="normal")
ax.set_ylabel("True label", fontsize=17, fontweight="normal")
ax.set_title(f"Confusion Matrix (Accuracy = {acc:.1%})", fontsize=16.5, fontweight="semibold")
fig.savefig(HERE / "py_4_confusion.png")
plt.close(fig)

# 5. per-generator AUC (unseen diffusion generators)
gens = [("sd", "Stable Diffusion"), ("sdxl", "SDXL"), ("flux_dev", "FLUX Dev"), ("flux_pro", "FLUX Pro")]


def gen_auc(name):
    d = load(name, "test_ood")
    real = d[d.y == 1]
    out = []
    for g, _ in gens:
        sub = pd.concat([real, d[(d.y == 0) & (d.generator == g)]])
        out.append(roc_curve(*fake_score(sub))[2])
    return out


base, final = gen_auc("base_logo_diffusion_crop"), gen_auc("final_logo_diffusion_crop")
x = np.arange(len(gens)); w = 0.38
fig, ax = plt.subplots(figsize=(6.5, 4.6))
b1 = ax.bar(x - w / 2, base, w, color="tab:gray", alpha=0.6, label="Baseline")
b2 = ax.bar(x + w / 2, final, w, color="tab:blue", label="Final (VAE+SBI)")
ax.bar_label(b1, fmt="%.2f", padding=3, fontsize=15, fontweight="normal")
ax.bar_label(b2, fmt="%.2f", padding=3, fontsize=16, fontweight="semibold")
ax.axhline(0.5, color="k", ls="--", lw=1.2, label="Random")
ax.set_xticks(x); ax.set_xticklabels(["Stable\nDiffusion", "SDXL", "FLUX\nDev", "FLUX\nPro"],
                                     fontsize=16, fontweight="normal")
ax.tick_params(axis="y", labelsize=14)
plt.setp(ax.get_yticklabels(), fontweight="normal")
ax.set_ylim(0.4, 1.12)
ax.set_ylabel("ROC-AUC", fontsize=16, fontweight="normal")
ax.set_title("생성기별 ROC-AUC (학습에 없던 Diffusion 생성기)", fontsize=16, fontweight="semibold")
leg = ax.legend(loc="upper center", ncol=3, fontsize=13, columnspacing=1.2, handlelength=1.5)
plt.setp(leg.get_texts(), fontweight="normal")
ax.grid(alpha=0.3, axis="y")
fig.savefig(HERE / "py_5_auc_by_generator.png")
plt.close(fig)
print("by generator:", [round(v, 3) for v in base], [round(v, 3) for v in final])

print("done:", family, f"acc={acc:.4f}", cm.tolist())
