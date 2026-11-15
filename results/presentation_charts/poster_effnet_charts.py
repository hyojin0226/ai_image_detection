"""포스터 '최종 모델 성능' 그래프 - 단독 실행용 (데이터 내장, 다른 파일 필요 없음)

실행:  python poster_effnet_charts.py
결과:  같은 폴더에 chart_A_generalization.png, chart_B_by_generator.png, chart_C_robustness.png

색·글자 크기·문구는 아래 '설정'/'데이터' 부분만 바꾸면 됩니다.
숫자 출처: results/phase9_efficientnet (EfficientNetV2S 최종 t4, 얼굴 crop)
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# ============================== 설정 ==============================
NAVY = "#24489a"     # 강조 (포스터 남색)
MID = "#6f8ac7"      # 보조 파랑
LIGHT = "#c9d3ea"    # 비교 대상
TEXT = "#1f2430"
SUB = "#6b7080"
LINE = "#d5d8e0"
DPI = 300
OUT = Path(__file__).resolve().parent

# 한글 폰트: Windows 맑은 고딕 → 없으면 나눔고딕 (Colab: !apt -qq install fonts-nanum)
for path in ["C:/Windows/Fonts/malgun.ttf", "C:/Windows/Fonts/malgunbd.ttf",
             "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
             "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"]:
    if Path(path).exists():
        font_manager.fontManager.addfont(path)
names = {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams["font.family"] = "Malgun Gothic" if "Malgun Gothic" in names else "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False

# ============================== 데이터 ==============================
# A. 처음 보는 생성기 일반화: Diffusion 계열을 학습에서 빼고 Diffusion 으로 테스트 (ROC-AUC)
# 최종 모델로 보여줄 것: "vae_sbi" (일반화 최고) 또는 "vae" (배포 t4)
FINAL = "vae_sbi"

GEN_LABELS = ["기본 학습", "학습 설정\n튜닝", "+ VAE\n의사 fake", "+ VAE + SBI\n의사 fake"]
GEN_AUC = [0.728, 0.722, 0.876, 0.935]

# B. 생성기별 ROC-AUC (기본 학습 vs 최종 모델)
GENERATORS = ["Stable\nDiffusion", "SDXL", "FLUX\nDev", "FLUX\nPro"]
BASE_AUC = [0.882, 0.733, 0.639, 0.624]
FINAL_AUC = {"vae_sbi": [0.966, 0.934, 0.920, 0.933], "vae": [0.963, 0.877, 0.852, 0.786]}[FINAL]

# C. 이미지 열화 강건성: 추가 테스트 20,000장 ROC-AUC
DEGRADE = ["원본", "JPEG\n압축(50)", "해상도\n축소(½)", "블러"]
DEGRADE_AUC = {"vae_sbi": [0.992, 0.987, 0.989, 0.990], "vae": [0.994, 0.989, 0.992, 0.993]}[FINAL]


def style(ax):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(LINE)
    ax.set_yticks([])
    ax.tick_params(axis="x", length=0, pad=8, labelsize=13, colors=TEXT)


# ---------------- A. 3막대 (기본 → 튜닝 → 최종) ----------------
fig, ax = plt.subplots(figsize=(5.6, 4.6), facecolor="white")
style(ax)
n = len(GEN_AUC)
hi = 3 if FINAL == "vae_sbi" else 2          # 강조할 막대 (최종 모델)
colors = [NAVY if i == hi else (MID if i > 1 else LIGHT) for i in range(n)]
if FINAL == "vae":
    colors[3] = LIGHT
ax.bar(range(n), [v - 0.5 for v in GEN_AUC], 0.62, bottom=0.5, color=colors, zorder=2)
for i, v in enumerate(GEN_AUC):
    final = i == hi
    ax.text(i, v + 0.012, f"{v:.3f}", ha="center", va="bottom",
            fontsize=21 if final else 15, fontweight="bold", color=NAVY if final else SUB)
ax.text(hi, GEN_AUC[hi] + 0.085, f"기본 대비 ▲{GEN_AUC[hi] - GEN_AUC[0]:.3f}", ha="center",
        fontsize=11.5, fontweight="bold", color="white",
        bbox=dict(boxstyle="round,pad=0.4", fc=NAVY, ec="none"))
ax.set_xlim(-0.6, n - 0.4); ax.set_ylim(0.5, 1.07)
ax.set_xticks(range(n)); ax.set_xticklabels(GEN_LABELS, fontsize=12)
ax.get_xticklabels()[hi].set_fontweight("bold"); ax.get_xticklabels()[hi].set_color(NAVY)
fig.tight_layout()
fig.savefig(OUT / "chart_A_generalization.png", dpi=DPI)
plt.close(fig)

# ---------------- B. 생성기별 묶음 막대 ----------------
fig, ax = plt.subplots(figsize=(6, 4.6), facecolor="white")
style(ax)
n = len(GENERATORS); w = 0.36; gap = 0.02
xb = [i - w / 2 - gap for i in range(n)]
xf = [i + w / 2 + gap for i in range(n)]
ax.bar(xb, [v - 0.5 for v in BASE_AUC], w, bottom=0.5, color=LIGHT, zorder=2, label="기본 학습")
ax.bar(xf, [v - 0.5 for v in FINAL_AUC], w, bottom=0.5, color=NAVY, zorder=2, label="최종 모델")
for x, v in zip(xb, BASE_AUC):
    ax.text(x, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=11, color=SUB)
for x, v in zip(xf, FINAL_AUC):
    ax.text(x, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=12.5, fontweight="bold", color=NAVY)
ax.set_xlim(-0.6, n - 0.4); ax.set_ylim(0.5, 1.05)
ax.set_xticks(range(n)); ax.set_xticklabels(GENERATORS, fontsize=12)
ax.legend(loc="upper right", frameon=False, fontsize=11.5, ncol=2, handlelength=1.0,
          handleheight=1.0, bbox_to_anchor=(1.0, 1.08))
fig.tight_layout()
fig.savefig(OUT / "chart_B_by_generator.png", dpi=DPI)
plt.close(fig)

# ---------------- C. 열화 강건성 ----------------
fig, ax = plt.subplots(figsize=(5.6, 4.0), facecolor="white")
style(ax)
n = len(DEGRADE)
ax.bar(range(n), DEGRADE_AUC, 0.58, color=[NAVY] + [MID] * (n - 1), zorder=2)
for i, v in enumerate(DEGRADE_AUC):
    ax.text(i, v + 0.025, f"{v:.3f}", ha="center", va="bottom", fontsize=16, fontweight="bold", color=NAVY)
ax.set_xlim(-0.6, n - 0.4); ax.set_ylim(0.0, 1.16)
ax.set_xticks(range(n)); ax.set_xticklabels(DEGRADE, fontsize=12)
fig.tight_layout()
fig.savefig(OUT / "chart_C_robustness.png", dpi=DPI)
plt.close(fig)

print("saved:", *[p.name for p in sorted(OUT.glob("chart_*.png"))])
