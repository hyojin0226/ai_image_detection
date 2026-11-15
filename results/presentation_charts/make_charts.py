"""발표용 핵심 그래프 (EfficientNetV2S, Phase 9).

숫자 출처
- results/phase9_efficientnet/ : tuning.csv, logo.csv, diffusion_by_generator.csv, robustness_t4/summary.csv
  (WSL scripts/43_summarize_phase9.py, 52_eval_robustness_final.py 결과 복사본)
- results/experiment_b/model_ablation/ : 보정 사진 오판율 (experiments/edited_photos.py --n 150 --no-explain)
- 5번 영상 그래프: 두현 영상 실험 숫자 (모델 교체와 무관)

실행: server/.venv/Scripts/python.exe results/presentation_charts/make_charts.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parents[2]
P9 = ROOT / "results" / "phase9_efficientnet"
ABL = ROOT / "results" / "experiment_b" / "model_ablation"
OUT = Path(__file__).resolve().parent
FONT = "C:/Windows/Fonts/malgun.ttf"
font_manager.fontManager.addfont(FONT)
plt.rcParams.update({
    "font.family": font_manager.FontProperties(fname=FONT).get_name(),
    "axes.unicode_minus": False,
    "figure.dpi": 100,
    "savefig.dpi": 200,
})

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT2 = "#52514e"
GRID = "#e4e3df"
BEFORE = "#b9b8b2"   # 이전 단계: 중립 회색
AFTER = "#2a78d6"    # 개선 후: 범주 1번(파랑)
PSEUDO_COLORS = ["#b9b8b2", "#8fb6e8", "#5a96df", "#2a78d6"]   # 없음 → VAE → SBI → VAE+SBI (같은 색조 단계)

PSEUDO = ["없음", "VAE", "SBI", "VAE+SBI"]
GEN_LABEL = {"flux_dev": "FLUX-dev", "flux_pro": "FLUX-pro", "sdxl": "SDXL", "sd": "SD"}
RETOUCH = [("skin_smooth", 2, "피부 보정 (중)"), ("skin_smooth", 3, "피부 보정 (강)"),
           ("beauty_combo", 1, "뷰티 조합 (중)"), ("beauty_combo", 2, "뷰티 조합 (강)")]


def base(ax, title, subtitle):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT2, length=0, labelsize=11)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.figure.text(0.04, 0.95, title, fontsize=17, fontweight="bold", color=TEXT, va="top")
    ax.figure.text(0.04, 0.885, subtitle, fontsize=11.5, color=TEXT2, va="top")


def bar_values(ax, bars, fmt, color=TEXT, pad=0.01, size=12, bold=False):
    for b in bars:
        v = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, v + pad, fmt(v), ha="center", va="bottom",
                fontsize=size, color=color, fontweight="bold" if bold else "normal")


def save(fig, name):
    fig.savefig(OUT / name, facecolor=SURFACE)
    plt.close(fig)
    print("saved", name)


def pseudo_of(model: str) -> str:
    return "기준선" if "기준선" in model else model.split("의사 fake ")[-1]


logo = pd.read_csv(P9 / "logo.csv")
logo = logo[logo.view == "crop"].assign(pseudo=lambda d: d.model.map(pseudo_of))
gen = pd.read_csv(P9 / "diffusion_by_generator.csv").assign(pseudo=lambda d: d.model.map(pseudo_of))
rob = pd.read_csv(P9 / "robustness_t4" / "summary.csv")
tuning = pd.read_csv(P9 / "tuning.csv")


def logo_auc(holdout, pseudo):
    return float(logo[(logo.holdout == holdout) & (logo.pseudo == pseudo)].ood_auc.iloc[0])


# 1. 처음 보는 Diffusion 생성기 AUC: 의사 fake 종류별
fig, ax = plt.subplots(figsize=(10, 6), facecolor=SURFACE)
fig.subplots_adjust(left=0.08, right=0.97, top=0.76, bottom=0.17)
keys = ["기준선"] + PSEUDO
labels = ["튜닝 전\n(의사 fake 없음)", "튜닝 후\n의사 fake 없음", "+ VAE", "+ SBI", "+ VAE + SBI"]
values = [logo_auc("diffusion", k) for k in keys]
bars = ax.bar(labels, values, width=0.58, color=[BEFORE, BEFORE, AFTER, AFTER, AFTER], edgecolor=SURFACE, linewidth=2)
bar_values(ax, bars, lambda v: f"{v:.3f}", pad=0.012, size=14, bold=True)
ax.axhline(0.5, color=TEXT2, linewidth=1, linestyle=(0, (4, 3)))
ax.text(-0.42, 0.505, "무작위 = 0.5", fontsize=10, color=TEXT2, ha="left", va="bottom")
ax.set_ylim(0.4, 1.02)
ax.set_ylabel("ROC-AUC", color=TEXT2, fontsize=11)
base(ax, f"처음 보는 Diffusion 생성기 탐지: AUC {values[1]:.3f} → {values[-1]:.3f}",
     "EfficientNetV2S · Diffusion 계열(FLUX·SDXL·SD)을 학습에서 빼고 그 계열로만 테스트 · 얼굴 crop 보기")
save(fig, "1_unseen_diffusion_auc.png")

# 2. 처음 보는 생성기별 AUC: 의사 fake 전후
fig, ax = plt.subplots(figsize=(10, 6), facecolor=SURFACE)
fig.subplots_adjust(left=0.08, right=0.97, top=0.72, bottom=0.12)
gens = ["flux_dev", "flux_pro", "sdxl", "sd"]
g = gen.set_index(["pseudo", "generator"]).ood_auc
before = [g[("없음", k)] for k in gens]
after = [g[("VAE+SBI", k)] for k in gens]
x = range(len(gens)); w = 0.36
b1 = ax.bar([i - w / 2 - 0.01 for i in x], before, w, color=BEFORE, label="의사 fake 없음", edgecolor=SURFACE, linewidth=2)
b2 = ax.bar([i + w / 2 + 0.01 for i in x], after, w, color=AFTER, label="+ 의사 fake (VAE + SBI)", edgecolor=SURFACE, linewidth=2)
bar_values(ax, b1, lambda v: f"{v:.3f}", color=TEXT2, pad=0.006, size=11)
bar_values(ax, b2, lambda v: f"{v:.3f}", pad=0.006, size=11, bold=True)
ax.set_xticks(list(x)); ax.set_xticklabels([GEN_LABEL[k] for k in gens], fontsize=12)
ax.set_ylim(0.5, 1.02)
ax.set_ylabel("ROC-AUC", color=TEXT2, fontsize=11)
ax.legend(loc="lower left", frameon=False, fontsize=11, ncol=2, labelcolor=TEXT, bbox_to_anchor=(0, 1.02))
base(ax, f"가장 어려운 FLUX도 AUC {min(after[:2]):.2f} 이상",
     f"Diffusion 계열 제외 학습 · 생성기별 테스트 (FLUX {min(before[:2]):.2f}~{max(before[:2]):.2f} → "
     f"{min(after[:2]):.2f}~{max(after[:2]):.2f})")
save(fig, "2_unseen_generators_auc.png")

# 3. SNS형 열화에 대한 강건성 (배포 모델)
fig, ax = plt.subplots(figsize=(10, 6), facecolor=SURFACE)
fig.subplots_adjust(left=0.08, right=0.97, top=0.72, bottom=0.15)
perts = ["none", "jpeg_q90", "jpeg_q70", "jpeg_q50", "resize_0.75", "resize_0.5", "blur_0.5", "blur_1.0"]
pert_lbl = ["원본", "JPEG 90", "JPEG 70", "JPEG 50", "축소 0.75", "축소 0.5", "블러 0.5", "블러 1.0"]
r = rob[rob.view == "face_crop"].set_index(["eval_set", "perturbation"]).roc_auc_real_positive
for ev, label, color, bold in [("test_id", "테스트 6,000장", BEFORE, False), ("held_out", "추가 테스트 20,000장", AFTER, True)]:
    ys = [r[(ev, p)] for p in perts]
    for a, b in [(0, 4), (4, 6), (6, 8)]:   # 압축 / 축소 / 블러는 다른 열화라 선을 끊음
        ax.plot(range(a, b), ys[a:b], color=color, lw=2.4, marker="o", ms=9, mec=SURFACE, mew=2,
                label=label if a == 0 else None)
    if bold:   # 추가 테스트 값만 점 아래에 (회색 선과 겹치지 않게)
        for i, v in enumerate(ys):
            ax.text(i, v - 0.0009, f"{v:.3f}", ha="center", va="top", fontsize=10.5, color=TEXT, fontweight="bold")
for xv in (3.5, 5.5):
    ax.axvline(xv, color=GRID, lw=1)
ax.set_xticks(range(len(perts))); ax.set_xticklabels(pert_lbl, fontsize=11)
ax.set_ylim(0.975, 1.0)
ax.set_ylabel("ROC-AUC (얼굴 crop)", color=TEXT2, fontsize=11)
ax.legend(loc="lower left", frameon=False, fontsize=11, ncol=2, labelcolor=TEXT, bbox_to_anchor=(0, 1.02))
worst = min(r[("held_out", p)] for p in perts)
base(ax, f"SNS처럼 압축·축소·블러된 사진에서도 AUC {worst:.3f} 이상",
     "배포 모델 (EfficientNetV2S + VAE 의사 fake + 축소·블러 증강) · y축 0.975부터")
save(fig, "3_degradation_robustness.png")

# 3b. 계열별 일반화(LOGO): 한 계열을 빼고 학습 → 그 계열로 테스트
fig, ax = plt.subplots(figsize=(10, 6), facecolor=SURFACE)
fig.subplots_adjust(left=0.08, right=0.97, top=0.72, bottom=0.12)
fams = [("diffusion", "Diffusion\n(FLUX·SDXL·SD)"), ("gan", "GAN\n(StyleGAN)"), ("manip", "영상 딥페이크\n(얼굴 교체 등)")]
w = 0.19
for k, (p, color) in enumerate(zip(PSEUDO, PSEUDO_COLORS)):
    xs = [i + (k - 1.5) * (w + 0.01) for i in range(len(fams))]
    vals = [logo_auc(f, p) for f, _ in fams]
    bars = ax.bar(xs, vals, w, color=color, edgecolor=SURFACE, linewidth=1.5, label=f"의사 fake {p}")
    bar_values(ax, bars, lambda v: f"{v:.2f}", color=TEXT if p == "VAE+SBI" else TEXT2, pad=0.008, size=9.5,
               bold=p == "VAE+SBI")
ax.axhline(0.5, color=TEXT2, linewidth=1, linestyle=(0, (4, 3)))
ax.text(1.5, 0.505, "무작위 0.5", fontsize=9.5, color=TEXT2, ha="center", va="bottom")
ax.set_xticks(range(len(fams))); ax.set_xticklabels([l for _, l in fams], fontsize=11.5)
ax.set_ylim(0.3, 1.02)
ax.set_ylabel("처음 보는 계열 ROC-AUC", color=TEXT2, fontsize=11)
ax.legend(loc="lower left", frameon=False, fontsize=10.5, ncol=4, labelcolor=TEXT, bbox_to_anchor=(0, 1.02))
base(ax, "의사 fake 는 Diffusion 일반화만 끌어올림 · GAN·영상 딥페이크는 무작위 수준",
     "한 계열을 학습에서 빼고 그 계열로 테스트 (LOGO) · 학습 데이터의 GAN 이 StyleGAN 1종뿐인 한계")
save(fig, "3b_logo_by_family.png")


# 4. 보정한 실제 사진의 AI 오판율 (낮을수록 좋음): SBI 포함 여부
def retouch_rates(folder):
    s = pd.read_csv(folder / "summary.csv").set_index(["edit", "level"])
    return [float(s.loc[(e, l), "fake_rate"]) * 100 for e, l, _ in RETOUCH], float(s.loc[("original", 0), "fake_rate"]) * 100


if (ABL / "deployed_t4" / "summary.csv").exists() and (ABL / "final_vae_sbi" / "summary.csv").exists():
    sbi, sbi0 = retouch_rates(ABL / "final_vae_sbi")
    dep, dep0 = retouch_rates(ABL / "deployed_t4")
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=SURFACE)
    fig.subplots_adjust(left=0.08, right=0.97, top=0.72, bottom=0.12)
    x = range(len(RETOUCH)); w = 0.36
    b1 = ax.bar([i - w / 2 - 0.01 for i in x], sbi, w, color=BEFORE, label="VAE + SBI 의사 fake", edgecolor=SURFACE, linewidth=2)
    b2 = ax.bar([i + w / 2 + 0.01 for i in x], dep, w, color=AFTER, label="배포 모델 (VAE, SBI 제외)", edgecolor=SURFACE, linewidth=2)
    top = max(sbi + dep)
    bar_values(ax, b1, lambda v: f"{v:.1f}%", color=TEXT2, pad=top * 0.012, size=11)
    bar_values(ax, b2, lambda v: f"{v:.1f}%", pad=top * 0.012, size=11, bold=True)
    ax.set_xticks(list(x)); ax.set_xticklabels([l for *_, l in RETOUCH], fontsize=12)
    ax.set_ylim(0, top * 1.18)
    ax.set_ylabel("실제 사진을 AI로 잘못 판정한 비율 (%)", color=TEXT2, fontsize=11)
    ax.legend(loc="lower left", frameon=False, fontsize=11, ncol=2, labelcolor=TEXT, bbox_to_anchor=(0, 1.02))
    base(ax, "보정한 실제 사진 오판: SBI 를 빼서 낮춤",
         f"실제 사진 150장에 보정 적용 · 낮을수록 좋음 · 보정 없는 사진 오판 {sbi0:.1f}% / {dep0:.1f}%")
    save(fig, "4_retouch_false_positive.png")
else:
    print("skip 4: experiment_b/model_ablation/{deployed_t4,final_vae_sbi} 없음")

# 5. 영상: 실제 영상 오판 감소 (두현 실험, 모델 교체와 무관)
fig, ax = plt.subplots(figsize=(10, 6), facecolor=SURFACE)
fig.subplots_adjust(left=0.08, right=0.97, top=0.76, bottom=0.17)
labels = ["길이별 구간 평균\n(이전 방식)", "전체 8프레임\n(현재 사이트)"]
values = [19.0, 11.6]
bars = ax.bar(labels, values, width=0.45, color=[BEFORE, AFTER], edgecolor=SURFACE, linewidth=2)
bar_values(ax, bars, lambda v: f"{v:.1f}%", pad=0.4, size=14, bold=True)
ax.set_ylim(0, 24)
ax.set_ylabel("실제 영상을 AI로 잘못 판정한 비율 (%)", color=TEXT2, fontsize=11)
base(ax, "영상: 실제 영상 오판 19.0% → 11.6%",
     "시험 영상 1,000개 (실제 500 · Sora 500) · 정확도는 80.3% → 81.5%로 유지 · 낮을수록 좋음")
save(fig, "5_video_real_false_positive.png")

# 11. 학습 설정 튜닝 (valid 기준으로만 선택)
fig, ax = plt.subplots(figsize=(10, 6), facecolor=SURFACE)
fig.subplots_adjust(left=0.08, right=0.97, top=0.76, bottom=0.2)
t = tuning.copy()
SHORT = {"t0": "튜닝 전", "t1": "+ 분류층\n0 초기화", "t2": "t1\n+ crop 혼합", "t3": "t2 + 8 epoch\nEMA·drop path",
         "t6": "t3\n- crop 혼합", "t4": "t3\n+ 입력 320", "t5": "t3\n+ lr 3e-4"}
labels = [f"{n.split()[0]}\n{SHORT[n.split()[0]]}" for n in t.name]
colors = [AFTER if s else BEFORE for s in t.selected]
bars = ax.bar(range(len(t)), t.valid_auc_mean, width=0.6, color=colors, edgecolor=SURFACE, linewidth=2)
bar_values(ax, bars, lambda v: f"{v:.4f}", pad=0.0003, size=10.5)
ax.set_xticks(range(len(t))); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylim(0.985, 1.0)
ax.set_ylabel("valid 평균 ROC-AUC (보기 5종)", color=TEXT2, fontsize=11)
sel = t[t.selected].iloc[0]
base(ax, f"학습 설정 튜닝: valid AUC {t.valid_auc_mean.iloc[0]:.4f} → {sel.valid_auc_mean:.4f}",
     "선택은 valid 점수로만 (test 는 고른 뒤 확인용) · 파랑 = 선택 (분류층 0 초기화·8 epoch·EMA·drop path·crop 혼합·입력 320)")
save(fig, "11_tuning_valid.png")
