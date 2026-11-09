#!/bin/bash
# Phase 7: 실험 A 후속. 의사 fake(VAE+SBI) + 축소·블러 증강(--degrade)
#   1) diffusion 제외 LOGO 실험으로 OOD 이득이 유지되는지 확인 (비교: convnext_tiny_pseudo_vae_sbi/diffusion)
#   2) 세 계열 전체로 최종 모델 재학습
#   3) 실험 A 축소판(52_eval_robustness_final.py)으로 강건성 재측정
# 사용 (저장소 루트): bash scripts/run_phase7.sh > outputs/exp2/phase7/run.log 2>&1
set -e
cd "$(dirname "$0")/.."
PY=.venv-exp2/bin/python
mkdir -p outputs/exp2/phase7
for holdout in diffusion all; do
    if [ -f "outputs/exp2/convnext_tiny_pseudo_vae_sbi_deg/$holdout/metrics.csv" ]; then echo "[skip] $holdout"; continue; fi
    echo "=== --pseudo vae+sbi --degrade / $holdout ($(date)) ==="
    $PY scripts/32_train_cnn.py --model convnext_tiny --holdout "$holdout" --pseudo vae+sbi --degrade
done
echo "=== 강건성 평가 ($(date)) ==="
$PY scripts/52_eval_robustness_final.py --run convnext_tiny_pseudo_vae_sbi_deg
echo "Phase 7 완료 ($(date))"
