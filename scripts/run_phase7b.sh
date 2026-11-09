#!/bin/bash
# Phase 7b: 실험 B 에서 SBI 의사 fake 가 피부 보정 real 을 FAKE 로 만드는 원인으로 확인됨
#   -> SBI 없이 VAE 재구성 + 축소·블러 증강으로 같은 절차를 반복해 대안 모델을 만든다.
#   1) diffusion 제외 LOGO 실험 (OOD 이득 확인)  2) 최종 모델(all)  3) 실험 A 축소판
# 사용 (저장소 루트): bash scripts/run_phase7b.sh > outputs/exp2/phase7b/run.log 2>&1
set -e
cd "$(dirname "$0")/.."
PY=.venv-exp2/bin/python
mkdir -p outputs/exp2/phase7b
for holdout in diffusion all; do
    if [ -f "outputs/exp2/convnext_tiny_pseudo_vae_deg/$holdout/metrics.csv" ]; then echo "[skip] $holdout"; continue; fi
    echo "=== --pseudo vae --degrade / $holdout ($(date)) ==="
    $PY scripts/32_train_cnn.py --model convnext_tiny --holdout "$holdout" --pseudo vae --degrade
done
echo "=== 강건성 평가 ($(date)) ==="
$PY scripts/52_eval_robustness_final.py --run convnext_tiny_pseudo_vae_deg
echo "Phase 7b 완료 ($(date))"
