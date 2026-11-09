#!/bin/bash
# Phase 6 실행: 의사 fake(SD VAE 재구성 / SBI) 추가 학습
#   Phase 3 기본 학습(--aug none) 위에 의사 fake 만 추가해 기준선과 비교합니다.
# 사전 준비: models_aux/sd-vae-ft-mse, data/manifests/face_boxes.csv
# 사용 (저장소 루트): bash scripts/run_phase6.sh > outputs/exp2/phase6/run.log 2>&1
# 이미 끝난 실험(metrics.csv 가 있는 폴더)은 건너뜁니다.
set -e
cd "$(dirname "$0")/.."
PY=.venv-exp2/bin/python
mkdir -p outputs/exp2/phase6

echo "=== VAE 재구성 (이미 만든 파일은 건너뜀) $(date) ==="
$PY scripts/51_vae_reconstruct.py

# 최근 AI 이미지와 가장 가까운 diffusion 부터, Phase 3 에서 diffusion OOD 가 가장 좋았던 ConvNeXt-T 로
for holdout in diffusion gan manip; do
    for pseudo in vae sbi vae+sbi; do
        name="convnext_tiny_pseudo_${pseudo/+/_}"
        if [ -f "outputs/exp2/$name/$holdout/metrics.csv" ]; then
            echo "[skip] $name/$holdout"
            continue
        fi
        echo "=== convnext_tiny --pseudo $pseudo / $holdout ($(date)) ==="
        $PY scripts/32_train_cnn.py --model convnext_tiny --holdout "$holdout" --pseudo "$pseudo"
    done
done
echo "Phase 6 완료 ($(date))"
