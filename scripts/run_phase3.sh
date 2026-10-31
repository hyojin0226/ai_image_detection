#!/bin/bash
# Phase 3 전체 실행: CLIP 특징 추출 → CLIP 선형 분류기 → CNN 2종 × LOGO 3종
# 사용 (저장소 루트): nohup bash scripts/run_phase3.sh > outputs/exp2/phase3_run.log 2>&1 &
# 이미 끝난 실험(metrics.csv가 있는 폴더)은 건너뜁니다.
set -e
cd "$(dirname "$0")/.."
PY=.venv-exp2/bin/python

$PY scripts/30_extract_clip_features.py
$PY scripts/31_train_clip_probe.py

for holdout in gan diffusion manip; do
    for model in efficientnetv2s convnext_tiny; do
        if [ -f "outputs/exp2/$model/$holdout/metrics.csv" ]; then
            echo "[skip] $model/$holdout"
            continue
        fi
        echo "=== $model / $holdout ($(date)) ==="
        $PY scripts/32_train_cnn.py --model "$model" --holdout "$holdout"
    done
done
echo "Phase 3 완료 ($(date))"
