#!/bin/bash
# Phase 5 실행: 기존 데이터 다양성·증강 실험
#   1) Phase 3 기준선(best.pt)을 데모 서버와 같은 얼굴 crop 보기로 다시 평가
#   2) --aug strong (얼굴 crop 무작위화 + SNS 열화 + 생성기 균형 샘플링) 학습
# 사전 준비: data/manifests/face_boxes.csv (scripts/50_detect_faces.py)
# 사용 (저장소 루트): nohup bash scripts/run_phase5.sh > outputs/exp2/phase5/run.log 2>&1 &
# 이미 끝난 실험(metrics.csv가 있는 폴더)은 건너뜁니다.
set -e
cd "$(dirname "$0")/.."
PY=.venv-exp2/bin/python
mkdir -p outputs/exp2/phase5

echo "=== 동작 확인 (smoke) $(date) ==="
$PY scripts/32_train_cnn.py --model convnext_tiny --holdout diffusion --aug strong --limit 200 --epochs 1

for holdout in diffusion gan manip; do
    for model in convnext_tiny efficientnetv2s; do
        if [ ! -f "outputs/exp2/$model/$holdout/crop_view/metrics.csv" ]; then
            echo "=== 기준선 crop 보기 평가: $model / $holdout ($(date)) ==="
            $PY scripts/32_train_cnn.py --model "$model" --holdout "$holdout" --eval-only
        fi
    done
done

# 최근 AI 이미지와 가장 가까운 diffusion 을 먼저, 모델은 Phase 3 에서 diffusion OOD 가 가장 좋았던 ConvNeXt-T 부터
for model in convnext_tiny efficientnetv2s; do
    for holdout in diffusion gan manip; do
        if [ -f "outputs/exp2/${model}_aug/$holdout/metrics.csv" ]; then
            echo "[skip] ${model}_aug/$holdout"
            continue
        fi
        echo "=== ${model} --aug strong / $holdout ($(date)) ==="
        $PY scripts/32_train_cnn.py --model "$model" --holdout "$holdout" --aug strong
    done
done
echo "Phase 5 완료 ($(date))"
