#!/bin/bash
# ================================================================
# 새 데이터 추가 스크립트
# - fake: 1_Million_Fake_Faces (4만 장) → train 32,000 / valid 4,000 / test 4,000
# - real: FFHQ (4만 장)              → train 32,000 / valid 4,000 / test 4,000
# ================================================================

set -e  # 오류 발생 시 즉시 중단

DATA_DIR="/home/hyojin/ai_image_detection/data"
FAKE_SRC="/mnt/c/Users/user/Downloads/1_Million_Fake_Faces"
REAL_SRC="/mnt/c/Users/user/Downloads/Flickr-Faces-HQ_Dataset_(FFHQ)-256-4/Flickr-Faces-HQ Dataset (FFHQ)-256-4"

TRAIN_FAKE=32000
VALID_FAKE=4000
TEST_FAKE=4000

TRAIN_REAL=32000
VALID_REAL=4000
TEST_REAL=4000

# ────────────────────────────────────────────
# 1단계: 소스 파일 목록 수집
# ────────────────────────────────────────────
echo "=== 소스 파일 목록 수집 중... ==="

# fake: 00~03 폴더 안의 이미지 전부 수집 후 셔플
mapfile -t FAKE_FILES < <(find "$FAKE_SRC" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | shuf --random-source=/dev/urandom)
echo "fake 소스 총 이미지: ${#FAKE_FILES[@]}장"

# real: FFHQ 폴더 안의 이미지 전부 수집 후 셔플
mapfile -t REAL_FILES < <(find "$REAL_SRC" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | shuf --random-source=/dev/urandom)
echo "real 소스 총 이미지: ${#REAL_FILES[@]}장"

# ────────────────────────────────────────────
# 2단계: 수량 검증
# ────────────────────────────────────────────
FAKE_NEEDED=$((TRAIN_FAKE + VALID_FAKE + TEST_FAKE))
REAL_NEEDED=$((TRAIN_REAL + VALID_REAL + TEST_REAL))

if [ ${#FAKE_FILES[@]} -lt $FAKE_NEEDED ]; then
    echo "❌ fake 이미지 부족: ${#FAKE_FILES[@]}장 < 필요 ${FAKE_NEEDED}장"
    exit 1
fi

if [ ${#REAL_FILES[@]} -lt $REAL_NEEDED ]; then
    echo "❌ real 이미지 부족: ${#REAL_FILES[@]}장 < 필요 ${REAL_NEEDED}장"
    exit 1
fi

echo "✅ 수량 검증 통과"
echo ""

# ────────────────────────────────────────────
# 3단계: 복사 함수 정의
# ────────────────────────────────────────────
copy_files() {
    local -n FILES=$1      # 파일 목록 배열 (nameref)
    local START=$2         # 시작 인덱스
    local COUNT=$3         # 복사할 개수
    local DEST=$4          # 목적지 폴더

    mkdir -p "$DEST"
    local END=$((START + COUNT))
    local COPIED=0

    for (( i=START; i<END; i++ )); do
        cp "${FILES[$i]}" "$DEST/"
        COPIED=$((COPIED + 1))
        if (( COPIED % 5000 == 0 )); then
            echo "  ... $COPIED / $COUNT 복사 중"
        fi
    done
    echo "  ✅ $DEST → $COUNT장 복사 완료"
}

# ────────────────────────────────────────────
# 4단계: fake 복사 (train → valid → test 순서로 인덱스 이동)
# ────────────────────────────────────────────
echo "=== fake 이미지 복사 중 ==="
copy_files FAKE_FILES 0              $TRAIN_FAKE "$DATA_DIR/train/fake"
copy_files FAKE_FILES $TRAIN_FAKE    $VALID_FAKE "$DATA_DIR/valid/fake"
copy_files FAKE_FILES $((TRAIN_FAKE + VALID_FAKE)) $TEST_FAKE "$DATA_DIR/test/fake"

echo ""

# ────────────────────────────────────────────
# 5단계: real 복사
# ────────────────────────────────────────────
echo "=== real 이미지 복사 중 ==="
copy_files REAL_FILES 0              $TRAIN_REAL "$DATA_DIR/train/real"
copy_files REAL_FILES $TRAIN_REAL    $VALID_REAL "$DATA_DIR/valid/real"
copy_files REAL_FILES $((TRAIN_REAL + VALID_REAL)) $TEST_REAL "$DATA_DIR/test/real"

echo ""

# ────────────────────────────────────────────
# 6단계: 최종 결과 확인
# ────────────────────────────────────────────
echo "=== 최종 데이터 현황 ==="
for split in train valid test; do
    for cls in fake real; do
        folder="$DATA_DIR/$split/$cls"
        count=$(find "$folder" -type f | wc -l)
        echo "  $split/$cls: ${count}장"
    done
done

echo ""
echo "🎉 완료! 이제 노트북을 실행하세요."
