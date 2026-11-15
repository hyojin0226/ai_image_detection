# AI 얼굴 이미지 탐지 크롬 확장 프로젝트 계획서

> 이 문서는 다른 채팅/세션에 그대로 전달해 작업을 이어가기 위한 요구사항 정리본입니다.

## 1. 한 줄 목표
웹페이지(특히 인스타그램)를 보다가 작은 버튼을 누르면, 페이지 안의 얼굴 이미지를 찾아 **REAL / FAKE(AI 생성)** 를 판정하고, **왜 그렇게 판단했는지(Grad-CAM)** 와 **실제 편집 흔적(ELA)** 을 함께 보여주는 크롬 확장 프로그램 + 이를 뒷받침하는 연구 실험.

## 2. 현재 보유 자산 (이미 있음)
- 학습 노트북: `efficientnetv2s_fake_face_FINAL_locaL.ipynb`
  - 모델: EfficientNetV2S (Keras), 입력 `256x256`, 모델 내부 전처리 사용 → 별도 `/255.` 하지 않음
  - 클래스 순서 고정: `fake=0, real=1`
  - 최종 모델 파일: `models/efficientnetv2s_fake_face_final.keras` (노트북의 `FINAL_MODEL_PATH`)
  - OOD 평가용 source 설정 존재 (`SOURCE_TEST_DIRS`, `OOD_SOURCE_NAMES = {"flux_sdxl"}`)
- 데이터 (WSL `\\wsl.localhost\ubuntu\home\hyojin\ai_image_detection\data`, `manifests/split_manifest.csv` 기준, 클래스별 장수)
  | source | train | valid | test | test_by_source |
  |---|---|---|---|---|
  | stylegan | 50,000 | 10,000 | 10,000 | 10,000 |
  | gravex | 36,800 | 4,600 | 4,600 | 4,600 |
  | flux_sdxl | 36,800 | 4,600 | 4,600 | 4,600 |
  - `add_dataset.sh`: 1 Million Fake Faces 40,000장 + FFHQ 256 40,000장 추가용 스크립트
  - 원본 소스는 `C:\Users\user\Downloads` (`FLUX_DEV`, `FLUX_PRO`, `SDXL.zip`, `1_Million_Fake_Faces`, FFHQ 등)
  - **새 데이터 수집은 하지 않기로 결정함**
- **알려진 문제:** 노트북은 `OOD_SOURCE_NAMES = {"flux_sdxl"}`인데 flux_sdxl은 train에 포함되어 있어 OOD가 아님. 진짜 OOD는 leave-one-generator-out(예: Flux로 학습, SDXL로 테스트) 재분할로 만들 예정 (실험 단계에서 처리)
- 노트북 `DATA_DIR`은 `C:\Users\user\Desktop\ai_image\data`를 가리킴 — 실제 학습 데이터 구성이 WSL 데이터와 같은지 확인 필요
- 작업 환경: Windows + WSL(Ubuntu, `/home/hyojin/ai_image_detection`), 시연은 노트북에서 해야 함

## 3. 최종 결과물 모습

### 3-1. 시연 흐름
1. 인스타그램 피드를 띄우면 화면 우측 하단에 작은 플로팅 버튼이 보임
2. 버튼 클릭 → 현재 보이는 이미지 스캔 → 얼굴 검출
3. 각 얼굴 이미지 위에 작은 배지 표시: `REAL 94%` / `FAKE 87%` / `판단 불가`(얼굴 없음·불확실)
4. 배지 클릭 → 사이드 패널에 다음을 보여줌
   - 잘라낸 얼굴 원본
   - **Grad-CAM 히트맵**: 모델이 판단 근거로 삼은 영역
   - **ELA 맵**: 재압축 오차 기반의 실제 편집 의심 영역
   - (선택) FFT 스펙트럼

### 3-2. 구성 요소
```
extension/            # 크롬 확장 (Manifest V3)
  manifest.json
  content.js          # 플로팅 버튼, 이미지 수집, 배지/패널 UI
  background.js       # 이미지 fetch(CORS 우회) → 서버 요청
  panel.css
server/               # 로컬 추론 서버 (FastAPI)
  app.py              # /predict, /explain 엔드포인트
  face.py             # 얼굴 검출 + crop (MediaPipe 등)
  gradcam.py
  ela.py
  requirements.txt
  run_server.bat      # 노트북에서 더블클릭 실행용
experiments/          # 연구 실험 노트북/스크립트
  robustness.ipynb    # SNS 열화(압축/리사이즈/필터) 실험
  edited_photos.ipynb # 보정 사진 오탐 + Grad-CAM vs 보정영역 겹침 실험
```

### 3-3. 연구(발표) 파트
- **실험 A: SNS 강건성**
  - test set에 JPEG 재압축(품질 90/75/50), 리사이즈, 인스타식 필터를 적용해 정확도/AUC 변화 측정
  - 압축·리사이즈 augmentation을 넣어 재학습한 뒤 성능 회복 정도 비교
- **실험 B: 보정 사진 오탐 분석**
  - real 사진 → 보정본(피부 보정, 눈 키우기, 리퀴파이, AI 뷰티 필터) 쌍 준비
  - 보정 강도별 FAKE 오탐률 측정
  - 원본-보정본 픽셀 차이로 "실제 보정 영역 마스크"를 만들고, Grad-CAM과의 겹침(IoU, Pointing Game) 측정
  - 같은 마스크로 ELA와도 비교
- **결론 형태 예:** "SNS 업로드 시 정확도 X% → Y%, augmentation 재학습으로 Z% 회복", "강한 피부 보정은 N% 확률로 FAKE 오탐, 이때 Grad-CAM은 보정 영역과 IoU M"

## 4. 진행 순서 (우선순위)

> **2026-10-31 결정:** 본 계획은 WSL 저장소의 `docs/EXPERIMENT_PLAN.md`(2차 실험)이며, 이 문서의 확장 프로그램은 그 계획의 **Phase 4 "데모 반영"** 단계에서 진행합니다.
> 데모에 넣을 모델은 Phase 3 비교 결과로 정해지므로, 실험(Phase 0~3)을 먼저 끝냅니다.
> - 실험 A(SNS 강건성)는 EXPERIMENT_PLAN Phase 1과 같으므로 그쪽으로 통합합니다.
> - 진짜 OOD 평가는 EXPERIMENT_PLAN Phase 2(LOGO)로 대체합니다.
> - 확장 MVP 작업은 중단했고, 일부 작성된 `server/`(app.py, face.py, model.py) 코드가 이 폴더에 남아 있습니다(미완성, 미검증).

EXPERIMENT_PLAN Phase 0~3 완료 후:
1. **MVP**: FastAPI 서버(`/predict`) + 확장 프로그램(버튼 → 이미지 수집 → 배지 표시). 얼굴 검출 포함. Phase 3에서 고른 모델 사용, 모델 경로·전처리는 설정 파일로 분리
2. Grad-CAM 패널 추가 (`/explain`)
3. ELA 패널 추가
4. 실험 B (보정 사진 + Grad-CAM 겹침)
5. 노트북 시연 환경 패키징 (`requirements.txt`, `run_server.bat`, 설치 안내)
6. (선택) 3-클래스(REAL / EDITED / GENERATED), FFT 패널, 불확실성 보정

## 5. 예상되는 어려운 부분 / 막힐 수 있는 지점

| 영역 | 문제 | 대응 |
|---|---|---|
| 학습-실사용 분포 차이 | FFHQ와 1M Fake Faces는 **정렬된 정면 얼굴 crop**인데, 인스타 사진은 각도·크기·배경이 제각각 | 얼굴 검출 후 FFHQ와 비슷하게 정렬/여백 crop, 256으로 리사이즈. 그래도 정확도는 test set보다 낮을 것으로 예상하고 발표에서 한계로 언급 |
| 오탐 | 인스타 real 사진은 대부분 보정·압축되어 있어 FAKE로 오탐될 가능성 높음 | 실험 A/B로 정량화 → 오히려 연구 결과로 활용. 시연 전 테스트 이미지 미리 선별 |
| 최신 생성기 | 학습 fake는 StyleGAN 계열이라 Midjourney/Flux/SDXL 얼굴에는 약할 수 있음 | `flux_sdxl` OOD 평가 결과를 같이 제시, "판단 불가" 임계값 설정 |
| 인스타 DOM | 클래스명이 자주 바뀜, lazy load, 캐러셀은 현재 장만 DOM에 존재 | `img` 태그 + 크기 기준 수집, `MutationObserver` 또는 재스캔 버튼 |
| CORS | 인스타 CDN 이미지를 content script에서 canvas로 읽으면 차단됨 | background service worker에서 fetch → 서버로 전송 |
| Mixed content / 권한 | https 페이지에서 `http://localhost` 호출 | background에서 요청하고 `host_permissions`에 localhost 등록 |
| Grad-CAM 해석 | 마지막 feature map이 저해상도(약 8x8)라 히트맵이 뭉개짐. 판단 근거일 뿐 "보정 위치 탐지기"가 아님 | 발표에서 정확히 구분해서 설명. 실험 B로 실제 겹침 정도를 수치로 검증 |
| ELA 한계 | 인스타가 전체를 재압축해서 ELA 신호가 약해짐 | 원본 파일로 하는 실험(B)과 인스타 시연을 구분. 시연에서는 참고용으로만 표시 |
| 보정 데이터 확보 | 원본-보정본 쌍 데이터가 공개된 게 적음 | FFHQ real 일부를 직접 보정 앱/스크립트(피부 블러, 리퀴파이)로 만들어 쌍 생성 |
| Mixed precision 모델 | 노트북에서 mixed_precision 사용 → 저장된 모델 로드/추론 시 dtype 이슈 가능 | 서버에서 float32 정책으로 로드 확인 |
| 노트북 성능 | GPU 없는 노트북에서 페이지당 이미지 수십 장 처리 시 수 초 소요 | 화면에 보이는 이미지만 처리, 배치 추론, 결과 캐시 |
| 재학습 시간 | 실험 A의 augmentation 재학습은 시간이 오래 걸림 | fine-tuning 단계만 짧게 재학습 또는 데이터 일부만 사용 |

## 6. 완료 기준
- 노트북에서 `run_server.bat` 실행 + 확장 로드만으로 인스타그램에서 버튼 → 배지 → Grad-CAM/ELA 패널이 동작
- 실험 A, B의 결과 표/그래프가 `results/`에 저장됨
- README에 설치·실행 방법과 한계점이 정리됨

## 7. 제약/원칙
- 1차 목표는 **시연 가능한 프로토타입**이며, 클라우드 배포나 브라우저 내 추론(TF.js)은 범위 밖
- 기존 학습 노트북의 전처리(256x256, 내부 전처리, 클래스 순서)를 서버 추론에서도 반드시 동일하게 유지
- 결과는 단정이 아닌 확률 + "판단 불가"로 표시
