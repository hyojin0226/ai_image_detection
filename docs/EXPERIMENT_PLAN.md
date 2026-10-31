# 2차 실험 계획: 생성기 일반화 검증 & 모델 비교

> 작성일: 2026-10-31 · 작성: hyojin0226 · 상태: **초안 (팀 확인 대기)**
> 이 문서는 1차 실험(EfficientNetV2S) 결과를 바탕으로 다음에 무엇을, 왜, 어디서, 어떻게 할지 정리한 팀 공유용 계획서입니다.

---

## 1. 한눈에 보기

| 항목 | 내용 |
|---|---|
| 1차 결과 | EfficientNetV2S, ID Test 정확도 **97.8%** / AUC **0.998** |
| 발견한 문제 | "OOD"로 표기한 FLUX·SDXL이 실제로는 학습에 포함됨 → **진짜 일반화 성능은 아직 측정되지 않음** |
| 2차 목표 | ① 처음 보는 생성기에 대한 진짜 일반화 성능 측정 ② CNN vs CLIP 모델 비교 ③ 실사용 환경(압축·리사이즈) 강건성 확인 |
| 작업 위치 | WSL Ubuntu `~/ai_image_detection` (데이터·GPU·1차 모델이 모두 여기 있음) |
| 브랜치 | `feature/main-model-hyojin`에서 분기 → `feature/generalization-study` |

---

## 2. 배경: 1차 실험 결과와 한계

### 2.1 1차 결과 (threshold = 0.24, valid Macro F1 기준 선정)

| 테스트셋 | 이미지 수 | 정확도 | ROC-AUC | Fake Recall |
|---|---:|---:|---:|---:|
| ID Test (통합) | 26,876 | 97.8% | 0.998 | 96.5% |
| GRAVEX | 9,200 | **91.7%** | 0.984 | **85.2%** |
| StyleGAN (140k) | 20,000 | 99.5% | 0.9999 | 99.5% |
| FLUX·SDXL | 9,200 | 99.6% | 0.9999 | 99.6% |

결과 원본: `outputs/efficientnetv2s_experiment_01/results/evaluation_summary.csv`

### 2.2 한계 (2차 실험을 하는 이유)

1. **FLUX·SDXL은 OOD가 아니었음.**
   `data/manifests/split_manifest.csv` 기준으로 flux_sdxl이 train에 클래스별 36,800장씩 포함되어 있습니다.
   따라서 99.6%는 "처음 보는 생성기" 성능이 아닙니다.
2. **소스별 테스트셋이 ID Test의 부분집합임.**
   `test_by_source/*`가 test split에서 복사된 것이라 독립적인 평가셋이 아닙니다.
3. **지름길 학습 위험.**
   생성기마다 짝지어진 real 이미지의 출처(FFHQ / GRAVEX real / K130 real)가 다릅니다.
   모델이 "진짜/가짜" 대신 해상도·JPEG 압축 흔적처럼 "어느 데이터셋에서 왔는지"를 배웠을 가능성이 있습니다.
4. **GRAVEX가 가장 약함.** Fake 이미지의 약 15%를 Real로 놓칩니다.

### 2.3 Phase 0에서 추가로 확인한 사항 (2026-10-31)

`scripts/00_verify_baseline_metrics.py`로 확인했습니다. 1차 지표는 `src/metrics.py`로 다시 계산해도 **완전히 일치**합니다(최대 오차 1e-16).
그런데 **실제 `data/` 폴더 구성이 `split_manifest.csv`와 다릅니다.**

| split | source | fake | real | manifest 기준 |
|---|---|---:|---:|---|
| train | stylegan | 50,000 | 50,000 | 일치 |
| train | gravex | 36,800 | 36,800 | 일치 |
| train | flux_sdxl | 36,800 | 36,800 | 일치 |
| train | **ffhq_1mff** (접두어 없음) | 31,997 | 32,000 | **manifest에 없음** |
| valid | stylegan / gravex / flux_sdxl / ffhq_1mff | 6,298 / 2,918 / 2,923 / 2,523 | 4,734 / 4,599 / 1,329 / 4,000 | manifest는 10,000 / 4,600 / 4,600 / – |
| test | stylegan / gravex / flux_sdxl / ffhq_1mff | 5,768 / 2,688 / 2,679 / 2,303 | 4,119 / 4,599 / 720 / 4,000 | manifest는 10,000 / 4,600 / 4,600 / – |

- **4번째 소스가 학습에 들어가 있음:** `add_dataset.sh`로 추가한 1M Fake Faces(fake) + FFHQ(real)가 접두어 없이 섞여 있습니다. `src/data.py`에서는 `ffhq_1mff`로 부릅니다. 1M Fake Faces도 StyleGAN 계열이라, LOGO-B(StyleGAN 제외)를 할 때 **`ffhq_1mff`도 함께 빼야** 진짜 OOD가 됩니다.
- **valid/test가 manifest 이후 조정됨:** 클래스 합계는 맞췄지만, 소스별 fake/real 비율이 크게 다릅니다(예: test flux_sdxl은 fake 2,679 / real 720). 이 상태로는 소스별 정확도에서 real 출처 효과가 섞입니다.
- **소스별 테스트셋(`test_by_source`)은 조정 전 test에서 복사된 것**이라 현재 ID test와 구성이 다릅니다. split 간 완전 중복 파일은 없습니다(`cross_split_exact_duplicates.csv`가 비어 있음).
- **세부 생성기 구분이 가능함:** flux_sdxl fake는 원본 경로로 FLUX_DEV / FLUX_PRO / SDXL을 나눌 수 있습니다(train 기준 4,210 / 1,858 / 30,732장). SDXL이 대부분입니다.
- **조치:** Phase 2의 LOGO split은 manifest가 아니라 `src.data.scan_data_dir()`(실제 폴더 기준)로 만듭니다. real 출처(FFHQ 계열이 stylegan·flux_sdxl·ffhq_1mff에 모두 있음)가 겹칠 수 있으므로, 출처 판별 대조 실험에서 함께 확인합니다.

---

## 3. 코드 작업 위치와 구조

### 3.1 원칙

- **모든 작업은 WSL 저장소에서 진행합니다.** 데이터(약 37만 장), GPU(RTX 4060 Ti 8GB), 1차 학습 결과가 모두 WSL에 있습니다.
  Windows 네이티브 TensorFlow에서는 GPU를 쓸 수 없습니다.
- **기존 노트북은 수정하지 않습니다.** `efficientnetv2s_fake_face_FINAL_locaL*.ipynb`는 1차 실험(baseline) 기록으로 그대로 둡니다.
- **실행 코드는 `.py`, 결과 확인·설명은 `.ipynb`로 나눕니다.**

| 구분 | 형식 | 이유 |
|---|---|---|
| 학습·평가·데이터 생성 | `.py` 스크립트 | 몇 시간씩 걸리는 학습을 백그라운드(`nohup`)로 돌릴 수 있음. git diff가 읽기 쉬움. 팀원 누구나 같은 명령어로 같은 결과를 재현 가능 |
| 공통 함수 | `src/` 모듈 | 데이터 로딩·평가 지표를 한 곳에서 관리해서 모델 간 비교 조건을 동일하게 유지 |
| 결과 확인·발표 자료 | `.ipynb` 노트북 | 저장된 결과 CSV/그림을 읽어서 표·그래프로 보여주기만 함 (학습 코드 없음) |

### 3.2 폴더 구조 (추가되는 부분만)

```text
ai_image_detection/
├── docs/
│   └── EXPERIMENT_PLAN.md        # 이 문서
├── src/                          # 공통 모듈 (import해서 사용)
│   ├── config.py                 # 경로·클래스 매핑(fake=0, real=1)·시드
│   ├── data.py                   # 데이터셋 로딩, manifest 기반 split
│   ├── models.py                 # EfficientNetV2S / ConvNeXt-T / CLIP 선형 분류기
│   ├── metrics.py                # Accuracy, Macro F1, ROC-AUC, Fake P/R/F1
│   └── perturb.py                # JPEG 재압축·리사이즈·블러 변환
├── scripts/                      # 실행 스크립트 (번호 = 실행 순서)
│   ├── organize_dataset.py       # (기존) 1차 데이터 구성
│   ├── 00_verify_baseline_metrics.py  # Phase 0 확인: 1차 지표 재계산 + 실제 폴더 구성
│   ├── 10_eval_robustness.py
│   ├── 20_build_logo_splits.py
│   ├── 30_extract_clip_features.py
│   ├── 31_train_clip_probe.py
│   ├── 32_train_cnn.py
│   └── 40_summarize_results.py
├── configs/                      # 실험 설정 (yaml)
├── notebooks/                    # 결과 확인용
│   ├── 01_robustness_report.ipynb
│   └── 02_generalization_report.ipynb
├── outputs/
│   ├── efficientnetv2s_experiment_01/   # (기존) 1차 결과
│   └── exp2/                            # 2차 결과 (실험별 하위 폴더)
└── requirements-exp2.txt
```

- `data/`와 모델 가중치(`*.h5`, `*.keras`, `*.pt`)는 용량 문제로 **git에 올리지 않습니다.**
- 결과 **CSV와 그림(PNG)**은 git에 올려서 팀원이 바로 볼 수 있게 합니다.

### 3.3 실행 환경

- 기존 `.venv`(TensorFlow 2.21): 1차 모델 재평가(Phase 1)에 사용
- 신규 `.venv-exp2`(PyTorch + `timm` + `open_clip`): Phase 2~3에 사용
  - CLIP, ConvNeXt, EfficientNetV2S를 같은 프레임워크로 맞춰야 비교 조건이 공정해집니다.
- 모든 명령은 저장소 루트에서 실행합니다. 예시:

```bash
source .venv-exp2/bin/activate
python scripts/31_train_clip_probe.py --config configs/logo_holdout_flux_sdxl.yaml
```

---

## 4. 단계별 계획

### Phase 0. 준비
| 작업 | 산출물 | 완료 기준 |
|---|---|---|
| 브랜치 생성, 폴더 구조 생성 | `feature/generalization-study` | 브랜치 push 완료 |
| PyTorch 환경 구성 | `requirements-exp2.txt`, `.venv-exp2` | `torch.cuda.is_available() == True` |
| 공통 모듈 작성 | `src/*.py` | 1차 결과 CSV를 `metrics.py`로 다시 계산했을 때 수치가 일치 |

### Phase 1. 강건성 테스트 (재학습 없음)
**질문:** 1차 모델은 SNS 업로드처럼 압축·축소된 이미지에서도 잘 작동하는가? 압축만 바꿔도 성능이 크게 떨어진다면 지름길 학습을 했다는 신호입니다.

| 변환 | 강도 |
|---|---|
| JPEG 재압축 | quality 90 / 70 / 50 |
| 리사이즈(축소 후 복원) | 0.75× / 0.5× |
| 가우시안 블러 | σ = 0.5 / 1.0 |

- 스크립트: `scripts/10_eval_robustness.py` (1차 `.keras` 모델 사용)
- 추가 분석: GRAVEX에서 놓친 Fake 이미지 상위 사례 Grad-CAM (`gravex_false_negative_cases_threshold_024.csv` 활용)
- 산출물: `outputs/exp2/robustness/robustness_summary.csv`, `notebooks/01_robustness_report.ipynb`

### Phase 2. Leave-One-Generator-Out 데이터 분할
**질문:** 처음 보는 생성기도 잡아낼 수 있는가?

생성 소스 3종 중 1종을 학습에서 완전히 빼고, 빠진 소스로만 테스트합니다.

| 실험 | 학습 소스 | 테스트 소스 (진짜 OOD) |
|---|---|---|
| LOGO-A | StyleGAN + FLUX·SDXL | GRAVEX |
| LOGO-B | GRAVEX + FLUX·SDXL | StyleGAN |
| LOGO-C | GRAVEX + StyleGAN | FLUX·SDXL |

- 기존 `split_manifest.csv`를 기준으로 **파일 복사 없이** 목록(CSV)만 새로 만듭니다. 디스크 공간을 아끼고 split을 추적하기 쉽게 하기 위해서입니다.
- 학습 시간을 고려해 각 실험은 **클래스당 최대 30,000장**으로 샘플링합니다. 소스별 비율은 동일하게 맞춥니다.
- 지름길 학습 점검용 대조 실험: real 이미지끼리 "출처 맞히기" 분류기를 학습해 봅니다. 정확도가 높게 나오면 real 출처마다 쉽게 구별되는 차이(압축·해상도 등)가 있다는 뜻입니다.
- 스크립트: `scripts/20_build_logo_splits.py` → `data/manifests/logo_*.csv`

### Phase 3. 모델 비교
| 모델 | 방식 | 선택 이유 |
|---|---|---|
| EfficientNetV2S | 전체 fine-tuning | 1차와 같은 모델 (기준선) |
| ConvNeXt-Tiny | 전체 fine-tuning | 최신 CNN 대표 모델 |
| CLIP ViT-L/14 + 선형 분류기 | 백본 고정, 특징 추출 후 Logistic Regression | UnivFD(Ojha et al., CVPR 2023)에서 처음 보는 생성기에 강하다고 보고된 방식 |

- CLIP은 특징을 한 번만 추출(`30_extract_clip_features.py`)하면 분류기 학습은 몇 분이면 끝나므로, LOGO 3종을 빠르게 반복할 수 있습니다.
- 3개 모델 × LOGO 3종 = **9개 학습**. 모델마다 threshold는 valid에서 정하고 테스트에 그대로 적용합니다(1차와 같은 규칙).
- 8GB VRAM 대응: CNN은 batch 32 + mixed precision, CLIP 특징 추출은 fp16으로 진행합니다.
- 산출물: `outputs/exp2/<model>/<logo_id>/` (metrics.csv, predictions.csv, roc.png)

### Phase 4. 결과 정리 & 데모 반영
- `scripts/40_summarize_results.py`로 아래 형태의 최종 표를 자동 생성합니다.

| 모델 | ID Macro F1 | OOD GRAVEX | OOD StyleGAN | OOD FLUX·SDXL | OOD 평균 | JPEG q50 |
|---|---|---|---|---|---|---|
| EfficientNetV2S | | | | | | |
| ConvNeXt-T | | | | | | |
| CLIP + Linear | | | | | | |

- 가장 좋은 모델을 Streamlit/HTML 데모에 반영할지는 결과를 보고 팀에서 결정합니다.

### (선택) Phase 5. 데이터 보강
결과에 따라 필요하면 진행합니다.
- 테스트 전용으로 최신 생성기 이미지를 추가합니다(예: GenImage 공개 데이터셋의 Midjourney, DALL·E 계열).
- real 이미지 출처를 생성기와 상관없이 섞어서 지름길 학습 위험을 줄입니다.

---

## 5. 작업 규칙

- **라벨 고정:** `fake = 0`, `real = 1` (모델 출력 = real일 확률)
- **시드 고정:** `SEED = 42`
- **threshold:** 반드시 valid에서만 정하고, test에서 조정하지 않습니다.
- **커밋 단위:** Phase 단위로 커밋하고, 커밋 메시지에 Phase 번호를 붙입니다. 예: `[P1] JPEG 강건성 평가 추가`
- **결과 기록:** 각 실험 폴더에 `config.yaml`(사용한 설정)과 `metrics.csv`를 함께 저장해서 나중에 재현할 수 있게 합니다.

## 6. 진행 체크리스트

- [x] Phase 0 — 브랜치·환경·공통 모듈 (브랜치 push는 남음)
- [ ] Phase 1 — 강건성 테스트 & GRAVEX 오탐 분석
- [ ] Phase 2 — LOGO split 생성 & 출처 판별 대조 실험
- [ ] Phase 3 — CLIP 선형 분류기 (LOGO 3종)
- [ ] Phase 3 — EfficientNetV2S (LOGO 3종)
- [ ] Phase 3 — ConvNeXt-T (LOGO 3종)
- [ ] Phase 4 — 최종 비교표 & 리포트 노트북
- [ ] (선택) Phase 5 — 데이터 보강

## 7. 역할 분담 (팀 논의 후 기입)

| Phase | 담당 | 비고 |
|---|---|---|
| 0–1 | | |
| 2 | | |
| 3 | | |
| 4 | | |
