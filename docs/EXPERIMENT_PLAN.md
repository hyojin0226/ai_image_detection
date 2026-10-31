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

### 2.4 Phase 2 준비 중 확인한 중복 문제 (2026-10-31)

같은 사진이 **여러 소스와 여러 split에 중복**으로 들어가 있습니다. 같은 ID/파일명끼리 dHash 거리를 재 보면 중앙값 0~1이고(다른 사진끼리는 약 120), 100쌍 모두 같은 사진이었습니다.

- **Real:** stylegan(140k real), flux_sdxl(K130 real), ffhq_1mff(FFHQ 256), gravex real의 숫자 파일명은 모두 **FFHQ 원본(ID 00000~69999)의 복사본**입니다. 전체 real 183,700장 중 FFHQ 사진은 고유 70,000장뿐입니다.
- **Fake:** stylegan fake(140k)와 ffhq_1mff fake는 모두 **1M Fake Faces(StyleGAN)**에서 왔고, gravex fake 중 17,832장도 같은 이미지입니다. fake 183,697장 중 고유 이미지는 154,316장입니다.
- **1차 실험 test에 미친 영향:**

| 1차 test | 이미지 | 같은 사진이 train에도 있음 |
|---|---:|---:|
| real | 13,438 | **9,762 (73%)** |
| fake | 13,438 | **4,329 (32%)** |

  1차 ID Test 97.8%와 Phase 1 결과의 절대 수치는 **외운 사진이 섞인 값이라 실제보다 높게 나왔을 수 있습니다.** 변환에 따른 상대적인 변화는 여전히 참고할 수 있습니다.
- **GRAVEX 구성 (fake 46,000장 기준, 디스크 전체 split):** StyleGAN(1MFF) 17,832 / DFDC 8,688 / CelebDF 4,987 / FaceForensics++ 4,626 / SD 3,144 / aug 2,042 / 기타 생성 약 1,100.
  real은 FFHQ 31,350 / 영상 프레임(real_·CelebDF·DFDC) 약 14,300입니다.
- **결론:** "데이터셋(source)" 단위로 하나를 빼는 기존 LOGO-A/B/C는 같은 생성기와 같은 사진이 다른 소스에 남아 있어서 진짜 OOD가 되지 않습니다. 생성기 계열 단위로 다시 나누고, 사진 단위 중복 제거가 필요합니다(Phase 2 설계 변경 참고).

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
python scripts/31_train_clip_probe.py --config configs/logo_diffusion.yaml
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

#### Phase 1 결과 (2026-10-31)

실행: `source scripts/tf_gpu_env.sh && python scripts/10_eval_robustness.py && python scripts/11_gradcam_gravex_fn.py`
리포트: `notebooks/01_robustness_report.ipynb` (`scripts/12_build_robustness_report.py`로 생성)

- 변환은 256x256으로 맞춘 뒤 적용했습니다. 확인해 보니 **test 이미지는 모든 소스가 이미 256x256**이라 해상도 차이에 따른 지름길은 없습니다.
- 변환 없음(none)의 정확도는 97.78%로 1차(97.79%)와 거의 같습니다. 차이는 이미지 디코딩·리사이즈 라이브러리(PIL vs TF) 차이입니다.

| 변환 | 정확도 | AUC | Fake Recall | Real Recall |
|---|---:|---:|---:|---:|
| none | 97.78% | 0.9983 | 96.43% | 99.12% |
| JPEG q90 / q70 / q50 | 97.49 / 97.32 / **96.60%** | 0.9977 / 0.9972 / 0.9957 | 96.44 / 96.22 / 96.06% | 98.55 / 98.41 / **97.14%** |
| 리사이즈 0.75× / 0.5× | 97.74 / **96.99%** | 0.9980 / 0.9965 | 96.66 / 96.10% | 98.82 / 97.89% |
| 블러 σ0.5 / σ1.0 | 97.78 / 97.34% | 0.9981 / 0.9974 | 96.58 / 96.48% | 98.97 / 98.19% |

- **결론 1: 압축·축소에 꽤 강건합니다.** 가장 센 JPEG q50에서도 정확도가 1.2%p만 떨어집니다. 지름길 학습을 했다는 강한 신호는 없습니다.
- **결론 2: 떨어지는 쪽은 Real입니다.** 압축·축소가 세질수록 Fake recall은 거의 그대로이고 **Real을 Fake로 오판하는 비율이 늘어납니다**(q50에서 real recall 99.1% → 97.1%). SNS에 올라간 진짜 사진이 FAKE로 뜰 위험과 같은 방향입니다.
- **결론 3: GRAVEX 약점의 원인은 "부분 조작" 이미지입니다.** GRAVEX fake에는 전체 생성 얼굴뿐 아니라 실제 영상에서 얼굴만 바꾼 이미지가 섞여 있습니다. 원본 파일명 기준 출처별 놓친 비율:

| GRAVEX fake 출처 | 이미지 | 놓침 | 놓친 비율 |
|---|---:|---:|---:|
| 생성 얼굴 (SD/StyleGAN 등) | 1,381 | 13 | **0.9%** |
| DFDC (영상 딥페이크) | 555 | 48 | 8.6% |
| CelebDF (영상 딥페이크) | 314 | 58 | 18.5% |
| aug | 121 | 22 | 18.2% |
| **FaceForensics++** (FaceSwap, NeuralTextures, Deepfakes, FaceShifter, Face2Face) | 291 | **286** | **98.3%** |

  놓친 428장 중 286장(67%)이 FaceForensics++ 조작 이미지입니다. 이 모델은 **"AI가 통째로 만든 얼굴"은 거의 다 잡지만, "실제 영상에서 얼굴 일부만 바꾼 딥페이크"는 사실상 못 잡습니다.** Grad-CAM에서도 놓친 사례는 저해상도·블록 노이즈가 많은 영상 프레임이고, 히트맵이 얼굴이 아닌 배경·가장자리에 퍼져 있습니다.
- **Phase 2~3에 주는 시사점:** GRAVEX는 성격이 다른 두 과제(전체 생성 / 부분 조작)가 섞인 소스입니다. LOGO 결과를 해석할 때 GRAVEX를 "생성 얼굴"과 "부분 조작"으로 나눠서 보고하는 것을 권장합니다.

### Phase 2. Leave-One-Generator-Out 데이터 분할
**질문:** 처음 보는 생성기도 잡아낼 수 있는가?

> **설계 변경 (2026-10-31):** 처음 계획한 소스 단위 LOGO-A/B/C는 같은 생성기·같은 사진이 다른 소스에 중복되어 있어(2.4절) 진짜 OOD가 되지 않습니다. **생성기 계열 단위**로 바꾸고 **사진 단위 중복 제거**를 적용했습니다.

생성기 계열 3종 중 1종을 학습에서 완전히 빼고, 빠진 계열로만 OOD 테스트합니다. 모델은 계열을 모르고 항상 real/fake 이진 분류만 합니다(계열은 데이터를 나누고 결과를 보고할 때만 사용).

| 계열 | 포함 이미지 (중복 제거 후 고유 장수) |
|---|---|
| gan | 1M Fake Faces(StyleGAN): stylegan·ffhq_1mff fake 전체 + GRAVEX 일부 (87,340) |
| diffusion | FLUX_DEV·FLUX_PRO·SDXL·GRAVEX SD (45,546) |
| manip | 영상 얼굴 조작: FaceForensics++·CelebDF·DFDC (18,301 / 영상 묶음 5,031) |
| real | FFHQ 사진 70,000 + 영상 프레임 14,304 |
| 제외 | 생성기를 특정할 수 없는 aug_ai·1024_man_men·ai_detect 등 3,473 |

| 실험 | 학습 fake | 학습 real | OOD 테스트 |
|---|---|---|---|
| LOGO-gan | diffusion + manip | FFHQ + 영상 프레임 | gan fake vs 학습에 안 쓴 FFHQ |
| LOGO-diffusion | gan + manip | FFHQ + 영상 프레임 | diffusion fake vs 학습에 안 쓴 FFHQ |
| LOGO-manip | gan + diffusion | **FFHQ만** | manip fake vs 영상 프레임 real (학습에 전혀 안 씀) |

- **분할 규칙:** 같은 사진(FFHQ ID, 1MFF 파일명)은 한 번만 사용하고, 같은 묶음(같은 사진, 같은 영상·대상 영상의 프레임)은 해시로 같은 bucket(train/valid/test = 70/10/20)에 넣습니다. test bucket의 FFHQ는 test_id와 test_ood가 겹치지 않게 나눕니다.
- **크기:** 실험마다 train 30,000 / valid 5,000 / test_id 3,000 / test_ood 10,000 (클래스당). train은 계열 비율을 유지해 샘플링합니다.
- **누수 검사:** 스크립트가 실행할 때마다 train·valid와 test_id·test_ood 사이에 같은 사진·같은 묶음이 없는지, 빼야 할 계열이 학습에 없는지 확인합니다(3종 모두 통과).
- 스크립트: `scripts/20_build_logo_splits.py` → `data/manifests/logo_{gan,diffusion,manip}.csv`, 구성표 `outputs/exp2/logo_splits/split_summary.csv`

#### 대조 실험: 파일 형식만으로 real/fake를 맞힐 수 있는가

real이 대부분 같은 FFHQ 사진이라 원래 계획한 "real 출처 맞히기"는 의미가 없어서, **이미지 내용 없이 파일 정보(PNG 여부, JPEG 양자화 테이블, 픽셀당 파일 크기, 크기)만으로** real/fake를 맞히는 분류기로 대체했습니다(`scripts/21_shortcut_probe.py`).

| 데이터 | test_id 정확도 / AUC | test_ood 정확도 / AUC |
|---|---|---|
| 1차 데이터 | 67.8% / 0.772 | – |
| LOGO-gan | **89.6% / 0.970** | 50.5% / 0.528 |
| LOGO-diffusion | 69.0% / 0.752 | **71.3% / 0.831** |
| LOGO-manip | 73.9% / 0.854 | 51.7% / 0.556 |

- 원인: ffhq_1mff는 real이 PNG, fake가 JPEG이고, FLUX·SDXL fake는 real보다 낮은 JPEG 품질(양자화 평균 약 11 vs 5)로 저장되어 있습니다.
- **파일 압축 정보만으로 최대 90%까지 맞힐 수 있으므로**, 모델이 진짜/가짜 대신 압축 차이를 배울 위험이 있습니다. LOGO-diffusion은 OOD에서도 71%라서, 정규화하지 않으면 OOD 성능이 실제보다 좋아 보일 수 있습니다.
- **Phase 3 규칙 추가:** 모든 이미지를 읽은 뒤 **같은 방식으로 JPEG 재인코딩**합니다. 학습 때는 real·fake 모두 무작위 품질(60~95), 평가 때는 고정 품질 90을 씁니다. 파일 형식 단서가 남지 않게 하기 위해서입니다. 이미 픽셀에 남은 이중 압축 흔적까지 완전히 지울 수는 없으므로, 결과 해석 시 한계로 적습니다.

### Phase 3. 모델 비교
| 모델 | 방식 | 선택 이유 |
|---|---|---|
| EfficientNetV2S | 전체 fine-tuning | 1차와 같은 모델 (기준선) |
| ConvNeXt-Tiny | 전체 fine-tuning | 최신 CNN 대표 모델 |
| CLIP ViT-L/14 + 선형 분류기 | 백본 고정, 특징 추출 후 Logistic Regression | UnivFD(Ojha et al., CVPR 2023)에서 처음 보는 생성기에 강하다고 보고된 방식 |

- CLIP은 특징을 한 번만 추출(`30_extract_clip_features.py`)하면 분류기 학습은 몇 분이면 끝나므로, LOGO 3종을 빠르게 반복할 수 있습니다.
- 3개 모델 × LOGO 3종 = **9개 학습**. 모델마다 threshold는 valid에서 정하고 테스트에 그대로 적용합니다(1차와 같은 규칙).
- 8GB VRAM 대응: CNN은 batch 32 + mixed precision, CLIP 특징 추출은 fp16으로 진행합니다.
- 산출물: `outputs/exp2/<model>/<logo_id>/` (metrics.csv, config.yaml, CNN은 training_history.csv). predictions.csv·best.pt는 용량 때문에 git에서 제외합니다.
- 실제 설정: CNN은 timm ImageNet 사전학습 가중치, AdamW(lr 1e-4) + OneCycle, bf16, 4 epoch, batch 32(학습 1회 약 17분). CLIP은 openai ViT-L/14 특징 + LogisticRegression(C는 valid AUC로 선택, LOGO 3종 모두 C=10).

#### Phase 3 결과 (2026-10-31, `scripts/40_summarize_results.py` → `outputs/exp2/summary/`)

| 모델 | ID Macro F1 | OOD gan (F1 / AUC) | OOD diffusion (F1 / AUC) | OOD manip (F1 / AUC) | OOD 평균 (F1 / AUC) | JPEG q50 Macro F1 |
|---|---|---|---|---|---|---|
| EfficientNetV2S | 0.978 | 0.333 / 0.474 | 0.342 / 0.692 | 0.414 / 0.442 | 0.363 / 0.536 | 0.971 |
| ConvNeXt-T | 0.982 | 0.333 / 0.377 | 0.337 / **0.871** | 0.423 / 0.433 | 0.365 / 0.560 | **0.978** |
| CLIP + Linear | 0.942 | 0.393 / **0.785** | 0.339 / 0.308 | **0.491** / 0.553 | 0.408 / 0.549 | 0.911 |

ID·JPEG q50은 LOGO 3종 평균입니다. OOD Macro F1 0.333은 "전부 REAL로 예측"한 것과 같은 값입니다.

**해석**
1. **학습에서 본 계열은 3개 모델 모두 잘 잡습니다.** ID Macro F1 0.94~0.98이고, JPEG q50에서도 1~3%p만 떨어집니다. CNN 2종이 CLIP보다 3~4%p 높습니다.
2. **처음 보는 계열은 3개 모델 모두 사실상 못 잡습니다.** valid threshold를 그대로 쓰면 OOD fake의 0~25%만 FAKE로 판정하고 대부분을 REAL로 넘깁니다. 어떤 모델도 LOGO 3종 모두에서 AUC가 0.5보다 확실히 높지 않았습니다.
3. **AUC가 0.87이어도 threshold가 맞지 않으면 쓸 수 없습니다.** ConvNeXt-T의 OOD diffusion은 순위로는 꽤 구분하지만(AUC 0.871), valid threshold에서는 fake recall이 0.4%입니다. OOD real의 95%를 REAL로 맞히도록 threshold를 다시 잡아도 fake recall은 48%입니다(`runs.csv`의 `ood_fake_recall_at_real95`). 새 계열마다 threshold 보정용 데이터가 필요하다는 뜻입니다.
4. **모델마다 강한 계열이 다릅니다.** CLIP은 처음 보는 GAN에 상대적으로 강하지만(AUC 0.785) diffusion에서는 오히려 반대로 판단합니다(AUC 0.308: FLUX·SDXL을 FFHQ 실사보다 더 진짜로 봄). CNN은 반대로 diffusion에서 높고 GAN에서는 0.5 아래입니다. UnivFD 논문의 "CLIP이 새 생성기에 강하다"는 결과는 이 엄격한 조건(중복 제거·형식 정규화)에서 일관되게 재현되지 않았습니다.
5. **OOD manip은 데이터셋별로 갈립니다** (`ood_by_generator.csv`). CNN은 FF++ 조작 6종을 valid threshold에서도 57~83% 잡지만, CelebDF·DFDC는 AUC 0.27~0.57로 거의 못 잡습니다. manip을 뺀 학습에는 영상 real이 없어서, 영상 출처(화질·압축) 차이가 판단에 섞였을 가능성이 큽니다.

**결론:** 이 데이터 안에서는 모델을 바꾸는 것만으로 일반화 문제를 풀 수 없습니다. 데모에 쓰는 최종 모델은 "학습한 생성기 계열에서만 신뢰할 수 있다"는 한계를 명시해야 하고, 실제로 쓰려면 탐지하려는 계열의 데이터를 학습에 넣어야 합니다(Phase 4 결정 사항).

### Phase 4. 결과 정리 & 데모 반영
- `scripts/40_summarize_results.py`로 아래 형태의 최종 표를 자동 생성합니다.

| 모델 | ID Macro F1 | OOD gan | OOD diffusion | OOD manip | OOD 평균 | JPEG q50 |
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
- [x] Phase 1 — 강건성 테스트 & GRAVEX 오탐 분석
- [x] Phase 2 — LOGO split 생성(계열 단위·중복 제거) & 파일 형식 대조 실험
- [x] Phase 3 — CLIP 선형 분류기 (LOGO 3종)
- [x] Phase 3 — EfficientNetV2S (LOGO 3종)
- [x] Phase 3 — ConvNeXt-T (LOGO 3종)
- [ ] Phase 4 — 최종 비교표(완료: `40_summarize_results.py`) & 리포트 노트북 & 데모 반영 결정
- [ ] (선택) Phase 5 — 데이터 보강

## 7. 역할 분담 (팀 논의 후 기입)

| Phase | 담당 | 비고 |
|---|---|---|
| 0–1 | | |
| 2 | | |
| 3 | | |
| 4 | | |
