# 2차 실험 계획: 생성기 일반화 검증 & 모델 비교

> 작성일: 2026-10-31 · 작성: hyojin0226 · 상태: **Phase 0~7b·실험 B·리포트 완료, 배포 기본 = VAE+축소·블러(SBI 없음)** (최종 갱신 2026-11-09)
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
│   ├── perturb.py                # JPEG 재압축·리사이즈·블러 변환, random_degrade (Phase 7)
│   ├── augment.py                # Phase 5 얼굴 crop 보기·SNS 열화 증강
│   └── pseudo.py                 # Phase 6 의사 fake (VAE 재구성 행 추가, SBI)
├── scripts/                      # 실행 스크립트 (번호 = 실행 순서)
│   ├── organize_dataset.py       # (기존) 1차 데이터 구성
│   ├── 00_verify_baseline_metrics.py  # Phase 0 확인: 1차 지표 재계산 + 실제 폴더 구성
│   ├── 10_eval_robustness.py
│   ├── 20_build_logo_splits.py
│   ├── 30_extract_clip_features.py
│   ├── 31_train_clip_probe.py
│   ├── 32_train_cnn.py           # --aug strong / --pseudo / --degrade / --holdout all
│   ├── 23_build_final_split.py   # Phase 7 최종 split (logo_all.csv)
│   ├── 40_summarize_results.py
│   ├── 41_summarize_final.py     # Phase 5~7b 리포트용 표 (outputs/exp2/summary/report/)
│   ├── 42_build_final_report.py  # 02_generalization_report.ipynb 생성
│   ├── 50_detect_faces.py        # Phase 5 얼굴 박스 (Windows Python 으로 실행: WSL 에 MediaPipe GL 라이브러리 없음)
│   ├── 51_vae_reconstruct.py     # Phase 6 SD VAE 재구성 의사 fake
│   ├── 52_eval_robustness_final.py  # Phase 7 실험 A 축소판
│   └── run_phase{3,5,6,7,7b}.sh
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
- Phase 3 비교표: `scripts/40_summarize_results.py` → `outputs/exp2/summary/` (위 표).
- **데모 반영 결정 (2026-11-08):** Phase 3 모델 그대로는 처음 보는 생성기를 못 잡고, 실사용(인스타그램)에서 최근 AI 얼굴을 REAL 로 넘기는 문제가 확인됐습니다.
  새 생성기 데이터 수집은 시간·방법 제약으로 하지 않기로 하고, **기존 데이터만으로 일반화·강건성을 높이는 Phase 5~7** 을 추가로 진행한 뒤
  그 결과(ConvNeXt-T + 의사 fake + 축소·블러 증강)를 데모 서버에 넣었습니다.
- 데모: Windows 저장소 `server/`(FastAPI, CPU) + `extension/`(크롬 MV3). 얼굴 검출(MediaPipe) → crop x2.0 → JPEG 90 → ConvNeXt-T,
  판정 근거 패널(모델 입력 crop · Grad-CAM · ELA). 서버 예측은 WSL 평가 예측과 같은 이미지 120장씩(얼굴 crop / 전체) 판정이 100% 일치합니다.

> 번호 안내: 아래 Phase 5~7 은 원래 계획의 "(선택) 데이터 보강" 대신 진행한 추가 실험입니다. 결과 폴더 `outputs/exp2/phase{5,6,7}/`,
> 실행 스크립트 `scripts/run_phase{5,6,7}.sh` 와 번호를 맞췄습니다. 원래의 데이터 보강은 Phase 8(보류)로 옮겼습니다.

### Phase 5. 데이터 증강 (기존 데이터 다양성) — 효과 없음
**질문:** 구도·압축·보정 같은 방해 단서를 지우면 처음 보는 계열을 더 잘 잡는가?

- 먼저 모든 LOGO 이미지(143,264장)의 얼굴 박스를 데모와 같은 검출기로 계산했습니다(`scripts/50_detect_faces.py` → `data/manifests/face_boxes.csv`, 미검출 0.33%).
  얼굴이 이미지에서 차지하는 비율이 diffusion fake 0.36~0.40, FFHQ real 0.60 으로 달라 **구도 지름길**이 있습니다(Windows 저장소 `results/framing_shortcut/`).
- `--aug strong` (`src/augment.py`): 얼굴 crop 배율 무작위(1.4~3.0) + SNS 열화(축소·블러/샤픈·색·피부 스무딩·JPEG 40~95) + 생성기 균형 샘플링.
  평가는 기존 전체 이미지 보기와 함께 데모와 같은 **얼굴 crop x2.0 보기**(`crop_view/metrics.csv`)도 계산합니다(Phase 3 기준선도 같은 보기로 재평가).

| OOD AUC (얼굴 crop 보기) | 기준선 → 증강 |
|---|---|
| ConvNeXt-T diffusion / gan / manip | 0.872 → **0.683** / 0.376 → 0.502 / 0.473 → 0.442 |
| EfficientNetV2S diffusion / gan / manip | 0.728 → 0.678 / 0.473 → 0.480 / 0.458 → 0.438 |

- **결론:** 증강은 방해 단서를 지울 뿐 새 생성기의 흔적을 만들어 주지 못합니다. diffusion AUC 하락은 기준선 점수 일부가 구도 지름길에서 나왔음을 시사합니다.
  manip 의 fake recall 상승(0.12 → 0.75)은 threshold 이동일 뿐입니다(AUC < 0.5, real recall 0.86 → 0.20).
- 결과: `outputs/exp2/<model>_aug/`, 요약 `results/phase5_augmentation/` (Windows 저장소)

### Phase 6. 의사 fake (pseudo-fake) — diffusion 일반화 크게 개선
**질문:** real 이미지만으로 만든 가짜 예시로 처음 보는 생성기의 공통 흔적을 배울 수 있는가?

- **SD VAE 재구성** (Rajan et al., ICLR 2025 "Aligned Datasets"): train real 35,222장을 `stabilityai/sd-vae-ft-mse` 로 인코딩·디코딩(`scripts/51_vae_reconstruct.py`).
  원본과 같은 파일 형식·같은 JPEG 양자화 테이블로 저장해 형식 지름길을 막았습니다(PSNR 평균 29.6dB).
- **SBI** (Shiohara & Yamasaki, CVPR 2022, 단순화): 색·선명도·크기를 살짝 바꾼 자기 사본을 얼굴 박스 기반 타원 마스크로 다시 합성(학습 중 즉석 생성, `src/pseudo.py`).
- 종류마다 real train 장수의 50% 를 fake 로 추가. valid/test 는 진짜 데이터만. `32_train_cnn.py --pseudo vae|sbi|vae+sbi` (Phase 3 기본 증강 위).

| ConvNeXt-T, 얼굴 crop 보기 | OOD diffusion AUC | fake recall | real recall | OOD gan AUC | OOD manip AUC |
|---|---|---|---|---|---|
| 기준선 | 0.872 | 0.010 | 0.999 | 0.376 | 0.473 |
| + VAE | 0.952 | 0.209 | 0.998 | 0.537 | 0.517 |
| + SBI | 0.929 | 0.233 | 0.994 | 0.353 | 0.506 |
| **+ VAE + SBI** | **0.965** | **0.585** | 0.988 | 0.489 | 0.508 |

- 생성기별(VAE+SBI): FLUX-dev 0.770 → 0.945, FLUX-pro 0.743 → 0.942, SDXL 0.886 → 0.968, SD 0.958 → 0.981 (AUC).
  VAE 는 VAE 디코더를 쓰는 SD/SDXL 에, SBI 는 FLUX 에 더 잘 일반화해서 둘을 합쳤을 때 가장 좋았습니다.
- GAN 은 그대로(두 방법 모두 GAN 흔적을 겨냥하지 않음), manip 은 SBI 로 FF++ 6종만 AUC 0.52 → 0.72~0.76 (DFDC·CelebDF 는 그대로).
- 전체 이미지 보기에서는 개선이 작습니다(diffusion 0.871 → 0.889). 시드 1회 결과입니다.
- 결과: `outputs/exp2/convnext_tiny_pseudo_*/`, 요약 `results/phase6_pseudo_fake/`

### Phase 7. 최종 모델 & 강건성 (실험 A 재측정)
- **최종 split** (`scripts/23_build_final_split.py` → `data/manifests/logo_all.csv`): 세 LOGO split 의 train/valid/test_id 합집합(한 이미지는 어느 split 에서든 같은 role 임을 확인).
  train real 30k / fake 30k(diffusion 11.7k · gan 13.6k · manip 4.7k) + 의사 fake 30k. 학습에 안 쓴 test bucket 2만 장을 추가 held-out 테스트로 씁니다(계열을 빼지 않았으므로 OOD 아님).
- **실험 A 축소판** (`scripts/52_eval_robustness_final.py`, 재학습 없음): Phase 1 과 같은 변환을 서버와 같은 순서(열화 → crop → JPEG 90)로 적용, threshold 는 배포값 고정.
  VAE+SBI 최종 모델은 JPEG 에는 강했지만 **축소·블러에서 GAN fake 를 크게 놓쳤습니다**(축소 0.5x 에서 GAN fake recall 99.5% → 64.6%).
- 그래서 축소(0.4~1.0x)·블러 증강을 더해 재학습했습니다(`--degrade`, `src/perturb.random_degrade`, `scripts/run_phase7.sh`). Phase 5 와 달리 crop 배율 무작위화는 넣지 않았습니다.

| 얼굴 crop 보기 AUC (test_id / held-out) | VAE+SBI | VAE+SBI+축소·블러 (**배포**) |
|---|---|---|
| 변환 없음 | 0.996 / 0.986 | 0.996 / 0.986 |
| JPEG q50 | 0.990 / 0.972 | 0.992 / 0.978 |
| 축소 0.5x | 0.917 / 0.884 | **0.995 / 0.984** |
| 블러 σ1.0 | 0.952 / 0.936 | **0.996 / 0.985** |

- 축소 0.5x 에서 GAN fake recall 64.6% → 98.7%, threshold 바로 위에 몰린 real 비율 13% → 4%.
- **대가:** diffusion 을 뺀 LOGO 실험에서 OOD diffusion AUC(얼굴 crop) 0.965 → 0.951, valid threshold 에서 fake recall 0.585 → 0.245 (전체 이미지 보기는 0.889 → 0.921).
  실사용 이미지 대부분이 축소돼 있다는 점을 우선해 배포 모델로 골랐습니다.
- (2026-11-08 까지) 배포 모델: `outputs/exp2/convnext_tiny_pseudo_vae_sbi_deg/all/best.pt`, threshold 얼굴 crop 0.30 / 전체 0.34. 2026-11-09 Phase 7b 대안 모델로 교체. 영상 딥페이크(manip) fake recall 은 약 75~80% 로 가장 약합니다.
- 결과: `outputs/exp2/robustness_final/`, 요약 Windows 저장소 `results/experiment_a_final/`

### 실험 B. 보정 사진 오탐 & Grad-CAM/ELA 겹침 (Windows 저장소 `experiments/edited_photos.py`)
**질문:** 진짜 사진을 보정하면 FAKE 로 오판하는가? 그때 Grad-CAM·ELA 는 실제 보정 영역을 가리키는가?

- 데이터: 학습에 안 쓴 FFHQ real 400장(`logo_all.csv` test 역할). 카메라 JPEG(q90) → 스크립트 보정 → JPEG(q95) 저장. 대조군은 편집 없이 재저장.
  보정 4종: 피부 보정(bilateral, 강도 3단계) · 눈 키우기(국소 확대) · 얼굴 슬림(턱선 리퀴파이) · 뷰티 조합(피부+눈+슬림+톤, 2단계).
  얼굴 위치는 MediaPipe keypoint. 실제 앱의 AI 뷰티 필터(생성 모델)는 재현하지 못해 조합으로 근사했습니다. 피부 보정 "강"은 눈으로 봐도 과한 수준입니다.
- 판정·설명은 데모 서버 코드 그대로(얼굴 검출 → crop → JPEG 90 → 배포 모델, threshold 0.30).
- 보정 마스크 = 같은 crop 에서 |원본 − 보정본| > 2 (저장 전 비교). 겹침은 IoU, Pointing Game(최댓값 위치), 마스크 안 비중으로 재고,
  "항상 가운데를 가리키는" 기준선과 무작위 기준선(넓힌 마스크 넓이)을 함께 봅니다.

| 보정 (배포 모델, 400장) | FAKE 오판율 | Grad-CAM Pointing | ELA Pointing | 기준선 (가운데 / 무작위) |
|---|---|---|---|---|
| 없음 (재저장) | 0.8% | – | – | – |
| 피부 보정 약 / 중 / 강 | 1.8% / **57%** / **97%** | 0.17 / 0.91 / 0.94 | 0.51 / 0.95 / 0.95 | 중·강: 1.00 / 0.45~0.48 |
| 눈 키우기 약~강 | 1.5% | 0.34 ~ 0.45 | **0.68 ~ 0.72** | 0.22~0.55 / 0.07~0.09 |
| 얼굴 슬림 약~강 | 0.8 ~ 2.5% | 0.24 ~ 0.56 | **0.83 ~ 0.88** | 0.00 / 0.20~0.22 |
| 뷰티 조합 중 / 강 | **81%** / **99%** | 0.98 | 0.94 | 1.00 / 0.51~0.53 |

- **결론 1 — 피부를 매끄럽게 하는 보정은 FAKE 로 오판됩니다.** 모양을 바꾸는 보정(눈·슬림)은 거의 영향이 없습니다.
- **결론 2 — 원인은 SBI 의사 fake 입니다.** 같은 보정 사진 150장을 모델별로 판정(`results/experiment_b/model_ablation/fake_rate_by_model.csv`):

| 모델 (diffusion 제외 LOGO, 최종은 all) | 피부 보정 중 / 강 | 뷰티 조합 중 / 강 |
|---|---|---|
| 기준선 (의사 fake 없음) | 0% / 0% | 0% / 0% |
| + VAE | 3% / 9% | 3% / 5% |
| + SBI | 63% / 99% | 87% / 99% |
| + VAE + SBI | 69% / 100% | 90% / 99% |
| 최종 VAE+SBI / 배포(VAE+SBI+축소·블러) | 57% / 97% · 57% / 99% | 87% / 100% · 81% / 99% |

  SBI 는 "얼굴 영역에 살짝 흐린 사본을 합성"하는 방식이라 피부 보정과 같은 흔적을 fake 로 가르칩니다. FLUX 일반화 이득(Phase 6)과 맞바꾼 부작용입니다.
- **결론 3 — Grad-CAM 은 보정 위치를 알려주지 못합니다.** 넓은 보정(피부·뷰티)에서 Pointing 0.9 이상이지만 "항상 가운데" 기준선이 1.00 이라 의미가 없고,
  마스크 안 비중은 마스크 넓이의 약 1.4배로 약하게만 쏠립니다. 국소 보정(눈·슬림)은 무작위보다는 높지만 0.24~0.56 입니다.
- **결론 4 — ELA 는 원본 파일에서는 국소 보정 위치를 잘 짚습니다**(눈 0.68~0.72, 슬림 0.83~0.88, IoU@area 0.40~0.50 vs Grad-CAM 0.07~0.20).
  단, SNS 재압축을 거치지 않은 경우입니다. 데모 패널에서는 참고 지표로만 표시합니다.
- 대안 모델(SBI 없이 VAE + 축소·블러)은 아래 Phase 7b 참고.
- 결과: Windows 저장소 `results/experiment_b/` (`summary.csv`, `overlap_summary.csv`, `overlap_when_fake.csv`, `examples.jpg`, `model_ablation/`)

### Phase 7b. SBI 없는 대안 모델 (VAE + 축소·블러)
실험 B 원인 분석에 따라 SBI 를 빼고 Phase 7 과 같은 절차를 반복했습니다(`scripts/run_phase7b.sh`, `--pseudo vae --degrade`).

| 항목 (얼굴 crop 보기) | VAE+SBI+축소·블러 (이전 배포) | **VAE+축소·블러 (현재 배포)** |
|---|---|---|
| 같은 분포 test AUC / real recall | 0.996 / 0.968 | **0.998 / 0.981** |
| 추가 held-out AUC / fake recall / real recall | 0.986 / 0.940 / 0.918 | **0.990** / 0.938 / **0.954** |
| held-out AUC 축소 0.5x / 블러 σ1.0 | 0.984 / 0.985 | **0.989 / 0.990** |
| 보정 오탐: 피부 보정 중 / 강 (400장) | 57% / 97% | **4.8% / 16%** |
| 보정 오탐: 뷰티 조합 중 / 강 | 81% / 99% | **6.2% / 18%** |
| 보정 오탐: 눈·슬림 / 보정 없음 | ≤2.5% / 0.8% | ≤1.2% / 0.5% |
| LOGO-diffusion OOD AUC (얼굴 crop / 전체) | **0.951** / 0.921 | 0.925 / **0.926** |
| └ FLUX-dev / FLUX-pro / SDXL / SD | **0.941 / 0.941 / 0.951** / 0.981 | 0.881 / 0.804 / 0.934 / **0.987** |
| manip fake recall (held-out, 변환 없음 / 축소 0.5x) | 0.791 / **0.846** | 0.775 / 0.721 |
| threshold (얼굴 crop / 전체) | 0.30 / 0.34 | 0.16 / 0.16 |

- **맞바꿈:** SBI 는 처음 보는 FLUX 를 잡는 데 도움이 되지만(AUC +0.06~0.14), 피부 보정·뷰티 필터 real 을 대부분 FAKE 로 만듭니다.
  대안 모델은 학습한 계열·강건성·real 오탐에서 같거나 낫고, 처음 보는 FLUX 일반화만 약합니다.
- 실험 B 의 Grad-CAM 결론은 대안 모델에서도 같습니다(국소 보정 Pointing 0.09~0.24, 무작위 기준 0.07~0.22).
- **결정 (2026-11-09): 배포 기본 = VAE+축소·블러.** SNS 사진은 피부 보정이 흔해 real 오탐을 줄이는 쪽을 택했습니다. 처음 보는 FLUX 탐지가 약한 점은 한계로 명시합니다.
  데모 서버 기본 `convnext_tiny_fake_face.json`(이 모델, threshold 0.16), 이전 모델은 `convnext_tiny_fake_face_sbi.json` 으로 남겨 `FAKE_FACE_MODEL_CONFIG` 로 선택.
  서버 vs WSL 평가 예측 120장×2 보기 판정 99.2% 일치 (불일치 1장은 P(real) 0.161 vs 0.130, threshold 경계).
- 결과: `outputs/exp2/convnext_tiny_pseudo_vae_deg/`, `outputs/exp2/robustness_final/convnext_tiny_pseudo_vae_deg/`,
  Windows 저장소 `results/experiment_a_final/vae_degrade/`, `results/experiment_b/model_ablation/final_vae_deg/`, `deployed_vs_vae_deg.csv`

### (보류) Phase 8. 데이터 보강
- 테스트 전용으로 최신 생성기 이미지를 추가합니다(예: GenImage 공개 데이터셋의 Midjourney, DALL·E 계열).
- real 이미지 출처를 생성기와 상관없이 섞어서 지름길 학습 위험을 줄입니다.
- 2026-11-07 결정: 시간·방법 제약으로 새 데이터 수집은 하지 않고 Phase 5~7 로 대체했습니다.

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
- [x] Phase 4 — 최종 비교표(`40_summarize_results.py`) & 데모 반영 결정 (리포트 노트북은 남음)
- [x] Phase 5 — 데이터 증강 (효과 없음)
- [x] Phase 6 — 의사 fake (VAE 재구성 + SBI)
- [x] Phase 7 — 최종 모델 + 축소·블러 증강 + 강건성 재측정(실험 A 축소판) + 데모 서버 교체
- [x] 실험 B — 보정 사진 오탐 & Grad-CAM/ELA 겹침 (원인: SBI)
- [x] Phase 7b — SBI 없는 대안 모델(VAE + 축소·블러) 비교
- [x] 배포 기본 모델 결정 → VAE+축소·블러 (2026-11-09)
- [x] 리포트 노트북 — `notebooks/02_generalization_report.ipynb` (2026-11-09)
- [ ] (보류) Phase 8 — 데이터 보강

## 7. 역할 분담 (팀 논의 후 기입)

| Phase | 담당 | 비고 |
|---|---|---|
| 0–1 | | |
| 2 | | |
| 3 | | |
| 4 | | |
