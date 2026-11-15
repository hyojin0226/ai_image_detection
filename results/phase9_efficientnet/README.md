# Phase 9 — EfficientNetV2S 최종 결과

WSL `scripts/43_summarize_phase9.py --tag p9 --final-run efficientnetv2s_pseudo_vae_deg_t4` 출력(`outputs/exp2/summary/phase9/`)과
강건성 평가(`scripts/52_eval_robustness_final.py`) 결과의 복사본입니다. 설계·해석은 WSL `docs/EXPERIMENT_PLAN.md` 의 Phase 9 절을 보세요.

| 파일 | 내용 |
|---|---|
| `tuning.csv` | B단계 학습 설정 튜닝 t0~t6. valid 보기 5종 평균 AUC·log loss(선택 기준) + 고른 뒤 확인한 test 지표 |
| `logo.csv` | C단계 LOGO 3종 × 의사 fake {없음, VAE, SBI, VAE+SBI} + Phase 3 기준선, 얼굴 crop / 전체 보기 |
| `diffusion_by_generator.csv` | diffusion 을 뺀 LOGO 실험의 생성기별 OOD AUC·fake recall (얼굴 crop) |
| `final_robustness.csv`, `final_robustness_by_family.csv` | 배포(VAE)·대안(VAE+SBI) 모델의 압축·축소·블러 강건성 (두 모델 합본) |
| `robustness_t4/`, `robustness_vae_sbi/` | 위 강건성 결과를 모델별로 나눈 원본 |

## 핵심 수치 (얼굴 crop 보기)

**배포 모델 t4** (`efficientnetv2s_pseudo_vae_deg_t4`, threshold 0.34)

| 평가 | AUC | 정확도 | fake recall | real recall |
|---|---|---|---|---|
| 같은 분포 test 6,000장 | 0.998 | 97.6% | 0.965 | 0.988 |
| 같은 분포 test, JPEG 50 | 0.997 | 96.9% | 0.971 | 0.967 |
| 학습에 안 쓴 추가 test 20,000장 | 0.994 | 95.5% | 0.937 | 0.973 |

**튜닝** (valid 평균 AUC): t0 0.9908 → t1 0.9945 → t3 0.9967 → **t4 0.9975** (t5 0.9972, t6 0.9957, t2 0.9927)

**처음 보는 계열 (LOGO OOD AUC)**

| 의사 fake | diffusion | gan | manip |
|---|---|---|---|
| Phase 3 기준선 | 0.728 | 0.473 | 0.458 |
| 없음 | 0.722 | 0.489 | 0.393 |
| VAE (배포와 같은 조합) | 0.876 | 0.530 | 0.429 |
| SBI | 0.907 | 0.383 | 0.527 |
| VAE + SBI | 0.935 | 0.466 | 0.507 |

**강건성** (추가 test AUC, 배포 모델): 원본 0.994 · JPEG 50 0.989 · 축소 0.5 0.992 · 블러 1.0 0.993. manip fake recall 0.77 안팎.
