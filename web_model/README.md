# 최종 모델 브라우저용 변환 (ONNX)

최종 EfficientNetV2S(Phase 9, `server/models/efficientnetv2s_fake_face.pt`)를 재학습 없이 onnxruntime-web 용 ONNX 로 바꾼다.

```bat
server\.venv\Scripts\python.exe -m pip install onnx onnxruntime onnxscript
server\.venv\Scripts\python.exe web_model\export_onnx.py
server\.venv\Scripts\python.exe web_model\verify_onnx.py --per-folder 150
```

| 파일 (`out/`) | 크기 | 용도 |
|---|---|---|
| `efficientnetv2s_fake_face.fp32.onnx` | 82 MB | 검증 기준 (커밋 안 함) |
| `efficientnetv2s_fake_face.onnx` | 42 MB | 배포용. 가중치만 fp16 저장, 연산은 fp32 (커밋 안 함) |
| `model_config.json` | – | 브라우저 전처리·기준값 |
| `verify_report.json` | – | PyTorch 대비 검증 결과 |

입력은 지금까지와 같은 (N,3,256,256) 얼굴 crop(JPEG 90 후 정규화). 모델은 320 입력으로 학습했으므로 256 → 320 확대를 그래프 안에서 한다.
출력: `logit` (N,1) → sigmoid = P(real), `cam_map` (N,10,10) → REAL 근거는 `relu(cam_map)`, FAKE 근거는 `relu(-cam_map)`.
EfficientNet 머리(bn2 특징맵 → 평균 풀링 → Linear)의 기울기는 모든 위치에서 `fc_w / (h·w)` 라서, 이 닫힌 식을 그래프에 넣어 브라우저에서 역전파 없이 Grad-CAM 을 낸다.

**256 → 320 확대는 PIL bicubic 을 비트 단위로 재현한다.** 처음에는 ONNX `Resize`(cubic, a=-0.5)를 썼는데,
PIL 은 가로 → uint8 → 세로 → uint8 로 22비트 정수 계수를 써서 픽셀이 ±1 씩 달라졌고, 이 탐지 모델은 그 차이에도
P(real) 이 최대 0.25 까지 움직였다(1,200장 중 판정 5개 불일치). 그래서 PIL 계수 행렬(상위·하위 11비트로 나눠 float32 에서도 정확) 곱과
정수 반올림으로 바꿨고, 확대 결과가 PIL 과 픽셀 완전 일치한다 (`export_onnx.py` 가 내보낼 때마다 확인).

## 검증 (2026-11-14, 테스트 1,200장: test + flux_sdxl / gravex / stylegan, 폴더당 150장)

같은 전처리 텐서를 넣고 서버 `FakeFaceModel` 과 비교 (기준값 full 0.52).

| | 점수 최대 차이 | 판정 불일치 | AUC | Grad-CAM 상관 (최소) |
|---|---|---|---|---|
| PyTorch | – | – | 0.99867 | – |
| ONNX fp32 | 0.000006 | 0 / 1200 | 0.99867 | 0.999998 |
| ONNX 배포용 | 0.0021 | 0 / 1200 | 0.99866 | 0.99999 |

## 브라우저 전체 파이프라인 검증 (2026-11-14, onnxruntime-web 1.20.1 wasm 1스레드)

`fakit-face-model.js` 가 서버 `predict_image` 를 그대로 재현하는지 `test.html` 의 **서버와 비교** 로 확인했다.
기준 결과는 `make_reference.py --per-folder 12` (테스트 96장 + 얼굴 여러 개 격자 4장, 넓은 배경 속 작은 얼굴 2장, 2600px 1장 = 103장, 얼굴 131개).

| 항목 | 결과 |
|---|---|
| 256 리사이즈 (Pillow BILINEAR 재현) | 103장 모두 픽셀 완전 일치 |
| JPEG 90 재인코딩 (브라우저 인코더 vs PIL) | 103장 모두 픽셀 완전 일치 |
| 얼굴 검출 개수 / crop 영역 | 131개 모두 같은 얼굴, crop 영역 같음 |
| full 모드 P(real) 차이 | 최대 0.0020, 평균 0.00006, 판정 불일치 0 |
| face 모드 P(real) 차이 | 최대 0.0020, 평균 0.00010, 판정 불일치 0 |
| 속도 | 모델 로드 약 2초. 103장 비교 418초 (창이 가려진 상태라 실제보다 느림) |

확인 방법:
```bat
server\.venv\Scripts\python.exe web_model\make_reference.py --per-folder 12
server\.venv\Scripts\python.exe -m http.server 8765 --bind 127.0.0.1
```
브라우저에서 http://127.0.0.1:8765/web_model/test.html → **비교 실행**.

## 사이트 1차 EfficientNetV2S(Keras) vs 최종 EfficientNetV2S(Phase 9) (2026-11-14)

`compare_models.py` → `compare_scene.py` → `compare.html` (`?set=compare`, `?set=compare_scene`).
1차 모델은 사이트(`fakit-integration/web/model`)의 TF.js 모델과 `app.js` 전처리(이미지 전체, 기준 0.24)를 그대로 썼다
(`reference/jei_model/` 에 복사해 둠).
두 모델 모두 학습·검증에 쓰지 않은 이미지만 사용 (data/test 중 같은 사진이 1차 train/valid, 최종 train/valid 어디에도 없는 2,591장 → real 286 / fake 702).
얼굴만 잘린 데이터셋 이미지라 생성기 종류는 학습 때와 같다 (처음 보는 생성기 비교는 아님).

| 조건 | 1차 (사이트) AUC / 균형 정확도 | 최종 얼굴 crop AUC / 균형 정확도 |
|---|---|---|
| 원본 | 1.000 / 98.9% | 1.000 / 98.9% |
| JPEG q50 | 0.998 / 98.5% | 1.000 / 99.2% |
| 축소 0.5x | 0.999 / 98.4% | 1.000 / 99.3% |
| 블러 σ1.0 | 0.999 / 98.7% | 1.000 / 99.4% |
| **얼굴이 작은 사진** (900×700 배경 속 200px 얼굴, 400장) | **0.659 / 56.0%** (real 정답률 32.5%) | **0.999 / 98.0%** |

- 보정 사진 FAKE 오판 (283장): 보정 없음 2.5% → 1.4%, 피부 보정(중간) 0.7% → 3.9%, 뷰티 조합(중간) 1.1% → 4.6%.
  보정 사진에서는 1차 모델보다 FAKE 로 더 많이 본다 (283장 중 2~3장 → 11~13장). 실험 B(`results/experiment_b/`)와 같은 경향.
- 최종 모델도 얼굴 검출 없이 이미지 전체를 보면 얼굴이 작은 사진에서 AUC 0.557 로 무너진다 → 이 차이는 모델보다 **얼굴 crop 파이프라인** 덕분이다.
- 얼굴 crop 은 얼굴을 못 찾으면 판단 불가(원본 988장 중 3~7장, 작은 얼굴 400장 중 7장).

## 브라우저 전처리 메모

서버 순서: 긴 변 제한 리사이즈 → MediaPipe BlazeFace(short range) 얼굴 검출 → 얼굴 bbox ×2.0 정사각 crop(중심 5% 위로) → 256×256 → **JPEG 품질 90 재인코딩** → /255 → mean/std 0.5 (320 확대는 ONNX 그래프 안).

- 얼굴 검출: `@mediapipe/tasks-vision` FaceDetector + 같은 `blaze_face_short_range.tflite`. 서버의 타일 재검출·필터 규칙(`server/face.py`)도 옮겨야 한다.
- JPEG 90: `canvas.toBlob('image/jpeg', 0.9)` 로 재현. 인코더가 PIL(libjpeg)과 달라 점수가 조금 달라질 수 있으므로 같은 이미지로 서버와 비교해 확인한다.
- 리사이즈: 서버는 PIL BILINEAR. 캔버스 `drawImage` 보간과 차이가 있으니 함께 측정한다.
