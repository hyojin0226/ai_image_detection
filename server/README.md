# AI 얼굴 탐지 크롬 확장 – 로컬 서버 실행 방법 (MVP)

웹페이지(인스타그램 등)에서 우측 하단 **AI 얼굴 스캔** 버튼을 누르면, 화면의 이미지를 로컬 서버로 보내
얼굴을 찾고 얼굴마다 `REAL 94%` / `FAKE 87%` / `판단 불가` 배지를 표시합니다.
GPU 없는 노트북(CPU)에서도 동작합니다.

```
server/      FastAPI 로컬 추론 서버 (http://127.0.0.1:8000)
extension/   크롬 확장 (Manifest V3)
```

## 1. 준비물
- Windows + **Python 3.12 (64-bit)** — `python --version` 으로 확인 (3.10~3.12 권장, 3.13 이상은 TensorFlow 설치가 안 될 수 있음)
- Chrome
- 학습된 모델 파일 2개 → `server/models/` 에 둡니다
  - `convnext_tiny_fake_face.pt` (약 110MB, git 제외) — 원본: WSL `\\wsl.localhost\ubuntu\home\hyojin\ai_image_detection\outputs\exp2\convnext_tiny_pseudo_vae_deg\all\best.pt` 를 이 이름으로 복사
  - `convnext_tiny_fake_face.json` (모델 이름·전처리·REAL 판정 threshold. 저장소에 포함)

### 사용 모델
ConvNeXt-T (ImageNet-22k 사전학습) 를 gan / diffusion / manip 세 계열 전체 데이터와
**의사 fake**(real 사진을 SD VAE 로 재구성한 이미지)로 학습하고,
SNS 축소·블러에 견디도록 **축소(0.4~1.0x)·블러·JPEG 증강**을 더한 모델입니다.
- VAE 의사 fake: diffusion 계열을 학습에서 뺀 실험에서 처음 보는 diffusion 생성기 탐지를 높였습니다
  (OOD AUC 0.872 → 0.952, `results/phase6_pseudo_fake/`)
- 축소·블러 증강: 증강 없이는 0.5배 축소에서 AUC 0.917 까지 떨어지고 GAN fake 의 35% 를 놓쳤지만, 증강 후 0.99 수준 유지 (`results/experiment_a_final/vae_degrade/`)
- 피부 보정·뷰티 필터를 건 진짜 사진을 FAKE 로 오판하는 비율이 5~6% 로 낮습니다 (실험 B, `results/experiment_b/`)

| 평가 (얼굴 crop x2.0, 서버와 같은 방식) | AUC | fake recall | real recall |
|---|---|---|---|
| 같은 분포 테스트 | 0.998 | – | 0.981 |
| 학습에 안 쓴 추가 테스트 2만 장 | 0.990 | 0.938 | 0.954 |
| 추가 테스트, 0.5배 축소 | 0.989 | – | – |
| 추가 테스트, 블러 σ1.0 | 0.990 | – | – |

이전 Keras 모델(`efficientnetv2s_fake_face_final.keras`)은 더 이상 쓰지 않습니다.

### 이전 모델 (SBI 포함: 처음 보는 FLUX 에 강하지만 보정 사진 오탐이 많음)
얼굴 영역 Self-Blended Images(SBI) 의사 fake 를 더한 모델도 넣어 두었습니다
(`convnext_tiny_fake_face_sbi.pt` / `.json`, 원본 WSL `outputs\exp2\convnext_tiny_pseudo_vae_sbi_deg\all\best.pt`).
실험 B 에서 이 모델이 피부 보정·뷰티 필터 real 을 FAKE 로 많이 판정했고 원인이 SBI 로 확인돼 2026-11-09 기본값에서 내렸습니다.

| | **기본 (VAE+축소·블러)** | 이전 (VAE+SBI+축소·블러) |
|---|---|---|
| 피부 보정(중간) real 을 FAKE 로 오판 | **4.8%** | 57% |
| 뷰티 조합(중간) real 을 FAKE 로 오판 | **6.2%** | 81% |
| 추가 테스트 2만 장 AUC / real recall | **0.990 / 0.954** | 0.986 / 0.918 |
| 처음 보는 FLUX 얼굴 (diffusion 제외 실험 AUC) | 0.80~0.88 | **0.94** |

이전 모델로 실행 (server 폴더에서):
```bat
set FAKE_FACE_MODEL_CONFIG=models\convnext_tiny_fake_face_sbi.json
.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

얼굴 검출 모델(`blaze_face_short_range.tflite`, 약 230KB)은 서버 첫 실행 때 자동으로 `server/models/` 에 내려받습니다.
인터넷이 없는 시연 환경이라면 미리 한 번 실행해 두거나 파일을 복사해 두세요.

## 2. 서버 실행
`server/run_server.bat` 더블클릭.

- 처음 실행 시 `server/.venv` 가상환경을 만들고 `requirements.txt` 를 설치합니다 (PyTorch CPU 때문에 수 분 소요).
- `Uvicorn running on http://127.0.0.1:8000` 이 보이면 준비 완료. 창을 닫으면 서버가 꺼집니다.
- 저장소는 **짧은 경로**(예: `C:\ai_image_detection\...`)에 두세요. PyTorch 설치 파일 안에 아주 긴 경로가 있어서,
  폴더 경로가 길면(대략 100자 이상) Windows 260자 제한에 걸려 `Could not install packages due to an OSError ... Long Path` 오류가 납니다.
  (`C:\ai_image_detection\ai_image_detection\server\.venv` 위치에서는 설치·실행을 확인했습니다.)
- 확인: 브라우저에서 http://127.0.0.1:8000/health → `"status":"ok"`

직접 실행하려면:
```bash
cd server
pip install -r requirements.txt
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

## 3. 크롬 확장 로드
1. Chrome 주소창에 `chrome://extensions` 입력
2. 우측 상단 **개발자 모드** 켜기
3. **압축해제된 확장 프로그램을 로드합니다** → 저장소의 `extension` 폴더 선택
4. 인스타그램 등 페이지를 **새로고침** → 우측 하단 **AI 얼굴 스캔** 버튼 클릭

- 다시 누르면 재스캔(스크롤 후 새로 보이는 이미지 포함), `✕` 는 배지 지우기
- 한 번에 화면에 보이는 이미지(+위아래 한 화면) 중 최대 24장만 분석합니다. 모달 등에 가려진 이미지는 건너뜁니다
- 배지의 %는 **판정 확신도**입니다. 모델 threshold(얼굴 기준 P(real) 16%)를 50%로 맞춰 늘린 값이라,
  `REAL 66%` 는 "기준을 조금 넘겨 REAL" 정도의 뜻입니다. 마우스를 올리면 모델 원 출력 P(real)이 보입니다
- REAL/FAKE 배지를 **클릭하면 판정 근거 패널**이 열립니다: 모델이 실제로 본 얼굴 crop, Grad-CAM, ELA 맵

## 4. API
`POST /predict` — 이미지 1장. 아래 중 아무 형식이나 가능
- multipart: `file` 필드
- JSON: `{"image_base64": "..."}` (data URL 형식도 가능)
- raw 바이트 본문 (`Content-Type: image/jpeg` 등)

쿼리 `mode=face`(기본, 얼굴 검출 후 얼굴별 판정) / `mode=full`(검출 없이 이미지 전체를 판정 — 이미 얼굴 crop 인 이미지용)

```bash
curl -F "file=@data/sample/test/fake/SDXL_image_0000902.jpg" http://127.0.0.1:8000/predict
```
```json
{"status":"ok","label":"fake",
 "faces":[{"bbox":[63,82,110,110],"bbox_norm":[0.246,0.320,0.430,0.430],"crop_box":[8,16,220,220],
           "detection_score":0.96,"prob_fake":1.0,"prob_real":0.0,"label":"fake"}],
 "image_size":[256,256],"threshold_real":0.16,"mode":"face","elapsed_ms":80}
```
얼굴이 없으면 `{"status":"no_face","label":"no_face","message":"판단 불가(얼굴 없음)","faces":[]}`.

`POST /explain?face_index=0` — `/predict` 와 같은 입력·같은 얼굴 순서. 얼굴마다 아래가 추가됩니다.
- `crop_image`: 모델 입력 256×256 crop (JPEG 90 재인코딩 후, data URL)
- `gradcam_image`, `gradcam_target`: 판정한 쪽(fake/real) 기준 Grad-CAM (ConvNeXt 마지막 stage, 8×8 → 256 업샘플)
- `ela_image`, `ela_stats`: ELA 맵(품질 90 재저장 오차, 원본 해상도에서 계산 후 crop) — 모델 판정과 별개인 참고 지표

## 5. 추론 방식 (WSL 학습·평가 코드와 동일하게 유지)
- 얼굴 검출: MediaPipe BlazeFace(short range, CPU). 큰 이미지는 겹치는 타일에서도 검출해 작은 얼굴을 보완
- crop: 얼굴 박스 한 변 × 2.0 정사각형, 중심을 살짝 위로(이마·머리 포함). 이미지 밖으로 나가면 안쪽으로 밀어서 맞춤 (패딩 없음)
  — 학습 때 얼굴 박스 계산(`scripts/50_detect_faces.py`)과 평가 crop(`src/augment.py`)이 같은 규칙
- 256×256 bilinear 리사이즈 → JPEG 품질 90 재인코딩(평가 조건과 동일) → /255 → ImageNet mean/std 정규화
- 출력은 P(real). threshold 는 얼굴 crop(`mode=face`)·전체 이미지(`mode=full`) 모두 0.16 (valid Macro F1 기준, 이전 모델은 0.30 / 0.34)
- 확인: WSL 평가 예측과 같은 이미지 120장씩 비교해 판정 99.2% 일치. 다른 1장은 threshold 바로 위(WSL 0.161 vs 서버 0.130)라 CPU/GPU 연산 차이로 봅니다 (얼굴당 CPU 약 60ms)
- 조정용 환경변수: `FACE_CROP_SCALE`, `FACE_CROP_SHIFT_UP`, `FACE_MIN_CONFIDENCE`, `FACE_MAX_FACES`, `FAKE_FACE_MODEL_CONFIG`(모델 설정 json 경로 직접 지정)

## 6. 한계
- 학습 데이터는 정렬된 얼굴 crop 이라 SNS 사진(각도·조명·보정·압축)에서는 정확도가 test set 보다 낮을 수 있습니다
- 학습에 없던 생성기는 여전히 놓칠 수 있습니다. 특히 GAN·영상 딥페이크 계열은 의사 fake 로 일반화가 거의 좋아지지 않았고(LOGO 실험),
  영상 딥페이크(manip) fake recall 은 테스트에서도 약 75~80% 입니다
- **처음 보는 생성기, 특히 FLUX 계열을 놓칠 수 있습니다** (diffusion 제외 실험 AUC: FLUX-dev 0.881, FLUX-pro 0.804). 보정 사진 오탐을 줄이려고 SBI 를 뺀 대가입니다
- 0.5배 축소된 영상 딥페이크(manip)는 fake recall 이 약 72% 로 이전 모델(85%)보다 낮습니다
- 피부 보정을 아주 강하게 한 사진은 여전히 16~18% 정도 FAKE 로 판정됩니다 (실험 B, 중간 강도는 5~6%)
- Grad-CAM 은 "모델이 본 곳"이지 "보정·조작된 곳"이 아닙니다. 실험 B 에서 보정 영역을 우연 수준 이상으로 짚지 못했습니다.
  ELA 는 원본 파일에서는 눈·턱선 같은 국소 보정 위치를 더 잘 짚지만(Pointing 0.7~0.9), SNS 재압축 후에는 약해집니다
- 얼굴 검출은 정면에 가까운 얼굴 위주이며, 옆모습·아주 작은 얼굴은 놓칠 수 있습니다
- 결과는 확률 추정이며 단정이 아닙니다
