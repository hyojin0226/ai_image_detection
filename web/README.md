# Fakit — HTML 버전

`feature/main-model-hyojin` 브랜치의 `streamlit_app.py`를 서버 없이 브라우저에서 돌아가는 정적 사이트로 옮긴 버전입니다.
같은 EfficientNetV2S 모델을 [TensorFlow.js](https://www.tensorflow.org/js)로 변환해 사용하고, 판정 기준(threshold 0.24)도 동일합니다.
업로드한 이미지는 서버로 전송되지 않습니다.

## 페이지 구성

주소 해시(`#/…`)로 페이지를 전환하는 단일 HTML입니다. 모델은 한 번만 불러오고 페이지를 옮겨도 유지됩니다.

| 주소 | 내용 |
| --- | --- |
| `#/` | 이미지 탐지. 드래그 앤 드롭·붙여넣기(Ctrl+V)·파일 선택으로 한 번에 최대 20장 판별 |
| `#/history` | 기록. 이 기기에서 판별한 결과(축소 미리보기 + 판정)를 localStorage에 보관 |
| `#/model` | 모델 설명. 파이프라인, 학습 방법, 평가, 사양, 한계 |
| `#/text` | AI 글 탐지 — 틀만 있음 (이후 추가 예정) |
| `#/video` | AI 영상 탐지 — 틀만 있음 (이후 추가 예정) |
| `#/guide` | 사용 가이드와 FAQ |
| `#/about` | 소개와 로드맵 |

랜딩 상단의 3D 장면(블러된 `Fakit.` 글자 위를 커서를 따라다니는 렌즈가 선명하게 비추고, 스크롤하면 글자가 뒤집히는 효과)은
Spline 커뮤니티의 [HOVER + SCROLL EFFECT](https://app.spline.design/community/file/355705e7-5f04-4051-93d5-5007f7034ada)(pixlspace, CC BY 4.0)를 참고해
CSS/JS로 직접 구현한 것입니다. `prefers-reduced-motion` 환경에서는 정적인 글자만 보입니다.

실제 Spline 씬으로 바꾸려면 ① 위 링크에서 본인 계정으로 Remix → ② 글자를 `Fakit` 또는 `REAL or AI`로 수정 →
③ Export → Viewer에서 받은 `scene.splinecode` 주소와 viewer 스크립트 주소를 `ui.js` 맨 위의 `SPLINE_SCENE`, `SPLINE_VIEWER_SRC`에 넣으면
내장 장면 대신 Spline 씬이 표시됩니다.

간편 로그인(카카오 → 네이버 → 구글) 창은 체험 모드입니다. 실제 OAuth는 연결되어 있지 않으며 로그인 상태는 이 브라우저에만 저장됩니다.

## 폴더 구성

| 파일 | 설명 |
| --- | --- |
| `index.html` | 모든 페이지 마크업 |
| `style.css` | 스타일 (보라 포인트 컬러, 흰 배경) |
| `ui.js` | 라우터, 상단 메뉴, 로그인 창, 토스트 등 화면 연출 (예측과 무관) |
| `app.js` | 모델 로딩, 이미지 큐 분석, 기록 저장, 결과 상세 창 |
| `icon.svg`, `icon.ico` | Fakit 로고 (파비콘, 바로가기 아이콘) |
| `model/model.json`, `model/group1-shard*.bin` | TF.js로 변환한 모델 (float16, 약 41MB) |
| `model/inference_config.json` | 입력 크기, threshold 등 (원본과 동일) |

## 로컬에서 실행

저장소 맨 위에 있는 `AI Face Detector.bat`을 더블클릭하면 서버가 켜지고 브라우저가 열립니다 (창을 닫으면 서버도 꺼집니다).
한 번 실행하면 같은 자리에 로고가 달린 바로가기 `AI Face Detector.lnk`가 생기니 다음부터는 그걸 눌러도 됩니다.
Python 3이 설치되어 있어야 합니다.

터미널에서 직접 실행하려면:

```bash
python -m http.server 8000 --directory web
```

그다음 브라우저에서 <http://localhost:8000> 접속.

## 배포 (GitHub Pages)

`web/` 폴더 전체가 정적 파일이라 GitHub Pages, Netlify, Vercel 등에 그대로 올리면 됩니다.
모델 파일이 4MB 단위로 나뉘어 있어 Git LFS 없이 push할 수 있습니다.

## 원본(Streamlit)과의 차이

- 브라우저에서 실행되므로 처음 접속할 때 모델(약 41MB)을 내려받습니다. 이후엔 브라우저 캐시를 사용합니다.
- 모델 가중치를 float16으로 줄였고, 이미지 리사이즈를 PIL 대신 브라우저 canvas로 합니다.
  테스트 이미지에서 원본 Keras 모델과의 확률 차이는 약 0.001 이하였습니다.

## 모델 다시 변환하기

모델을 새로 학습했다면 아래 스크립트로 `web/model/`을 다시 만들 수 있습니다 (Python 3.10+, `tensorflow==2.21.0`).

```bash
pip install tensorflow==2.21.0 tensorflow_hub
pip install --no-deps tensorflowjs
python scripts/convert_to_tfjs.py path/to/efficientnetv2s_fake_face_final.keras web/model
```

> Windows에서 TensorFlow 설치 중 경로 길이 오류가 나면 가상환경을 `C:\tfv`처럼 짧은 경로에 만드세요.
