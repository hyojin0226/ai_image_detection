# AI Face Detector — HTML 버전

`feature/main-model-hyojin` 브랜치의 `streamlit_app.py`를 서버 없이 브라우저에서 돌아가는 정적 사이트로 옮긴 버전입니다.
같은 EfficientNetV2S 모델을 [TensorFlow.js](https://www.tensorflow.org/js)로 변환해 사용하고, 판정 기준(threshold 0.24)도 동일합니다.
업로드한 이미지는 서버로 전송되지 않습니다.

## 폴더 구성

| 파일 | 설명 |
| --- | --- |
| `index.html`, `style.css`, `app.js` | 사이트 화면과 예측 로직 |
| `ui.js` | 링 모션그래픽, 배경 효과, 상단 메뉴 패널 등 화면 연출 (예측과 무관) |
| `icon.svg`, `icon.ico` | 서비스 아이콘 (파비콘, 바로가기 로고) |
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
