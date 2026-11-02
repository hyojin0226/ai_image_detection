# Fakit — 소셜 로그인 · 이미지 가져오기 · 인터랙티브 웹

기준 소스: `feature/html-site`의 `1c1263f25389fe6811585dba0c7f25b9953e7614` (최신 Fakit 화면). 기존 다중 이미지 분석, 기록, 가이드, 모델 페이지를 유지하고 실제 서버 인증 및 확장 프로그램 URL 가져오기를 통합했습니다.

## 통합 사이트 구성

- 보라색 렌즈 효과가 파일 입력 영역에 적용됩니다. 포인터를 따라 선명한 영역이 이동하고 업로드 후 실제 이미지와 처리 상태를 표시합니다.
- 이미지·영상·문서 유형을 같은 입력창에서 구분합니다. 현재 실제 분석은 얼굴 이미지 모델만 지원합니다. 글 입력창은 모델 연결 전입니다.
- 카카오·네이버·구글 OAuth 서버 연동 코드가 포함되어 있습니다. 각 제공자 앱 설정과 로컬 환경변수가 필요하며, 실제 계정 로그인 완료는 별도로 검증해야 합니다.
- Chrome 확장 프로그램 소스와 다운로드 ZIP을 포함합니다. 개발자 모드로 설치하며 스토어 배포판은 아닙니다.
- 로그인 정보는 서버 세션, 분석 기록은 브라우저 로컬 저장소에 보관합니다. 계정별 기록 동기화는 없습니다.
- 디자인 참고: https://community.spline.design/file/355705e7-5f04-4051-93d5-5007f7034ada . 장면 파일은 포함하지 않고 CSS로 별도 구현했습니다.

`feature/main-model-hyojin` 브랜치의 `streamlit_app.py`를 브라우저에서 분석하는 웹사이트로 옮긴 버전입니다. 소셜 로그인은 저장소 루트의 Node.js 서버가 처리합니다.
같은 EfficientNetV2S 모델을 [TensorFlow.js](https://www.tensorflow.org/js)로 변환해 사용하고, 판정 기준(threshold 0.24)도 동일합니다.
업로드한 이미지는 서버로 전송되지 않습니다.

## 폴더 구성

| 파일 | 설명 |
| --- | --- |
| `index.html`, `style.css`, `app.js` | 사이트 화면과 예측 로직 |
| `ui.js` | 페이지 라우팅, 실제 로그인 팝업/로그아웃, 계정 표시 |
| `experience.css`, `lens.js` | 렌즈 입력 디자인과 포인터 반응 |
| `file-types.js` | 통합 입력의 파일 종류 분류 |
| `image-import.js` | 확장 프로그램 URL 수신, 크기 제한이 있는 이미지 다운로드 |
| `extension.zip` | 개발 모드로 설치하는 Chrome 확장 프로그램 소스 |
| `model/model.json`, `model/group1-shard*.bin` | TF.js로 변환한 모델 (float16, 약 41MB) |
| `model/inference_config.json` | 입력 크기, threshold 등 (원본과 동일) |

## 팀 공유용 통합본

`fakit-priview` 브랜치에는 최신 렌즈 UI, 최대 20장 이미지 순차 분석, 카카오·네이버·구글 OAuth 서버, 크롬 확장 프로그램이 함께 있습니다. 비밀 키는 각 실행 환경에 별도로 설정합니다.

루트의 `render.yaml`로 무료 단일 인스턴스 Web Service를 생성할 수 있습니다. Render의 공개 주소는 서버가 `RENDER_EXTERNAL_URL`에서 읽습니다. 사용자 지정 도메인은 `APP_ORIGIN`으로 설정하세요. 각 제공자 콘솔에 실제 공개 주소의 `/auth/kakao/callback`, `/auth/naver/callback`, `/auth/google/callback`을 등록해야 외부 로그인까지 동작합니다.

구현·확인 범위: [통합 입력 문서](../docs/unified-input.md).

## 로컬에서 실행

`index.html`을 더블클릭해서 열면 브라우저 보안 정책 때문에 모델을 불러오지 못합니다. 간단한 웹 서버로 실행하세요.

Node.js 22 이상을 설치하세요. 추가 npm 패키지는 필요하지 않습니다. 저장소 루트에서:

```powershell
Copy-Item .env.example .env
npm start
```

`.env`가 없어도 사이트는 실행되며 미설정 로그인 버튼은 준비 중으로 표시됩니다. `.env`를 수정한 뒤에는 서버를 재시작하세요. Windows에서는 `web/start.bat`으로 실행할 수도 있습니다.

로그인 없이 이미지 분석만 실행하려면 기존 정적 서버도 사용할 수 있습니다:

```bash
python -m http.server 8000 --directory web
```

그다음 브라우저에서 <http://localhost:8000> 접속.

## 소셜 로그인 설정

`npm run check:auth`로 비밀 값 노출 없이 설정 여부와 콜백 주소를 확인할 수 있습니다. `.env`와 Google에서 내려받은 `client_secret*.json`은 Git에 올리지 않습니다. 키를 입력한 뒤 서버를 재시작해야 적용됩니다. 개발자 콘솔의 로그인 활성화·동의 화면·허용 사용자·콜백 등록과 실제 계정 로그인까지 확인해야 연동이 완료됩니다.

각 개발자 콘솔에 웹 앱을 등록하고 아래 콜백 주소를 정확히 입력합니다. `localhost`와 `127.0.0.1`을 섞지 말고 **http://localhost:8000**으로 접속하세요.

| 서비스 | `.env` 항목 | 로컬 Redirect / Callback URL |
| --- | --- | --- |
| 카카오 | `KAKAO_CLIENT_ID` = REST API 키, `KAKAO_CLIENT_SECRET` = 활성화한 경우 해당 비밀 키 | `http://localhost:8000/auth/kakao/callback` |
| 네이버 | `NAVER_CLIENT_ID`, `NAVER_CLIENT_SECRET` | `http://localhost:8000/auth/naver/callback` |
| 구글 | `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET` (웹 애플리케이션 OAuth 클라이언트) | `http://localhost:8000/auth/google/callback` |

- 카카오: 카카오 로그인을 활성화하고 닉네임을 표시하려면 프로필 닉네임 동의 항목을 설정하세요. 이메일 권한은 요청하지 않습니다.
- 네이버: 서비스 URL은 `http://localhost:8000`, Callback URL은 위 주소로 등록하고 별명 등 사용할 프로필 항목을 설정하세요. 개발 중에는 앱에 허용된 테스트 계정으로 확인하세요.
- 구글: OAuth 동의 화면을 설정하고 앱이 테스트 상태라면 테스트 사용자를 등록하세요. 이 구현은 `openid profile`만 요청합니다.
- 비밀 키는 루트 `.env`에만 입력하세요. `.env`는 Git에서 제외되어 있고 `web/` 파일로 노출되지 않습니다.
- 로그인 성공 시 서비스별 ID와 표시 이름만 서버 메모리에 보관합니다. 토큰은 프로필 조회 후 보관하지 않습니다. 브라우저에는 임의 세션 ID를 HttpOnly 쿠키로 전달합니다.
- 세션은 8시간 후 만료되고 서버를 재시작하면 초기화됩니다. 로그아웃은 Fakit 세션만 종료하며 카카오/네이버/구글 계정 자체를 로그아웃하거나 연결 해제하지 않습니다.
- 계정 DB, 다른 제공자 계정 통합, 회원 탈퇴는 아직 구현하지 않았습니다. 표시 이름 제공에 동의하지 않아도 `사용자`로 로그인할 수 있습니다.

공식 문서: [카카오](https://developers.kakao.com/docs/ko/kakaologin/rest-api), [네이버](https://developers.naver.com/docs/login/api/api.md), [구글](https://developers.google.com/identity/protocols/oauth2/web-server).

## 배포

이미지 분석만 사용한다면 `web/`을 GitHub Pages 같은 정적 호스팅에 배포할 수 있지만 **소셜 로그인은 동작하지 않습니다**.
로그인을 사용하려면 Node.js 서버를 실행할 수 있는 호스팅이 필요합니다. HTTPS를 적용하고 `APP_ORIGIN=https://배포도메인`, 호스팅 환경에 맞는 `PORT` 및 `HOST`를 설정한 뒤 각 서비스의 콜백 주소도 변경하세요. HTTPS에서는 Secure 쿠키를 사용합니다.
현재 메모리 세션 구현은 단일 프로세스 개발/시연용입니다. 여러 인스턴스를 사용하는 운영 배포에는 공유 세션 저장소와 요청 제한을 추가해야 합니다.
모델 파일이 4MB 단위로 나뉘어 있어 Git LFS 없이 push할 수 있습니다.

## 원본(Streamlit)과의 차이

- 브라우저에서 실행되므로 처음 접속할 때 모델(약 41MB)을 내려받습니다. 이후엔 브라우저 캐시를 사용합니다.
- 모델 가중치를 float16으로 줄였고, 이미지 리사이즈를 PIL 대신 브라우저 canvas로 합니다.
  테스트 이미지에서 원본 Keras 모델과의 확률 차이는 약 0.001 이하였습니다.

## 크롬 우클릭 확장 프로그램

[`../extension/README.md`](../extension/README.md)에 설치 방법이 있습니다. 사이트의 **확장 프로그램 받기**에서 개발용 ZIP을 내려받을 수 있습니다. 확장 프로그램은 이미지 주소를 Fakit 입력란으로 전달합니다. **가져와 분석하기**를 누르면 브라우저에서 가져오며, 원본 사이트의 CORS 제한이 있으면 직접 업로드해야 합니다.

## 테스트

저장소 루트에서 `npm test`를 실행합니다. 제공자 API를 모의 처리해 세 가지 로그인 흐름, state 검증/재사용/만료, 로그아웃 CSRF 방어, 비공개 파일 차단과 확장 프로그램 URL 전달을 검증합니다. 실제 제공자 로그인을 검증하려면 앱 키를 설정하고 각각의 계정으로 브라우저 테스트를 진행해야 합니다.

## 모델 다시 변환하기

모델을 새로 학습했다면 아래 스크립트로 `web/model/`을 다시 만들 수 있습니다 (Python 3.10+, `tensorflow==2.21.0`).

```bash
pip install tensorflow==2.21.0 tensorflow_hub
pip install --no-deps tensorflowjs
python scripts/convert_to_tfjs.py path/to/efficientnetv2s_fake_face_final.keras web/model
```

> Windows에서 TensorFlow 설치 중 경로 길이 오류가 나면 가상환경을 `C:\tfv`처럼 짧은 경로에 만드세요.
