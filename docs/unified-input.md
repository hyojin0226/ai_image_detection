# Fakit 통합 입력

첫 화면은 SVG 워드마크와 렌즈, 한 개의 입력창으로 구성한다. 워드마크 바깥에는 RGB 채널 오프셋, 렌즈 안에는 같은 벡터의 선명한 복제본을 사용한다. 이미지·결과·오류가 있을 때는 워드마크와 렌즈를 분리해 글자를 가리지 않는다.

## 실제 지원 범위

| 입력 | 처리 | 한도 |
| --- | --- | --- |
| 이미지 파일·촬영 사진 | 기존 TF.js 얼굴 이미지 이진 분류 | JPG/PNG/WEBP, 15MB |
| 클립보드 이미지 | 기존 모델, 브라우저 디코더 | 15MB |
| 이미지 URL | 브라우저가 원본에 CORS 요청 후 기존 모델 | JPG/PNG/WEBP, 15MB, 15초 |
| 영상 파일 | MIME 판별, 로컬 메타데이터로 길이 표시, 미지원 응답 | 모델과 분석 한도 미정 |
| 영상 URL | 응답 Content-Type 확인 후 본문 취소, 미지원 응답 | 모델 없음 |
| 글 | 입력/문자 수 표시, 미지원 응답 | 기존 입력 한도 20,000자 |
| 웹페이지 URL | HTML Content-Type 확인 후 본문 취소, 미지원 응답 | 본문 추출 기능 없음 |

짧은 단독 http(s) 문자열(200자 미만)은 링크로, 그 외 문자열은 글로 분류한다. 파일은 확장자가 아닌 MIME을 따른다. 여러 파일은 최대 20개까지 추가하고 순서대로 검사한다. 칩에서 종류를 수정할 수 있지만 파일을 다른 형식으로 변환하지는 않는다.

## 연결 구조

- `web/lens.js`: `handleInput(source, payload)` → 입력 확정/미리보기 → `analyze(input)`. 드롭은 즉시 검사, 선택·붙여넣기·입력은 검사 버튼 또는 Enter로 시작. Shift+Enter는 줄바꿈, Esc는 취소 및 초기화.
- `web/input-adapters.js`: 모델별 공통 인터페이스. `analyze(input, {signal, onState, onResolved, imageAnalyzer})`는 `{status:'result', real, fake, verdict}` 또는 `{status:'unsupported', message}`를 반환한다. 통신·디코딩·크기 오류는 예외로 전달한다.
- `web/app.js`: `Fakit.analyzeImage` 연결부만 새 입력 레이어에 제공한다. 기존 모델 로딩/전처리/predict/0.24 판정 기준/기록 저장 로직은 유지한다.
- `web/image-import.js`: 인증정보와 referrer 없이 제한된 크기의 스트림을 가져오며, 취소를 전달한다. 원본 서버의 CORS 허용이 필요하다.
- `server.mjs`: 새 어댑터 정적 파일을 허용 목록에 추가했다. 소셜 로그인은 변경하지 않았다.

## 상태와 정직한 표시

idle, dragover, detecting, loading, analyzing, result, error, unsupported를 사용한다. 결과 링은 기존 real/fake 판정의 초록/빨강이다. 현재 모델에는 중간 ‘의심’ 등급이 없어 주황 등급을 임의로 만들지 않는다. 퍼센트는 REAL 출력 점수이며 확정 진위나 정확도가 아니다.

영상 프레임 샘플링·필름 스트립·의심 구간, 글의 문장별 점수·하이라이트는 해당 모델이 생길 때 어댑터 계약을 확장한다. 지금은 가짜 진행률이나 결과를 출력하지 않는다. 스캔선은 이미지 전체 분석 중의 시각적 상태 표시이며 국소 위조 탐지를 뜻하지 않는다.

미리보기 object URL은 교체·취소·결과·오류·페이지 종료 때 해제한다. 이미지 분석 연결부도 원본 URL과 DOM 이미지 참조, 임시 입력 캔버스를 정리한다. 성공 시 입력의 File 참조를 제거하며, 기존 기록 기능은 작은 썸네일과 점수만 브라우저에 남긴다.

## 확인

`node --test tests/*.test.mjs` — 19개 통과. 입력 분류, 지원 안 되는 타입의 무추론 응답, 원본 점수 보존, 용량·글자 한도, URL 취소, 영상/HTML 응답 취소, 소셜 로그인 회귀, 확장 프로그램 URL 전달을 포함한다.

브라우저 확인: 이미지 파일 선택 후 실제 모델 결과, 입력창 내부 PNG 붙여넣기, Enter 검사, Esc 초기화, 이미지 URL 분석, 영상 URL 미지원, 글 미지원, 칩 수정, 360×800 / 768×1024 / 1440×1000 배치와 가로 넘침 검사.

실제 모바일 카메라·앨범 선택과 가상 키보드, OS에서 브라우저로 직접 드래그하는 동작은 해당 기기에서 추가 확인한다. `prefers-reduced-motion`에서는 이동·스캔·회전이 꺼진다. 렌즈는 SVG이므로 Canvas/WebGL 미지원 환경에서도 표시되지만, 기존 TF.js 모델의 실행 가능 여부는 브라우저에 따른다.

## 변경 파일

`web/index.html`, `web/experience.css`, `web/lens.js`, `web/input-adapters.js`, `web/app.js`, `web/image-import.js`, `server.mjs`, `tests/auth.test.mjs`, `tests/image-import.test.mjs`, `tests/input-adapters.test.mjs`, `docs/unified-input.md`.

## 렌즈 모션과 여러 사진 업데이트

첫 진입에서 선명한 워드마크를 보여준 뒤 1.1초 인트로가 끝나면 입력창을 연다. 바깥 워드마크는 흐려지고 렌즈 안은 확대·굴절 표현을 사용한다. 헤더의 중복 로고는 제거했다. 모션은 SVG/CSS 원근과 감쇠 스프링으로 구현하며 외부 3D 라이브러리는 추가하지 않았다.

사진은 최대 20개까지 더 첨부하거나 검사 전에 개별 삭제할 수 있다. 여러 장은 카드 덱으로 표시되고 실제 검사 중인 사진에 맞춰 앞으로 움직인다. batch-input.js는 기존 단일 이미지 어댑터를 순차 호출하며 하나가 실패해도 다음 파일을 처리한다. 전체 완료 후 파일별 점수를 확인한다. 썸네일만 남기고 성공한 원본 File 참조는 해제한다.

글은 동일한 textarea가 보라색 사각 프레임 안으로 이동한다. 로고에는 textLength 강제 맞춤을 사용하지 않으며 균등 scale로만 축소한다. 문장 분석 모델은 여전히 미지원이다. 추가 변경: web/batch-input.js, tests/batch-input.test.mjs.
