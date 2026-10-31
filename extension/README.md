# Fakit 크롬 확장 프로그램 (개발용)

이미지 우클릭 → **Fakit으로 이미지 가져오기** → Fakit의 이미지 주소 입력란에 자동 입력됩니다. 페이지에서 **가져와 분석하기**를 누르면 브라우저가 이미지를 가져와 다중 이미지 분석 목록에 추가합니다.

## 실행

1. 저장소 루트에서 `npm start`로 Fakit을 실행합니다 (Node.js 22 이상).
2. Chrome 주소창에 `chrome://extensions`를 입력합니다.
3. **개발자 모드**를 켜고 **압축해제된 확장 프로그램을 로드합니다**에서 이 `extension` 폴더를 선택합니다.
4. 일반 웹페이지에서 이미지를 우클릭하고 **Fakit으로 이미지 가져오기**를 선택합니다.
5. 열린 Fakit 페이지에 주소가 입력되었는지 확인하고 **가져와 분석하기**를 누릅니다.

사이트의 **확장 프로그램 받기**에서도 소스 ZIP을 내려받을 수 있습니다. 소스 수정 후에는 저장소 루트의 PowerShell에서 `Compress-Archive -Path extension/* -DestinationPath web/extension.zip -Force`로 다운로드 파일을 갱신하세요.

배포 사이트에 연결하려면 `background.js`의 `FAKIT_URL`을 배포 주소로 수정한 뒤 확장 프로그램 페이지에서 새로고침합니다. HTTPS 사이트나 하위 경로 배포도 지정할 수 있습니다.

## 범위와 제한

- Manifest V3이며 `contextMenus` 권한만 요청합니다. 방문 기록, 페이지 내용, 로그인 쿠키를 수집하지 않습니다.
- 사용자가 선택한 이미지 주소만 URL fragment(`#image=...`)로 전달합니다. 서버 요청에는 포함되지 않으며 Fakit은 입력 후 주소창에서 제거합니다.
- 이 버전은 **주소 전달**용입니다. 원본 서버가 CORS를 허용하지 않거나, 쿠키/로그인이 필요하거나, HTTPS 페이지에서 HTTP 이미지를 불러오는 경우 실제 다운로드는 실패할 수 있습니다. 해당 이미지를 저장한 뒤 직접 업로드하세요.
- `blob:`, `data:`, `file:` 주소는 다른 탭에서 재사용하기 어려워 지원하지 않습니다. CSS 배경 이미지도 일반 이미지 우클릭 메뉴의 대상이 아닐 수 있습니다.
- 가져오기는 JPG/PNG/WEBP 및 최대 15MB로 제한합니다. 원본 이미지 호스트에는 브라우저의 이미지 요청이 전달되지만, Fakit 서버에 이미지를 업로드하지 않습니다.
- Chrome Web Store 배포나 실제 Chrome 설치 테스트는 별도로 필요합니다.

공식 문서: [Context menus](https://developer.chrome.com/docs/extensions/reference/api/contextMenus), [Cross-origin requests](https://developer.chrome.com/docs/extensions/develop/concepts/network-requests).
