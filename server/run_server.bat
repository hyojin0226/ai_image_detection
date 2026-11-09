@echo off
chcp 65001 > nul
REM AI 얼굴 탐지 로컬 서버 실행 (더블클릭)
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [setup] 가상환경을 만들고 패키지를 설치합니다. 처음 한 번만 몇 분 걸립니다.
    python -m venv .venv || goto :error
    ".venv\Scripts\python.exe" -m pip install --upgrade pip
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :error
)

if not exist "models\convnext_tiny_fake_face.pt" (
    echo [warn] models\convnext_tiny_fake_face.pt 가 없습니다. README 의 모델 준비 단계를 확인하세요.
)

echo [run] http://127.0.0.1:8000  (종료: Ctrl+C)
".venv\Scripts\python.exe" -m uvicorn app:app --host 127.0.0.1 --port 8000
pause
exit /b 0

:error
echo [error] 설치 또는 실행에 실패했습니다. 위 메시지를 확인하세요.
pause
exit /b 1
