@echo off
title AI Face Detector
cd /d "%~dp0"

rem 처음 실행하면 같은 폴더에 로고 달린 바로가기(AI Face Detector.lnk)를 만들어 둡니다.
if not exist "%~dp0AI Face Detector.lnk" (
  powershell -NoProfile -ExecutionPolicy Bypass -Command "$s=(New-Object -ComObject WScript.Shell).CreateShortcut('%~dp0AI Face Detector.lnk'); $s.TargetPath='%~f0'; $s.WorkingDirectory='%~dp0'; $s.IconLocation='%~dp0web\icon.ico'; $s.WindowStyle=7; $s.Description='AI Face Detector 실행'; $s.Save()" >nul 2>nul
)

rem 파이썬 찾기 (py 런처 우선)
set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY where python >nul 2>nul && set "PY=python"
if not defined PY (
  echo Python이 필요합니다. https://www.python.org/downloads/ 에서 설치한 뒤 다시 실행해 주세요.
  pause
  exit /b 1
)

rem 이미 서버가 켜져 있으면 브라우저만 엽니다.
netstat -ano | findstr /r /c:":8000 .*LISTENING" >nul
if not errorlevel 1 (
  start "" http://localhost:8000
  exit /b 0
)

rem 서버가 뜰 시간을 준 뒤 브라우저를 엽니다.
start "" /b cmd /c "timeout /t 1 /nobreak >nul & start "" http://localhost:8000"
echo AI Face Detector 실행 중: http://localhost:8000
echo 이 창을 닫으면 서버가 꺼집니다.
%PY% -m http.server 8000 --directory web
