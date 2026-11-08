@echo off
setlocal
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "WEB_PYTHON=%LOCALAPPDATA%\Programs\Python\Python312\pythonw.exe"
if not exist "%WEB_PYTHON%" (
  echo Python 3.12 was not found. Contact the project owner.
  pause
  exit /b 1
)
start "" "%WEB_PYTHON%" "%~dp0deploy\start_web_site.py"
