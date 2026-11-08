@echo off
chcp 65001 >nul
set "PYTHONUTF8=1"
set "FINAL_VIDEO_PYTHON=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if not exist "%FINAL_VIDEO_PYTHON%" set "FINAL_VIDEO_PYTHON=python"
cd /d "%~dp0"
"%FINAL_VIDEO_PYTHON%" -u "%~dp0src\predict_final_research.py" --interactive
pause
