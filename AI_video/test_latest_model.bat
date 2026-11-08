@echo off
chcp 65001 >nul
set "PYTHONUTF8=1"
set "LATEST_VIDEO_PYTHON=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if not exist "%LATEST_VIDEO_PYTHON%" set "LATEST_VIDEO_PYTHON=python"
"%LATEST_VIDEO_PYTHON%" "%~dp0src\test_model_window.py"
if errorlevel 1 pause
