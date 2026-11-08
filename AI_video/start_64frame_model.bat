@echo off
chcp 65001 >nul
set "PYTHONUTF8=1"
set "VIDEO64_PYTHON=%~dp0.venv\Scripts\python.exe"
if not exist "%VIDEO64_PYTHON%" set "VIDEO64_PYTHON=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if not exist "%VIDEO64_PYTHON%" set "VIDEO64_PYTHON=python"
cd /d "%~dp0"
"%VIDEO64_PYTHON%" "%~dp0src\analysis64_window.py"
if errorlevel 1 pause
