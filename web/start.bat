@echo off
rem Double-click to run the site locally (the model can't load from file://).
cd /d "%~dp0"
start "" http://localhost:8000
python -m http.server 8000
