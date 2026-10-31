@echo off
rem Requires Node.js 22+; configuration is read from the root .env.
cd /d "%~dp0.."
start "" http://localhost:8000
node --env-file-if-exists=.env server.mjs
pause
