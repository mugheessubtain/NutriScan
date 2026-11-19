@echo off
rem NutriScan: starts the Python API (port 8000) and the Next.js website (port 3000), then opens the browser.
rem Close the two black windows to stop everything.
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Python environment not found. Run:  py -3.14 -m venv .venv  and  .venv\Scripts\pip install -r requirements.txt
  pause
  exit /b 1
)

start "NutriScan API (models load in about 1 minute)" cmd /k .venv\Scripts\python.exe -m uvicorn server:app --app-dir runtime/api --host 127.0.0.1 --port 8000

cd web
if not exist node_modules (
  echo Installing website packages, first run only...
  call npm install
)
if not exist .next\BUILD_ID (
  echo Building the website, first run only...
  call npm run build
)
start "NutriScan Web" cmd /k npm run start -- -p 3000

timeout /t 6 /nobreak >nul
start "" http://localhost:3000
