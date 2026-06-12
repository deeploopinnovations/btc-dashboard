@echo off
REM =====================================================================
REM kronos_local\run.bat — one-click launcher for the local Kronos server
REM =====================================================================
REM Uses the verified venv (C:\Users\DELL\kronos-env) and the verified
REM repo  (C:\Users\DELL\Kronos). Edit the two paths below if you move them.

set KRONOS_ENV=C:\Users\DELL\kronos-env
set KRONOS_REPO=C:\Users\DELL\Kronos

REM ── tunables (uncomment to override) ─────────────────────────────────
REM set KRONOS_MODEL=NeoQuasar/Kronos-mini      & REM lowest RAM (4.1M params)
REM set KRONOS_MODEL=NeoQuasar/Kronos-small     & REM default (24.7M params)
REM set KRONOS_SAMPLES=12                       & REM fewer MC samples = faster

if not exist "%KRONOS_ENV%\Scripts\activate.bat" (
  echo [run.bat] venv not found at %KRONOS_ENV% — edit KRONOS_ENV in this file.
  pause & exit /b 1
)
if not exist "%KRONOS_REPO%\model" (
  echo [run.bat] Kronos repo not found at %KRONOS_REPO% — edit KRONOS_REPO in this file.
  pause & exit /b 1
)

call "%KRONOS_ENV%\Scripts\activate.bat"
cd /d "%~dp0"
echo [run.bat] starting local Kronos server on http://127.0.0.1:8899/ ...
python app.py
pause
