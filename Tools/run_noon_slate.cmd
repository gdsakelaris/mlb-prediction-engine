@echo off
REM Noon safety-net job (Task Scheduler 12:00 PM, safe to run by hand):
REM   1. Tools/1) Get Todays Games.py   slate scrape -> todays_games.json
REM                                     + Data/slates archive
REM   2. Tools/2) Scrape Odds.py        odds capture (pins the opening
REM                                     price; a later manual rerun near
REM                                     first pitch tightens the close)
REM Guarantees every game day has at least one archived slate and one
REM early-ish odds capture even if the manual game-day workflow is missed.
REM Manual serving is unchanged; grading + the served tracker were folded
REM into the 6 AM job 2026-07-23 (Scrapers\run_daily_update.cmd).
REM Logs to Logs\noon_YYYY-MM-DD.log; exit 1 if any step failed.

set "ROOT=%~dp0.."
REM preferred interpreter (the install with the full ML stack). If that exact
REM folder is gone (a Python upgrade/reinstall moves it, e.g. Python313 ->
REM Python314), fall back to the PATH python so the job survives the
REM transition - then either reinstall the packages there or update this line.
set "PY=C:\Users\gdsak\AppData\Local\Programs\Python\Python313\python.exe"
if not exist "%PY%" set "PY=python"
if not exist "%ROOT%\Logs" mkdir "%ROOT%\Logs"

REM locale-independent date (%date% includes the weekday on some locales)
for /f %%d in ('powershell -NoProfile -Command "Get-Date -Format yyyy-MM-dd"') do set "TODAY=%%d"
set "LOG=%ROOT%\Logs\noon_%TODAY%.log"
set "FAIL="

echo ==================================================================>> "%LOG%"
echo Noon slate run started %date% %time% (python: %PY%) >> "%LOG%"

"%PY%" "%ROOT%\Tools\1) Get Todays Games.py" >> "%LOG%" 2>&1
if errorlevel 1 set "FAIL=1"

"%PY%" "%ROOT%\Tools\2) Scrape Odds.py" >> "%LOG%" 2>&1
if errorlevel 1 set "FAIL=1"

echo Noon slate run finished %date% %time% (fail=%FAIL%) >> "%LOG%"
if defined FAIL exit /b 1
exit /b 0
