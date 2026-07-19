@echo off
REM Morning job (Task Scheduler 6:00 AM, safe to run by hand):
REM   1. update_all --retrain   scrape + validate everything, retrain models
REM   2. Sundays only           CLV gate refresh over the captured-odds window
REM Everything game-day-facing is MANUAL by design (user preference
REM 2026-07-19): slate fetch, odds capture, serving and grading are run by
REM hand (Tools/1, Tools/2, GUI or Model/serve_slate.py, Tools/4).
REM Logs to Logs\update_YYYY-MM-DD.log; exit 1 if any step failed.

set "ROOT=%~dp0.."
REM preferred interpreter (the install with the full ML stack). If that exact
REM folder is gone (a Python upgrade/reinstall moves it, e.g. Python313 ->
REM Python314), fall back to the PATH python so the job survives the
REM transition - then either reinstall the packages there or update this line.
set "PY=C:\Users\gdsak\AppData\Local\Programs\Python\Python313\python.exe"
if not exist "%PY%" set "PY=python"
if not exist "%ROOT%\Logs" mkdir "%ROOT%\Logs"

REM locale-independent date (%date% includes the weekday on some locales,
REM which produced misnamed logs like update_06-Mon-07.log)
for /f %%d in ('powershell -NoProfile -Command "Get-Date -Format yyyy-MM-dd"') do set "TODAY=%%d"
for /f %%w in ('powershell -NoProfile -Command "(Get-Date).DayOfWeek"') do set "DOW=%%w"
set "LOG=%ROOT%\Logs\update_%TODAY%.log"
set "FAIL="

echo ==================================================================>> "%LOG%"
echo Morning run started %date% %time% (python: %PY%) >> "%LOG%"

"%PY%" "%ROOT%\Scrapers\update_all.py" --retrain >> "%LOG%" 2>&1
if errorlevel 1 set "FAIL=1"

if /I not "%DOW%"=="Sunday" goto :finish
for /f %%y in ('powershell -NoProfile -Command "Get-Date (Get-Date).AddDays(-1) -Format yyyy-MM-dd"') do set "YDAY=%%y"
"%PY%" "%ROOT%\Model\evaluate.py" --gate --start 2026-07-15 --end %YDAY% --sims 4000 >> "%LOG%" 2>&1
if errorlevel 1 set "FAIL=1"

:finish
echo Morning run finished %date% %time% (fail=%FAIL%) >> "%LOG%"
if defined FAIL exit /b 1
exit /b 0
