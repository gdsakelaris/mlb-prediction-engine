@echo off
REM Morning job (Task Scheduler 6:00 AM, safe to run by hand):
REM   1. update_all --retrain   scrape + validate everything, retrain models
REM   2. Tools/4 + Tools/6      grade newest served workbook + refresh the
REM                             served top-K tracker (automated 2026-07-23;
REM                             skipped if the update failed so grading only
REM                             ever sees validated finals)
REM   3. Sundays only           CLV gate refresh over the captured-odds window
REM Serving stays MANUAL by design (user preference 2026-07-19): GUI or
REM "Model/predict.py --serve". Slate fetch + odds capture (Tools/1, Tools/2)
REM also run by hand, but a noon safety-net task (Tools\run_noon_slate.cmd,
REM added 2026-07-19) guarantees one slate and an early odds capture every
REM game day regardless.
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

REM Grade the newest served workbook against last night's finals, then
REM refresh the served top-K precision tracker (feeds the late-Aug W4.21
REM decision). Both are display/ledger-only - no model input. Grading is
REM idempotent, so a failure here (e.g. the workbook left open in Excel
REM locks the file) is retried harmlessly tomorrow. Tools/6 reads the
REM workbook + CSVs directly, so it runs even if Tools/4 failed.
if defined FAIL goto :gate
"%PY%" "%ROOT%\Tools\4) Grade Results.py" >> "%LOG%" 2>&1
if errorlevel 1 set "FAIL=1"
"%PY%" "%ROOT%\Tools\6) Goal Tracker.py" >> "%LOG%" 2>&1
if errorlevel 1 set "FAIL=1"

:gate
if /I not "%DOW%"=="Sunday" goto :finish
for /f %%y in ('powershell -NoProfile -Command "Get-Date (Get-Date).AddDays(-1) -Format yyyy-MM-dd"') do set "YDAY=%%y"
"%PY%" "%ROOT%\Model\evaluate.py" --gate --start 2026-07-15 --end %YDAY% --sims 4000 >> "%LOG%" 2>&1
if errorlevel 1 set "FAIL=1"

:finish
echo Morning run finished %date% %time% (fail=%FAIL%) >> "%LOG%"
if defined FAIL exit /b 1
exit /b 0
