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

REM ---- cross-task mutex (shared with Tools\run_noon_slate.cmd) -----------
REM StartWhenAvailable catch-up runs can land next to the noon task (they
REM overlapped 2026-07-23); a lock directory serializes them - mkdir is
REM atomic, exactly one process wins. A lock older than 180 min is from a
REM crashed/killed run (both tasks' ExecutionTimeLimit <= 3h): it is broken
REM by an atomic rename (move), so two waiters can never both claim the
REM break, and every loop iteration is counted and slept so a lock that
REM will not die can never spin the script. The release checks a token
REM written at acquire, so a run whose lock was stale-broken cannot delete
REM the next holder's lock. After 30 minutes of waiting we proceed WITHOUT
REM the lock: a lost odds capture is irreplaceable data, a rare overlap
REM only risks a retry.
set "LOCKDIR=%ROOT%\.task_lock"
set "LOCKOWNED="
set "LOCKTOKEN="
set /a LOCKTRIES=0
:acquire
mkdir "%LOCKDIR%" 2>nul && goto :locked
powershell -NoProfile -Command "exit [int]((Test-Path -LiteralPath '%LOCKDIR%') -and (((Get-Date) - (Get-Item -LiteralPath '%LOCKDIR%').CreationTime).TotalMinutes -lt 180))"
if errorlevel 1 goto :lockwait
set "STALE=%LOCKDIR%.stale%RANDOM%"
move "%LOCKDIR%" "%STALE%" >nul 2>&1
if errorlevel 1 goto :lockwait
echo Broke stale task lock %date% %time% >> "%LOG%"
rmdir /s /q "%STALE%" 2>nul
goto :acquire
:lockwait
set /a LOCKTRIES+=1
if %LOCKTRIES% geq 30 (
    echo Task lock still held after 30 min - proceeding without it %date% %time% >> "%LOG%"
    goto :run
)
if %LOCKTRIES% equ 1 echo Waiting for task lock %date% %time% >> "%LOG%"
powershell -NoProfile -Command "Start-Sleep -Seconds 60"
goto :acquire
:locked
set "LOCKOWNED=1"
set "LOCKTOKEN=%RANDOM%%RANDOM%"
type nul > "%LOCKDIR%\t%LOCKTOKEN%" 2>nul
:run

REM ship-gate backstop: a red fast suite must never feed a retrain, but
REM the DATA capture still runs (odds/lineups vanish forever and must
REM not be blocked by a broken test tree) - the job degrades to
REM data-only, marks FAIL, and exits 1 so the red day is visible.
REM --basetemp: this task runs elevated, and an elevated pytest leaves
REM admin-owned dirs in the user's shared pytest temp root that a later
REM interactive pytest cannot clean (crashes its teardown, exit 1 -
REM which blocked a commit via the pre-commit hook on 2026-07-24). A
REM task-private temp root keeps the two contexts apart for good.
set "TESTS_RED="
"%PY%" -m pytest "%ROOT%\Tests" -q --basetemp "%ROOT%\Logs\pytest_tmp" >> "%LOG%" 2>&1
if errorlevel 1 set "TESTS_RED=1"
if defined TESTS_RED echo pytest RED - degrading to data-only update (no retrain) >> "%LOG%"
if defined TESTS_RED set "FAIL=1"

if defined TESTS_RED (
    "%PY%" "%ROOT%\Scrapers\update_all.py" >> "%LOG%" 2>&1
) else (
    "%PY%" "%ROOT%\Scrapers\update_all.py" --retrain >> "%LOG%" 2>&1
)
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
REM --start auto = day after the served calibrators' fit window ends, so
REM the Sunday verdict is never graded in-sample and the window resets
REM itself at every calibration refresh (bounded runtime).
"%PY%" "%ROOT%\Model\evaluate.py" --gate --start auto --end %YDAY% --sims 4000 >> "%LOG%" 2>&1
if errorlevel 1 set "FAIL=1"
REM weekly slow lane: golden replay + artifact-dependent tests run where
REM they can never be skipped by a fast local loop (same private temp
REM root - see the --basetemp note above)
"%PY%" -m pytest "%ROOT%\Tests" -q -m slow --basetemp "%ROOT%\Logs\pytest_tmp" >> "%LOG%" 2>&1
if errorlevel 1 set "FAIL=1"

:finish
echo Morning run finished %date% %time% (fail=%FAIL%) >> "%LOG%"
if defined LOCKOWNED if exist "%LOCKDIR%\t%LOCKTOKEN%" rmdir /s /q "%LOCKDIR%" 2>nul
if defined FAIL exit /b 1
exit /b 0
