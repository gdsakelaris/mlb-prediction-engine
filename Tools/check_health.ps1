# MLB Engine health watchdog ("MLB Engine Health Check" scheduled task:
# at logon + 08:15 + 12:45). The in-script exit codes and status JSON
# can only report a run that HAPPENED — this watchdog is the one
# mechanism that catches a task that never started (the 2026-07-23
# failure mode). Alerts via MessageBox once per day PER PROBLEM KIND
# (an early false alarm's ack must never mute a later, different real
# failure). FAIL-CLOSED (2026-07-25): a corrupt/unreadable status file
# or a crashed check IS an alert, never a silent pass — so no blanket
# SilentlyContinue; every check is try/caught individually. Runs the
# same under powershell.exe 5.1 (the scheduled context) and pwsh 7
# (the user's shell): pwsh 7's ConvertFrom-Json auto-converts ISO-8601
# strings to [DateTime], so date fields are handled as either type.
# Exit 0 = healthy, 1 = problems (recorded in Logs\health_alert_*.txt).
param([switch]$NoPopup)   # manual/dry run: report + record, no modal, no ack burn
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
$logs = Join-Path $root "Logs"
$today = Get-Date -Format yyyy-MM-dd
$now = Get-Date
$problems = @()

function Add-Problem([string]$kind, [string]$msg) {
    # kind = stable slug for the per-day ack marker (one popup per kind)
    $script:problems += [pscustomobject]@{ Kind = $kind; Msg = $msg }
}

function Test-TaskRunning([string]$name) {
    # a Running task legitimately has stale status/log files (the morning
    # ExecutionTimeLimit is PT3H, so an 08:15 check can land mid-run —
    # that false alarm used to burn the day's single ack). Unknown state
    # counts as NOT running so the file checks still fire (fail closed).
    try { return ((Get-ScheduledTask -TaskName $name -ErrorAction Stop).State -eq "Running") }
    catch { return $false }
}

function Get-DayString($v) {
    # 'finished'/'date' is ISO-8601 text under 5.1 but a [DateTime]
    # under pwsh 7 (ConvertFrom-Json auto-conversion). Throws — i.e.
    # alerts — on anything that resolves to neither.
    if ($v -is [datetime]) { return $v.ToString("yyyy-MM-dd") }
    $s = [string]$v
    if ($s.Length -ge 10) { $s = $s.Substring(0, 10) }
    if ($s -notmatch '^\d{4}-\d{2}-\d{2}$') { throw "unparseable date '$v'" }
    return $s
}

# 1) morning job produced a status file today and it says ok
#    (time-gated: only meaningful after 07:00 so a 6:05 logon can't
#    false-alarm mid-run; skipped while the task is actually Running)
if ($now.Hour -ge 7 -and -not (Test-TaskRunning "MLB Engine Daily Update")) {
    $ranToday = $false
    try {
        $sf = Join-Path $logs "last_run_status.json"
        if (-not (Test-Path $sf)) {
            Add-Problem "morning-missing" "No last_run_status.json - the 6AM job has never recorded a run."
        } else {
            $st = Get-Content $sf -Raw | ConvertFrom-Json
            if ($null -eq $st) { throw "empty JSON" }
            if ((Get-DayString $st.finished) -ne $today) {
                Add-Problem "morning-stale" "6AM data update has NOT run today (last: $($st.finished)). Run Scrapers\run_daily_update.cmd."
            } elseif (-not $st.ok) {
                $ranToday = $true
                Add-Problem "morning-failed" "6AM data update FAILED today (jobs: $($st.failed_jobs -join ', ')). See newest Logs\update_*.log."
            } else { $ranToday = $true }
        }
    } catch {
        # a corrupt status file IS an alert (fail closed, 2026-07-25)
        Add-Problem "morning-status-corrupt" "last_run_status.json is unreadable ($($_.Exception.Message)) - treating the 6AM run as FAILED. See newest Logs\update_*.log."
    }
    # 1b) cmd-stage sidecar (written by run_daily_update.cmd): a red
    #     suite (retrain silently skipped), grading, Sunday gate or
    #     slow-lane failures never reach update_all's ok:true, so a
    #     "healthy" morning status alone can hide a degrading engine.
    try {
        $cf = Join-Path $logs "cmd_status.json"
        $cs = $null
        if (Test-Path $cf) {
            $cs = Get-Content $cf -Raw | ConvertFrom-Json
            if ($null -ne $cs -and (Get-DayString $cs.date) -ne $today) { $cs = $null }
        }
        if ($null -eq $cs) {
            if ($ranToday) { Add-Problem "stage-missing" "update ran today but cmd_status.json is missing/stale - run_daily_update.cmd crashed mid-run (grading/gate state unknown). See Logs\update_$today.log." }
        } else {
            if ($cs.tests_red) { Add-Problem "suite-red" "6AM fast suite was RED - retrain was SKIPPED (data-only update), the served model is going stale. Fix the suite: Logs\update_$today.log." }
            if (-not $cs.data_update_ok) { Add-Problem "data-fail" "6AM data update FAILED - grading, gate refresh and slow lane were skipped; Data\ may be the restored backup. See Logs\update_$today.log." }
            if ([string]$cs.grading -eq "failed") { Add-Problem "grading-fail" "Post-update grading (Tools 4/6) FAILED today - see Logs\update_$today.log." }
            if ([string]$cs.gate -eq "failed") { Add-Problem "gate-fail" "Sunday CLV-gate refresh FAILED - market_gate_report.csv is stale, staking eligibility may be wrong. See Logs\update_$today.log." }
            if ([string]$cs.slow_lane -eq "failed") { Add-Problem "slow-fail" "Weekly slow lane (golden/parity tests) FAILED - see Logs\update_$today.log." }
        }
    } catch {
        Add-Problem "stage-corrupt" "cmd_status.json is unreadable ($($_.Exception.Message)) - cmd-stage outcomes unknown. See Logs\update_$today.log."
    }
}

# 2) noon slate/odds capture ran (only meaningful after 12:30; skipped
#    while the task is actually Running, e.g. waiting on the task lock)
if ((($now.Hour -gt 12) -or ($now.Hour -eq 12 -and $now.Minute -ge 30)) -and
    -not (Test-TaskRunning "MLB Engine Noon Slate")) {
    try {
        $noonLog = Join-Path $logs "noon_$today.log"
        if (-not (Test-Path $noonLog)) {
            Add-Problem "noon-missing" "Noon slate/odds capture has NOT run today - opening odds are being lost. Run Tools\run_noon_slate.cmd."
        } else {
            $fin = Select-String -Path $noonLog -Pattern "Noon slate run finished" | Select-Object -Last 1
            if ($null -eq $fin) {
                Add-Problem "noon-crashed" "Noon slate run STARTED but never finished (crashed or killed?) - see Logs\noon_$today.log."
            } elseif ($fin.Line -match "fail=1") {
                Add-Problem "noon-fail" "Noon slate/odds capture FAILED today - see Logs\noon_$today.log."
            }
        }
    } catch {
        Add-Problem "noon-check-error" "Noon health check itself failed ($($_.Exception.Message)) - inspect Logs\noon_$today.log by hand."
    }
}

# 3) both tasks still registered and enabled
foreach ($t in "MLB Engine Daily Update", "MLB Engine Noon Slate") {
    $slug = $t -replace '[^A-Za-z0-9]', ''
    try { $task = Get-ScheduledTask -TaskName $t -ErrorAction Stop } catch { $task = $null }
    if (-not $task) { Add-Problem "task-missing-$slug" "Scheduled task '$t' is MISSING." }
    elseif ($task.State -eq "Disabled") { Add-Problem "task-disabled-$slug" "Scheduled task '$t' is DISABLED." }
}

if ($problems.Count -eq 0) { exit 0 }

# written record + console echo on every problem run; popup once per
# day PER KIND (per-kind ack markers): a morning false alarm can no
# longer consume the ack and silence a distinct afternoon failure.
$msgs = @($problems | ForEach-Object { $_.Msg })
try { Set-Content -Path (Join-Path $logs "health_alert_$today.txt") -Value ($msgs -join "`r`n") } catch {}
$msgs | ForEach-Object { Write-Output "HEALTH: $_" }
if ($NoPopup) { exit 1 }
$unacked = @($problems | Where-Object { -not (Test-Path (Join-Path $logs ".health_ack_${today}_$($_.Kind)")) })
if ($unacked.Count -eq 0) { exit 1 }
foreach ($p in $problems) {
    try { New-Item -ItemType File -Path (Join-Path $logs ".health_ack_${today}_$($p.Kind)") -Force | Out-Null } catch {}
}
try {
    Add-Type -AssemblyName System.Windows.Forms
    [void][System.Windows.Forms.MessageBox]::Show(
        ($msgs -join "`n`n"), "MLB Engine health",
        [System.Windows.Forms.MessageBoxButtons]::OK,
        [System.Windows.Forms.MessageBoxIcon]::Warning)
} catch {}
exit 1
