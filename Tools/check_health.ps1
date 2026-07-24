# MLB Engine health watchdog ("MLB Engine Health Check" scheduled task:
# at logon + 08:15 + 12:45). The in-script exit codes and status JSON
# can only report a run that HAPPENED — this watchdog is the one
# mechanism that catches a task that never started (the 2026-07-23
# failure mode). Alerts once per day via MessageBox (WinRT toasts are
# fragile on Windows Home; a modal box is unmissable and zero-dep).
$ErrorActionPreference = "SilentlyContinue"
$root = Split-Path $PSScriptRoot -Parent
$logs = Join-Path $root "Logs"
$today = Get-Date -Format yyyy-MM-dd
$now = Get-Date
$problems = @()

# 1) morning job produced a status file today and it says ok
#    (time-gated: only meaningful after 07:00 so a 6:05 logon can't
#    false-alarm mid-run)
if ($now.Hour -ge 7) {
    $sf = Join-Path $logs "last_run_status.json"
    if (-not (Test-Path $sf)) {
        $problems += "No last_run_status.json - the 6AM job has never recorded a run."
    } else {
        $st = Get-Content $sf -Raw | ConvertFrom-Json
        if ($st.finished.Substring(0, 10) -ne $today) {
            $problems += "6AM data update has NOT run today (last: $($st.finished)). Run Scrapers\run_daily_update.cmd."
        } elseif (-not $st.ok) {
            $problems += "6AM data update FAILED today (jobs: $($st.failed_jobs -join ', ')). See newest Logs\update_*.log."
        }
    }
}

# 2) noon slate/odds capture ran (only meaningful after 12:30)
if (($now.Hour -gt 12) -or ($now.Hour -eq 12 -and $now.Minute -ge 30)) {
    $noonLog = Join-Path $logs "noon_$today.log"
    if (-not (Test-Path $noonLog)) {
        $problems += "Noon slate/odds capture has NOT run today - opening odds are being lost. Run Tools\run_noon_slate.cmd."
    } elseif (Select-String -Path $noonLog -Pattern "fail=1" -Quiet) {
        $problems += "Noon slate/odds capture FAILED today - see Logs\noon_$today.log."
    }
}

# 3) both tasks still registered and enabled
foreach ($t in "MLB Engine Daily Update", "MLB Engine Noon Slate") {
    $task = Get-ScheduledTask -TaskName $t
    if (-not $task) { $problems += "Scheduled task '$t' is MISSING." }
    elseif ($task.State -eq "Disabled") { $problems += "Scheduled task '$t' is DISABLED." }
}

if ($problems.Count -eq 0) { exit 0 }

# once-per-day alert (ack marker) + a written record either way
$ack = Join-Path $logs ".health_ack_$today"
Set-Content -Path (Join-Path $logs "health_alert_$today.txt") -Value ($problems -join "`r`n")
if (Test-Path $ack) { exit 0 }
New-Item -ItemType File -Path $ack -Force | Out-Null
Add-Type -AssemblyName System.Windows.Forms
[void][System.Windows.Forms.MessageBox]::Show(
    ($problems -join "`n`n"), "MLB Engine health",
    [System.Windows.Forms.MessageBoxButtons]::OK,
    [System.Windows.Forms.MessageBoxIcon]::Warning)
exit 0
