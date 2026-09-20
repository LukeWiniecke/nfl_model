# Registers a Windows Scheduled Task that runs weekly_refresh.ps1 every
# Tuesday at 8:00 AM (after Monday Night Football has finished, so the prior
# week's results are final, and the next week's early lines are often
# already posted). Run this ONCE to set it up; safe to rerun (re-registering
# just replaces the existing task definition with the same one).
#
# To change the schedule: edit $trigger below and rerun this script.
# To check it's registered:   Get-ScheduledTask -TaskName "NFL Model Weekly Refresh"
# To run it immediately:      Start-ScheduledTask -TaskName "NFL Model Weekly Refresh"
# To see its run history:     Get-ScheduledTaskInfo -TaskName "NFL Model Weekly Refresh"
# To remove it entirely:      Unregister-ScheduledTask -TaskName "NFL Model Weekly Refresh" -Confirm:$false
#
# --- 2026-09-20: why the battery settings below are explicit ---
#
# The task silently did nothing on 2026-08-18 and 2026-09-15: it launched,
# exited about seven seconds later with 0xC000013A (STATUS_CONTROL_C_EXIT),
# and left 0-byte logs. Both times it started late, right after the laptop
# woke from sleep.
#
# New-ScheduledTaskSettingsSet defaults BOTH battery guards to $true, and the
# original version of this script never overrode them:
#   DisallowStartIfOnBatteries -- the task will not start at all on battery.
#     Reproduced directly on 2026-09-20: a manual Start-ScheduledTask on
#     battery sat in state "Queued" for over ten minutes and never ran, with
#     LastTaskResult 0x0 and no log file at all.
#   StopIfGoingOnBatteries -- a RUNNING task is forcibly terminated the moment
#     the machine drops to battery. That is the better explanation for the two
#     0xC000013A deaths than the visible-console theory this file used to
#     carry: a laptop waking from sleep is exactly when it is likely to be on
#     battery, and a forced stop mid-run produces an abrupt exit code and a
#     truncated log.
#
# A weekly data refresh is a two-minute, low-power job. It should run on
# battery, so both guards are turned off. -WindowStyle Hidden is kept: it
# removes a console window that was a second, independent way for the run to
# be killed, and there is no reason to show one.
#
# An S4U principal (runs whether or not anyone is logged on, no stored
# password) is attempted first and needs the "log on as a batch job" right,
# which this account does not have unelevated; it falls back to the
# interactive principal. Rerun this script from an elevated PowerShell to get
# S4U, which also makes the task independent of the desktop session.

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$scriptPath = Join-Path $root "scripts\weekly_refresh.ps1"
$taskName = "NFL Model Weekly Refresh"

$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$scriptPath`""
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Tuesday -At 8:00AM
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -DontStopOnIdleEnd `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Hours 1) -MultipleInstances IgnoreNew
$description = "Weekly refresh for the NFL ATS/ML model: pulls fresh data, reruns the pipeline, logs upcoming picks."

$registered = $false
try {
    $principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType S4U -RunLevel Limited
    Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings `
        -Principal $principal -Description $description -Force | Out-Null
    $registered = $true
    Write-Output "Registered '$taskName' with an S4U principal (runs whether or not you are logged on)."
} catch {
    Write-Warning "S4U principal refused ($($_.Exception.Message.Trim())) -- falling back to the interactive principal. Rerun elevated to get S4U."
}
if (-not $registered) {
    Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings `
        -Description $description -Force | Out-Null
    Write-Output "Registered '$taskName' with the interactive principal."
}

$t = Get-ScheduledTask -TaskName $taskName
Write-Output "Runs every Tuesday at 8:00 AM. State: $($t.State)"
Write-Output "Principal: $($t.Principal.UserId) / $($t.Principal.LogonType)"
Write-Output "Action:    $($t.Actions[0].Execute) $($t.Actions[0].Arguments)"
Write-Output "Battery:   DisallowStartIfOnBatteries=$($t.Settings.DisallowStartIfOnBatteries) StopIfGoingOnBatteries=$($t.Settings.StopIfGoingOnBatteries)"