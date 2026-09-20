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
# 2026-09-20: hidden console window. The task died on 2026-08-18 and
# 2026-09-15 with result 0xC000013A (STATUS_CONTROL_C_EXIT) about seven
# seconds after launching, both times when it started late right after the
# laptop woke from sleep, leaving 0-byte logs. It runs as an interactive
# task, which pops a visible PowerShell console for its 1-2 minute run --
# closing that window is exactly a Ctrl-C exit. -WindowStyle Hidden removes
# the window. The root cause is inferred from the exit code and timing, not
# proven; see BUGS.md item 4.
#
# A non-interactive S4U principal (runs whether or not anyone is logged on,
# no stored password) would be the more robust fix and is tried first, but
# registering one needs the "log on as a batch job" right, which this
# account does not have unelevated -- it falls back to the interactive
# principal. Rerun this script from an elevated PowerShell to get S4U.

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$scriptPath = Join-Path $root "scripts\weekly_refresh.ps1"
$taskName = "NFL Model Weekly Refresh"

$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$scriptPath`""
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Tuesday -At 8:00AM
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -DontStopOnIdleEnd `
    -ExecutionTimeLimit (New-TimeSpan -Hours 1) -MultipleInstances IgnoreNew
$description = "Weekly refresh for the NFL ATS/ML model: pulls fresh data, reruns the pipeline, logs upcoming picks."

$registered = $false
try {
    $principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType S4U -RunLevel Limited
    Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings `
        -Principal $principal -Description $description -Force | Out-Null
    $registered = $true
    Write-Output "Registered '$taskName' with an S4U principal (runs whether or not you are logged on), hidden window."
} catch {
    Write-Warning "S4U principal refused ($($_.Exception.Message.Trim())) -- falling back to the interactive principal. Rerun elevated to get S4U."
}
if (-not $registered) {
    Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings `
        -Description $description -Force | Out-Null
    Write-Output "Registered '$taskName' with the interactive principal, hidden window."
}

Write-Output "Runs every Tuesday at 8:00 AM."
$t = Get-ScheduledTask -TaskName $taskName
$t | Select-Object TaskName, State
Write-Output "Principal: $($t.Principal.UserId) / $($t.Principal.LogonType)"
Write-Output "Action:    $($t.Actions[0].Execute) $($t.Actions[0].Arguments)"