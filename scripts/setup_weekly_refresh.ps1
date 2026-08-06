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

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$scriptPath = Join-Path $root "scripts\weekly_refresh.ps1"
$taskName = "NFL Model Weekly Refresh"

$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$scriptPath`""
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Tuesday -At 8:00AM
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -DontStopOnIdleEnd -ExecutionTimeLimit (New-TimeSpan -Hours 1)

Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings `
    -Description "Weekly refresh for the NFL ATS/ML model: pulls fresh data, reruns the pipeline, logs upcoming picks." `
    -Force

Write-Output "Registered scheduled task '$taskName' -- runs every Tuesday at 8:00 AM."
Get-ScheduledTask -TaskName $taskName | Select-Object TaskName, State
