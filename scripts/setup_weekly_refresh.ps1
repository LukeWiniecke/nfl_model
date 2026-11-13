# Registers a Windows Scheduled Task that runs weekly_refresh.ps1 every
# Tuesday at 8:00 AM (after Monday Night Football has finished, so the prior
# week's results are final, and the next week's early lines are often
# already posted). Run this ONCE to set it up; safe to rerun.
#
# To check it's registered:   Get-ScheduledTask -TaskName "NFL Model Weekly Refresh"
# To run it immediately:      Start-ScheduledTask -TaskName "NFL Model Weekly Refresh"
# To see its run history:     Get-ScheduledTaskInfo -TaskName "NFL Model Weekly Refresh"
# To remove it entirely:      Unregister-ScheduledTask -TaskName "NFL Model Weekly Refresh" -Confirm:$false
#
# --- Why this task kept dying, and what actually fixes it ---
#
# It failed silently on 2026-08-18, 2026-09-15 and again on 2026-09-22, every
# time with 0xC000013A (STATUS_CONTROL_C_EXIT) and no usable log.
#
# FIRST DIAGNOSIS, WRONG: the battery guards. New-ScheduledTaskSettingsSet
# defaults DisallowStartIfOnBatteries and StopIfGoingOnBatteries to $true, and
# the original script never overrode them. That IS a real misconfiguration --
# reproduced directly on 2026-09-20, where a manual start on battery sat in
# state "Queued" for ten minutes and never ran -- so both are turned off
# below. But it was not what killed the runs: the guards were already off on
# 2026-09-22 and the task died anyway.
#
# SECOND DIAGNOSIS, better evidenced. The event log for 2026-09-22 reads:
#   08:25:20  Kernel-Power 507  "The system is exiting Modern Standby"
#   08:25:24  Task Scheduler launches the task (4 seconds into the resume)
#   08:25:27  Kernel-Power 105  "Power source change"
#   08:26:14  action exits 3221225786 = 0xC000013A, after 50 seconds
# and NO log file was created at all. weekly_refresh.ps1 opens its log within
# milliseconds of starting, so fifty seconds with no file means PowerShell
# never got that far -- the process was starved and then torn down while the
# machine was still coming out of modern standby. StartWhenAvailable is what
# puts it there: the 08:00 trigger is missed while asleep, so it fires the
# instant the machine stirs, which is the worst possible moment.
#
# THE FIX, in three parts, none needing elevation:
#   -WakeToRun          wake the machine FOR the task instead of catching it
#                       mid-resume.
#   RandomDelay 5 min   if it does fire off a resume, start after the system
#                       has settled rather than four seconds in.
#   RestartCount 3      and if it is killed anyway, retry every 15 minutes.
#                       This is the part that does not depend on the
#                       diagnosis being right, which matters given the first
#                       one was not.
#
# Still optional, needs an elevated shell: an S4U principal, so the task does
# not depend on the interactive desktop session at all. Attempted first below
# and falls back with a warning.

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$scriptPath = Join-Path $root "scripts\weekly_refresh.ps1"
$taskName = "NFL Model Weekly Refresh"

$action = New-ScheduledTaskAction -Execute "powershell.exe" `
    -Argument "-NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File `"$scriptPath`""

$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Tuesday -At 8:00AM
$trigger.RandomDelay = "PT5M"

$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -DontStopOnIdleEnd `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -WakeToRun `
    -ExecutionTimeLimit (New-TimeSpan -Hours 1) -MultipleInstances IgnoreNew `
    -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 15)

$description = "Weekly refresh for the NFL ATS/ML model: pulls fresh data, reruns the pipeline, logs upcoming picks."

$registered = $false
try {
    $principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType S4U -RunLevel Limited
    Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings `
        -Principal $principal -Description $description -Force | Out-Null
    $registered = $true
    Write-Output "Registered '$taskName' with an S4U principal (runs whether or not you are logged on)."
} catch {
    Write-Warning "S4U principal refused ($($_.Exception.Message.Trim())) -- using the interactive principal. Rerun this script from an elevated PowerShell to upgrade."
}
if (-not $registered) {
    Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings `
        -Description $description -Force | Out-Null
    Write-Output "Registered '$taskName' with the interactive principal."
}

$t = Get-ScheduledTask -TaskName $taskName
Write-Output "Tuesday 8:00 AM (+ up to 5 min random delay). State: $($t.State)"
Write-Output "Principal:  $($t.Principal.UserId) / $($t.Principal.LogonType)"
Write-Output "Wake to run: $($t.Settings.WakeToRun)   Retries: $($t.Settings.RestartCount) every $($t.Settings.RestartInterval)"
Write-Output "Battery:     DisallowStart=$($t.Settings.DisallowStartIfOnBatteries) StopIfGoing=$($t.Settings.StopIfGoingOnBatteries)"