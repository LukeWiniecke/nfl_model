# Weekly data refresh for the NFL model project.
#
# Pulls fresh data (new results, new market lines, updated injury reports),
# reruns the full pipeline, and logs a timestamped snapshot of that week's
# Upcoming picks to the permanent picks log. Meant to run on a schedule
# (see setup_weekly_refresh.ps1) so the dashboard's Upcoming picks / Track
# record tabs stay current without manual intervention.
#
# Run manually with:
#   powershell -File scripts\weekly_refresh.ps1

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$logDir = Join-Path $root "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$timestamp = Get-Date -Format "yyyy-MM-dd_HHmmss"
$outLog = Join-Path $logDir "refresh_$timestamp.out.log"
$errLog = Join-Path $logDir "refresh_$timestamp.err.log"

$python = Join-Path $root ".venv\Scripts\python.exe"

Write-Output "[$timestamp] Starting weekly refresh -- stdout: $outLog"

$proc = Start-Process -FilePath $python -ArgumentList "main.py", "--refresh" `
    -WorkingDirectory $root -NoNewWindow -Wait -PassThru `
    -RedirectStandardOutput $outLog -RedirectStandardError $errLog

if ($proc.ExitCode -ne 0) {
    Write-Output "[$timestamp] Refresh FAILED (exit code $($proc.ExitCode)) -- see $errLog"
    Get-Content $errLog -Tail 30
    exit $proc.ExitCode
}

Write-Output "[$timestamp] Refresh completed successfully."
