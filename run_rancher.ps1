# Окно Rancher: запускает main.py и пишет лог в data\logs (для разбора проблем).
$host.UI.RawUI.WindowTitle = "Rancher"
[Console]::OutputEncoding = [Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"
$logDir = "$PSScriptRoot\data\logs"
New-Item -ItemType Directory -Force $logDir | Out-Null
$log = Join-Path $logDir ("rancher_" + (Get-Date -Format "yyyy-MM-dd_HH-mm") + ".log")
Set-Location "$PSScriptRoot\src"
& "$PSScriptRoot\.venv\Scripts\python.exe" main.py -c -o -f 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log
