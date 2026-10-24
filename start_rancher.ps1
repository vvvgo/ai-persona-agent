# Запуск Rancher одним кликом: Docker-сервисы, VTube Studio, OBS, простой голос и сам Rancher.
$repo = $PSScriptRoot
$py = "$repo\.venv\Scripts\python.exe"
$host.UI.RawUI.WindowTitle = "Запуск Rancher"

function Test-Port([int]$port) {
    $c = New-Object System.Net.Sockets.TcpClient
    try { return $c.ConnectAsync("127.0.0.1", $port).Wait(500) -and $c.Connected } catch { return $false } finally { $c.Close() }
}

function Wait-Port([int]$port, [string]$name, [int]$timeoutSec) {
    Write-Host "Жду: $name..." -NoNewline
    $deadline = (Get-Date).AddSeconds($timeoutSec)
    while ((Get-Date) -lt $deadline) {
        if (Test-Port $port) { Write-Host " готово" -ForegroundColor Green; return $true }
        Start-Sleep 2
        Write-Host "." -NoNewline
    }
    Write-Host " НЕ ЗАПУСТИЛОСЬ за $timeoutSec сек" -ForegroundColor Red
    return $false
}

function Test-NetTyanRunning([string]$pattern) {
    return [bool](Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object { $_.CommandLine -match $pattern })
}

Write-Host "=== 1. Docker: база, Redis, распознавание речи ===" -ForegroundColor Cyan
wsl.exe --cd "$repo" -- bash -lc "docker compose up -d db redis parakeet 2>&1 | grep -v SPEECH_CUDA"
$ok = (Wait-Port 5432 "база данных" 60) -and (Wait-Port 6379 "Redis" 60)
[void](Wait-Port 22233 "распознавание речи" 120)
if (-not $ok) { Write-Host "Без базы и Redis Rancher не запустится. Проверь Docker в WSL." -ForegroundColor Red; Read-Host "Enter - выход"; exit 1 }

Write-Host "=== 2. VTube Studio и OBS ===" -ForegroundColor Cyan
if (-not (Get-Process "VTube Studio" -ErrorAction SilentlyContinue)) { Start-Process "steam://rungameid/1325860" }
if (-not (Get-Process "obs64" -ErrorAction SilentlyContinue)) {
    Start-Process -FilePath "C:\Program Files\obs-studio\bin\64bit\obs64.exe" -WorkingDirectory "C:\Program Files\obs-studio\bin\64bit"
}
[void](Wait-Port 22232 "VTube Studio" 120)
[void](Wait-Port 4455 "OBS" 60)

Write-Host "=== 3. Голос ===" -ForegroundColor Cyan
if (-not (Test-Port 22231)) {
    Start-Process -FilePath $py -ArgumentList "simple_tts_server.py" -WorkingDirectory $repo
}
if (-not (Wait-Port 22231 "голос" 300)) { Write-Host "Rancher запустится без голоса." -ForegroundColor Yellow }

Write-Host "=== 4. Rancher ===" -ForegroundColor Cyan
if (Test-NetTyanRunning "main\.py") {
    Write-Host "Rancher уже запущен." -ForegroundColor Yellow
} else {
    Start-Process -FilePath "powershell.exe" -ArgumentList "-NoProfile -ExecutionPolicy Bypass -File `"$repo\run_rancher.ps1`""
    [void](Wait-Port 22228 "панель управления" 120)
}

Write-Host ""
Write-Host "Готово! Зайди в Minecraft (Rancher подключится к игре сам)." -ForegroundColor Green
Write-Host "Панель управления: http://localhost:22228"
Write-Host "Остановить Rancher: закрой окно «Rancher» и чёрное окно python (голос)."
Read-Host "Enter - закрыть это окно"
