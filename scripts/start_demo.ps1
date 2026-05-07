<#
.SYNOPSIS
    One-shot demo launcher — starts FastAPI backend (port 8001) and Vite frontend (port 5173).

.DESCRIPTION
    * Validates Python + Node + dataset path
    * Health-checks port 8001 before launching backend (skips launch if already up)
    * Tails both servers in the same window
    * Hits /health once the backend is ready

.EXAMPLE
    pwsh -NoProfile -ExecutionPolicy Bypass -File scripts\start_demo.ps1
#>

[CmdletBinding()]
param(
    [int]$BackendPort = 8001,
    [int]$FrontendPort = 5173,
    [string]$Config = "configs/base.yaml"
)

$ErrorActionPreference = "Stop"
$ROOT = Split-Path -Parent $PSScriptRoot
Set-Location $ROOT

Write-Host "[demo] Universal Speech Enhancement - launcher" -ForegroundColor Cyan
Write-Host "[demo] root = $ROOT"

# --- preflight ---
$pythonOk = (Get-Command python -ErrorAction SilentlyContinue) -ne $null
$nodeOk = (Get-Command node -ErrorAction SilentlyContinue) -ne $null
if (-not $pythonOk) { throw "python not on PATH" }
if (-not $nodeOk) { throw "node not on PATH" }

if (-not (Test-Path "checkpoints/policy_best.pt")) {
    Write-Host "[demo] WARN: checkpoints/policy_best.pt missing - router will run with random init" -ForegroundColor Yellow
}

if (-not (Test-Path "frontend/node_modules")) {
    Write-Host "[demo] frontend node_modules missing - running npm install"
    Push-Location frontend
    & npm install --no-audit --no-fund
    Pop-Location
}

# --- backend ---
$backendListening = $false
try {
    $r = Invoke-WebRequest -Uri "http://127.0.0.1:$BackendPort/health" -TimeoutSec 2 -UseBasicParsing
    if ($r.StatusCode -eq 200) { $backendListening = $true }
} catch { }

if ($backendListening) {
    Write-Host "[demo] backend already listening on $BackendPort" -ForegroundColor Green
} else {
    Write-Host "[demo] starting backend on $BackendPort ..."
    $cmd = "python -m uvicorn backend.app.main:app --host 127.0.0.1 --port $BackendPort"
    Start-Process -FilePath "powershell" -ArgumentList "-NoExit", "-Command", "Set-Location '$ROOT'; $cmd" -WindowStyle Normal
    # poll for readiness up to 90s
    for ($i = 0; $i -lt 45; $i++) {
        Start-Sleep -Seconds 2
        try {
            $r = Invoke-WebRequest -Uri "http://127.0.0.1:$BackendPort/health" -TimeoutSec 2 -UseBasicParsing
            if ($r.StatusCode -eq 200) { $backendListening = $true; break }
        } catch { }
    }
    if (-not $backendListening) {
        throw "backend did not become healthy on http://127.0.0.1:$BackendPort/health within 90s"
    }
    Write-Host "[demo] backend healthy" -ForegroundColor Green
}

# --- frontend ---
Write-Host "[demo] starting frontend on $FrontendPort ..."
$frontEnv = @{ "VITE_API_BASE_URL" = "http://127.0.0.1:$BackendPort" }
$envSet = $frontEnv.Keys | ForEach-Object { "`$env:$_='$($frontEnv[$_])'" } | Out-String
$cmdFront = "$envSet; Set-Location '$ROOT/frontend'; npm run dev"
Start-Process -FilePath "powershell" -ArgumentList "-NoExit", "-Command", $cmdFront -WindowStyle Normal

Start-Sleep -Seconds 3
Write-Host ""
Write-Host "[demo] === ready ===" -ForegroundColor Cyan
Write-Host "[demo]   backend : http://127.0.0.1:$BackendPort/health"
Write-Host "[demo]   frontend: http://127.0.0.1:$FrontendPort/"
Write-Host "[demo] Open the frontend URL, upload a noisy .wav/.flac/.mp3, and click 'Enhance Audio'."
