param()
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if (-not (Test-Path ".venv")) {
    python -m venv .venv
}
& ".venv\Scripts\python.exe" -m pip install -q -e ".[dev]"

if (-not (Test-Path ".env")) {
    Write-Host "No .env found -- copy .env.example to .env and fill in LiveKit credentials first." -ForegroundColor Yellow
    exit 1
}

& ".venv\Scripts\python.exe" -m roxroom.main
