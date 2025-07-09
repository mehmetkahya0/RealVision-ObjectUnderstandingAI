# RealVision AI - PowerShell Launcher
# =====================================

Write-Host "🎯 RealVision Object Understanding AI" -ForegroundColor Cyan
Write-Host "=" * 50 -ForegroundColor Gray
Write-Host ""

# Check if Python is installed
try {
    $pythonVersion = python --version 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ $pythonVersion found" -ForegroundColor Green
    } else {
        throw "Python not found"
    }
} catch {
    Write-Host "❌ Python is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Python 3.8 or higher from https://python.org" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

# Change to script directory
$scriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptPath

Write-Host ""
Write-Host "🚀 Starting RealVision AI..." -ForegroundColor Yellow

# Run the GUI launcher
try {
    python launch_gui.py
} catch {
    Write-Host ""
    Write-Host "❌ Application failed to start: $($_.Exception.Message)" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "❌ Application closed with an error (Exit code: $LASTEXITCODE)" -ForegroundColor Red
    Read-Host "Press Enter to exit"
}
