# RealVision-ObjectUnderstandingAI Environment Setup
# ================================================

Write-Host ""
Write-Host "🤖 RealVision-ObjectUnderstandingAI Environment Setup" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "Activating virtual environment..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"

Write-Host ""
Write-Host "✅ Virtual environment activated!" -ForegroundColor Green
Write-Host "📁 Current directory: $(Get-Location)" -ForegroundColor Blue
Write-Host "🐍 Python version:" -ForegroundColor Blue
& python --version

Write-Host ""
Write-Host "🚀 Available commands:" -ForegroundColor Yellow
Write-Host "  python run.py                         - Run with default camera" -ForegroundColor White
Write-Host "  python run.py --input media/video.mp4 - Process video file" -ForegroundColor White
Write-Host "  python run.py --list-cameras           - List available cameras" -ForegroundColor White
Write-Host "  python run.py --help                   - Show all options" -ForegroundColor White

Write-Host ""
Write-Host "📊 Data science & analytics:" -ForegroundColor Yellow
Write-Host "  python src/demo_analytics.py          - Demo analytics" -ForegroundColor White
Write-Host "  python tests/test_data_science.py     - Test analytics features" -ForegroundColor White
Write-Host "  python tests/test_imports.py          - Verify library imports" -ForegroundColor White

Write-Host ""
Write-Host "📈 Visualization tools:" -ForegroundColor Yellow
Write-Host "  python visualization/launch_visualizer.py - Launch visualizer" -ForegroundColor White
Write-Host "  scripts/visualize_data.bat             - Quick Windows launcher" -ForegroundColor White

Write-Host ""
Write-Host "💡 Press Ctrl+C to exit the application when running" -ForegroundColor Magenta
Write-Host "   Type 'deactivate' to exit the virtual environment" -ForegroundColor Magenta
Write-Host ""
