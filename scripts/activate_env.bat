@echo off
echo.
echo 🤖 RealVision-ObjectUnderstandingAI Environment Setup
echo ==========================================
echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo.
echo ✅ Virtual environment activated!
echo 📁 Current directory: %CD%
echo 🐍 Python version:
python --version
echo.
echo 🚀 Available commands:
echo   python run.py                         - Run with default camera
echo   python run.py --input media/video.mp4 - Process video file
echo   python run.py --list-cameras           - List available cameras
echo   python run.py --help                   - Show all options
echo.
echo 📊 Data science ^& analytics:
echo   python src/demo_analytics.py          - Demo analytics
echo   python tests/test_data_science.py     - Test analytics features
echo   python tests/test_imports.py          - Verify library imports
echo.
echo 📈 Visualization tools:
echo   python visualization/launch_visualizer.py - Launch visualizer
echo   scripts/visualize_data.bat             - Quick Windows launcher
echo.
echo 💡 Press Ctrl+C to exit the application when running
echo    Type 'deactivate' to exit the virtual environment
echo.
