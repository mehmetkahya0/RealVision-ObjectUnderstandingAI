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
echo   python run.py                    - Run with default camera
echo   python run.py --input video.mp4 - Process video file
echo   python run.py --list-cameras    - List available cameras
echo   python run.py --help            - Show all options
echo.
echo 📊 To test data science features:
echo   python demo_analytics.py        - Demo analytics
echo   python test_data_science.py     - Test analytics features
echo.
echo 💡 Press Ctrl+C to exit the application when running
echo    Type 'deactivate' to exit the virtual environment
echo.
