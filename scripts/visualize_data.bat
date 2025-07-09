@echo off
echo.
echo 📊 RealVision Performance Data Visualizer
echo ==========================================
echo.
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo.
echo 🚀 Launching Performance Data Visualizer...
python visualization\launch_visualizer.py
echo.
pause
