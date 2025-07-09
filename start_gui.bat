@echo off
title RealVision AI - Object Understanding
echo ========================================
echo  RealVision Object Understanding AI
echo ========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://python.org
    pause
    exit /b 1
)

echo ✓ Python found
echo.

REM Run the GUI launcher
echo 🚀 Starting RealVision AI...
python launch_gui.py

REM Keep window open if there's an error
if errorlevel 1 (
    echo.
    echo ❌ Application closed with an error
    pause
)
