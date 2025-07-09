#!/usr/bin/env python3
"""
RealVision Object Understanding AI - Quick Launcher
==================================================

Simple launcher for the RealVision Object Understanding AI application.

Usage:
    python run.py [options]           # Main object detection application

Author: Mehmet Kahya
Date: July 2025
"""

import sys
import os
from pathlib import Path

# Add src directory to Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

if __name__ == "__main__":
    # Import and run the main application
    from run import main
    main()
