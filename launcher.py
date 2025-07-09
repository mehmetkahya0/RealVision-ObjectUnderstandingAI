#!/usr/bin/env python3
"""
RealVision AI - Professional Launcher Script
============================================

This script serves as the main entry point for the RealVision Object Understanding AI application.
It handles dependency checking, error handling, and provides multiple launch modes.

Author: Mehmet Kahya
Date: July 2025
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_modules = [
        ('cv2', 'opencv-python'),
        ('numpy', 'numpy'),
        ('PyQt6', 'PyQt6'),
        ('torch', 'torch'),
        ('ultralytics', 'ultralytics'),
        ('PIL', 'Pillow'),
        ('matplotlib', 'matplotlib'),
        ('pandas', 'pandas'),
        ('seaborn', 'seaborn'),
    ]
    
    missing = []
    for module, package in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    
    if missing:
        print("❌ Missing required dependencies:")
        for package in missing:
            print(f"  - {package}")
        print("\nPlease install them using:")
        print(f"  pip install {' '.join(missing)}")
        return False
    
    return True

def launch_gui():
    """Launch the GUI version"""
    try:
        from gui import RealVisionGUI
        from PyQt6.QtWidgets import QApplication
        
        app = QApplication(sys.argv)
        app.setApplicationName("RealVision Object Understanding AI")
        app.setApplicationVersion("1.0")
        app.setStyle('Fusion')
        
        # Set application icon if available
        icon_path = Path("media/icon.ico")
        if icon_path.exists():
            from PyQt6.QtGui import QIcon
            app.setWindowIcon(QIcon(str(icon_path)))
        
        main_window = RealVisionGUI()
        main_window.show()
        
        return app.exec()
        
    except Exception as e:
        print(f"❌ Failed to launch GUI: {e}")
        return 1

def launch_console():
    """Launch the console version"""
    try:
        from src.main import ObjectUnderstandingApp
        
        app = ObjectUnderstandingApp()
        return app.run()
        
    except Exception as e:
        print(f"❌ Failed to launch console application: {e}")
        return 1

def main():
    """Main entry point"""
    print("🎯 RealVision Object Understanding AI")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        input("Press Enter to exit...")
        return 1
    
    parser = argparse.ArgumentParser(
        description="RealVision Object Understanding AI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--console', 
        action='store_true',
        help='Launch console version instead of GUI'
    )
    
    parser.add_argument(
        '--camera', 
        type=int, 
        default=0,
        help='Camera index (default: 0)'
    )
    
    parser.add_argument(
        '--video', 
        type=str,
        help='Path to video file'
    )
    
    args = parser.parse_args()
    
    if args.console:
        return launch_console()
    else:
        return launch_gui()

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n⚠️  Application interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        input("Press Enter to exit...")
        sys.exit(1)
