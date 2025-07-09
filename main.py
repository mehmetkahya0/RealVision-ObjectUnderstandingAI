#!/usr/bin/env python3
"""
RealVision Object Understanding AI - Main Entry Point
====================================================

Professional main entry point for the RealVision Object Understanding AI application.

Usage:
    python main.py [options]           # Main object detection application
    python main.py --gui               # Launch GUI version
    python main.py --demo              # Run analytics demo
    python main.py --help              # Show help

Author: Mehmet Kahya
Date: July 2025
"""

import sys
import os
import argparse
from pathlib import Path

# Add src directory to Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

def main():
    """Main launcher function"""
    parser = argparse.ArgumentParser(
        description="RealVision Object Understanding AI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--gui', 
        action='store_true',
        help='Launch GUI version'
    )
    
    parser.add_argument(
        '--demo', 
        action='store_true',
        help='Run analytics demo'
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
    
    parser.add_argument(
        '--model', 
        type=str, 
        default='auto',
        choices=['auto', 'yolo', 'dnn', 'mediapipe', 'onnx', 'efficientdet'],
        help='Object detection model to use'
    )
    
    args = parser.parse_args()
    
    try:
        if args.gui:
            # Launch GUI version
            from gui import RealVisionGUI
            from PyQt6.QtWidgets import QApplication
            
            app = QApplication(sys.argv)
            app.setApplicationName("RealVision Object Understanding AI")
            app.setApplicationVersion("1.0")
            app.setStyle('Fusion')
            
            main_window = RealVisionGUI()
            main_window.show()
            
            return app.exec()
            
        elif args.demo:
            # Run analytics demo
            from src.demo_analytics import main as demo_main
            return demo_main()
            
        else:
            # Run console version
            from src.main import ObjectUnderstandingApp
            
            app = ObjectUnderstandingApp(preferred_model=args.model)
            
            if args.video:
                return app.run(camera_index=args.video)
            else:
                return app.run(camera_index=args.camera)
                
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Please install required dependencies:")
        print("  pip install -r requirements.txt")
        return 1
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
