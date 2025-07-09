#!/usr/bin/env python3
"""
Simple launcher script for the Real-Time Object Understanding Application
"""

import sys
import os
import argparse
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'cv2', 'numpy', 'torch', 'torchvision', 'ultralytics'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'numpy':
                import numpy
            elif package == 'torch':
                import torch
            elif package == 'torchvision':
                import torchvision
            elif package == 'ultralytics':
                import ultralytics
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\n💡 Run: pip install -r requirements.txt")
        return False
    
    return True

def get_available_cameras():
    """Get list of available cameras with better error handling"""
    try:
        import cv2
        import sys
        import io
        from contextlib import redirect_stderr
        
        cameras = []
        
        # Suppress OpenCV warnings/errors during camera detection
        for i in range(5):  # Check first 5 cameras
            try:
                # Redirect stderr to suppress OpenCV warnings
                f = io.StringIO()
                with redirect_stderr(f):
                    cap = cv2.VideoCapture(i)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    
                    if cap.isOpened():
                        ret, frame = cap.read()
                        if ret and frame is not None and frame.size > 0:
                            cameras.append(i)
                    cap.release()
            except Exception:
                # Skip cameras that cause exceptions
                continue
                
        return cameras
    except ImportError:
        return []

def main():
    """Main launcher function"""
    parser = argparse.ArgumentParser(
        description="Real-Time Object Understanding Application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                              # Use default camera (0)
  python run.py --camera 1                   # Use camera 1
  python run.py --input video.mp4            # Use video file as input
  python run.py --list-cameras               # List available cameras
  python run.py --confidence 0.7             # Set confidence threshold to 0.7
  python run.py --input video.mp4 --model yolo  # Use YOLO with video file
        """
    )
    
    parser.add_argument(
        '--camera', '-c',
        type=int,
        default=0,
        help='Camera index to use (default: 0)'
    )
    
    parser.add_argument(
        '--input', '-i',
        type=str,
        help='Input video file path (if specified, camera will be ignored)'
    )
    
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.5,
        help='Initial confidence threshold (default: 0.5)'
    )
    
    parser.add_argument(
        '--model', '-m',
        choices=['yolo', 'dnn', 'mediapipe', 'onnx', 'efficientdet', 'auto'],
        default='auto',
        help='Object detection model to use (default: auto)'
    )
    
    parser.add_argument(
        '--list-cameras',
        action='store_true',
        help='List available cameras and exit'
    )
    
    parser.add_argument(
        '--no-gui',
        action='store_true',
        help='Run without GUI (headless mode)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='screenshots',
        help='Directory for screenshots (default: screenshots)'
    )
    
    args = parser.parse_args()
    
    print("🎯 Real-Time Object Understanding Application Launcher")
    print("=" * 55)
    
    # Check requirements
    if not check_requirements():
        return 1
     # List cameras if requested
    if args.list_cameras:
        print("📹 Available cameras:")
        cameras = get_available_cameras()
        if cameras:
            for camera_id in cameras:
                print(f"   Camera {camera_id}")
        else:
            print("   No working cameras found")
            print("   💡 Tip: You can use video files instead with --input parameter")
        return 0

    # Check if using video file input
    if args.input:
        if not os.path.exists(args.input):
            print(f"❌ Input video file not found: {args.input}")
            return 1
        print(f"📹 Using video file: {args.input}")
        video_source = args.input
    else:
        # Validate camera
        available_cameras = get_available_cameras()
        if not available_cameras:
            print("❌ No working cameras found!")
            print("💡 Make sure your camera is connected and not being used by another application")
            print("💡 Alternatively, you can use a video file with --input parameter")
            return 1

        if args.camera not in available_cameras:
            print(f"❌ Camera {args.camera} not available!")
            print(f"💡 Available cameras: {available_cameras}")
            return 1
        
        print(f"📹 Using camera: {args.camera}")
        video_source = args.camera
    
    # Import and run the application
    try:
        from main import ObjectUnderstandingApp
        
        print(f"🎯 Confidence threshold: {args.confidence}")
        print(f"🤖 Detection model: {args.model}")
        print(f"📁 Output directory: {args.output_dir}")
        print("=" * 55)
        
        # Create application instance
        app = ObjectUnderstandingApp(preferred_model=args.model)
        
        # Configure application
        app.confidence_threshold = args.confidence
        app.screenshots_dir = args.output_dir
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Run application with video source (camera index or file path)
        app.run(camera_index=video_source)
        
    except ImportError as e:
        print(f"❌ Failed to import main application: {e}")
        return 1
    except Exception as e:
        print(f"❌ Application error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
