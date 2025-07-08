#!/usr/bin/env python3
"""
Setup script for Real-Time Object Understanding Application
This script helps install optional AI models and dependencies.
"""

import subprocess
import sys
import os

def print_banner():
    print("\n" + "="*70)
    print("🎯 REAL-TIME OBJECT UNDERSTANDING - SETUP")
    print("="*70)
    print("This script will help you install additional AI models for enhanced detection.")
    print("="*70 + "\n")

def install_package(package_name, description):
    """Install a package with user confirmation"""
    print(f"\n📦 {description}")
    print(f"Package: {package_name}")
    
    response = input("Install this package? [y/N]: ").lower().strip()
    if response in ['y', 'yes']:
        try:
            print(f"Installing {package_name}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
            print(f"✅ {package_name} installed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package_name}: {e}")
            return False
    else:
        print("⏭️  Skipped")
        return False

def check_package(package_name):
    """Check if a package is already installed"""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False

def main():
    print_banner()
    
    # Check current installation
    print("🔍 Checking current installation...")
    
    packages = [
        ("cv2", "opencv-python", "Basic computer vision (REQUIRED)"),
        ("numpy", "numpy", "Numerical computing (REQUIRED)"),
        ("torch", "torch", "PyTorch deep learning (REQUIRED)"),
        ("ultralytics", "ultralytics", "YOLO v8 models (RECOMMENDED)"),
        ("mediapipe", "mediapipe", "MediaPipe hand/object tracking"),
        ("onnxruntime", "onnxruntime", "ONNX model runtime"),
    ]
    
    installed = []
    missing = []
    
    for import_name, package_name, description in packages:
        if check_package(import_name):
            installed.append((package_name, description))
            print(f"✅ {package_name} - {description}")
        else:
            missing.append((package_name, description))
            print(f"❌ {package_name} - {description}")
    
    print(f"\n📊 Summary: {len(installed)} installed, {len(missing)} missing")
    
    if not missing:
        print("\n🎉 All packages are already installed!")
        print("You have access to all AI models!")
        return
    
    # Install missing packages
    print("\n" + "="*70)
    print("OPTIONAL PACKAGE INSTALLATION")
    print("="*70)
    
    required_missing = [pkg for pkg in missing if pkg[0] in ["opencv-python", "numpy", "torch", "ultralytics"]]
    optional_missing = [pkg for pkg in missing if pkg[0] not in ["opencv-python", "numpy", "torch", "ultralytics"]]
    
    # Install required packages first
    if required_missing:
        print("\n🚨 REQUIRED PACKAGES (application won't work without these):")
        for package_name, description in required_missing:
            install_package(package_name, description)
    
    # Install optional packages
    if optional_missing:
        print("\n💡 OPTIONAL PACKAGES (enable additional AI models):")
        
        for package_name, description in optional_missing:
            if package_name == "mediapipe":
                print("\n🤚 MediaPipe Features:")
                print("   - Hand detection and tracking")
                print("   - 3D object detection (cups, chairs, etc.)")
                print("   - Face detection and mesh")
                install_package(package_name, description)
                
            elif package_name == "onnxruntime":
                print("\n🧠 ONNX Runtime Features:")
                print("   - Support for ONNX models (YOLOv5, custom models)")
                print("   - Cross-platform optimized inference")
                print("   - Support for GPU acceleration")
                install_package(package_name, description)
    
    # Final check
    print("\n" + "="*70)
    print("INSTALLATION COMPLETE")
    print("="*70)
    
    print("\n🔍 Final check...")
    available_models = []
    
    if check_package("ultralytics"):
        available_models.append("YOLO v8")
    if check_package("cv2"):
        available_models.append("MobileNet-SSD (OpenCV DNN)")
    if check_package("mediapipe"):
        available_models.append("MediaPipe")
    if check_package("onnxruntime"):
        available_models.append("ONNX Runtime")
    
    print(f"\n🤖 Available AI Models: {', '.join(available_models)}")
    
    if len(available_models) >= 2:
        print("🎉 Multiple AI models available! You can switch between them using 'M' key.")
    
    print("\n🚀 Ready to run the application!")
    print("Use: python run.py")
    
    # Show advanced usage
    print("\n💡 Advanced Usage Examples:")
    print("   python run.py --model yolo        # Force YOLO model")
    print("   python run.py --model mediapipe   # Use MediaPipe")
    print("   python run.py --model onnx        # Use ONNX model")
    print("   python run.py --model auto        # Auto-select best model")
    print("   python run.py --list-cameras      # List available cameras")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Setup interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ Setup error: {e}")
        print("Please check your Python installation and try again.")
