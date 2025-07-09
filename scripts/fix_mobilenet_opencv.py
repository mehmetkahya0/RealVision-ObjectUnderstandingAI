#!/usr/bin/env python3
"""
Comprehensive MobileNet-SSD OpenCV Compatibility Fix
This script downloads and verifies OpenCV-compatible MobileNet-SSD model files.
"""

import os
import sys
import urllib.request
import hashlib
import requests
from pathlib import Path

def get_project_root():
    """Get the project root directory"""
    return Path(__file__).parent.parent

def download_file_with_progress(url, filepath, expected_size=None):
    """Download a file with progress indication"""
    try:
        print(f"Downloading from: {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        if expected_size and total_size != expected_size:
            print(f"Warning: Expected size {expected_size}, got {total_size}")
        
        downloaded = 0
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rProgress: {percent:.1f}% ({downloaded}/{total_size} bytes)", end='')
        
        print(f"\nDownloaded: {filepath}")
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False

def verify_file_hash(filepath, expected_hash, hash_type='md5'):
    """Verify file integrity using hash"""
    try:
        hash_func = hashlib.new(hash_type)
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_func.update(chunk)
        
        file_hash = hash_func.hexdigest()
        if file_hash == expected_hash:
            print(f"✓ Hash verification passed for {filepath}")
            return True
        else:
            print(f"✗ Hash mismatch for {filepath}")
            print(f"  Expected: {expected_hash}")
            print(f"  Got:      {file_hash}")
            return False
    except Exception as e:
        print(f"Hash verification failed: {e}")
        return False

def backup_existing_files(models_dir):
    """Backup existing model files"""
    prototxt_path = models_dir / "MobileNetSSD_deploy.prototxt"
    caffemodel_path = models_dir / "MobileNetSSD_deploy.caffemodel"
    
    if prototxt_path.exists():
        backup_path = models_dir / f"MobileNetSSD_deploy.prototxt.old"
        prototxt_path.rename(backup_path)
        print(f"Backed up existing prototxt to {backup_path}")
    
    if caffemodel_path.exists():
        backup_path = models_dir / f"MobileNetSSD_deploy.caffemodel.old"
        caffemodel_path.rename(backup_path)
        print(f"Backed up existing caffemodel to {backup_path}")

def test_opencv_dnn_loading(models_dir):
    """Test if OpenCV can load the DNN model"""
    try:
        import cv2
        print(f"Testing OpenCV DNN model loading...")
        print(f"OpenCV version: {cv2.__version__}")
        
        prototxt_path = str(models_dir / "MobileNetSSD_deploy.prototxt")
        caffemodel_path = str(models_dir / "MobileNetSSD_deploy.caffemodel")
        
        # Try to load the network
        net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
        print("✓ Successfully loaded MobileNet-SSD with OpenCV DNN!")
        
        # Test with a dummy input
        blob = cv2.dnn.blobFromImage(
            cv2.imread("media/sample.jpg") if os.path.exists("media/sample.jpg") else cv2.Mat.zeros((300, 300, 3), cv2.CV_8UC3),
            0.017, (300, 300), (103.94, 116.78, 123.68)
        )
        net.setInput(blob)
        detections = net.forward()
        print(f"✓ Successfully ran inference! Output shape: {detections.shape}")
        return True
        
    except Exception as e:
        print(f"✗ OpenCV DNN loading failed: {e}")
        return False

def main():
    """Main function to fix MobileNet-SSD OpenCV compatibility"""
    print("=" * 60)
    print("MobileNet-SSD OpenCV Compatibility Fix")
    print("=" * 60)
    
    project_root = get_project_root()
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)
    
    print(f"Project root: {project_root}")
    print(f"Models directory: {models_dir}")
    
    # Backup existing files
    print("\n1. Backing up existing files...")
    backup_existing_files(models_dir)
    
    # Download URLs for OpenCV-compatible MobileNet-SSD
    urls = {
        "prototxt": {
            "url": "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/MobileNetSSD_deploy.prototxt",
            "filepath": models_dir / "MobileNetSSD_deploy.prototxt",
            "size": None
        },
        "caffemodel": {
            "url": "https://drive.google.com/uc?export=download&id=0B3gersZ2cHIxRm5PMWRoTkdHdHc",
            "filepath": models_dir / "MobileNetSSD_deploy.caffemodel",
            "size": 23147564  # ~23MB
        }
    }
    
    # Alternative URL for caffemodel if Google Drive fails
    alternative_caffemodel_url = "https://github.com/chuanqi305/MobileNet-SSD/raw/master/MobileNetSSD_deploy.caffemodel"
    
    success = True
    
    # Download prototxt file
    print("\n2. Downloading prototxt file...")
    if not download_file_with_progress(
        urls["prototxt"]["url"], 
        urls["prototxt"]["filepath"]
    ):
        print("Failed to download prototxt file")
        success = False
    
    # Download caffemodel file
    print("\n3. Downloading caffemodel file...")
    if not download_file_with_progress(
        urls["caffemodel"]["url"], 
        urls["caffemodel"]["filepath"],
        urls["caffemodel"]["size"]
    ):
        print("Trying alternative URL...")
        if not download_file_with_progress(
            alternative_caffemodel_url,
            urls["caffemodel"]["filepath"],
            urls["caffemodel"]["size"]
        ):
            print("Failed to download caffemodel file from both URLs")
            success = False
    
    if not success:
        print("\n⚠️  Download failed. Manual download instructions:")
        print("1. Download MobileNetSSD_deploy.prototxt from:")
        print("   https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/MobileNetSSD_deploy.prototxt")
        print("2. Download MobileNetSSD_deploy.caffemodel from:")
        print("   https://github.com/chuanqi305/MobileNet-SSD/raw/master/MobileNetSSD_deploy.caffemodel")
        print(f"3. Place both files in: {models_dir}")
        return False
    
    # Test OpenCV loading
    print("\n4. Testing OpenCV DNN loading...")
    if test_opencv_dnn_loading(models_dir):
        print("\n✅ MobileNet-SSD OpenCV compatibility fix completed successfully!")
        print("\nYou can now use the DNN model with:")
        print("  python run.py --model dnn --input media/traffic.mp4")
        return True
    else:
        print("\n❌ OpenCV DNN loading test failed.")
        print("The model files may be incompatible with your OpenCV version.")
        print("Consider using YOLO or ONNX models instead:")
        print("  python run.py --model yolo --input media/traffic.mp4")
        print("  python run.py --model onnx --input media/traffic.mp4")
        return False

if __name__ == "__main__":
    # Install requests if not available
    try:
        import requests
    except ImportError:
        print("Installing requests...")
        os.system("pip install requests")
        import requests
    
    main()
