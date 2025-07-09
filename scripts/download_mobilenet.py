#!/usr/bin/env python3
"""
Simple MobileNet-SSD Model Downloader
Downloads OpenCV-compatible MobileNet-SSD model files using urllib.
"""

import os
import urllib.request
import urllib.error
from pathlib import Path

def download_mobilenet_files():
    """Download MobileNet-SSD files for OpenCV compatibility"""
    project_root = Path(__file__).parent.parent
    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)
    
    print("Downloading OpenCV-compatible MobileNet-SSD files...")
    print(f"Target directory: {models_dir}")
    
    # URLs for OpenCV-compatible files
    prototxt_url = "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/MobileNetSSD_deploy.prototxt"
    
    # Alternative URLs for the caffemodel
    caffemodel_urls = [
        "https://github.com/chuanqi305/MobileNet-SSD/raw/master/MobileNetSSD_deploy.caffemodel",
        "https://drive.google.com/uc?export=download&id=0B3gersZ2cHIxRm5PMWRoTkdHdHc"
    ]
    
    success = True
    
    # Download prototxt
    prototxt_path = models_dir / "MobileNetSSD_deploy.prototxt"
    print(f"\n1. Downloading prototxt file...")
    try:
        urllib.request.urlretrieve(prototxt_url, prototxt_path)
        print(f"✓ Downloaded: {prototxt_path}")
    except Exception as e:
        print(f"✗ Failed to download prototxt: {e}")
        success = False
    
    # Download caffemodel
    caffemodel_path = models_dir / "MobileNetSSD_deploy.caffemodel"
    print(f"\n2. Downloading caffemodel file...")
    
    caffemodel_success = False
    for i, url in enumerate(caffemodel_urls, 1):
        try:
            print(f"   Trying URL {i}: {url}")
            urllib.request.urlretrieve(url, caffemodel_path)
            
            # Check file size (should be around 23MB)
            file_size = os.path.getsize(caffemodel_path)
            if file_size > 20_000_000:  # At least 20MB
                print(f"✓ Downloaded: {caffemodel_path} ({file_size:,} bytes)")
                caffemodel_success = True
                break
            else:
                print(f"   File too small ({file_size:,} bytes), trying next URL...")
        except Exception as e:
            print(f"   Failed: {e}")
    
    if not caffemodel_success:
        print("✗ Failed to download caffemodel from all URLs")
        success = False
    
    return success

def test_model_loading():
    """Test if the downloaded models work with OpenCV"""
    try:
        import cv2
        project_root = Path(__file__).parent.parent
        models_dir = project_root / "models"
        
        prototxt_path = str(models_dir / "MobileNetSSD_deploy.prototxt")
        caffemodel_path = str(models_dir / "MobileNetSSD_deploy.caffemodel")
        
        print(f"\n3. Testing OpenCV DNN loading...")
        print(f"   OpenCV version: {cv2.__version__}")
        print(f"   Prototxt: {prototxt_path}")
        print(f"   Caffemodel: {caffemodel_path}")
        
        # Load the network
        net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
        print("✓ Successfully loaded MobileNet-SSD with OpenCV DNN!")
        return True
        
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 50)
    print("Simple MobileNet-SSD Model Downloader")
    print("=" * 50)
    
    if download_mobilenet_files():
        if test_model_loading():
            print("\n🎉 Success! MobileNet-SSD is ready to use.")
            print("\nTry it with:")
            print("   python run.py --model dnn --input media/traffic.mp4")
        else:
            print("\n⚠️  Download succeeded but model loading failed.")
            print("This might be an OpenCV compatibility issue.")
    else:
        print("\n❌ Download failed. Manual download required:")
        print("\n1. Download these files:")
        print("   • MobileNetSSD_deploy.prototxt from:")
        print("     https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/MobileNetSSD_deploy.prototxt")
        print("   • MobileNetSSD_deploy.caffemodel from:")
        print("     https://github.com/chuanqi305/MobileNet-SSD/raw/master/MobileNetSSD_deploy.caffemodel")
        print(f"\n2. Place both files in: models/")
        print("\n3. Run this script again to test loading.")
