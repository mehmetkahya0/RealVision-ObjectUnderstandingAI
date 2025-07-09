#!/usr/bin/env python3
"""
Download MobileNet-SSD models for OpenCV DNN
============================================

This script downloads the correct MobileNet-SSD model files that are compatible
with OpenCV DNN module.
"""

import os
import urllib.request
import sys
from pathlib import Path

def download_file(url, filename, description):
    """Download a file with progress indication"""
    print(f"📥 Downloading {description}...")
    print(f"   URL: {url}")
    print(f"   File: {filename}")
    
    def progress_hook(block_num, block_size, total_size):
        if total_size > 0:
            percent = min(100, (block_num * block_size * 100) // total_size)
            print(f"\r   Progress: {percent}% ({block_num * block_size:,} / {total_size:,} bytes)", end='')
        else:
            print(f"\r   Downloaded: {block_num * block_size:,} bytes", end='')
    
    try:
        urllib.request.urlretrieve(url, filename, progress_hook)
        print(f"\n✅ Successfully downloaded {filename}")
        return True
    except Exception as e:
        print(f"\n❌ Failed to download {filename}: {e}")
        return False

def main():
    """Main function to download MobileNet-SSD models"""
    print("🤖 MobileNet-SSD Model Downloader for OpenCV DNN")
    print("=" * 55)
    print()
    
    # Ensure models directory exists
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Model URLs (OpenCV compatible versions)
    models = [
        {
            "url": "https://github.com/chuanqi305/MobileNet-SSD/raw/master/deploy.prototxt",
            "filename": "models/MobileNetSSD_deploy.prototxt",
            "description": "MobileNet-SSD Architecture (deploy.prototxt)"
        },
        {
            "url": "https://drive.google.com/uc?export=download&id=0B3gersZ2cHIxRm5PMWRoTkdHdHc",
            "filename": "models/MobileNetSSD_deploy.caffemodel",
            "description": "MobileNet-SSD Weights (caffemodel)"
        }
    ]
    
    # Alternative source if first fails
    alternative_models = [
        {
            "url": "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/dnn/MobileNetSSD_deploy.prototxt",
            "filename": "models/MobileNetSSD_deploy.prototxt", 
            "description": "MobileNet-SSD Architecture (OpenCV official)"
        },
        {
            "url": "https://github.com/opencv/opencv_extra/raw/master/testdata/dnn/MobileNetSSD_deploy.caffemodel",
            "filename": "models/MobileNetSSD_deploy.caffemodel",
            "description": "MobileNet-SSD Weights (OpenCV official)"
        }
    ]
    
    success_count = 0
    
    # Try primary sources first
    for model in models:
        if download_file(model["url"], model["filename"], model["description"]):
            success_count += 1
        else:
            print(f"⚠️  Primary source failed, trying alternative...")
    
    # If primary failed, try alternatives
    if success_count < 2:
        print("\n🔄 Trying alternative sources...")
        for model in alternative_models:
            if not os.path.exists(model["filename"]) or os.path.getsize(model["filename"]) < 1000:
                if download_file(model["url"], model["filename"], model["description"]):
                    success_count += 1
    
    print("\n" + "=" * 55)
    if success_count >= 2:
        print("✅ Model download completed successfully!")
        print("\n📋 Downloaded files:")
        for model_file in ["models/MobileNetSSD_deploy.prototxt", "models/MobileNetSSD_deploy.caffemodel"]:
            if os.path.exists(model_file):
                size = os.path.getsize(model_file)
                print(f"   📄 {model_file} ({size:,} bytes)")
        
        print("\n🚀 You can now use the DNN model:")
        print("   python run.py --model dnn")
    else:
        print("❌ Model download failed. Manual download required.")
        print("\n📖 Manual download instructions:")
        print("   1. Download deploy.prototxt from:")
        print("      https://github.com/chuanqi305/MobileNet-SSD/raw/master/deploy.prototxt")
        print("   2. Download MobileNetSSD_deploy.caffemodel from:")
        print("      https://drive.google.com/file/d/0B3gersZ2cHIxRm5PMWRoTkdHdHc/view")
        print("   3. Place both files in the models/ directory")

if __name__ == "__main__":
    main()
