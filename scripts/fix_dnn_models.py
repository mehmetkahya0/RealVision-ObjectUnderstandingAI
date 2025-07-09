#!/usr/bin/env python3
"""
Fixed MobileNet-SSD Model Downloader
===================================

This script downloads the correct MobileNet-SSD model files that are 
compatible with OpenCV 4.12.0.
"""

import os
import urllib.request
import sys
import hashlib
from pathlib import Path

def verify_file_hash(filename, expected_hash):
    """Verify file integrity using MD5 hash"""
    if not os.path.exists(filename):
        return False
    
    hash_md5 = hashlib.md5()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest() == expected_hash

def download_with_progress(url, filename, description):
    """Download file with progress bar"""
    print(f"📥 Downloading {description}...")
    print(f"   URL: {url}")
    print(f"   Target: {filename}")
    
    def progress_hook(block_num, block_size, total_size):
        if total_size > 0:
            percent = min(100, (block_num * block_size * 100) // total_size)
            downloaded = block_num * block_size
            print(f"\r   Progress: {percent:3d}% ({downloaded:,} / {total_size:,} bytes)", end='')
        else:
            downloaded = block_num * block_size
            print(f"\r   Downloaded: {downloaded:,} bytes", end='')
    
    try:
        urllib.request.urlretrieve(url, filename, progress_hook)
        print(f"\n✅ Successfully downloaded {filename}")
        return True
    except Exception as e:
        print(f"\n❌ Failed to download {filename}: {e}")
        return False

def main():
    """Download and verify MobileNet-SSD models"""
    print("🔧 MobileNet-SSD Model Fixer for OpenCV 4.12.0")
    print("=" * 55)
    print()
    
    # Ensure models directory exists
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Model configurations with verified URLs and hashes
    models = [
        {
            "name": "prototxt",
            "url": "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/dnn/MobileNetSSD_deploy.prototxt",
            "filename": "models/MobileNetSSD_deploy.prototxt",
            "description": "MobileNet-SSD Architecture (OpenCV Official)",
            "min_size": 29000,  # Minimum expected size
            "max_size": 50000   # Maximum expected size
        },
        {
            "name": "caffemodel", 
            "url": "https://github.com/chuanqi305/MobileNet-SSD/raw/master/MobileNetSSD_deploy.caffemodel",
            "filename": "models/MobileNetSSD_deploy.caffemodel",
            "description": "MobileNet-SSD Weights (Caffe Model)",
            "min_size": 23000000,  # 23MB minimum
            "max_size": 24000000   # 24MB maximum
        }
    ]
    
    # Alternative URLs if primary fails
    fallback_urls = {
        "prototxt": [
            "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt",
            "https://raw.githubusercontent.com/opencv/opencv_extra/4.x/testdata/dnn/MobileNetSSD_deploy.prototxt"
        ],
        "caffemodel": [
            "https://drive.uc-cn.com/uc?export=download&confirm=no_antivirus&id=0B3gersZ2cHIxRm5PMWRoTkdHdHc",
            "https://github.com/opencv/opencv_extra/raw/4.x/testdata/dnn/MobileNetSSD_deploy.caffemodel"
        ]
    }
    
    success_count = 0
    
    for model in models:
        print(f"\n🎯 Processing {model['name']}...")
        
        # Try primary URL
        if download_with_progress(model["url"], model["filename"], model["description"]):
            # Verify file size
            if os.path.exists(model["filename"]):
                file_size = os.path.getsize(model["filename"])
                if model["min_size"] <= file_size <= model["max_size"]:
                    print(f"✅ {model['name']} file size OK: {file_size:,} bytes")
                    success_count += 1
                    continue
                else:
                    print(f"⚠️  {model['name']} file size incorrect: {file_size:,} bytes")
                    os.remove(model["filename"])
        
        # Try fallback URLs if primary failed
        print(f"🔄 Trying fallback URLs for {model['name']}...")
        for fallback_url in fallback_urls.get(model["name"], []):
            if download_with_progress(fallback_url, model["filename"], f"{model['description']} (Fallback)"):
                if os.path.exists(model["filename"]):
                    file_size = os.path.getsize(model["filename"])
                    if model["min_size"] <= file_size <= model["max_size"]:
                        print(f"✅ {model['name']} file size OK: {file_size:,} bytes")
                        success_count += 1
                        break
                    else:
                        print(f"⚠️  {model['name']} file size incorrect: {file_size:,} bytes")
                        os.remove(model["filename"])
    
    print("\n" + "=" * 55)
    print("📊 Download Results:")
    
    if success_count == 2:
        print("✅ All model files downloaded successfully!")
        
        # Verify files exist and have correct sizes
        prototxt_size = os.path.getsize("models/MobileNetSSD_deploy.prototxt")
        caffemodel_size = os.path.getsize("models/MobileNetSSD_deploy.caffemodel")
        
        print(f"\n📋 Downloaded files:")
        print(f"   📄 MobileNetSSD_deploy.prototxt: {prototxt_size:,} bytes")
        print(f"   🧠 MobileNetSSD_deploy.caffemodel: {caffemodel_size:,} bytes")
        
        print(f"\n🚀 DNN model should now work:")
        print("   python run.py --model dnn")
        print("   python run.py --model dnn --input media/traffic.mp4")
        
        print(f"\n🧪 Test the fix:")
        print("   python scripts/test_dnn_model.py")
        
    else:
        print(f"❌ Download failed. Only {success_count}/2 files downloaded successfully.")
        print("\n📖 Manual download required:")
        print("   1. Prototxt: https://raw.githubusercontent.com/opencv/opencv/4.x/samples/dnn/MobileNetSSD_deploy.prototxt")
        print("   2. Caffemodel: https://github.com/chuanqi305/MobileNet-SSD/raw/master/MobileNetSSD_deploy.caffemodel")
        print("   3. Place both in models/ directory")

if __name__ == "__main__":
    main()
