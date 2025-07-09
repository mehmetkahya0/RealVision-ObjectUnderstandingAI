import os
import requests
import hashlib
from pathlib import Path

# URLs for the correct MobileNet-SSD model files
PROTOTXT_URL = "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt"
MODEL_URL = "https://drive.google.com/uc?export=download&id=0B3gersZ2cHIxRm5PMWRoTkdHdHc"

# Alternative URLs if the above don't work
ALT_PROTOTXT_URL = "https://github.com/chuanqi305/MobileNet-SSD/raw/master/deploy.prototxt"
ALT_MODEL_URL = "https://github.com/djmv/MobilNet_SSD_OpenCV/raw/master/MobileNetSSD_deploy.caffemodel"

def download_file(url, filepath, chunk_size=8192):
    """Download a file from URL to filepath with progress indication."""
    try:
        print(f"Downloading from {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"\rProgress: {percent:.1f}%", end="", flush=True)
        
        print(f"\nDownloaded: {filepath}")
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def verify_file_size(filepath, min_size_mb=5):
    """Verify that the downloaded file has a reasonable size."""
    if os.path.exists(filepath):
        size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"File size: {size_mb:.2f} MB")
        return size_mb >= min_size_mb
    return False

def main():
    # Create models directory if it doesn't exist
    models_dir = Path("../models")
    models_dir.mkdir(exist_ok=True)
    
    # Define file paths
    prototxt_path = models_dir / "MobileNetSSD_deploy.prototxt"
    model_path = models_dir / "MobileNetSSD_deploy.caffemodel"
    
    # Backup existing files
    if prototxt_path.exists():
        backup_path = models_dir / "MobileNetSSD_deploy.prototxt.old"
        prototxt_path.rename(backup_path)
        print(f"Backed up existing prototxt to {backup_path}")
    
    if model_path.exists():
        backup_path = models_dir / "MobileNetSSD_deploy.caffemodel.old"
        model_path.rename(backup_path)
        print(f"Backed up existing model to {backup_path}")
    
    # Download prototxt file
    print("Downloading MobileNet-SSD prototxt file...")
    if not download_file(PROTOTXT_URL, prototxt_path):
        print("Trying alternative prototxt URL...")
        if not download_file(ALT_PROTOTXT_URL, prototxt_path):
            print("Failed to download prototxt file")
            return False
    
    # Download model file
    print("\nDownloading MobileNet-SSD model file...")
    if not download_file(ALT_MODEL_URL, model_path):
        print("Trying Google Drive URL...")
        if not download_file(MODEL_URL, model_path):
            print("Failed to download model file")
            return False
    
    # Verify files
    print("\nVerifying downloaded files...")
    if not verify_file_size(prototxt_path, 0.01):  # At least 10KB for prototxt
        print("Prototxt file seems too small or corrupted")
        return False
    
    if not verify_file_size(model_path, 5):  # At least 5MB for model
        print("Model file seems too small or corrupted")
        return False
    
    print("\nMobileNet-SSD model files downloaded successfully!")
    print(f"Prototxt: {prototxt_path}")
    print(f"Model: {model_path}")
    
    return True

if __name__ == "__main__":
    main()
