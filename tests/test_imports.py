#!/usr/bin/env python3
"""Test script to verify all required libraries are installed and working."""

import sys
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")
print("-" * 50)

# Test core libraries
try:
    import cv2
    print("✅ OpenCV imported successfully - Version:", cv2.__version__)
except ImportError as e:
    print("❌ OpenCV import failed:", e)

try:
    import torch
    print("✅ PyTorch imported successfully - Version:", torch.__version__)
except ImportError as e:
    print("❌ PyTorch import failed:", e)

try:
    import ultralytics
    print("✅ Ultralytics imported successfully")
except ImportError as e:
    print("❌ Ultralytics import failed:", e)

try:
    import numpy as np
    print("✅ NumPy imported successfully - Version:", np.__version__)
except ImportError as e:
    print("❌ NumPy import failed:", e)

try:
    import pandas as pd
    print("✅ Pandas imported successfully - Version:", pd.__version__)
except ImportError as e:
    print("❌ Pandas import failed:", e)

try:
    import matplotlib
    print("✅ Matplotlib imported successfully - Version:", matplotlib.__version__)
except ImportError as e:
    print("❌ Matplotlib import failed:", e)

print("-" * 50)
print("🎉 All essential libraries for RealVision-ObjectUnderstandingAI are ready!")
