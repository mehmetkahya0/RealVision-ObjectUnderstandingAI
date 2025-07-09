#!/usr/bin/env python3
"""
Model Compatibility Test
=======================

This script tests which object detection models are working correctly.
"""

import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_model(model_name):
    """Test if a specific model can be loaded"""
    print(f"\n🧪 Testing {model_name.upper()} model...")
    
    try:
        # Import the main application
        from main import ObjectUnderstandingApp
        
        # Create app instance
        app = ObjectUnderstandingApp()
        
        # Try to initialize the model
        if model_name == 'yolo':
            success = app.init_yolo_model()
        elif model_name == 'onnx':
            success = app.init_onnx_model()
        elif model_name == 'dnn':
            success = app.init_dnn_model()
        else:
            print(f"❓ Unknown model: {model_name}")
            return False
            
        if success:
            print(f"✅ {model_name.upper()} model loaded successfully!")
            return True
        else:
            print(f"❌ {model_name.upper()} model failed to load")
            return False
            
    except Exception as e:
        print(f"❌ {model_name.upper()} model error: {e}")
        return False

def main():
    """Test all available models"""
    print("🤖 RealVision Model Compatibility Test")
    print("=" * 45)
    
    models_to_test = ['yolo', 'onnx', 'dnn']
    working_models = []
    
    for model in models_to_test:
        if test_model(model):
            working_models.append(model)
    
    print("\n" + "=" * 45)
    print("📊 Test Results:")
    print(f"✅ Working models: {', '.join(working_models) if working_models else 'None'}")
    
    if working_models:
        print(f"\n🚀 Recommended usage:")
        for model in working_models:
            print(f"   python run.py --model {model}")
            
        if 'yolo' in working_models:
            print(f"\n💡 For best performance, use: python run.py --model yolo")
        elif 'onnx' in working_models:
            print(f"\n💡 For best performance, use: python run.py --model onnx")
    else:
        print("\n⚠️  No models are working. Please check your installation:")
        print("   pip install -r requirements.txt")
    
    if 'dnn' not in working_models:
        print(f"\n🔧 DNN (MobileNet-SSD) troubleshooting:")
        print("   • Download correct model files from:")
        print("     https://github.com/chuanqi305/MobileNet-SSD/")
        print("   • Or use working alternatives: --model yolo or --model onnx")

if __name__ == "__main__":
    main()
