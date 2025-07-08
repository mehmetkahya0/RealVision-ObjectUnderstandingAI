# 🤖 AI Models Guide - Real-Time Object Understanding

## Available AI Models

Your application now supports **4 different AI models**, each with unique strengths:

### 📊 **Model Comparison**

| Model | Speed | Accuracy | Specialization | GPU Required |
|-------|-------|----------|----------------|--------------|
| **YOLO v8** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | General objects | No |
| **MobileNet-SSD** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | Speed optimized | No |
| **MediaPipe** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Hands + 3D objects | No |
| **ONNX Runtime** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | Custom models | Optional |

---

## 🎯 **1. YOLO v8** (Ultralytics)
**Best for: General object detection with high accuracy**

### Features:
- ✅ 80+ object classes (COCO dataset)
- ✅ Excellent accuracy and speed balance
- ✅ Real-time performance (25-30 FPS)
- ✅ Advanced object tracking

### Usage:
```bash
python run.py --model yolo
```

### Detection Classes:
- People, vehicles, animals, household items, food, sports equipment, etc.

---

## ⚡ **2. MobileNet-SSD** (OpenCV DNN)
**Best for: Maximum speed on any hardware**

### Features:
- ✅ Fastest inference (~40+ FPS)
- ✅ Lightweight and efficient
- ✅ No additional dependencies
- ✅ Works on older hardware

### Usage:
```bash
python run.py --model dnn
```

### Detection Classes:
- 21 optimized classes: person, car, bicycle, dog, cat, chair, bottle, etc.

---

## 🤚 **3. MediaPipe** (Google)
**Best for: Hand tracking and specialized 3D detection**

### Features:
- ✅ Real-time hand tracking (21 landmarks per hand)
- ✅ 3D object detection (cups, chairs, cameras, shoes)
- ✅ Pose estimation capabilities
- ✅ Optimized for mobile/edge devices

### Installation:
```bash
pip install mediapipe
```

### Usage:
```bash
python run.py --model mediapipe
```

### Specialized Detection:
- **Hands**: Full hand tracking with finger positions
- **3D Objects**: Spatial understanding of objects
- **Pose**: Body pose detection (can be enabled)

---

## 🧠 **4. ONNX Runtime** (Cross-platform)
**Best for: Custom models and cross-platform optimization**

### Features:
- ✅ Support for YOLOv5 and custom ONNX models
- ✅ Optimized inference engine
- ✅ GPU acceleration support
- ✅ Custom model compatibility

### Installation:
```bash
pip install onnxruntime
# For GPU support:
pip install onnxruntime-gpu
```

### Usage:
```bash
python run.py --model onnx
```

### Detection Classes:
- 80 COCO classes (same as YOLO v8)
- Support for custom trained models

---

## 🚀 **Quick Setup Guide**

### 1. **Automatic Setup** (Recommended)
```bash
python setup.py
```
This interactive script will guide you through installing optional models.

### 2. **Manual Installation**
```bash
# Core requirements (always needed)
pip install -r requirements.txt

# Optional models
pip install mediapipe      # For hand tracking
pip install onnxruntime    # For ONNX models
```

### 3. **Test Installation**
```bash
python run.py --list-cameras  # Shows available models
```

---

## 💡 **Usage Recommendations**

### **For General Use:**
```bash
python run.py --model auto    # Auto-selects best available
```

### **For Maximum Accuracy:**
```bash
python run.py --model yolo --confidence 0.3
```

### **For Maximum Speed:**
```bash
python run.py --model dnn --confidence 0.6
```

### **For Hand Tracking:**
```bash
python run.py --model mediapipe
```

### **For Custom Models:**
```bash
python run.py --model onnx
```

---

## ⌨️ **Runtime Model Switching**

During application runtime:
- **Press 'M'** to cycle through available models
- **No restart required** - instant switching!
- Performance stats show current model

---

## 🔧 **Model Performance Tuning**

### **For Better FPS:**
1. Use MobileNet-SSD (`--model dnn`)
2. Increase confidence threshold (`--confidence 0.7`)
3. Lower camera resolution (edit `config.py`)

### **For Better Accuracy:**
1. Use YOLO v8 (`--model yolo`)
2. Lower confidence threshold (`--confidence 0.3`)
3. Ensure good lighting conditions

### **For Specialized Tasks:**
1. Use MediaPipe for hand gestures
2. Use ONNX for custom-trained models
3. Combine models by switching during runtime

---

## 📈 **Performance Benchmarks**

**On typical hardware (mid-range CPU):**
- **YOLO v8**: ~25-30 FPS, 35-50ms inference
- **MobileNet-SSD**: ~35-45 FPS, 20-30ms inference  
- **MediaPipe**: ~30-40 FPS, 25-35ms inference
- **ONNX Runtime**: ~20-30 FPS, 50-70ms inference (Optimized with NMS)

*Results may vary based on hardware and camera resolution*

### **ONNX Performance Notes:**
- ✅ **Fixed duplicated detections** with Non-Maximum Suppression (NMS)
- ✅ **Optimized post-processing** for YOLOv5 ONNX format
- ✅ **Reduced memory usage** with vectorized operations
- ⚡ **Better performance** compared to previous version

---

## 🆘 **Troubleshooting**

### **ONNX Model Issues:**
```bash
# If ONNX is slow or shows duplicates:
python run.py --model onnx --confidence 0.6  # Try higher confidence
python run.py --model yolo                   # Compare with YOLO

# Check ONNX installation:
python -c "import onnxruntime; print('ONNX Version:', onnxruntime.__version__)"
```

### **Model Not Loading:**
```bash
# Check installation
python -c "import mediapipe; print('MediaPipe OK')"
python -c "import onnxruntime; print('ONNX OK')"

# Reinstall if needed
pip uninstall mediapipe onnxruntime
pip install mediapipe onnxruntime
```

### **Low Performance:**
- Try different models with `M` key
- Lower confidence threshold
- Check CPU usage in system monitor

### **No Detections:**
- Lower confidence threshold (`+`/`-` keys)
- Ensure good lighting
- Try different models for different object types

---

## 🎉 **Enjoy Your Multi-AI Object Detection System!**

You now have access to **4 powerful AI models** that can detect:
- 🎯 General objects (80+ classes)
- 🤚 Hands and gestures  
- 🏠 Household items
- 🚗 Vehicles and transportation
- 🍎 Food and beverages
- 🧸 And much more!

**Switch between models in real-time and find the perfect AI for your use case!**
