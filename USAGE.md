# 🎯 Real-Time Object Understanding Application

## Quick Start Guide

### 1. Basic Usage

**Using with webcam (default):**
```bash
python run.py
```

**Using with specific camera:**
```bash
python run.py --camera 1
```

**Using with video file:**
```bash
python run.py --input video.mp4
```

**List available cameras:**
```bash
python run.py --list-cameras
```

### 2. Advanced Options

**Set confidence threshold:**
```bash
python run.py --confidence 0.7
```

**Choose detection model:**
```bash
python run.py --model yolo         # Use YOLO v8 (best accuracy)
python run.py --model dnn          # Use MobileNet-SSD (fastest)
python run.py --model mediapipe    # Use MediaPipe (hands + objects)
python run.py --model onnx         # Use ONNX Runtime (YOLOv5)
python run.py --model auto         # Auto-select best model (default)
```

**Custom output directory:**
```bash
python run.py --output-dir my_screenshots
```

### 3. Keyboard Controls (During Runtime)

| Key | Action |
|-----|--------|
| `Q` or `ESC` | Quit application |
| `SPACE` | Pause/Resume video |
| `S` | Take screenshot |
| `M` | Switch between YOLO and DNN models |
| `C` | Toggle confidence display |
| `T` | Toggle tracking IDs |
| `P` | Toggle performance stats |
| `+` or `=` | Increase confidence threshold |
| `-` | Decrease confidence threshold |
| `F` | Toggle fullscreen mode |
| `R` | Reset settings to default |

### 4. Features

✅ **Multiple AI Models**: YOLO v8, MobileNet-SSD, MediaPipe, ONNX Runtime
✅ **Real-time Processing**: 25-30 FPS on most hardware
✅ **Object Tracking**: Persistent object IDs across frames
✅ **Multiple Input Sources**: Webcam, video files, or IP cameras
✅ **Performance Monitoring**: Real-time FPS and inference time display
✅ **Screenshot Capture**: Save frames with detections
✅ **Adjustable Settings**: Real-time confidence threshold adjustment
✅ **Hand Detection**: MediaPipe hand tracking and gestures
✅ **3D Object Detection**: MediaPipe 3D object recognition

### 5. Supported Object Classes & Models

#### **YOLO v8** (Best Accuracy)
- 80+ COCO dataset classes
- **People**: person
- **Vehicles**: car, truck, bus, motorcycle, bicycle, airplane, train, boat
- **Animals**: bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe
- **Objects**: bottle, cup, fork, knife, spoon, bowl, chair, couch, bed, laptop, cell phone
- **Food**: banana, apple, sandwich, orange, pizza, donut, cake
- And many more...

#### **MobileNet-SSD** (Fastest)
- 21 classes optimized for speed
- person, car, bicycle, dog, cat, chair, bottle, etc.

#### **MediaPipe** (Specialized)
- **Hand Detection**: Real-time hand tracking with 21 landmarks
- **3D Objects**: Cups, chairs, cameras, shoes (3D detection)
- **Pose Detection**: Full body pose estimation

#### **ONNX Runtime** (Flexible)
- Supports YOLOv5 and custom ONNX models
- 80 COCO classes (same as YOLO v8)
- Optimized for cross-platform performance

### 6. Performance Tips

**For better FPS:**
- Use MobileNet-SSD: `python run.py --model dnn`
- Use ONNX Runtime: `python run.py --model onnx`
- Lower camera resolution in config.py
- Increase confidence threshold

**For better accuracy:**
- Use YOLO v8: `python run.py --model yolo`
- Lower confidence threshold: `python run.py --confidence 0.3`
- Enable GPU acceleration
- Use higher camera resolution

**For specialized detection:**
- Use MediaPipe for hands: `python run.py --model mediapipe`
- MediaPipe excels at hand tracking and 3D objects
- YOLO v8 best for general object detection

### 7. Troubleshooting

**Camera not working?**
- Check camera permissions in System Preferences (macOS)
- Try a different camera index: `python run.py --camera 1`
- Use a video file instead: `python run.py --input sample_video.mp4`

**Low FPS?**
- Close other applications using the camera
- Lower the video resolution in config.py
- Switch to DNN model: `python run.py --model dnn`

**No detections?**
- Lower confidence threshold: `python run.py --confidence 0.3`
- Ensure good lighting conditions
- Point camera at recognizable objects

### 8. Installing Optional AI Models

**Automatic setup (recommended):**
```bash
python setup.py
```

**Manual installation:**
```bash
# For MediaPipe (hand tracking + 3D objects)
pip install mediapipe

# For ONNX Runtime (YOLOv5 + custom models)
pip install onnxruntime

# For GPU acceleration (if you have NVIDIA GPU)
pip install onnxruntime-gpu
```

**Check available models:**
```bash
python run.py --list-cameras  # Also shows available AI models
```

A sample video is included for testing:
```bash
python create_sample_video.py  # Generate sample video
python run.py --input sample_video.mp4  # Test with sample video
```

### 9. Configuration

Edit `config.py` to customize:
- Camera resolution and FPS
- Detection thresholds
- UI appearance
- Color schemes
- Model parameters

### 10. Requirements

- Python 3.8+
- OpenCV 4.5+
- PyTorch 2.0+
- YOLO v8 (ultralytics)
- NumPy
- Good webcam or video file

Enjoy your powerful real-time object understanding application! 🚀
