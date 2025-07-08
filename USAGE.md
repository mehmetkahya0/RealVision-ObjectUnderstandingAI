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
python run.py --model yolo      # Use YOLO only
python run.py --model dnn       # Use MobileNet-SSD only
python run.py --model auto      # Auto-select (default)
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

✅ **Dual Model Support**: YOLO v8 and MobileNet-SSD
✅ **Real-time Processing**: 25-30 FPS on most hardware
✅ **Object Tracking**: Persistent object IDs across frames
✅ **Multiple Input Sources**: Webcam, video files, or IP cameras
✅ **Performance Monitoring**: Real-time FPS and inference time display
✅ **Screenshot Capture**: Save frames with detections
✅ **Adjustable Settings**: Real-time confidence threshold adjustment

### 5. Supported Object Classes

The application can detect 80+ object classes from the COCO dataset, including:
- **People**: person
- **Vehicles**: car, truck, bus, motorcycle, bicycle, airplane, train, boat
- **Animals**: bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe
- **Objects**: bottle, cup, fork, knife, spoon, bowl, chair, couch, bed, laptop, cell phone
- **Food**: banana, apple, sandwich, orange, pizza, donut, cake
- And many more...

### 6. Performance Tips

- **For better FPS**: Use YOLO nano model (default) or switch to DNN model
- **For better accuracy**: Use YOLO small/medium model (modify config.py)
- **Lower confidence threshold**: See more detections (may include false positives)
- **Higher confidence threshold**: See only high-confidence detections

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

### 8. Sample Video

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
