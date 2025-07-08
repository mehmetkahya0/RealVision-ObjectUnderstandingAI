# 🎯 Real-Time Object Understanding Application

A powerful, real-time object detection and understanding application using Python, OpenCV, and state-of-the-art AI models. Features dual model support (YOLO v8 + MobileNet-SSD), object tracking, performance monitoring, and modern GUI interface.

![Application Demo](https://img.shields.io/badge/Python-3.8%2B-blue) ![OpenCV](https://img.shields.io/badge/OpenCV-4.9.0-green) ![YOLOv8](https://img.shields.io/badge/YOLOv8-Latest-orange)

## 🚀 Features

### Core Functionality
- **Real-time object detection** with multiple model support (YOLOv8, MobileNet-SSD)
- **Advanced object tracking** with unique ID assignment
- **Performance optimization** for smooth real-time processing
- **Multi-camera support** with automatic camera detection

### Detection Models
- **YOLOv8**: State-of-the-art accuracy and speed
- **MobileNet-SSD**: Lightweight and efficient for lower-end devices
- **Automatic model switching** during runtime
- **GPU acceleration** support when available

### User Interface
- **Modern overlay interface** with real-time statistics
- **Adjustable confidence thresholds** on-the-fly
- **Performance metrics display** (FPS, processing time, detection count)
- **Object tracking visualization** with unique IDs
- **Customizable display options**

### Advanced Features
- **Screenshot capture** with detection metadata
- **Session statistics** and performance analytics
- **Detection logging** for analysis
- **Configurable settings** via configuration files
- **Keyboard shortcuts** for all functions

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- macOS, Windows, or Linux
- Webcam or USB camera

### Quick Setup

1. **Clone or download the project:**
   ```bash
   cd ~/Desktop
   # Project is already in ObjectUnderstanding folder
   cd ObjectUnderstanding
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   python run.py
   ```

### Alternative Installation Methods

#### Using Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
python run.py
```

#### Using Conda
```bash
conda create -n object_detection python=3.9
conda activate object_detection
pip install -r requirements.txt
python run.py
```

## 🎯 Usage

### Basic Usage
```bash
# Run with default settings
python run.py

# Use specific camera
python run.py --camera 1

# Set confidence threshold
python run.py --confidence 0.7

# List available cameras
python run.py --list-cameras
```

### Advanced Usage
```bash
# Use specific model
python run.py --model yolo
python run.py --model dnn

# Custom output directory
python run.py --output-dir my_screenshots

# Headless mode (no GUI)
python run.py --no-gui
```

### Direct Python Usage
```python
from main import ObjectUnderstandingApp

app = ObjectUnderstandingApp()
app.confidence_threshold = 0.6
app.run(camera_index=0)
```

## ⌨️ Keyboard Controls

| Key | Function |
|-----|----------|
| **Q** or **ESC** | Quit application |
| **SPACE** | Pause/Resume detection |
| **S** | Take screenshot |
| **M** | Switch between models (YOLO ↔ MobileNet) |
| **C** | Toggle confidence display |
| **T** | Toggle tracking IDs |
| **P** | Toggle performance statistics |
| **+** / **=** | Increase confidence threshold |
| **-** | Decrease confidence threshold |
| **F** | Toggle fullscreen mode |
| **R** | Reset all settings |

## 📊 Performance Features

### Real-time Metrics
- **FPS (Frames Per Second)**: Current and average
- **Processing Time**: Per-frame detection time
- **Detection Count**: Number of objects detected
- **Session Duration**: Total runtime
- **Frame Count**: Total frames processed

### Optimization Features
- **Multi-threading**: Separate threads for capture and processing
- **Frame buffering**: Efficient memory management
- **GPU acceleration**: Automatic CUDA detection
- **Adaptive processing**: Dynamic quality adjustment

## 🎯 Object Detection

### Supported Object Classes
The application can detect 80+ object classes including:
- **People**: person
- **Vehicles**: car, truck, bus, motorcycle, bicycle
- **Animals**: dog, cat, bird, horse, cow, etc.
- **Household items**: chair, table, TV, laptop, phone, etc.
- **Food items**: apple, banana, pizza, cake, etc.
- **And many more...**

### Detection Accuracy
- **High accuracy** with YOLOv8 model (mAP 50-95: ~37%)
- **Fast inference** with MobileNet-SSD (suitable for real-time)
- **Adjustable confidence** thresholds (0.05 - 0.95)
- **Non-Maximum Suppression** to eliminate duplicate detections

## 📷 Screenshot Features

### Automatic Saving
- **High-quality screenshots** with detection overlays
- **Metadata preservation** (JSON format with detection data)
- **Timestamp-based naming** for easy organization
- **Batch processing** support

### Screenshot Data
Each screenshot includes:
```json
{
  "timestamp": "20250708_143022",
  "model": "yolo",
  "confidence_threshold": 0.5,
  "detections": [
    {
      "bbox": [100, 150, 300, 400],
      "confidence": 0.85,
      "class_name": "person",
      "class_id": 0
    }
  ],
  "total_detections": 1
}
```

## 🔧 Configuration

### Configuration File (config.py)
```python
# Camera settings
CAMERA_CONFIG = {
    'default_camera_index': 0,
    'frame_width': 1280,
    'frame_height': 720,
    'fps': 30
}

# Detection settings
DETECTION_CONFIG = {
    'default_confidence_threshold': 0.5,
    'nms_threshold': 0.4,
    'max_detections_per_frame': 100
}
```

### Runtime Configuration
All settings can be adjusted during runtime using keyboard shortcuts or by modifying the configuration file.

## 🚨 Troubleshooting

### Common Issues

#### Camera Not Found
```bash
# List available cameras
python run.py --list-cameras

# Try different camera index
python run.py --camera 1
```

#### Low Performance
- Try MobileNet-SSD model: `python run.py --model dnn`
- Lower confidence threshold: `python run.py --confidence 0.3`
- Reduce camera resolution in config.py

#### Model Loading Errors
```bash
# Reinstall ultralytics
pip uninstall ultralytics
pip install ultralytics

# Clear model cache
rm -rf ~/.ultralytics
```

#### GPU Issues
- Ensure CUDA is properly installed
- Check PyTorch CUDA compatibility
- Fall back to CPU if needed

### Performance Optimization

#### For Better Speed
1. Use MobileNet-SSD model (`--model dnn`)
2. Lower camera resolution
3. Increase confidence threshold
4. Disable tracking IDs display

#### For Better Accuracy
1. Use YOLOv8 model (`--model yolo`)
2. Lower confidence threshold
3. Enable GPU acceleration
4. Use higher camera resolution

## 📁 Project Structure

```
ObjectUnderstanding/
├── main.py              # Main application code
├── run.py               # Application launcher
├── config.py            # Configuration settings
├── utils.py             # Utility functions
├── requirements.txt     # Python dependencies
├── README.md           # This file
├── screenshots/        # Screenshot output directory
├── models/            # Downloaded model files
└── logs/              # Application logs
```

## 🔬 Technical Details

### Architecture
- **Modular design** with separate classes for different components
- **Object-oriented** approach for maintainability
- **Event-driven** keyboard handling
- **Multi-threaded** processing for performance

### Dependencies
- **OpenCV**: Computer vision and image processing
- **PyTorch**: Deep learning framework
- **Ultralytics**: YOLOv8 implementation
- **NumPy**: Numerical computations
- **Pillow**: Image processing utilities

### Model Details
- **YOLOv8n**: Nano version for speed (6.2M parameters)
- **MobileNet-SSD**: Efficient architecture (5.8M parameters)
- **Input size**: 640x640 (YOLO), 300x300 (MobileNet)
- **Output**: Bounding boxes, confidence scores, class predictions

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a virtual environment
3. Install development dependencies
4. Make your changes
5. Test thoroughly
6. Submit a pull request

### Code Style
- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add docstrings to all functions
- Include unit tests for new features

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Ultralytics** for the excellent YOLOv8 implementation
- **OpenCV** community for the comprehensive computer vision library
- **PyTorch** team for the deep learning framework
- **Contributors** to all open-source libraries used

## 📧 Support

If you encounter any issues or have questions:

1. Check the troubleshooting section above
2. Review the GitHub issues page
3. Create a new issue with detailed information
4. Include system information and error messages

---

**Made with ❤️ for the computer vision community**

**Mehmet Kahya - July 2025**

*Last updated: July 8, 2025*
