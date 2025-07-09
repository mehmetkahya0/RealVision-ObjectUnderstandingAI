# RealVision AI - GUI Application Guide

## Overview
RealVision AI now includes a modern graphical user interface (GUI) that makes it easy to use object detection and computer vision features without command-line interaction.

## Features
- **Camera Integration**: Open and use your webcam for real-time object detection
- **Video Upload**: Upload and process video files with object detection
- **Model Selection**: Choose between different AI models (YOLO, DNN, ONNX, etc.)
- **Real-time Statistics**: View FPS, detection counts, and performance metrics
- **Analytics Dashboard**: Generate and view comprehensive analytics reports
- **Confidence Adjustment**: Adjust detection confidence thresholds in real-time
- **Modern UI**: Clean, intuitive interface with status indicators

## Quick Start

### Method 1: Using the Launcher Scripts (Recommended)
1. **Windows Batch File**: Double-click `start_gui.bat`
2. **PowerShell**: Right-click `start_gui.ps1` → "Run with PowerShell"
3. **Python**: Run `python launch_gui.py`

### Method 2: Direct GUI Launch
```bash
python gui.py
```

### Method 3: Through Main App
```bash
python app.py --gui
```

## Installation

### Automatic Installation
The launcher scripts will automatically install required dependencies. Just run any of the launcher scripts and they'll handle the setup.

### Manual Installation
```bash
# Install required packages
pip install -r requirements.txt

# Or install core packages manually
pip install PyQt6 opencv-python numpy ultralytics matplotlib pandas seaborn
```

## GUI Controls

### Main Interface
- **📹 Open Camera**: Start real-time object detection using your webcam
- **📁 Upload Video**: Select and process a video file
- **⏹️ Stop**: Stop current video processing
- **📊 Show Analytics**: Open the analytics dashboard in your browser
- **📋 Generate Report**: Create a new analytics report

### Settings Panel
- **Model Selection**: Choose AI model (YOLO, DNN, ONNX, etc.)
- **Confidence Slider**: Adjust detection confidence threshold (0.01 - 1.00)
- **Display Options**: Toggle FPS display, confidence scores, etc.

### Status Information
- **FPS**: Current frames per second
- **Detections**: Number of objects detected in current frame
- **Tracked Objects**: Number of objects being tracked
- **Status Bar**: Current application status

## Creating a Standalone .exe File

### Step 1: Install Build Tools
```bash
pip install pyinstaller
```

### Step 2: Run the Build Script
```bash
python build_exe.py
```

### Step 3: Test the Executable
The `.exe` file will be created in the `dist/` directory:
```
dist/RealVision-AI.exe
```

### Manual Build (Advanced)
If you prefer to build manually:

```bash
pyinstaller --onefile --windowed --name "RealVision-AI" ^
  --add-data "src;src" ^
  --add-data "models;models" ^
  --add-data "media;media" ^
  --hidden-import cv2 ^
  --hidden-import numpy ^
  --hidden-import PyQt6.QtCore ^
  --hidden-import PyQt6.QtGui ^
  --hidden-import PyQt6.QtWidgets ^
  --hidden-import ultralytics ^
  gui.py
```

## Distribution

### For End Users
1. **Simple Distribution**: Share the `RealVision-AI.exe` file
2. **Complete Package**: Share the entire `dist/` folder
3. **With Models**: Include the `models/` directory with pre-trained models

### Required Files for Distribution
- `RealVision-AI.exe` (main executable)
- `models/` directory (AI model files)
- `media/` directory (optional, for sample videos)

## Troubleshooting

### Common Issues

#### "PyQt6 not found"
```bash
pip install PyQt6
```

#### "OpenCV not found"
```bash
pip install opencv-python
```

#### "No models available"
- Ensure model files are in the `models/` directory
- Run `python scripts/download_models.py` to download models
- Check that model files are not corrupted

#### "Camera not accessible"
- Close other applications using the camera
- Check camera permissions in Windows Settings
- Try different camera indices (0, 1, 2, etc.)

#### "Analytics not found"
- Process some video first to generate analytics data
- Check that the `output/` directory exists
- Ensure pandas and matplotlib are installed

### Performance Tips
1. **Use YOLO models** for best performance
2. **Lower confidence threshold** for more detections
3. **Close other applications** to free up system resources
4. **Use smaller video files** for faster processing

## Model Information

### Available Models
- **YOLO v8**: Fast, accurate object detection
- **MobileNet-SSD**: Lightweight for mobile devices
- **ONNX**: Cross-platform optimized models
- **MediaPipe**: Google's ML solutions
- **EfficientDet**: Efficient object detection

### Model Installation
Models are automatically downloaded when first used. To manually download:
```bash
python scripts/download_models.py
```

## Advanced Features

### Analytics Integration
- Real-time performance monitoring
- Comprehensive reporting
- Export to multiple formats (HTML, PDF, CSV)
- Time-series analysis

### Custom Configuration
Edit `config.json` to customize:
- Default model selection
- Confidence thresholds
- UI preferences
- Performance settings

### Batch Processing
Use the command-line interface for batch processing:
```bash
python app.py --input video_folder/ --output results/
```

## System Requirements

### Minimum Requirements
- **OS**: Windows 10, macOS 10.14, Ubuntu 18.04
- **Python**: 3.8 or higher
- **RAM**: 4GB minimum, 8GB recommended
- **GPU**: Optional but recommended for better performance

### Recommended Requirements
- **OS**: Windows 11, macOS 12+, Ubuntu 20.04+
- **Python**: 3.9 or higher
- **RAM**: 16GB or more
- **GPU**: NVIDIA GPU with CUDA support
- **Storage**: 2GB free space for models and data

## Support

### Getting Help
1. Check the troubleshooting section above
2. Review the error messages in the console
3. Ensure all requirements are properly installed
4. Check GitHub issues for similar problems

### Reporting Issues
When reporting issues, please include:
- Operating system and version
- Python version
- Full error message
- Steps to reproduce the issue

## License
This project is licensed under the MIT License. See `LICENSE` file for details.

## Author
Mehmet Kahya - July 2025
