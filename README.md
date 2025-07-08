# 🎯 Real-Time Object Understanding Application

A powerful, real-time object detection and understanding application using Python, OpenCV, and state-of-the-art AI models. Features dual model support (YOLO v8 + MobileNet-SSD), object tracking, performance monitoring, and modern GUI interface.

![Application Demo](https://img.shields.io/badge/Python-3.8%2B-blue) ![OpenCV](https://img.shields.io/badge/OpenCV-4.9.0-green) ![YOLOv8](https://img.shields.io/badge/YOLOv8-Latest-orange)

## 🎬 Demo



Watch the application in action detecting objects in real-time:

[https://github.com/user-attachments/assets/traffic.mov](https://github.com/user-attachments/assets/528289d2-f751-4afb-b446-c4921c53e3f1)

*Alternative formats:*
- [Download Demo Video (MOV)](./traffic.mov)

> **Note**: The demo shows real-time object detection on traffic footage, demonstrating multi-object tracking, confidence adjustments, and model switching capabilities.



## 🚀 Features

### Core Functionality
- **Real-time object detection** with multiple model support (YOLOv8, MobileNet-SSD)
- **Video file processing** with support for MP4, AVI, MOV, and other formats
- **Live camera feed analysis** with automatic camera detection
- **Advanced object tracking** with unique ID assignment
- **Performance optimization** for smooth real-time processing
- **Multi-camera support** with automatic camera detection
- **Batch video processing** for multiple files

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
- **Video output generation** with processed detections
- **Session statistics** and performance analytics
- **Detection logging** for analysis
- **Configurable settings** via configuration files
- **Keyboard shortcuts** for all functions
- **Multiple input sources** (camera, video files, image sequences)

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- macOS, Windows, or Linux
- Webcam or USB camera (for live detection)
- Video files in supported formats (MP4, AVI, MOV, etc.) for video processing

### Quick Setup

1. **Clone or download the project:**
   ```bash
   cd ~/Desktop
   # Project is in RealVision-ObjectUnderstandingAI folder
   cd RealVision-ObjectUnderstandingAI
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
# Run with default settings (webcam)
python run.py

# Use specific camera
python run.py --camera 1

# Process video file
python run.py --input video.mp4

# Set confidence threshold
python run.py --confidence 0.7

# List available cameras
python run.py --list-cameras
```

### Video Processing
```bash
# Process various video formats
python run.py --input traffic.mp4
python run.py --input demo.avi
python run.py --input sample.mov

# Process video with specific model
python run.py --input video.mp4 --model yolo
python run.py --input video.mp4 --model dnn

# Save processed video output
python run.py --input video.mp4 --output processed_video.mp4

# Batch process multiple videos
python run.py --input-dir ./videos/ --output-dir ./processed/
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

# Process video with custom settings
python run.py --input video.mp4 --confidence 0.6 --output-dir results/
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
| **M** | Switch between models (YOLO ↔ MobileNet ↔ ONNX) |
| **C** | Toggle confidence display |
| **T** | Toggle tracking IDs |
| **P** | Toggle performance statistics |
| **A** | Generate analytics report |
| **D** | Toggle data logging on/off |
| **R** | Reset analytics data |
| **+** / **=** | Increase confidence threshold |
| **-** | Decrease confidence threshold |
| **F** | Toggle fullscreen mode |

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

## 📹 Video Processing Features

### Supported Video Formats
- **MP4**: Most common format, excellent compatibility
- **AVI**: Windows standard format
- **MOV**: QuickTime format, ideal for macOS
- **MKV**: Matroska container format
- **WMV**: Windows Media Video
- **FLV**: Flash Video format
- **WebM**: Web-optimized format

### Video Processing Capabilities
- **Frame-by-frame analysis** with object detection
- **Real-time playback** with detection overlays
- **Export processed videos** with bounding boxes and labels
- **Performance metrics** during video processing
- **Pause/resume** functionality during processing
- **Seek controls** for navigation through video
- **Batch processing** for multiple video files

### Video Output Options
- **Original video** with detection overlays
- **Detection-only output** (bounding boxes and labels)
- **Statistical overlay** showing detection counts and performance
- **Custom resolution** and quality settings
- **Multiple output formats** (MP4, AVI, MOV)

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

### Configuration Settings
Configuration is handled directly in the main application. Key settings can be adjusted:

```python
# Camera settings (in main.py)
CAMERA_CONFIG = {
    'default_camera_index': 0,
    'frame_width': 1280,
    'frame_height': 720,
    'fps': 30
}

# Detection settings (in main.py)
DETECTION_CONFIG = {
    'default_confidence_threshold': 0.5,
    'nms_threshold': 0.4,
    'max_detections_per_frame': 100
}
```

### Runtime Configuration
All settings can be adjusted during runtime using keyboard shortcuts:

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
- Use lower camera resolution or reduce detection frequency

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
RealVision-ObjectUnderstandingAI/
├── main.py                          # Main application code
├── run.py                           # Application launcher  
├── performance_analyzer.py          # Data science analytics module
├── analyze_performance.py           # Standalone analysis tool
├── demo_analytics.py                # Analytics demonstration script
├── demo_sample_analytics.py         # Sample data analytics demo
├── requirements.txt                 # Python dependencies
├── README.md                       # This documentation
├── USAGE.md                        # Usage instructions
├── setup.py                        # Installation setup
├── performance_analysis.ipynb      # Jupyter notebook for analysis
├── test_data_science.py            # Data science feature tests
├── test_models_analyze.py          # ModelsAnalyze folder tests
├── .gitignore                      # Git ignore rules
├── screenshots/                    # Screenshot output directory
├── ModelsAnalyze/                  # Model analysis and performance graphs
├── data/                           # Performance data exports
├── demo.mov                        # Demo video file
├── traffic.mp4                     # Sample video for testing
├── yolov8n.pt                      # YOLOv8 model weights
├── yolov5s.onnx                    # ONNX model weights
├── MobileNetSSD_deploy.prototxt    # MobileNet model architecture
└── MobileNetSSD_deploy.caffemodel  # MobileNet model weights
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

## 📊 Data Science & Performance Analysis

### Advanced Analytics Features
The application includes comprehensive data science capabilities for analyzing model performance, detection patterns, and system optimization.

#### Real-time Performance Monitoring
- **Automatic data logging** of inference times, detection counts, and model performance
- **Statistical analysis** with pandas, numpy, and scipy
- **Interactive visualizations** using matplotlib, seaborn, and plotly
- **Performance comparisons** between different AI models
- **Export capabilities** for offline analysis

#### Analytics Controls
| Key | Function |
|-----|----------|
| **A** | Generate comprehensive analytics report |
| **D** | Toggle data logging on/off |
| **R** | Reset analytics data |

#### Data Science Tools Included
```python
# Core data science libraries
pandas>=1.5.0          # Data manipulation and analysis
numpy>=1.9.0           # Numerical computing
matplotlib>=3.6.0      # Static visualizations
seaborn>=0.11.0        # Statistical visualizations
plotly>=5.0.0          # Interactive visualizations
scikit-learn>=1.2.0    # Machine learning metrics
scipy>=1.9.0           # Scientific computing
statsmodels>=0.14.0    # Statistical modeling
```

#### Analytics Outputs
The system generates several types of analysis in the `ModelsAnalyze/` folder:

1. **Performance Reports** (`ModelsAnalyze/performance_report.json`)
   - Model comparison statistics
   - FPS analysis over time
   - Detection accuracy trends
   - Resource utilization metrics

2. **Interactive Dashboards** (`ModelsAnalyze/performance_dashboard.html`)
   - Real-time performance charts
   - Model switching impact analysis
   - Confidence threshold optimization
   - Detection pattern analysis

3. **Statistical Plots** (`ModelsAnalyze/`)
   - `model_performance_comparison.png` - Box plots for inference time comparison
   - `detection_patterns_analysis.png` - Histogram distributions of detection counts
   - Time series analysis of performance metrics
   - Correlation matrices for performance factors

4. **Data Exports** (`data/`)
   - JSON format performance logs
   - CSV exports for external analysis
   - Timestamp-indexed datasets

#### Using the Analytics Features

##### During Application Runtime
```bash
# Run with analytics enabled (default)
python run.py

# During runtime:
# Press 'A' to generate reports
# Press 'D' to toggle logging
# Press 'M' to switch models and compare performance
```

##### Standalone Analysis
```bash
# Analyze saved performance data
python analyze_performance.py

# Interactive analysis with dashboard
python analyze_performance.py --interactive

# Analyze specific data file
python analyze_performance.py --data-file data/performance_data_20250709_143022.json
```

##### Analytics Testing & Demo
```bash
# Test data science features
python test_data_science.py

# Test ModelsAnalyze folder functionality  
python test_models_analyze.py

# Run analytics demo with camera
python demo_analytics.py

# Demo with video file
python demo_analytics.py --video traffic.mp4

# Generate sample analytics data
python demo_sample_analytics.py

# Analyze existing performance data
python analyze_performance.py --interactive
```

#### Performance Metrics Tracked

**System Performance:**
- Frame processing time (ms)
- Frames per second (FPS)
- Memory usage patterns
- CPU/GPU utilization

**Model Performance:**
- Inference time per model
- Detection accuracy rates
- Confidence score distributions
- Model switching overhead

**Detection Analytics:**
- Object class frequency
- Detection confidence trends
- Spatial detection patterns
- Temporal detection consistency

#### Jupyter Notebook Analysis
Interactive analysis notebook is included:

```bash
# Start Jupyter
jupyter notebook

# Open the analysis notebook
# File: performance_analysis.ipynb
```

The notebook includes:
- Data loading and preprocessing
- Statistical analysis and hypothesis testing
- Advanced visualizations
- Model performance optimization recommendations
- Custom analysis workflows

#### Advanced Analytics Examples

**Model Comparison Analysis:**
```python
from performance_analyzer import ModelPerformanceAnalyzer

analyzer = ModelPerformanceAnalyzer()
# Data is automatically logged during application use

# Generate comparison report
analyzer.analyze_model_comparison(save_plots=True)

# Create interactive dashboard
analyzer.create_interactive_dashboard()

# Statistical analysis
df = analyzer.get_performance_dataframe()
print(df.groupby('model')['inference_time_ms'].describe())
```

**Custom Analysis:**
```python
# Load performance data
import pandas as pd
df = pd.read_json('data/performance_data_latest.json')

# Analyze performance by time of day
df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
hourly_performance = df.groupby('hour')['fps'].mean()

# Find optimal confidence threshold
confidence_analysis = df.groupby('confidence_threshold').agg({
    'detection_count': 'mean',
    'inference_time_ms': 'mean'
})
```

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

*Last updated: July 9, 2025*
