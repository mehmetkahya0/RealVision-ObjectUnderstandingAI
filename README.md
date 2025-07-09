# 🤖 RealVision-ObjectUnderstandingAI

A powerful, real-time object detection and understanding application using Python, OpenCV, and state-of-the-art AI models. Features dual model support (YOLO v8 + MobileNet-SSD), object tracking, performance monitoring, and modern GUI interface.




![Application Demo](https://img.shields.io/badge/Python-3.8%2B-blue) ![OpenCV](https://img.shields.io/badge/OpenCV-4.9.0-green) ![YOLOv8](https://img.shields.io/badge/YOLOv8-Latest-orange)
![Visitors](https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fgithub.com%2Fmehmetkahya0%2FRealVision-ObjectUnderstandingAI&label=VISITOR&countColor=%23263759)


## 🎬 Demo
Watch the application in action detecting objects in real-time:

[https://github.com/user-attachments/assets/traffic.mov](https://github.com/user-attachments/assets/528289d2-f751-4afb-b446-c4921c53e3f1)

*Alternative formats:*
- [Download Demo Video (MOV)](./traffic.mov)

> **Note**: The demo shows real-time object detection on traffic footage, demonstrating multi-object tracking, confidence adjustments, and model switching capabilities



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

**Linux/macOS:**
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python run.py
```

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run.py
```

**Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
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

#### Windows Setup (Recommended for Windows Users)

For Windows users, we provide convenient activation scripts that automatically set up the virtual environment:

**Option 1: PowerShell Script (Recommended)**
```powershell
# Navigate to project directory
cd "c:\Users\[YourUsername]\Desktop\RealVision-ObjectUnderstandingAI"

# Run the setup script (creates venv and installs dependencies automatically)
scripts\activate_env.ps1
```

**Option 2: Batch File**
```cmd
# Double-click scripts\activate_env.bat or run from Command Prompt:
scripts\activate_env.bat
```

**Option 3: Manual Windows Setup**
```powershell
# Create virtual environment
python -m venv venv

# Activate environment
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Run application
python run.py
```

**Windows Features:**
- 🚀 **One-click activation scripts** (`scripts/activate_env.bat` and `scripts/activate_env.ps1`)
- 📊 **Automatic environment setup** with all dependencies
- 🐍 **Python version verification** and path confirmation
- 💡 **Built-in command reference** displayed on activation
- ✅ **Library verification** with `tests/test_imports.py`

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

### Windows Quick Start

After running `scripts/activate_env.bat` or `scripts/activate_env.ps1`, you'll see available commands. Here are the most common:

```powershell
# Basic operations
python run.py                          # Use default camera
python run.py --camera 1               # Use specific camera
python run.py --input media/traffic.mp4      # Process the included sample video
python run.py --list-cameras           # Check available cameras

# Advanced features
python run.py --confidence 0.7         # Adjust detection sensitivity
python run.py --model yolo             # Force YOLO model
python run.py --no-gui                 # Run without display (save results only)

# Data science features (Windows optimized)
python tests/test_imports.py                 # Verify all libraries are working
python src/demo_analytics.py               # Interactive analytics demo
python tests/test_data_science.py           # Test analytics features
jupyter notebook                       # Open Jupyter for analysis

# Performance data visualization
python visualization/launch_visualizer.py           # Launch visualization tool launcher
python visualization/visualize_performance.py       # Command-line data visualizer
python visualization/visualize_performance_gui.py   # GUI data visualizer
scripts/visualize_data.bat                     # Windows batch launcher
```

**Windows-Specific Tips:**
- Use **PowerShell** for best compatibility with the activation scripts
- The scripts automatically verify your Python environment and show library versions
- All paths are Windows-compatible (backslashes handled automatically)
- Screenshot and video outputs save to Windows-friendly directories

### Video Processing
```bash
# Process various video formats
python run.py --input media/traffic.mp4
python run.py --input media/demo.avi
python run.py --input media/sample.mov

# Process video with specific model
python run.py --input media/traffic.mp4 --model yolo
python run.py --input media/traffic.mp4 --model onnx

# Save processed video output
python run.py --input media/traffic.mp4 --output processed_video.mp4

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

### Windows-Specific Issues

#### PowerShell Execution Policy
If you get an execution policy error when running `.ps1` scripts:
```powershell
# Check current policy
Get-ExecutionPolicy

# Set policy for current user (if needed)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

#### Virtual Environment Activation Issues
```powershell
# If activation fails, try:
.\venv\Scripts\activate.bat  # Use batch file instead
# or
python -m venv venv --clear  # Recreate environment
```

#### Python Not Found
```powershell
# Verify Python installation
python --version
# or
py --version

# If Python not found, install from python.org or Microsoft Store
```

#### Library Import Errors
```powershell
# Run the test script to verify installation
python test_imports.py

# If imports fail, recreate virtual environment:
Remove-Item -Recurse -Force venv
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Common Issues

#### Camera Not Found
```bash
# List available cameras
python run.py --list-cameras

# Try different camera index
python run.py --camera 1
```

#### Low Performance
- Try ONNX model: `python run.py --model onnx`
- Use YOLOv8 with lower confidence: `python run.py --model yolo --confidence 0.3`
- Use lower camera resolution or reduce detection frequency

#### Model Loading Errors
```bash
# Reinstall ultralytics
pip uninstall ultralytics
pip install ultralytics

# Clear model cache
rm -rf ~/.ultralytics
```

#### DNN (MobileNet-SSD) Model Issues
If you encounter OpenCV DNN errors like "!blobs.empty() || inputs.size() > 1":

**Quick Solution - Use Alternative Models:**
```bash
# Use YOLOv8 (recommended, works out of the box)
python run.py --model yolo

# Use ONNX model (also works reliably)
python run.py --model onnx
```

**Manual MobileNet-SSD Fix:**
1. Download the correct model files manually:
   - **Prototxt**: [MobileNetSSD_deploy.prototxt](https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt)
   - **Caffemodel**: [MobileNetSSD_deploy.caffemodel](https://drive.google.com/uc?export=download&id=0B3gersZ2cHIxRm5PMWRoTkdHdHc) (23MB)

2. Place both files in the `models/` directory

3. Verify file sizes:
   ```bash
   # Should be ~44KB for prototxt and ~23MB for caffemodel
   dir models\MobileNet*
   ```

**Alternative Download Script:**
```bash
# Run the model downloader script
python scripts\download_models.py
```

#### GPU Issues
- Ensure CUDA is properly installed
- Check PyTorch CUDA compatibility
- Fall back to CPU if needed

### Performance Optimization

#### For Better Speed
1. Use ONNX model (`--model onnx`)
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
├── 📁 src/                           # Main source code
│   ├── main.py                       # Core application with GUI and analytics
│   ├── run.py                        # Application launcher and CLI
│   ├── performance_analyzer.py       # Real-time performance monitoring
│   ├── analyze_performance.py        # Standalone analysis tool
│   ├── demo_analytics.py             # Interactive analytics demo
│   └── demo_sample_analytics.py      # Sample data generation
├── 📁 models/                        # AI model files
│   ├── yolov8n.pt                    # YOLOv8 model weights
│   ├── yolov5s.onnx                  # ONNX model weights
│   ├── MobileNetSSD_deploy.prototxt  # MobileNet architecture
│   └── MobileNetSSD_deploy.caffemodel # MobileNet weights
├── 📁 visualization/                 # Performance data visualization tools
│   ├── launch_visualizer.py          # Visualization tool launcher
│   ├── visualize_performance.py      # CLI visualizer
│   └── visualize_performance_gui.py  # GUI visualizer
├── 📁 tests/                         # Test suite
│   ├── test_imports.py               # Library verification
│   ├── test_data_science.py          # Analytics tests
│   ├── test_models_analyze.py        # Model analysis tests
│   └── test_visualization_system.py  # Visualization tests
├── 📁 scripts/                       # Utility scripts and launchers
│   ├── activate_env.bat              # Windows batch activation
│   ├── activate_env.ps1              # PowerShell activation
│   ├── visualize_data.bat            # Visualization launcher
│   └── setup.py                      # Installation setup
├── 📁 notebooks/                     # Jupyter notebooks
│   └── performance_analysis.ipynb    # Interactive analysis notebook
├── 📁 media/                         # Demo videos and screenshots
│   ├── demo.mov                      # Demo video file
│   ├── traffic.mp4                   # Sample video for testing
│   └── screenshots/                  # Screenshot output directory
├── 📁 data/                          # Performance data exports
│   └── performance_data_*.json       # Generated performance logs
├── 📁 docs/                          # Documentation and analysis reports
│   └── ModelsAnalyze/                # Model analysis and performance graphs
├── 📁 output/                        # Generated reports and charts
│   ├── *.html                        # Interactive dashboards
│   ├── *.md                          # Analysis reports
│   └── *.png                         # Chart exports
├── run.py                            # Main application launcher (root)
├── app.py                            # Advanced launcher with subcommands
├── requirements.txt                  # Python dependencies
├── README.md                         # This documentation
├── LICENSE                           # MIT License
├── .gitignore                        # Git ignore rules
└── venv/                            # Virtual environment (after setup)
```

**Professional Organization:**
- 📁 **Structured directories** for different components (src/, models/, tests/, etc.)
- 📚 **Comprehensive documentation** with README files in each directory
- 🔧 **Utility scripts** in dedicated scripts/ folder
- 📊 **Organized outputs** in separate directories (data/, output/, docs/)

**Windows-Specific Features:**
- `scripts/activate_env.bat` - One-click environment setup for Command Prompt
- `scripts/activate_env.ps1` - PowerShell environment setup with enhanced features
- `tests/test_imports.py` - Comprehensive library verification for Windows
- `scripts/visualize_data.bat` - Quick launcher for performance data visualization
- `venv/` - Virtual environment directory (created during setup)

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
python src/analyze_performance.py

# Interactive analysis with dashboard
python src/analyze_performance.py --interactive

# Analyze specific data file
python src/analyze_performance.py --data-file data/performance_data_20250709_143022.json
```

##### Analytics Testing & Demo
```bash
# Test data science features
python tests/test_data_science.py

# Test ModelsAnalyze folder functionality  
python tests/test_models_analyze.py

# Run analytics demo with camera
python src/demo_analytics.py

# Demo with video file
python src/demo_analytics.py --video media/traffic.mp4

# Generate sample analytics data
python src/demo_sample_analytics.py

# Analyze existing performance data
python src/analyze_performance.py --interactive
```

### 📊 Performance Data Visualization System

The application includes a comprehensive performance data visualization system that allows users to analyze their detection performance data anytime after testing.

#### Visualization Tools Available

**1. 🚀 Quick Launcher (Recommended)**
```bash
# Launch the visualization tool selector
python visualization/launch_visualizer.py

# Or use Windows batch file
scripts/visualize_data.bat
```

**2. 🖥️ GUI Visualizer (User-Friendly)**
```bash
# Launch GUI application
python visualization/visualize_performance_gui.py
```

**3. 💻 Command-Line Visualizer (Advanced)**
```bash
# Launch interactive command-line tool
python visualization/visualize_performance.py
```

#### Features of the Visualization System

**📈 Dashboard Creation:**
- **Comprehensive Performance Dashboard** - Interactive HTML dashboards with multiple charts
- **Model Comparison Charts** - Side-by-side performance comparisons
- **Time Series Analysis** - Performance trends over time
- **Detection Pattern Analysis** - Object detection frequency and types

**📊 Chart Types:**
- FPS performance over time
- Inference time distributions
- Detection count trends
- Model performance box plots
- Detection type pie charts
- Performance correlation scatter plots

**💾 Export Options:**
- Interactive HTML dashboards (opens in browser)
- High-resolution PNG charts
- Detailed markdown reports
- CSV data exports

**🔍 Data Analysis:**
- Automatic data file detection
- File browser for data selection
- Real-time statistics display
- Performance summary reports

#### Using the Visualization System

**Quick Start:**
1. Run your object detection application to generate performance data
2. Launch `python visualization/launch_visualizer.py` or double-click `scripts/visualize_data.bat`
3. Select option 1 for GUI or option 3 for quick dashboard
4. Choose your data file and generate visualizations

**GUI Workflow:**
1. Launch the GUI: `python visualization/visualize_performance_gui.py`
2. Select a data file from the dropdown or browse for one
3. View data summary in the information panel
4. Click visualization buttons to generate charts
5. Charts automatically open in your browser or image viewer

**Advanced Analysis:**
```bash
# Command-line interactive mode
python visualization/visualize_performance.py

# Available actions:
# 1. List available data files
# 2. Load data file
# 3. Show data summary
# 4. Create performance dashboard
# 5. Create model comparison charts
# 6. Create time series analysis
# 7. Generate summary report
```

#### Sample Outputs

**Performance Dashboard** (`dashboard_YYYYMMDD_HHMMSS.html`):
- Interactive multi-panel dashboard
- Real-time performance metrics
- Model switching analysis
- Detection pattern visualization

**Model Comparison** (`model_comparison_YYYYMMDD_HHMMSS.png`):
- Statistical comparison between models
- Performance distribution analysis
- Speed vs accuracy trade-offs

**Summary Reports** (`performance_report_YYYYMMDD_HHMMSS.md`):
- Detailed performance statistics
- Model-specific analysis
- Recommendations for optimization

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

**Advanced Analysis:**
```python
# Load performance data
import sys
sys.path.append('src')
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
