# 🚀 RealVision Object Understanding AI

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyQt6](https://img.shields.io/badge/GUI-PyQt6-green.svg)](https://pypi.org/project/PyQt6/)
[![OpenCV](https://img.shields.io/badge/CV-OpenCV-red.svg)](https://opencv.org/)
[![YOLOv8](https://img.shields.io/badge/AI-YOLOv8-orange.svg)](https://github.com/ultralytics/ultralytics)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**A professional-grade real-time object detection and understanding system powered by state-of-the-art AI models. Transform your computer vision projects with advanced analytics, intuitive interfaces, and enterprise-level performance monitoring.**

---

## 🎯 Overview

RealVision Object Understanding AI is a comprehensive computer vision platform that combines cutting-edge AI models with professional-grade analytics and visualization tools. Whether you're a researcher, developer, or business looking to integrate object detection into your workflow, this system provides everything you need for production-ready applications.

### 🎬 Demo Video

![RealVision Demo](media/demo.mov)


*Watch our demo video showcasing real-time object detection, tracking, and analytics features.*

## ✨ Core Features

### 🤖 **Advanced AI Models**
- **YOLOv8**: Latest state-of-the-art object detection with 30+ FPS performance
- **MobileNet-SSD**: Lightweight model optimized for mobile and edge devices
- **ONNX Runtime**: Cross-platform inference with optimal performance
- **EfficientDet**: Google's efficient detection with superior accuracy
- **Dynamic Model Switching**: Switch between models in real-time

### 🎥 **Real-Time Processing**
- **Live Camera Feed**: Real-time detection from webcam or IP cameras
- **Video File Processing**: Batch processing of video files with progress tracking
- **Multi-Camera Support**: Connect and process multiple camera streams
- **Frame Rate Optimization**: Adaptive FPS control for smooth performance

### 📊 **Professional Analytics Dashboard**
- **Performance Metrics**: Real-time FPS, inference time, and resource usage
- **Model Comparison**: Side-by-side performance analysis across models
- **Detection Statistics**: Object count, confidence scores, and class distribution
- **Time Series Analysis**: Performance trends over time with interactive charts
- **Export Capabilities**: JSON, CSV, and visual report generation

### 🖥️ **Modern GUI Interface**
- **PyQt6 Framework**: Professional desktop application with native feel
- **Real-Time Controls**: Adjust confidence thresholds and model settings on-the-fly
- **Visual Feedback**: Live preview with bounding boxes and confidence scores
- **Multi-Threading**: Responsive interface with background processing
- **Customizable Layout**: Adjustable panels and workspace configuration

### 🎯 **Advanced Object Tracking**
- **Multi-Object Tracking**: Track multiple objects across frames with unique IDs
- **Trajectory Analysis**: Visualize object movement patterns and paths
- **Persistence Tracking**: Maintain object identity through occlusions
- **Speed Estimation**: Calculate object velocities and movement statistics

### 📈 **Data Science & Analytics**
- **Performance Profiling**: Comprehensive model performance analysis
- **Statistical Reports**: Detailed metrics with confidence intervals
- **Visualization Suite**: Interactive charts, graphs, and dashboards
- **A/B Testing**: Compare model performance across different conditions
- **Machine Learning Insights**: Pattern recognition in detection data

### �️ **Developer Tools**
- **API Integration**: Easy integration with existing systems
- **Command Line Interface**: Scriptable automation for batch processing
- **Extensive Documentation**: Comprehensive guides and API reference
- **Testing Suite**: Unit tests and integration tests for reliability
- **Custom Model Support**: Easy integration of custom trained models

---

## �🚀 Quick Start Guide

### 📦 Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/mehmetkahya0/RealVision-ObjectUnderstandingAI.git
   cd RealVision-ObjectUnderstandingAI
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download Model Files**
   ```bash
   python scripts/download_models.py
   ```

### 🎮 Usage Examples

#### **GUI Application (Recommended)**
```bash
# Launch the professional GUI interface
python app.py

# Launch with specific camera
python app.py --camera 1

# Process video file
python app.py --input video.mp4
```

#### **Command Line Interface**
```bash
# Run with default settings
python app.py run

# Use specific model
python app.py run --model yolo --confidence 0.7

# Process video with output directory
python app.py run --input video.mp4 --output-dir results/
```

#### **Analytics & Visualization**
```bash
# Launch analytics dashboard
python app.py --visualize

# Generate performance report
python app.py --demo

# Run comprehensive tests
python app.py --test
```

---

## 📊 Performance Benchmarks

### **Model Performance Comparison**

| Model | FPS (1080p) | Memory Usage | CPU Usage | Accuracy (mAP) |
|-------|-------------|--------------|-----------|----------------|
| **YOLOv8n** | 45-60 | 400MB | 45% | 37.3% |
| **YOLOv8s** | 35-45 | 600MB | 55% | 44.9% |
| **YOLOv8m** | 25-35 | 1.2GB | 70% | 50.2% |
| **MobileNet-SSD** | 50-65 | 250MB | 35% | 22.2% |
| **EfficientDet-D0** | 30-40 | 800MB | 60% | 34.6% |

### **System Requirements**

| Component | Minimum | Recommended | Optimal |
|-----------|---------|-------------|---------|
| **CPU** | Intel i5-6400 | Intel i7-8700K | Intel i9-10900K |
| **RAM** | 8GB | 16GB | 32GB |
| **GPU** | GTX 1060 6GB | RTX 3070 | RTX 4080+ |
| **Storage** | 5GB | 10GB | 20GB SSD |
| **Python** | 3.8+ | 3.9+ | 3.10+ |

---

## 🎯 Supported Object Classes

The system can detect and classify **80+ object classes** including:

### **Common Objects**
- **People**: person, clothing, accessories
- **Vehicles**: car, truck, bus, motorcycle, bicycle
- **Animals**: dog, cat, bird, horse, sheep, cow
- **Sports**: ball, frisbee, skateboard, surfboard
- **Food**: apple, banana, sandwich, pizza, donut

### **Household Items**
- **Furniture**: chair, couch, table, bed, toilet
- **Electronics**: TV, laptop, mouse, keyboard, cell phone
- **Kitchen**: microwave, oven, refrigerator, sink
- **Tools**: scissors, hair dryer, toothbrush

---

## 🏗️ Architecture & Project Structure

```
RealVision-ObjectUnderstandingAI/
├── 📁 app.py                    # Main application launcher
├── 📁 gui.py                    # PyQt6 GUI interface
├── 📁 main.py                   # Legacy entry point
├── 📁 requirements.txt          # Python dependencies
├── 📁 LICENSE                   # MIT License
│
├── 📁 src/                      # Core application logic
│   ├── 📄 main.py              # Object detection engine
│   ├── 📄 run.py               # Console runner
│   ├── 📄 performance_analyzer.py # Analytics engine
│   ├── 📄 demo_analytics.py    # Demo data generator
│   └── 📄 demo_sample_analytics.py # Sample analytics
│
├── 📁 models/                   # AI model files
│   ├── 📄 yolov8n.pt          # YOLOv8 nano model
│   ├── 📄 yolov5s.onnx        # YOLOv5 ONNX model
│   ├── 📄 MobileNetSSD_deploy.caffemodel # MobileNet model
│   └── 📄 MobileNetSSD_deploy.prototxt   # MobileNet config
│
├── 📁 data/                     # Performance data storage
│   └── 📄 performance_data_*.json # Collected metrics
│
├── 📁 visualization/            # Visualization tools
│   ├── 📄 launch_visualizer.py # Visualization launcher
│   ├── 📄 visualize_performance.py # CLI visualizer
│   └── 📄 visualize_performance_gui.py # GUI visualizer
│
├── 📁 media/                    # Media files
│   ├── 📄 demo.mov             # Demo video
│   ├── 📄 traffic.mp4          # Sample video
│   ├── 📄 icon.png             # Application icon
│   └── 📁 screenshots/         # Captured screenshots
│
├── 📁 ModelsAnalyze/           # Generated analysis reports
│   ├── 📄 model_performance_comparison.png
│   ├── 📄 detection_patterns_analysis.png
│   ├── 📄 performance_dashboard.html
│   └── 📄 performance_report.json
│
├── 📁 output/                   # Generated reports
│   ├── 📄 performance_report_*.md
│   ├── 📄 dashboard_*.html
│   └── 📄 time_series_*.html
│
├── 📁 tests/                    # Test suite
│   ├── 📄 test_imports.py      # Import validation
│   ├── 📄 test_data_science.py # Analytics tests
│   └── 📄 test_visualization_system.py # Visualization tests
│
├── 📁 scripts/                  # Utility scripts
│   ├── 📄 download_models.py   # Model downloader
│   ├── 📄 setup.py            # Environment setup
│   └── 📄 activate_env.bat    # Environment activation
│
├── 📁 docs/                     # Documentation
│   └── 📄 README.md           # Additional documentation
│
└── 📁 notebooks/               # Jupyter notebooks
    └── 📄 performance_analysis.ipynb # Interactive analysis
```

---

## 🎮 User Interface & Controls

### **GUI Interface (PyQt6)**

![GUI Interface](media/screenshots/gui_interface.png)

#### **Main Controls**
- **📹 Open Camera**: Start live webcam feed
- **📁 Upload Video**: Load and process video files
- **⏹️ Stop Processing**: Stop current detection session
- **📊 Show Analytics**: Launch performance dashboard
- **📝 Generate Report**: Create comprehensive analytics report
- **⚙️ Settings**: Configure models and parameters

#### **Real-Time Adjustments**
- **🎯 Confidence Slider**: Adjust detection confidence (0.1-1.0)
- **🤖 Model Selector**: Switch between AI models instantly
- **📏 Tracking Settings**: Configure object tracking parameters
- **🎨 Display Options**: Toggle bounding boxes, labels, and IDs

### **Keyboard Shortcuts (Console Mode)**

| Key | Action | Description |
|-----|---------|-------------|
| `Q` | Quit | Exit application |
| `SPACE` | Pause/Resume | Toggle processing |
| `S` | Screenshot | Capture current frame |
| `M` | Model Switch | Cycle through AI models |
| `C` | Confidence | Toggle confidence display |
| `T` | Tracking | Toggle tracking IDs |
| `+/-` | Threshold | Adjust confidence threshold |
| `A` | Analytics | Generate performance report |
| `D` | Data Logging | Toggle data collection |
| `F` | Full Screen | Toggle full screen mode |

---

## 📈 Analytics & Visualization

### **Performance Dashboard**

![Performance Dashboard](ModelsAnalyze/performance_dashboard.html)

Our comprehensive analytics dashboard provides:

#### **Real-Time Metrics**
- **FPS Monitoring**: Live frame rate tracking
- **Inference Time**: Model processing speed
- **Memory Usage**: System resource consumption
- **Detection Accuracy**: Confidence score distribution

#### **Model Comparison Analysis**

![Model Comparison](ModelsAnalyze/model_performance_comparison.png)

Compare different AI models across:
- **Speed vs Accuracy**: Performance trade-offs
- **Resource Usage**: Memory and CPU consumption
- **Detection Quality**: Precision and recall metrics
- **Stability**: Performance consistency over time

#### **Detection Patterns**

![Detection Patterns](ModelsAnalyze/detection_patterns_analysis.png)

Analyze object detection patterns:
- **Class Distribution**: Most detected object types
- **Confidence Trends**: Detection confidence over time
- **Spatial Analysis**: Object location patterns
- **Temporal Patterns**: Detection frequency changes

### **Export & Reporting**

#### **Report Formats**
- **📊 HTML Dashboard**: Interactive web-based reports
- **📈 JSON Data**: Raw performance metrics
- **📋 Markdown Reports**: Human-readable summaries
- **📊 CSV Export**: Spreadsheet-compatible data
- **🖼️ PNG Charts**: High-quality visualizations

#### **Sample Analytics Code**
```python
from src.performance_analyzer import ModelPerformanceAnalyzer

# Initialize analyzer
analyzer = ModelPerformanceAnalyzer()

# Load performance data
analyzer.load_data('data/performance_data.json')

# Generate comprehensive report
analyzer.generate_report(
    output_format='html',
    include_charts=True,
    export_path='reports/analysis.html'
)
```

---

## 🔧 Advanced Configuration

### **Environment Variables**

Set up your environment for optimal performance:

```bash
# Model preferences
export REALVISION_MODEL=yolo              # Default AI model
export REALVISION_CONFIDENCE=0.6          # Default confidence threshold
export REALVISION_DEVICE=cuda             # Processing device (cuda/cpu)

# Performance settings
export REALVISION_MAX_FPS=30              # Maximum frame rate
export REALVISION_BUFFER_SIZE=3           # Frame buffer size
export REALVISION_BATCH_SIZE=1            # Inference batch size

# Analytics settings
export REALVISION_ANALYTICS_ENABLED=true  # Enable performance tracking
export REALVISION_DATA_PATH=./data        # Analytics data directory
```

### **Custom Model Integration**

Integrate your own trained models:

```python
from src.main import ObjectUnderstandingApp

# Initialize with custom configuration
app = ObjectUnderstandingApp(
    model_path='path/to/your/model.pt',
    confidence_threshold=0.7,
    device='cuda',
    enable_tracking=True,
    enable_analytics=True
)

# Custom class names
app.set_class_names(['person', 'car', 'custom_object'])

# Run detection
app.run()
```

### **API Integration**

Use RealVision as a service:

```python
from src.api import RealVisionAPI

# Initialize API
api = RealVisionAPI(host='localhost', port=8080)

# Process image
result = api.detect_objects(image_path='image.jpg')
print(f"Detected {len(result['detections'])} objects")

# Process video
video_results = api.process_video(
    video_path='video.mp4',
    output_path='results.json'
)
```

---

## 🧪 Testing & Development

### **Running Tests**

```bash
# Run all tests
python app.py --test

# Run specific test categories
python app.py --test --imports          # Test imports and dependencies
python app.py --test --data-science     # Test analytics features
python app.py --test --visualization    # Test visualization system

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

### **Development Setup**

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Format code
black src/ gui.py app.py

# Type checking
mypy src/

# Linting
flake8 src/
```

### **Performance Profiling**

```bash
# Profile memory usage
python -m memory_profiler src/main.py

# Profile CPU usage
python -m cProfile -o profile.prof src/main.py

# Analyze profiling results
python -m snakeviz profile.prof
```

---

## 🌟 Use Cases & Applications

### **🏢 Enterprise Applications**
- **Security Systems**: Real-time surveillance and threat detection
- **Retail Analytics**: Customer behavior analysis and inventory management
- **Manufacturing**: Quality control and defect detection
- **Healthcare**: Medical imaging and diagnostic assistance

### **🎓 Research & Education**
- **Computer Vision Research**: Benchmark different AI models
- **Academic Projects**: Teaching tool for machine learning courses
- **Prototyping**: Rapid development of vision applications
- **Data Collection**: Automated annotation and labeling

### **👨‍💻 Developer Tools**
- **API Integration**: Embed object detection in existing applications
- **Model Comparison**: Evaluate different AI architectures
- **Performance Testing**: Benchmark system capabilities
- **Custom Training**: Prepare datasets for model training

---

## 🛠️ Troubleshooting

### **Common Issues**

#### **Camera Not Detected**
```bash
# List available cameras
python app.py run --list-cameras

# Try different camera indices
python app.py run --camera 1
python app.py run --camera 2
```

#### **Performance Issues**
```bash
# Reduce model complexity
python app.py run --model mobilenet

# Lower confidence threshold
python app.py run --confidence 0.3

# Disable analytics
export REALVISION_ANALYTICS_ENABLED=false
```

#### **Memory Issues**
```bash
# Reduce buffer size
export REALVISION_BUFFER_SIZE=1

# Use CPU instead of GPU
export REALVISION_DEVICE=cpu
```

### **System Requirements Check**

```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Check OpenCV installation
python -c "import cv2; print(cv2.__version__)"

# Check all dependencies
python scripts/check_requirements.py
```

---

## � Documentation & Resources

### **📖 Additional Documentation**
- [**API Reference**](docs/api_reference.md) - Complete API documentation
- [**Model Guide**](docs/model_guide.md) - Detailed model comparison
- [**Performance Tuning**](docs/performance_tuning.md) - Optimization tips
- [**Custom Training**](docs/custom_training.md) - Train your own models

### **🎥 Video Tutorials**
- [**Getting Started**](media/tutorials/getting_started.mp4) - Basic setup and usage
- [**Advanced Features**](media/tutorials/advanced_features.mp4) - Analytics and customization
- [**API Integration**](media/tutorials/api_integration.mp4) - Using RealVision in your projects

### **🔗 Useful Links**
- [**YOLOv8 Documentation**](https://docs.ultralytics.com/)
- [**OpenCV Tutorials**](https://opencv-python-tutroals.readthedocs.io/)
- [**PyQt6 Guide**](https://doc.qt.io/qtforpython/)
- [**Computer Vision Papers**](https://paperswithcode.com/area/computer-vision)

---

## 🤝 Contributing

We welcome contributions from the community! Whether you're fixing bugs, adding features, or improving documentation, your help is appreciated.

### **📋 How to Contribute**

1. **Fork the Repository**
   ```bash
   git clone https://github.com/mehmetkahya0/RealVision-ObjectUnderstandingAI.git
   cd RealVision-ObjectUnderstandingAI
   ```

2. **Create a Feature Branch**
   ```bash
   git checkout -b feature/amazing-feature
   ```

3. **Make Your Changes**
   - Follow the existing code style
   - Add tests for new features
   - Update documentation as needed

4. **Test Your Changes**
   ```bash
   python app.py --test
   python -m pytest tests/
   ```

5. **Submit a Pull Request**
   ```bash
   git commit -m 'Add amazing feature'
   git push origin feature/amazing-feature
   ```

### **🎯 Contribution Guidelines**

- **Code Style**: Follow PEP 8 and use Black for formatting
- **Testing**: Write tests for new features and bug fixes
- **Documentation**: Update README and docstrings
- **Performance**: Ensure changes don't degrade performance
- **Compatibility**: Maintain Python 3.8+ compatibility

### **🐛 Bug Reports**

Use our [Issue Template](https://github.com/mehmetkahya0/RealVision-ObjectUnderstandingAI/issues/new?template=bug_report.md) to report bugs with:
- System information
- Steps to reproduce
- Expected vs actual behavior
- Error messages and logs

### **💡 Feature Requests**

Submit feature requests using our [Feature Template](https://github.com/mehmetkahya0/RealVision-ObjectUnderstandingAI/issues/new?template=feature_request.md) with:
- Clear description
- Use case examples
- Implementation suggestions
- Impact assessment

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 Mehmet Kahya

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

## 🙏 Acknowledgments

We extend our gratitude to the following projects and contributors:

### **🔬 AI & Machine Learning**
- [**YOLOv8**](https://github.com/ultralytics/ultralytics) - Advanced object detection architecture
- [**PyTorch**](https://pytorch.org/) - Deep learning framework
- [**OpenCV**](https://opencv.org/) - Computer vision library
- [**ONNX**](https://onnx.ai/) - Open neural network exchange format

### **🖥️ User Interface & Visualization**
- [**PyQt6**](https://pypi.org/project/PyQt6/) - Professional GUI framework
- [**Matplotlib**](https://matplotlib.org/) - Data visualization
- [**Plotly**](https://plotly.com/) - Interactive charts
- [**Seaborn**](https://seaborn.pydata.org/) - Statistical visualization

### **📊 Data Science & Analytics**
- [**Pandas**](https://pandas.pydata.org/) - Data manipulation and analysis
- [**NumPy**](https://numpy.org/) - Numerical computing
- [**Scikit-learn**](https://scikit-learn.org/) - Machine learning tools
- [**Bokeh**](https://bokeh.org/) - Interactive visualization

### **🏗️ Development Tools**
- [**GitHub**](https://github.com/) - Version control and collaboration
- [**pytest**](https://pytest.org/) - Testing framework
- [**Black**](https://black.readthedocs.io/) - Code formatting
- [**MyPy**](https://mypy.readthedocs.io/) - Static type checking

---

## 📞 Support & Community

### **💬 Get Help**
- **📧 Email**: [mehmetkahyakas5@gmail.com](mailto:mehmetkahyakas5@gmail.com)
- **📖 Documentation**: [docs.realvision-ai.com](https://docs.realvision-ai.com)

### **🐛 Issue Reporting**
- **GitHub Issues**: [Report bugs and request features](https://github.com/mehmetkahya0/RealVision-ObjectUnderstandingAI/issues)
---

## 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=mehmetkahya0/RealVision-ObjectUnderstandingAI&type=Date)](https://star-history.com/#mehmetkahya0/RealVision-ObjectUnderstandingAI&Date)

---

## 📊 Project Stats

![GitHub stars](https://img.shields.io/github/stars/mehmetkahya0/RealVision-ObjectUnderstandingAI?style=social)
![GitHub forks](https://img.shields.io/github/forks/mehmetkahya0/RealVision-ObjectUnderstandingAI?style=social)
![GitHub watchers](https://img.shields.io/github/watchers/mehmetkahya0/RealVision-ObjectUnderstandingAI?style=social)
![GitHub issues](https://img.shields.io/github/issues/mehmetkahya0/RealVision-ObjectUnderstandingAI)
![GitHub pull requests](https://img.shields.io/github/issues-pr/mehmetkahya0/RealVision-ObjectUnderstandingAI)
![GitHub last commit](https://img.shields.io/github/last-commit/mehmetkahya0/RealVision-ObjectUnderstandingAI)

---

<div align="center">

**Author**: [Mehmet Kahya](https://github.com/mehmetkahya0)  
**Date**: July 2025  
**Version**: 1.0.0  

*Built with ❤️ for the computer vision community*

</div>
