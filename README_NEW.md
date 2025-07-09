# 🎯 RealVision Object Understanding AI

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyQt6](https://img.shields.io/badge/GUI-PyQt6-green.svg)](https://pypi.org/project/PyQt6/)
[![OpenCV](https://img.shields.io/badge/CV-OpenCV-red.svg)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A professional real-time object detection and understanding system powered by state-of-the-art AI models. Features advanced computer vision capabilities, comprehensive analytics, and an intuitive GUI interface.

## ✨ Key Features

- 🎥 **Real-time Detection**: Live camera feed and video file processing
- 🤖 **Multiple AI Models**: YOLOv8, MobileNet-SSD, ONNX, and EfficientDet support
- 📊 **Advanced Analytics**: Performance monitoring and detailed reporting
- 🖥️ **Professional GUI**: Modern PyQt6 interface with real-time controls
- 📈 **Data Science Integration**: Comprehensive performance analysis
- 🎯 **Object Tracking**: Advanced multi-object tracking capabilities
- 📷 **Export Features**: Screenshots, reports, and analytics data export

## 🚀 Quick Start

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/RealVision-ObjectUnderstandingAI.git
   cd RealVision-ObjectUnderstandingAI
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   # Launch GUI version
   python main.py --gui
   
   # Or use the console version
   python main.py
   ```

### Usage Examples

```bash
# Launch GUI application
python main.py --gui

# Run with specific camera
python main.py --camera 1

# Process video file
python main.py --video path/to/video.mp4

# Use specific AI model
python main.py --model yolo

# Run analytics demo
python main.py --demo
```

## 🏗️ Project Structure

```
RealVision-ObjectUnderstandingAI/
├── main.py                 # Main application entry point
├── gui.py                  # PyQt6 GUI interface
├── requirements.txt        # Project dependencies
├── src/                    # Core source code
│   ├── main.py            # Main application logic
│   ├── performance_analyzer.py  # Analytics engine
│   ├── demo_analytics.py  # Demo and sample data
│   └── run.py             # Console runner
├── models/                 # AI model files
├── data/                   # Performance data storage
├── output/                 # Generated reports and exports
├── screenshots/            # Captured screenshots
└── docs/                   # Documentation
```

## 🎮 Controls

### GUI Interface
- **Open Camera**: Start live camera feed
- **Upload Video**: Load and process video files
- **Stop**: Stop current processing
- **Show Analytics**: View performance dashboards
- **Generate Report**: Create comprehensive analytics reports

### Keyboard Controls (Console Mode)
- `Q` - Quit application
- `SPACE` - Pause/Resume
- `S` - Take screenshot
- `M` - Switch between AI models
- `C` - Toggle confidence scores
- `T` - Toggle tracking IDs
- `+/-` - Adjust confidence threshold
- `A` - Generate analytics report
- `D` - Toggle data logging

## 🤖 Supported AI Models

| Model | Description | Speed | Accuracy |
|-------|-------------|--------|----------|
| **YOLOv8** | Latest YOLO architecture | ⚡⚡⚡ | 🎯🎯🎯🎯 |
| **MobileNet-SSD** | Lightweight detection | ⚡⚡⚡⚡ | 🎯🎯🎯 |
| **ONNX** | Cross-platform inference | ⚡⚡⚡ | 🎯🎯🎯🎯 |
| **EfficientDet** | Google's efficient detection | ⚡⚡ | 🎯🎯🎯🎯🎯 |

## 📊 Analytics Features

- **Real-time Performance Monitoring**: FPS, processing time, detection counts
- **Model Comparison**: Side-by-side performance analysis
- **Statistical Analysis**: Comprehensive performance metrics
- **Interactive Dashboards**: Visual performance reports
- **Data Export**: JSON, CSV, and plot exports
- **Time Series Analysis**: Performance trends over time

## 🛠️ Advanced Configuration

### Environment Variables
```bash
# Model preferences
export REALVISION_MODEL=yolo
export REALVISION_CONFIDENCE=0.6

# Performance settings
export REALVISION_MAX_FPS=30
export REALVISION_BUFFER_SIZE=3
```

### Custom Model Integration
```python
from src.main import ObjectUnderstandingApp

# Custom model configuration
app = ObjectUnderstandingApp(preferred_model='yolo')
app.confidence_threshold = 0.7
app.run()
```

## 🔧 Development

### Running Tests
```bash
# Run all tests
python -m pytest tests/

# Test specific components
python tests/test_imports.py
python tests/test_data_science.py
```

### Code Quality
```bash
# Format code
black src/ gui.py main.py

# Type checking
mypy src/
```

## 📈 Performance Benchmarks

| Model | FPS (1080p) | Memory Usage | CPU Usage |
|-------|-------------|--------------|-----------|
| YOLOv8 | 30-45 | 800MB | 65% |
| MobileNet-SSD | 45-60 | 400MB | 45% |
| ONNX | 35-50 | 600MB | 55% |

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [YOLOv8](https://github.com/ultralytics/ultralytics) - Advanced object detection
- [OpenCV](https://opencv.org/) - Computer vision library
- [PyQt6](https://pypi.org/project/PyQt6/) - GUI framework
- [TensorFlow](https://tensorflow.org/) - Machine learning platform

## 📞 Support

For support, please open an issue in the GitHub repository or contact the development team.

---

**Author**: Mehmet Kahya  
**Date**: July 2025  
**Version**: 1.0.0
