# 💾 Main Source Code

This directory contains the core application source code.

## 📁 Source Files

### Main Application
- `main.py` - Core object detection application with GUI and analytics
- `run.py` - Application launcher and command-line interface

### Analytics & Performance
- `performance_analyzer.py` - Real-time performance monitoring and analysis
- `analyze_performance.py` - Standalone performance analysis tool

### Demo & Testing
- `demo_analytics.py` - Interactive analytics demonstration
- `demo_sample_analytics.py` - Sample data generation for testing

## 🏗️ Application Architecture

### Main Components

**ObjectUnderstandingApp** (`main.py`)
- Real-time object detection and tracking
- Multi-model support (YOLOv8, MobileNet-SSD, ONNX)
- Performance monitoring and analytics
- GUI interface and keyboard controls

**ModelPerformanceAnalyzer** (`performance_analyzer.py`)
- Real-time performance data collection
- Statistical analysis and reporting
- Data export and visualization support

### Core Features
- 🎥 **Multi-source input**: Camera, video files, image sequences
- 🤖 **AI Models**: YOLOv8, MobileNet-SSD, ONNX support
- 📊 **Analytics**: Real-time performance monitoring
- 🖼️ **Output**: Screenshots, processed videos, data exports
- ⌨️ **Controls**: Full keyboard control interface

## 🚀 Usage Examples

### Basic Usage
```python
# Import and run main application
from main import ObjectUnderstandingApp

app = ObjectUnderstandingApp()
app.run(camera_index=0)
```

### Advanced Configuration
```python
app = ObjectUnderstandingApp()
app.confidence_threshold = 0.7
app.model_type = 'yolo'
app.enable_analytics = True
app.run(input_source='video.mp4')
```

### Performance Analysis
```python
from performance_analyzer import ModelPerformanceAnalyzer

analyzer = ModelPerformanceAnalyzer()
# Analyzer automatically collects data during application runs
analyzer.generate_report()
```

## 🔧 Key Classes

### ObjectUnderstandingApp
- **Purpose**: Main application class
- **Methods**: `run()`, `process_frame()`, `handle_keyboard()`
- **Features**: Object detection, tracking, GUI, analytics

### ModelPerformanceAnalyzer  
- **Purpose**: Performance monitoring and analysis
- **Methods**: `log_performance()`, `analyze_data()`, `generate_report()`
- **Features**: Real-time metrics, statistical analysis, data export

## 📊 Analytics Features

### Real-time Monitoring
- FPS tracking and optimization
- Inference time measurement  
- Detection count analysis
- Model performance comparison

### Data Collection
- JSON export of performance metrics
- Session-based data logging
- Historical performance tracking
- Statistical analysis ready data

### Reporting
- Interactive HTML dashboards
- Statistical model comparison
- Performance optimization recommendations
- Custom analysis workflows

## ⚙️ Configuration

### Model Settings
```python
# Available models
MODELS = {
    'yolo': 'YOLOv8 nano model',
    'dnn': 'MobileNet-SSD with OpenCV DNN',
    'onnx': 'YOLOv5 ONNX model'
}
```

### Performance Settings
```python
# Default configuration
DEFAULT_CONFIG = {
    'confidence_threshold': 0.5,
    'enable_tracking': True,
    'enable_analytics': True,
    'max_objects': 100
}
```

## 🔍 Debugging

### Common Issues
- **Import Errors**: Check virtual environment activation
- **Model Loading**: Verify model files in `models/` directory
- **Camera Access**: Test camera permissions and availability
- **Performance**: Adjust confidence threshold and model selection

### Debug Mode
```bash
# Run with debug output
python src/run.py --debug
```

*Note: All source files are designed to work together as a cohesive application.*
