# 📊 Visualization Tools

This directory contains all the performance data visualization tools.

## 🔧 Tools Available

### Interactive Launcher
- `launch_visualizer.py` - Main launcher with menu system
- Provides easy access to all visualization tools

### GUI Tools
- `visualize_performance_gui.py` - User-friendly graphical interface
- File browser, data summary, and one-click chart generation

### Command-Line Tools  
- `visualize_performance.py` - Advanced command-line interface
- Interactive menu system for power users

## 🚀 Quick Start

### Easiest Method
```bash
# Launch the tool selector
python visualization/launch_visualizer.py

# Or use Windows batch file
scripts/visualize_data.bat
```

### Direct Access
```bash
# GUI version (recommended for beginners)
python visualization/visualize_performance_gui.py

# Command-line version (for advanced users)
python visualization/visualize_performance.py
```

## 📈 Features

### Chart Types
- **Performance Dashboards** - Interactive HTML with multiple charts
- **Model Comparison** - Side-by-side performance analysis
- **Time Series** - Performance trends over time  
- **Statistical Analysis** - Box plots, distributions, correlations

### Data Sources
- Automatically detects performance data files
- Supports data from `data/` directory
- Real-time data from current sessions

### Export Options
- Interactive HTML dashboards (opens in browser)
- High-resolution PNG charts
- Detailed markdown reports
- CSV data exports

## 🎯 Visualization Workflow

1. **Generate Data**: Run object detection application to collect performance data
2. **Launch Visualizer**: Use `launch_visualizer.py` or batch file
3. **Select Data**: Choose from available data files
4. **Create Charts**: Generate dashboards, comparisons, or custom analysis
5. **View Results**: Charts open automatically in browser or image viewer

## 🔍 Data Analysis

The visualization tools provide:
- **Model Performance**: Compare YOLOv8 vs MobileNet-SSD vs ONNX
- **FPS Analysis**: Frame rate trends and optimization insights
- **Inference Time**: Processing speed analysis
- **Detection Patterns**: Object frequency and confidence distributions
- **System Performance**: Memory and CPU utilization trends

## 🛠️ Advanced Usage

### Custom Analysis
```python
# Import visualization modules for custom analysis
from visualize_performance import PerformanceDataVisualizer

visualizer = PerformanceDataVisualizer()
visualizer.load_data('data/performance_data_latest.json')
visualizer.create_dashboard()
```

### Batch Processing
Process multiple data files and generate comparative reports across different sessions.

*Note: All visualization tools are designed to work from the project root directory.*
