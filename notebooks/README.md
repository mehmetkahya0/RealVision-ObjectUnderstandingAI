# 📓 Jupyter Notebooks

This directory contains Jupyter notebooks for interactive analysis and experimentation.

## 📒 Available Notebooks

### Performance Analysis
- `performance_analysis.ipynb` - Comprehensive performance analysis notebook
- Interactive data exploration, statistical analysis, and visualization

## 🚀 Getting Started

### Launch Jupyter
```bash
# Activate environment first
scripts/activate_env.ps1

# Start Jupyter
jupyter notebook

# Navigate to notebooks/ and open performance_analysis.ipynb
```

### Alternative: VS Code
```bash
# Open in VS Code with Jupyter extension
code notebooks/performance_analysis.ipynb
```

## 📊 Notebook Features

### Performance Analysis Notebook
- **Data Loading**: Import performance data from JSON files
- **Statistical Analysis**: Descriptive statistics, hypothesis testing
- **Visualizations**: Interactive plots with matplotlib, seaborn, plotly
- **Model Comparison**: Compare YOLOv8, MobileNet-SSD, ONNX performance
- **Time Series Analysis**: Performance trends over time
- **Export Tools**: Save charts and generate reports

### Interactive Features
- Real-time data visualization
- Parameter adjustment widgets
- Custom analysis workflows
- Export capabilities

## 🔬 Analysis Workflows

### Basic Performance Review
1. Load performance data from `data/` directory
2. Generate summary statistics
3. Create performance comparison charts
4. Analyze FPS and inference time trends

### Advanced Analysis
1. Statistical significance testing
2. Correlation analysis between variables
3. Predictive modeling for performance optimization
4. Custom visualization dashboards

### Model Optimization
1. Confidence threshold optimization
2. Model selection based on use case
3. Performance tuning recommendations
4. Resource utilization analysis

## 📈 Sample Analysis

The notebook includes examples for:
- Loading and preprocessing performance data
- Creating interactive dashboards
- Statistical model comparison
- Performance optimization recommendations
- Custom analysis workflows

## 💾 Data Sources

Notebooks can analyze:
- Real-time performance data from application runs
- Historical data from `data/` directory
- Custom datasets in JSON/CSV format
- Comparative studies across different models

## 🎯 Use Cases

**Research**: Detailed performance analysis and model comparison
**Optimization**: Find optimal settings for your specific use case  
**Reporting**: Generate professional analysis reports
**Experimentation**: Test new analysis methods and visualizations

## 🔧 Adding New Notebooks

To create new analysis notebooks:
1. Create `.ipynb` files in this directory
2. Use consistent naming convention
3. Include documentation and examples
4. Test with sample data

*Note: Ensure Jupyter is installed: `pip install jupyter` or use the included requirements.txt*
