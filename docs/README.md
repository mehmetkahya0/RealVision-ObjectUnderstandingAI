# 📚 Documentation & Analysis Reports

This directory contains generated documentation, analysis reports, and performance data visualizations.

## 📊 Contents

### Generated Reports
- **Performance Reports** (`.md` files) - Detailed performance analysis
- **Interactive Dashboards** (`.html` files) - Web-based performance charts
- **Model Analysis** - Comparative studies between different AI models

### ModelsAnalyze Directory
Contains detailed model performance analysis:
- `performance_report.json` - Raw performance metrics
- `performance_dashboard.html` - Interactive analysis dashboard
- `model_performance_comparison.png` - Visual model comparisons
- Additional charts and graphs

## 🔄 Auto-Generated Content

This directory is populated automatically when you:
1. Run the object detection application with analytics enabled
2. Generate performance reports using `python src/analyze_performance.py`
3. Create visualizations using the visualization tools

## 📈 Viewing Reports

**HTML Dashboards**: Open `.html` files in your web browser for interactive charts
**Markdown Reports**: View `.md` files in any text editor or markdown viewer
**JSON Data**: Raw data files that can be processed with custom analysis tools

## 🧹 Cleanup

To clean up old reports:
```bash
# Remove all generated reports (optional)
rm docs/performance_report_*.md
rm docs/dashboard_*.html
rm docs/time_series_*.html
```

*Note: This directory is automatically created and managed by the application.*
