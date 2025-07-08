#!/usr/bin/env python3
"""
Standalone Data Science Analysis Tool
=====================================

This script provides standalone analysis capabilities for 
object detection performance data.

Usage:
    python analyze_performance.py [--data-file performance_data.json]
    python analyze_performance.py --interactive

Features:
- Load and analyze performance data
- Generate comprehensive reports
- Create interactive visualizations
- Export analysis results
"""

import sys
import os
import argparse
import json
import pandas as pd
from datetime import datetime

try:
    from performance_analyzer import ModelPerformanceAnalyzer, analyze_session_performance
    ANALYZER_AVAILABLE = True
except ImportError:
    print("❌ Performance analyzer not available.")
    print("Install required packages: pip install pandas matplotlib seaborn plotly scikit-learn")
    sys.exit(1)

def load_performance_data(data_file):
    """Load performance data from file"""
    try:
        with open(data_file, 'r') as f:
            data = json.load(f)
        print(f"✅ Loaded {len(data)} performance records from {data_file}")
        return data
    except FileNotFoundError:
        print(f"❌ File not found: {data_file}")
        return None
    except json.JSONDecodeError:
        print(f"❌ Invalid JSON format in {data_file}")
        return None

def analyze_data(data_file=None, interactive=False):
    """Analyze performance data"""
    
    print("📊 Object Detection Performance Analysis")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = ModelPerformanceAnalyzer()
    
    if data_file:
        # Load data from file
        data = load_performance_data(data_file)
        if not data:
            return
        
        # Populate analyzer with loaded data
        for record in data:
            analyzer.log_performance(
                model_name=record.get('model', 'unknown'),
                inference_time=record.get('inference_time_ms', 0),
                detection_count=record.get('detection_count', 0),
                detections=record.get('detections', []),
                frame_number=record.get('frame_number', 0),
                timestamp=datetime.fromisoformat(record.get('timestamp', datetime.now().isoformat()))
            )
    else:
        print("📁 Looking for existing performance data...")
        # Try to find existing data files
        data_files = [f for f in os.listdir('.') if f.endswith('_performance.json')]
        
        if not data_files:
            print("❌ No performance data files found.")
            print("Run the main application first to generate data.")
            return
        
        latest_file = sorted(data_files)[-1]
        print(f"📄 Using latest data file: {latest_file}")
        
        data = load_performance_data(latest_file)
        if not data:
            return
        
        # Populate analyzer
        for record in data:
            analyzer.log_performance(
                model_name=record.get('model', 'unknown'),
                inference_time=record.get('inference_time_ms', 0),
                detection_count=record.get('detection_count', 0),
                detections=record.get('detections', []),
                frame_number=record.get('frame_number', 0),
                timestamp=datetime.fromisoformat(record.get('timestamp', datetime.now().isoformat()))
            )
    
    # Generate analysis
    print("\n📈 Generating comprehensive analysis...")
    
    # Get basic statistics
    df = analyzer.get_performance_dataframe()
    if df.empty:
        print("❌ No valid data for analysis")
        return
    
    print(f"\n📊 Dataset Overview:")
    print(f"Records: {len(df)}")
    print(f"Time span: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Models: {', '.join(df['model'].unique())}")
    print(f"Total detections: {df['detection_count'].sum()}")
    
    # Performance summary
    print(f"\n⚡ Performance Summary:")
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        avg_fps = 1000 / model_data['inference_time_ms'].mean() if model_data['inference_time_ms'].mean() > 0 else 0
        print(f"  {model.upper()}: {avg_fps:.1f} FPS avg, {model_data['detection_count'].mean():.1f} detections avg")
    
    # Generate comprehensive analysis
    try:
        analyzer.analyze_model_comparison(save_plots=True)
        print("✅ Model comparison analysis completed")
        
        analyzer.analyze_detection_patterns()
        print("✅ Detection pattern analysis completed")
        
        analyzer.create_interactive_dashboard()
        print("✅ Interactive dashboard created")
        
        report = analyzer.generate_performance_report()
        print("✅ Performance report generated")
        
        print("\n📁 Analysis files saved to:")
        print("  📊 ModelsAnalyze/ - Static visualizations")
        print("  🌐 ModelsAnalyze/ - Interactive dashboards and reports")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
    
    if interactive:
        print("\n🌐 Starting interactive analysis...")
        try:
            # Try to open interactive dashboard
            import webbrowser
            dashboard_file = "ModelsAnalyze/performance_dashboard.html"
            if os.path.exists(dashboard_file):
                webbrowser.open(f"file://{os.path.abspath(dashboard_file)}")
                print(f"✅ Interactive dashboard opened in browser")
            else:
                print("❌ Interactive dashboard file not found")
        except Exception as e:
            print(f"❌ Could not open interactive dashboard: {e}")

def main():
    """Main analysis function"""
    parser = argparse.ArgumentParser(
        description="Standalone Data Science Analysis Tool for Object Detection Performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_performance.py                           # Analyze latest data
  python analyze_performance.py --data-file data.json    # Analyze specific file
  python analyze_performance.py --interactive             # Open interactive dashboard
        """
    )
    
    parser.add_argument(
        '--data-file',
        type=str,
        help='Path to performance data JSON file'
    )
    
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Open interactive dashboard after analysis'
    )
    
    args = parser.parse_args()
    
    # Run analysis
    analyze_data(args.data_file, args.interactive)

if __name__ == "__main__":
    main()
