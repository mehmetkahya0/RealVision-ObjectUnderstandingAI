#!/usr/bin/env python3
"""
Performance Data Visualizer for RealVision-ObjectUnderstandingAI
================================================================

Interactive system for visualizing performance data from the object detection application.
Supports multiple data files, various chart types, and export capabilities.

Author: RealVision-ObjectUnderstandingAI Team
Date: July 2025
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import os
import glob
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import webbrowser
import tempfile

class PerformanceDataVisualizer:
    """Main class for performance data visualization"""
    
    def __init__(self):
        self.data = None
        self.df = None
        self.data_file = None
        self.setup_style()
        
    def setup_style(self):
        """Setup plotting styles"""
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
    def load_data_file(self, file_path=None):
        """Load performance data from JSON file"""
        if file_path is None:
            # Use file dialog to select data file
            root = tk.Tk()
            root.withdraw()  # Hide the main window
            
            # Look for data files in the data directory
            data_dir = os.path.join(os.path.dirname(__file__), 'data')
            initial_dir = data_dir if os.path.exists(data_dir) else os.getcwd()
            
            file_path = filedialog.askopenfilename(
                title="Select Performance Data File",
                initialdir=initial_dir,
                filetypes=[
                    ("JSON files", "*.json"),
                    ("All files", "*.*")
                ]
            )
            root.destroy()
            
            if not file_path:
                print("No file selected.")
                return False
                
        try:
            print(f"Loading data from: {file_path}")
            with open(file_path, 'r') as f:
                self.data = json.load(f)
            
            self.data_file = file_path
            self.df = pd.DataFrame(self.data)
            
            # Convert timestamp to datetime
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
            self.df['time_elapsed'] = (self.df['timestamp'] - self.df['timestamp'].iloc[0]).dt.total_seconds()
            
            print(f"✅ Loaded {len(self.df)} data points")
            print(f"📊 Data range: {self.df['timestamp'].min()} to {self.df['timestamp'].max()}")
            print(f"🤖 Models: {self.df['model'].unique()}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def get_data_summary(self):
        """Get summary statistics of the loaded data"""
        if self.df is None:
            return "No data loaded"
            
        summary = {
            'total_frames': len(self.df),
            'duration_seconds': self.df['time_elapsed'].max(),
            'models_used': list(self.df['model'].unique()),
            'avg_fps': self.df['fps'].mean(),
            'avg_inference_time': self.df['inference_time_ms'].mean(),
            'total_detections': self.df['detection_count'].sum(),
            'date_range': f"{self.df['timestamp'].min().strftime('%Y-%m-%d %H:%M:%S')} to {self.df['timestamp'].max().strftime('%Y-%m-%d %H:%M:%S')}"
        }
        
        return summary
    
    def create_performance_dashboard(self, save_html=True):
        """Create comprehensive performance dashboard"""
        if self.df is None:
            print("❌ No data loaded. Please load a data file first.")
            return None
            
        print("📊 Creating performance dashboard...")
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'FPS Over Time',
                'Inference Time Distribution',
                'Detection Count Over Time', 
                'Model Performance Comparison',
                'Detection Types Distribution',
                'Performance Metrics Correlation'
            ],
            specs=[
                [{"secondary_y": False}, {"type": "histogram"}],
                [{"secondary_y": True}, {"type": "box"}],
                [{"type": "pie"}, {"type": "scatter"}]
            ]
        )
        
        # 1. FPS Over Time
        for model in self.df['model'].unique():
            model_data = self.df[self.df['model'] == model]
            fig.add_trace(
                go.Scatter(
                    x=model_data['time_elapsed'],
                    y=model_data['fps'],
                    name=f'{model.upper()} FPS',
                    mode='lines+markers',
                    line=dict(width=2)
                ),
                row=1, col=1
            )
        
        # 2. Inference Time Distribution
        fig.add_trace(
            go.Histogram(
                x=self.df['inference_time_ms'],
                name='Inference Time',
                nbinsx=30,
                opacity=0.7
            ),
            row=1, col=2
        )
        
        # 3. Detection Count Over Time with Moving Average
        fig.add_trace(
            go.Scatter(
                x=self.df['time_elapsed'],
                y=self.df['detection_count'],
                name='Detection Count',
                mode='markers',
                opacity=0.6
            ),
            row=2, col=1
        )
        
        # Add moving average
        window_size = max(10, len(self.df) // 50)
        moving_avg = self.df['detection_count'].rolling(window=window_size).mean()
        fig.add_trace(
            go.Scatter(
                x=self.df['time_elapsed'],
                y=moving_avg,
                name=f'Moving Avg ({window_size})',
                mode='lines',
                line=dict(width=3)
            ),
            row=2, col=1
        )
        
        # 4. Model Performance Comparison (Box Plot)
        for model in self.df['model'].unique():
            model_data = self.df[self.df['model'] == model]
            fig.add_trace(
                go.Box(
                    y=model_data['inference_time_ms'],
                    name=f'{model.upper()}',
                    boxpoints='outliers'
                ),
                row=2, col=2
            )
        
        # 5. Detection Types Distribution (if available)
        if 'detections' in self.df.columns and len(self.df) > 0:
            # Count detection types across all frames
            detection_counts = {}
            for _, row in self.df.iterrows():
                if isinstance(row['detections'], list):
                    for detection in row['detections']:
                        if isinstance(detection, dict) and 'class_name' in detection:
                            class_name = detection['class_name']
                            detection_counts[class_name] = detection_counts.get(class_name, 0) + 1
            
            if detection_counts:
                labels = list(detection_counts.keys())
                values = list(detection_counts.values())
                
                fig.add_trace(
                    go.Pie(
                        labels=labels,
                        values=values,
                        name="Detection Types"
                    ),
                    row=3, col=1
                )
        
        # 6. Performance Metrics Correlation
        fig.add_trace(
            go.Scatter(
                x=self.df['inference_time_ms'],
                y=self.df['fps'],
                mode='markers',
                name='Inference Time vs FPS',
                text=self.df['model'],
                hovertemplate='<b>%{text}</b><br>Inference: %{x:.1f}ms<br>FPS: %{y:.1f}<extra></extra>'
            ),
            row=3, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=1200,
            title_text=f"Performance Analysis Dashboard - {os.path.basename(self.data_file)}",
            title_x=0.5,
            showlegend=True,
            template="plotly_white"
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Time (seconds)", row=1, col=1)
        fig.update_yaxes(title_text="FPS", row=1, col=1)
        
        fig.update_xaxes(title_text="Inference Time (ms)", row=1, col=2)
        fig.update_yaxes(title_text="Frequency", row=1, col=2)
        
        fig.update_xaxes(title_text="Time (seconds)", row=2, col=1)
        fig.update_yaxes(title_text="Detection Count", row=2, col=1)
        
        fig.update_yaxes(title_text="Inference Time (ms)", row=2, col=2)
        
        fig.update_xaxes(title_text="Inference Time (ms)", row=3, col=2)
        fig.update_yaxes(title_text="FPS", row=3, col=2)
        
        if save_html:
            # Save dashboard
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"performance_dashboard_{timestamp}.html"
            fig.write_html(output_file)
            print(f"📊 Dashboard saved as: {output_file}")
            
            # Open in browser
            webbrowser.open(f"file://{os.path.abspath(output_file)}")
        
        return fig
    
    def create_model_comparison_chart(self):
        """Create detailed model comparison charts"""
        if self.df is None:
            print("❌ No data loaded.")
            return None
            
        models = self.df['model'].unique()
        if len(models) < 2:
            print("❌ Need at least 2 different models for comparison.")
            return None
            
        print("📈 Creating model comparison charts...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # 1. FPS Comparison
        sns.boxplot(data=self.df, x='model', y='fps', ax=axes[0,0])
        axes[0,0].set_title('FPS Distribution by Model')
        axes[0,0].set_ylabel('Frames Per Second')
        
        # 2. Inference Time Comparison
        sns.boxplot(data=self.df, x='model', y='inference_time_ms', ax=axes[0,1])
        axes[0,1].set_title('Inference Time Distribution by Model')
        axes[0,1].set_ylabel('Inference Time (ms)')
        
        # 3. Detection Count Comparison
        sns.boxplot(data=self.df, x='model', y='detection_count', ax=axes[1,0])
        axes[1,0].set_title('Detection Count Distribution by Model')
        axes[1,0].set_ylabel('Number of Detections')
        
        # 4. Performance vs Time
        for model in models:
            model_data = self.df[self.df['model'] == model]
            axes[1,1].plot(model_data['time_elapsed'], 
                          model_data['fps'], 
                          label=f'{model.upper()} FPS', 
                          alpha=0.7)
        
        axes[1,1].set_title('FPS Over Time by Model')
        axes[1,1].set_xlabel('Time (seconds)')
        axes[1,1].set_ylabel('FPS')
        axes[1,1].legend()
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"model_comparison_{timestamp}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"📊 Model comparison saved as: {output_file}")
        
        plt.show()
        return fig
    
    def create_time_series_analysis(self):
        """Create detailed time series analysis"""
        if self.df is None:
            print("❌ No data loaded.")
            return None
            
        print("📈 Creating time series analysis...")
        
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=[
                'FPS Over Time',
                'Inference Time Over Time',
                'Detection Count Over Time',
                'System Performance Trend'
            ],
            shared_xaxes=True
        )
        
        # 1. FPS Over Time
        for model in self.df['model'].unique():
            model_data = self.df[self.df['model'] == model]
            fig.add_trace(
                go.Scatter(
                    x=model_data['timestamp'],
                    y=model_data['fps'],
                    name=f'{model.upper()} FPS',
                    mode='lines+markers'
                ),
                row=1, col=1
            )
        
        # 2. Inference Time Over Time
        for model in self.df['model'].unique():
            model_data = self.df[self.df['model'] == model]
            fig.add_trace(
                go.Scatter(
                    x=model_data['timestamp'],
                    y=model_data['inference_time_ms'],
                    name=f'{model.upper()} Inference',
                    mode='lines+markers'
                ),
                row=2, col=1
            )
        
        # 3. Detection Count Over Time
        fig.add_trace(
            go.Scatter(
                x=self.df['timestamp'],
                y=self.df['detection_count'],
                name='Detection Count',
                mode='lines+markers',
                line=dict(color='green')
            ),
            row=3, col=1
        )
        
        # 4. System Performance Trend (normalized metrics)
        df_normalized = self.df.copy()
        df_normalized['fps_norm'] = (df_normalized['fps'] - df_normalized['fps'].min()) / (df_normalized['fps'].max() - df_normalized['fps'].min())
        df_normalized['inference_norm'] = 1 - (df_normalized['inference_time_ms'] - df_normalized['inference_time_ms'].min()) / (df_normalized['inference_time_ms'].max() - df_normalized['inference_time_ms'].min())
        df_normalized['detection_norm'] = (df_normalized['detection_count'] - df_normalized['detection_count'].min()) / (df_normalized['detection_count'].max() - df_normalized['detection_count'].min())
        
        fig.add_trace(
            go.Scatter(
                x=df_normalized['timestamp'],
                y=df_normalized['fps_norm'],
                name='FPS (normalized)',
                mode='lines'
            ),
            row=4, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df_normalized['timestamp'],
                y=df_normalized['inference_norm'],
                name='Inference Speed (normalized)',
                mode='lines'
            ),
            row=4, col=1
        )
        
        fig.update_layout(
            height=1000,
            title_text="Time Series Performance Analysis",
            title_x=0.5
        )
        
        # Save and show
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"time_series_analysis_{timestamp}.html"
        fig.write_html(output_file)
        print(f"📊 Time series analysis saved as: {output_file}")
        
        webbrowser.open(f"file://{os.path.abspath(output_file)}")
        return fig
    
    def generate_summary_report(self):
        """Generate comprehensive summary report"""
        if self.df is None:
            print("❌ No data loaded.")
            return None
            
        summary = self.get_data_summary()
        
        report = f"""
# Performance Analysis Report
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Data File:** {os.path.basename(self.data_file)}
**Analysis Period:** {summary['date_range']}

## Summary Statistics
- **Total Frames Processed:** {summary['total_frames']:,}
- **Session Duration:** {summary['duration_seconds']:.1f} seconds
- **Models Used:** {', '.join(summary['models_used'])}
- **Average FPS:** {summary['avg_fps']:.2f}
- **Average Inference Time:** {summary['avg_inference_time']:.2f} ms
- **Total Detections:** {summary['total_detections']:,}

## Model Performance Breakdown
"""
        
        for model in self.df['model'].unique():
            model_data = self.df[self.df['model'] == model]
            report += f"""
### {model.upper()} Model
- **Frames:** {len(model_data):,}
- **Average FPS:** {model_data['fps'].mean():.2f}
- **Average Inference Time:** {model_data['inference_time_ms'].mean():.2f} ms
- **Detection Rate:** {model_data['detection_count'].mean():.1f} objects/frame
- **Best FPS:** {model_data['fps'].max():.2f}
- **Worst FPS:** {model_data['fps'].min():.2f}
"""
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"performance_report_{timestamp}.md"
        
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"📋 Summary report saved as: {report_file}")
        print(report)
        
        return report
    
    def list_available_data_files(self):
        """List all available performance data files"""
        data_dir = os.path.join(os.path.dirname(__file__), 'data')
        
        if not os.path.exists(data_dir):
            print("❌ Data directory not found.")
            return []
        
        json_files = glob.glob(os.path.join(data_dir, "*.json"))
        
        if not json_files:
            print("❌ No performance data files found in data directory.")
            return []
        
        print("📁 Available performance data files:")
        for i, file_path in enumerate(json_files, 1):
            file_name = os.path.basename(file_path)
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            print(f"  {i}. {file_name} ({file_size:.1f} MB) - {mod_time.strftime('%Y-%m-%d %H:%M')}")
        
        return json_files

def interactive_visualizer():
    """Interactive command-line interface for the visualizer"""
    visualizer = PerformanceDataVisualizer()
    
    print("🤖 RealVision Performance Data Visualizer")
    print("=" * 50)
    
    while True:
        print("\n📊 Available Actions:")
        print("1. List available data files")
        print("2. Load data file")
        print("3. Show data summary")
        print("4. Create performance dashboard")
        print("5. Create model comparison charts")
        print("6. Create time series analysis")
        print("7. Generate summary report")
        print("8. Exit")
        
        try:
            choice = input("\n🔹 Choose an action (1-8): ").strip()
            
            if choice == '1':
                visualizer.list_available_data_files()
                
            elif choice == '2':
                if visualizer.load_data_file():
                    print("✅ Data loaded successfully!")
                else:
                    print("❌ Failed to load data.")
                    
            elif choice == '3':
                if visualizer.df is not None:
                    summary = visualizer.get_data_summary()
                    print("\n📊 Data Summary:")
                    for key, value in summary.items():
                        print(f"  {key}: {value}")
                else:
                    print("❌ No data loaded. Please load a data file first.")
                    
            elif choice == '4':
                visualizer.create_performance_dashboard()
                
            elif choice == '5':
                visualizer.create_model_comparison_chart()
                
            elif choice == '6':
                visualizer.create_time_series_analysis()
                
            elif choice == '7':
                visualizer.generate_summary_report()
                
            elif choice == '8':
                print("👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid choice. Please select 1-8.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    interactive_visualizer()
