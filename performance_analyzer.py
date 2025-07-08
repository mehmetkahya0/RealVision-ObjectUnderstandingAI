"""
Performance Analysis Module for Object Detection Models
Provides comprehensive data science analysis of model performance metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import json
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ModelPerformanceAnalyzer:
    """Comprehensive performance analysis for object detection models"""
    
    def __init__(self):
        self.performance_data = []
        self.detection_history = []
        self.model_metrics = {
            'yolo': {'inference_times': [], 'detection_counts': [], 'confidences': []},
            'dnn': {'inference_times': [], 'detection_counts': [], 'confidences': []},
            'onnx': {'inference_times': [], 'detection_counts': [], 'confidences': []}
        }
        self.class_statistics = {}
        
    def log_performance(self, model_name, inference_time, detection_count, 
                       detections, frame_number, timestamp=None):
        """Log performance metrics for analysis"""
        if timestamp is None:
            timestamp = datetime.now()
            
        # Store in model-specific metrics
        if model_name in self.model_metrics:
            self.model_metrics[model_name]['inference_times'].append(inference_time)
            self.model_metrics[model_name]['detection_counts'].append(detection_count)
            
            # Extract confidence scores
            confidences = [det.get('confidence', 0) for det in detections]
            self.model_metrics[model_name]['confidences'].extend(confidences)
        
        # Store detailed performance data
        performance_entry = {
            'timestamp': timestamp,
            'model': model_name,
            'frame_number': frame_number,
            'inference_time_ms': inference_time,
            'detection_count': detection_count,
            'fps': 1000 / inference_time if inference_time > 0 else 0,
            'detections': detections
        }
        self.performance_data.append(performance_entry)
        
        # Update class statistics
        for detection in detections:
            class_name = detection.get('class_name', 'unknown')
            if class_name not in self.class_statistics:
                self.class_statistics[class_name] = {
                    'count': 0, 'confidences': [], 'models': set()
                }
            self.class_statistics[class_name]['count'] += 1
            self.class_statistics[class_name]['confidences'].append(
                detection.get('confidence', 0)
            )
            self.class_statistics[class_name]['models'].add(model_name)
    
    def get_performance_dataframe(self):
        """Convert performance data to pandas DataFrame for analysis"""
        if not self.performance_data:
            return pd.DataFrame()
            
        df = pd.DataFrame(self.performance_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    
    def analyze_model_comparison(self, save_plots=True):
        """Compare performance across different models"""
        if not self.performance_data:
            print("No performance data available for analysis")
            return
            
        df = self.get_performance_dataframe()
        
        # Create output directory
        import os
        output_dir = 'ModelsAnalyze'
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Model Performance Comparison Analysis', fontsize=16, fontweight='bold')
        
        # 1. Inference Time Comparison
        sns.boxplot(data=df, x='model', y='inference_time_ms', ax=axes[0,0])
        axes[0,0].set_title('Inference Time Distribution by Model')
        axes[0,0].set_ylabel('Inference Time (ms)')
        
        # 2. FPS Comparison
        sns.barplot(data=df, x='model', y='fps', ax=axes[0,1], estimator=np.mean)
        axes[0,1].set_title('Average FPS by Model')
        axes[0,1].set_ylabel('Frames Per Second')
        
        # 3. Detection Count Distribution
        sns.violinplot(data=df, x='model', y='detection_count', ax=axes[1,0])
        axes[1,0].set_title('Detection Count Distribution by Model')
        axes[1,0].set_ylabel('Number of Detections')
        
        # 4. Performance Over Time
        for model in df['model'].unique():
            model_data = df[df['model'] == model]
            axes[1,1].plot(range(len(model_data)), model_data['inference_time_ms'], 
                          label=f'{model}', alpha=0.7)
        axes[1,1].set_title('Inference Time Trends')
        axes[1,1].set_xlabel('Frame Number')
        axes[1,1].set_ylabel('Inference Time (ms)')
        axes[1,1].legend()
        
        plt.tight_layout()
        if save_plots:
            plt.savefig(f'{output_dir}/model_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Generate summary statistics
        self._generate_performance_summary(df)
    
    def analyze_detection_patterns(self, save_plots=True):
        """Analyze object detection patterns and class distributions"""
        if not self.class_statistics:
            print("No detection data available for analysis")
            return
            
        # Create output directory
        import os
        output_dir = 'ModelsAnalyze'
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare data for visualization
        class_data = []
        for class_name, stats in self.class_statistics.items():
            class_data.append({
                'class': class_name,
                'total_detections': stats['count'],
                'avg_confidence': np.mean(stats['confidences']),
                'min_confidence': np.min(stats['confidences']),
                'max_confidence': np.max(stats['confidences']),
                'std_confidence': np.std(stats['confidences']),
                'models_detected': len(stats['models'])
            })
        
        class_df = pd.DataFrame(class_data)
        class_df = class_df.sort_values('total_detections', ascending=False)
        
        # Create visualizations
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Object Detection Pattern Analysis', fontsize=16, fontweight='bold')
        
        # 1. Class frequency
        top_classes = class_df.head(10)
        sns.barplot(data=top_classes, x='total_detections', y='class', ax=axes[0,0])
        axes[0,0].set_title('Top 10 Detected Object Classes')
        axes[0,0].set_xlabel('Total Detections')
        
        # 2. Confidence distribution by class
        sns.scatterplot(data=top_classes, x='avg_confidence', y='total_detections', 
                       size='std_confidence', sizes=(50, 200), ax=axes[0,1])
        axes[0,1].set_title('Confidence vs Detection Frequency')
        axes[0,1].set_xlabel('Average Confidence')
        axes[0,1].set_ylabel('Total Detections')
        
        # 3. Detection confidence histogram
        all_confidences = []
        for stats in self.class_statistics.values():
            all_confidences.extend(stats['confidences'])
        axes[1,0].hist(all_confidences, bins=30, alpha=0.7, edgecolor='black')
        axes[1,0].set_title('Detection Confidence Distribution')
        axes[1,0].set_xlabel('Confidence Score')
        axes[1,0].set_ylabel('Frequency')
        
        # 4. Model coverage by class
        model_coverage = class_df['models_detected'].value_counts()
        axes[1,1].pie(model_coverage.values, labels=[f'{i} models' for i in model_coverage.index], 
                     autopct='%1.1f%%')
        axes[1,1].set_title('Model Coverage per Object Class')
        
        plt.tight_layout()
        if save_plots:
            plt.savefig(f'{output_dir}/detection_patterns_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return class_df
    
    def create_interactive_dashboard(self):
        """Create an interactive Plotly dashboard for performance analysis"""
        if not self.performance_data:
            print("No performance data available for dashboard")
            return
            
        # Create output directory
        import os
        output_dir = 'ModelsAnalyze'
        os.makedirs(output_dir, exist_ok=True)
        
        df = self.get_performance_dataframe()
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Model Performance Metrics', 'Detection Timeline', 
                          'FPS Distribution', 'Confidence Scores'),
            specs=[[{"secondary_y": True}, {"type": "scatter"}],
                   [{"type": "histogram"}, {"type": "box"}]]
        )
        
        # 1. Performance metrics over time
        for model in df['model'].unique():
            model_data = df[df['model'] == model]
            fig.add_trace(
                go.Scatter(x=model_data.index, y=model_data['inference_time_ms'],
                          name=f'{model} - Inference Time', line=dict(width=2)),
                row=1, col=1
            )
        
        # 2. Detection timeline
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['detection_count'],
                      mode='markers', name='Detections over Time',
                      marker=dict(size=6, color=df['fps'], colorscale='Viridis',
                                showscale=True)),
            row=1, col=2
        )
        
        # 3. FPS distribution
        for model in df['model'].unique():
            model_data = df[df['model'] == model]
            fig.add_trace(
                go.Histogram(x=model_data['fps'], name=f'{model} FPS',
                           opacity=0.7, nbinsx=20),
                row=2, col=1
            )
        
        # 4. Confidence box plots
        confidence_data = []
        for model_name, metrics in self.model_metrics.items():
            if metrics['confidences']:
                confidence_data.extend([(conf, model_name) for conf in metrics['confidences']])
        
        if confidence_data:
            conf_df = pd.DataFrame(confidence_data, columns=['confidence', 'model'])
            for model in conf_df['model'].unique():
                model_confs = conf_df[conf_df['model'] == model]['confidence']
                fig.add_trace(
                    go.Box(y=model_confs, name=f'{model} Confidence'),
                    row=2, col=2
                )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="Real-Time Object Detection Performance Dashboard",
            showlegend=True
        )
        
        # Save and show
        fig.write_html(f"{output_dir}/performance_dashboard.html")
        fig.show()
    
    def generate_performance_report(self, output_file="performance_report.json"):
        """Generate comprehensive performance report"""
        if not self.performance_data:
            return {"error": "No performance data available"}
            
        # Create output directory
        import os
        output_dir = 'ModelsAnalyze'
        os.makedirs(output_dir, exist_ok=True)
        
        df = self.get_performance_dataframe()
        
        # Calculate comprehensive metrics
        report = {
            "analysis_timestamp": datetime.now().isoformat(),
            "total_frames_analyzed": len(df),
            "analysis_duration_seconds": (df['timestamp'].max() - df['timestamp'].min()).total_seconds(),
            "models_tested": list(df['model'].unique()),
            "overall_metrics": {
                "avg_inference_time_ms": float(df['inference_time_ms'].mean()),
                "std_inference_time_ms": float(df['inference_time_ms'].std()),
                "avg_fps": float(df['fps'].mean()),
                "avg_detections_per_frame": float(df['detection_count'].mean()),
                "total_detections": int(df['detection_count'].sum())
            },
            "model_comparison": {},
            "object_class_analysis": {},
            "performance_insights": self._generate_insights(df)
        }
        
        # Model-specific metrics
        for model in df['model'].unique():
            model_data = df[df['model'] == model]
            report["model_comparison"][model] = {
                "frames_processed": len(model_data),
                "avg_inference_time_ms": float(model_data['inference_time_ms'].mean()),
                "avg_fps": float(model_data['fps'].mean()),
                "avg_detections": float(model_data['detection_count'].mean()),
                "inference_time_percentiles": {
                    "p50": float(model_data['inference_time_ms'].quantile(0.5)),
                    "p90": float(model_data['inference_time_ms'].quantile(0.9)),
                    "p95": float(model_data['inference_time_ms'].quantile(0.95))
                }
            }
        
        # Object class analysis
        for class_name, stats in self.class_statistics.items():
            report["object_class_analysis"][class_name] = {
                "total_detections": stats['count'],
                "avg_confidence": float(np.mean(stats['confidences'])),
                "confidence_std": float(np.std(stats['confidences'])),
                "detected_by_models": list(stats['models'])
            }
        
        # Save report
        with open(f"{output_dir}/{output_file}", 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return report
    
    def _generate_performance_summary(self, df):
        """Generate and print performance summary statistics"""
        print("\n" + "="*60)
        print("MODEL PERFORMANCE SUMMARY")
        print("="*60)
        
        for model in df['model'].unique():
            model_data = df[df['model'] == model]
            print(f"\n{model.upper()} Model:")
            print(f"  • Frames processed: {len(model_data)}")
            print(f"  • Avg inference time: {model_data['inference_time_ms'].mean():.2f} ms")
            print(f"  • Avg FPS: {model_data['fps'].mean():.2f}")
            print(f"  • Avg detections/frame: {model_data['detection_count'].mean():.2f}")
            print(f"  • Min/Max inference time: {model_data['inference_time_ms'].min():.2f}/{model_data['inference_time_ms'].max():.2f} ms")
    
    def _generate_insights(self, df):
        """Generate performance insights and recommendations"""
        insights = []
        
        # Speed comparison
        model_speeds = df.groupby('model')['fps'].mean()
        fastest_model = model_speeds.idxmax()
        slowest_model = model_speeds.idxmin()
        
        insights.append(f"Fastest model: {fastest_model} ({model_speeds[fastest_model]:.2f} FPS)")
        insights.append(f"Slowest model: {slowest_model} ({model_speeds[slowest_model]:.2f} FPS)")
        
        # Detection accuracy
        model_detections = df.groupby('model')['detection_count'].mean()
        most_detections = model_detections.idxmax()
        
        insights.append(f"Most detections per frame: {most_detections} ({model_detections[most_detections]:.2f} objects)")
        
        # Performance stability
        model_stability = df.groupby('model')['inference_time_ms'].std()
        most_stable = model_stability.idxmin()
        
        insights.append(f"Most stable performance: {most_stable} (std: {model_stability[most_stable]:.2f} ms)")
        
        return insights

# Usage example and integration
def analyze_session_performance(performance_analyzer):
    """Comprehensive analysis of a detection session"""
    print("Starting comprehensive performance analysis...")
    
    # Generate all analyses
    performance_analyzer.analyze_model_comparison()
    class_analysis = performance_analyzer.analyze_detection_patterns()
    performance_analyzer.create_interactive_dashboard()
    report = performance_analyzer.generate_performance_report()
    
    print("\nAnalysis complete! Check the ModelsAnalyze/ directory for:")
    print("• model_performance_comparison.png")
    print("• detection_patterns_analysis.png") 
    print("• performance_dashboard.html")
    print("• performance_report.json")
    
    return report

if __name__ == "__main__":
    # Example usage
    analyzer = ModelPerformanceAnalyzer()
    
    # Simulate some performance data
    models = ['yolo', 'dnn', 'onnx']
    for i in range(100):
        model = np.random.choice(models)
        inference_time = np.random.normal(40, 10) if model == 'yolo' else np.random.normal(25, 5)
        detection_count = np.random.poisson(8) if model == 'yolo' else np.random.poisson(5)
        
        # Simulate detections
        detections = []
        for j in range(detection_count):
            detections.append({
                'class_name': np.random.choice(['car', 'person', 'truck', 'bus']),
                'confidence': np.random.uniform(0.5, 0.95),
                'bbox': [100, 100, 200, 200]
            })
        
        analyzer.log_performance(model, inference_time, detection_count, detections, i)
    
    # Run analysis
    analyze_session_performance(analyzer)
