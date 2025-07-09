#!/usr/bin/env python3
"""
Generate Sample Analytics Data for Demonstration
=================================================

This script creates sample performance data to demonstrate
the data science analysis capabilities.
"""

from performance_analyzer import ModelPerformanceAnalyzer
import json
import random
import time
from datetime import datetime, timedelta
import os

def generate_sample_data():
    """Generate realistic sample performance data"""
    
    print("📊 Generating Sample Analytics Data...")
    
    analyzer = ModelPerformanceAnalyzer()
    
    # Simulate different scenarios
    models = ['yolo', 'dnn', 'onnx']
    object_classes = ['person', 'car', 'truck', 'bicycle', 'motorcycle', 'bus', 'traffic light', 'stop sign']
    
    # Generate data for different time periods and conditions
    base_time = datetime.now() - timedelta(hours=1)
    
    for i in range(200):  # 200 frames of data
        # Vary performance based on model
        if i % 3 == 0:
            model = 'yolo'
            base_inference_time = random.uniform(20, 30)  # ms
            detection_prob = 0.8
        elif i % 3 == 1:
            model = 'dnn'
            base_inference_time = random.uniform(15, 25)  # ms
            detection_prob = 0.7
        else:
            model = 'onnx'
            base_inference_time = random.uniform(18, 28)  # ms
            detection_prob = 0.75
        
        # Add some variation based on frame complexity
        complexity_factor = random.uniform(0.8, 1.4)
        inference_time = base_inference_time * complexity_factor
        
        # Generate detections
        num_detections = 0
        detections = []
        
        if random.random() < detection_prob:
            num_detections = random.randint(1, 8)
            
            for _ in range(num_detections):
                detection = {
                    'class_name': random.choice(object_classes),
                    'confidence': random.uniform(0.5, 0.95),
                    'bbox': [
                        random.randint(50, 500),
                        random.randint(50, 400),
                        random.randint(100, 200),
                        random.randint(100, 200)
                    ]
                }
                detections.append(detection)
        
        # Log the performance data
        timestamp = base_time + timedelta(seconds=i * 0.033)  # ~30 FPS
        analyzer.log_performance(
            model_name=model,
            inference_time=inference_time,
            detection_count=num_detections,
            detections=detections,
            frame_number=i,
            timestamp=timestamp
        )
    
    print(f"✅ Generated {i+1} performance records")
    return analyzer

def demonstrate_analysis():
    """Demonstrate the data science analysis capabilities"""
    
    print("\n🔬 Demonstrating Data Science Analysis...")
    
    # Generate sample data
    analyzer = generate_sample_data()
    
    # Get performance dataframe
    df = analyzer.get_performance_dataframe()
    print(f"\n📈 Dataset Overview:")
    print(f"Records: {len(df)}")
    print(f"Models tested: {df['model'].unique().tolist()}")
    print(f"Time span: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"Total detections: {df['detection_count'].sum()}")
    
    # Performance summary by model
    print(f"\n⚡ Performance Summary by Model:")
    for model in df['model'].unique():
        model_data = df[df['model'] == model]
        avg_fps = 1000 / model_data['inference_time_ms'].mean() if model_data['inference_time_ms'].mean() > 0 else 0
        avg_detections = model_data['detection_count'].mean()
        print(f"  {model.upper()}: {avg_fps:.1f} FPS avg, {avg_detections:.1f} detections avg")
    
    # Generate comprehensive analysis
    print(f"\n📊 Generating Analysis Reports...")
    
    try:
        # Model comparison analysis
        analyzer.analyze_model_comparison(save_plots=True)
        print("✅ Model comparison analysis completed")
        
        # Detection pattern analysis
        class_analysis = analyzer.analyze_detection_patterns()
        print("✅ Detection pattern analysis completed")
        
        # Interactive dashboard
        analyzer.create_interactive_dashboard()
        print("✅ Interactive dashboard created")
        
        # Performance report
        report = analyzer.generate_performance_report()
        print("✅ Performance report generated")
        
        print(f"\n📁 Analysis files generated:")
        print(f"  📊 plots/ - Statistical visualizations")
        print(f"  🌐 reports/ - Interactive dashboard and detailed reports")
        
        # Export the data for later analysis
        os.makedirs("data", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_file = f"data/sample_performance_data_{timestamp}.json"
        
        # Convert dataframe to JSON format
        data_export = []
        for _, row in df.iterrows():
            data_export.append({
                'timestamp': row['timestamp'].isoformat(),
                'model': row['model'],
                'frame_number': row['frame_number'],
                'inference_time_ms': row['inference_time_ms'],
                'detection_count': row['detection_count'],
                'fps': row['fps'],
                'detections': row['detections']
            })
        
        with open(export_file, 'w') as f:
            json.dump(data_export, f, indent=2)
        
        print(f"  📄 {export_file} - Raw data export")
        
        # Show some interesting insights
        print(f"\n🔍 Key Insights:")
        
        # Best performing model
        model_performance = df.groupby('model')['fps'].mean()
        best_model = model_performance.idxmax()
        print(f"  🏆 Best performing model: {best_model.upper()} ({model_performance[best_model]:.1f} FPS)")
        
        # Detection efficiency
        detection_efficiency = df.groupby('model')['detection_count'].mean()
        most_detections = detection_efficiency.idxmax()
        print(f"  🎯 Most detections: {most_detections.upper()} ({detection_efficiency[most_detections]:.1f} avg)")
        
        # Peak performance periods
        df['hour'] = df['timestamp'].dt.hour
        if len(df['hour'].unique()) > 1:
            hourly_perf = df.groupby('hour')['fps'].mean()
            peak_hour = hourly_perf.idxmax()
            print(f"  ⏰ Peak performance hour: {peak_hour}:00 ({hourly_perf[peak_hour]:.1f} FPS)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        return False

def main():
    """Main demonstration function"""
    print("🎯 Data Science Analytics Demonstration")
    print("=" * 60)
    print("This demonstration shows the comprehensive data science")
    print("capabilities for analyzing object detection performance.")
    print()
    
    success = demonstrate_analysis()
    
    if success:
        print(f"\n🎉 Demonstration completed successfully!")
        print(f"\nNext steps:")
        print(f"1. Check the generated files in plots/ and reports/ folders")
        print(f"2. Run: python analyze_performance.py --interactive")
        print(f"3. Try the real application: python run.py")
        print(f"4. During real-time detection, press 'A' to generate live analytics")
    else:
        print(f"\n❌ Demonstration failed. Check error messages above.")

if __name__ == "__main__":
    main()
