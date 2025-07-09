#!/usr/bin/env python3
"""
Data Science Analytics Demo
==========================

This script demonstrates the advanced data science capabilities
for analyzing model performance in the Object Understanding Application.

Usage:
    python demo_analytics.py [--video path/to/video.mp4] [--duration 30]

Features demonstrated:
- Real-time performance logging
- Model comparison analysis
- Statistical performance metrics
- Interactive dashboards
- Automated report generation
"""

import argparse
import time
import sys
import os
from datetime import datetime, timedelta
from main import ObjectUnderstandingApp

def run_analytics_demo(video_path=None, duration=30):
    """Run analytics demo with data science features"""
    
    print("🎯 Real-Time Object Understanding - Analytics Demo")
    print("=" * 60)
    print("This demo showcases advanced data science analysis capabilities")
    print("for monitoring and analyzing object detection model performance.")
    print()
    
    if video_path:
        print(f"📹 Processing video: {video_path}")
    else:
        print("📷 Using live camera feed")
    
    print(f"⏱️  Demo duration: {duration} seconds")
    print()
    print("🎮 Controls during demo:")
    print("  SPACE - Pause/Resume")
    print("  M - Switch models (YOLO ↔ MobileNet)")
    print("  A - Generate analytics report")
    print("  D - Toggle data logging")
    print("  +/- - Adjust confidence threshold")
    print("  Q - Quit demo")
    print()
    
    # Initialize application with analytics enabled
    app = ObjectUnderstandingApp()
    
    # Configure for demo
    if video_path and os.path.exists(video_path):
        print(f"✅ Loading video file: {video_path}")
        # The run method will handle video input
        import sys
        sys.argv = ['demo_analytics.py', '--input', video_path]
    else:
        if video_path:
            print(f"❌ Video file not found: {video_path}")
            print("🔄 Falling back to camera")
    
    # Set up models
    app.initialize_models()
    
    if not app.available_models:
        print("❌ No models available. Please install required packages.")
        return
    
    print(f"🤖 Available models: {', '.join(app.available_models)}")
    print(f"🎯 Starting with: {app.current_model}")
    print()
    
    # Track demo start time
    demo_start = datetime.now()
    
    print("🚀 Starting analytics demo...")
    print("📊 Performance data will be collected automatically")
    print("📈 You can generate reports anytime by pressing 'A'")
    print()
    
    try:
        # Start the application
        app.run()
        
    except KeyboardInterrupt:
        print("\n⚠️  Demo interrupted by user")
    
    # Generate final analytics report
    if app.enable_data_logging and app.data_analyzer:
        print("\n📊 Generating final analytics report...")
        app.generate_analytics_report()
        
        # Show summary statistics
        df = app.data_analyzer.get_performance_dataframe()
        if not df.empty:
            print("\n📈 Demo Summary:")
            print(f"Total duration: {datetime.now() - demo_start}")
            print(f"Frames processed: {len(df)}")
            print(f"Models tested: {df['model'].nunique()}")
            print(f"Average FPS: {(1000 / df['inference_time_ms']).mean():.1f}")
            print(f"Total detections: {df['detection_count'].sum()}")
            
            # Model performance comparison
            print("\n🏆 Model Performance Summary:")
            for model in df['model'].unique():
                model_data = df[df['model'] == model]
                avg_fps = 1000 / model_data['inference_time_ms'].mean()
                avg_detections = model_data['detection_count'].mean()
                print(f"  {model.upper()}: {avg_fps:.1f} FPS, {avg_detections:.1f} avg detections")
    
    print("\n✅ Analytics demo completed!")
    print("📁 Check the 'reports/' and 'plots/' folders for generated analysis files")

def main():
    """Main demo function"""
    parser = argparse.ArgumentParser(
        description="Data Science Analytics Demo for Object Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo_analytics.py                    # Use camera for 30 seconds
  python demo_analytics.py --duration 60      # Use camera for 60 seconds
  python demo_analytics.py --video traffic.mp4 # Process video file
  python demo_analytics.py --video traffic.mp4 --duration 120
        """
    )
    
    parser.add_argument(
        '--video', 
        type=str, 
        help='Path to video file for analysis (optional)'
    )
    
    parser.add_argument(
        '--duration', 
        type=int, 
        default=30,
        help='Demo duration in seconds (default: 30)'
    )
    
    parser.add_argument(
        '--model',
        type=str,
        choices=['yolo', 'dnn', 'auto'],
        default='auto',
        help='Preferred model to start with (default: auto)'
    )
    
    args = parser.parse_args()
    
    # Validate video file if provided
    if args.video and not os.path.exists(args.video):
        print(f"❌ Error: Video file '{args.video}' not found")
        sys.exit(1)
    
    # Run the demo
    run_analytics_demo(args.video, args.duration)

if __name__ == "__main__":
    main()
