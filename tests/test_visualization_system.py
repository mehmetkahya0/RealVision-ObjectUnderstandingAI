#!/usr/bin/env python3
"""
Test script for the Performance Data Visualization System
========================================================

Tests the core functionality of the visualization system.
"""

import sys
import os
import json
import tempfile
from datetime import datetime, timedelta
import random

def create_sample_performance_data():
    """Create sample performance data for testing"""
    print("📊 Creating sample performance data...")
    
    # Generate sample data
    start_time = datetime.now() - timedelta(minutes=5)
    sample_data = []
    
    models = ['yolo', 'dnn', 'efficientdet']
    classes = ['car', 'person', 'truck', 'bicycle', 'motorcycle']
    
    for i in range(100):
        timestamp = start_time + timedelta(seconds=i*3)
        model = models[i % len(models)]
        
        # Simulate different performance characteristics
        if model == 'yolo':
            inference_time = random.uniform(80, 120)
            fps = random.uniform(8, 12)
        else:  # dnn
            inference_time = random.uniform(30, 60)
            fps = random.uniform(15, 25)
        
        detection_count = random.randint(1, 8)
        detections = []
        
        for j in range(detection_count):
            detections.append({
                'class_name': random.choice(classes),
                'confidence': random.uniform(0.5, 0.95),
                'bbox': [
                    random.randint(0, 640),
                    random.randint(0, 480),
                    random.randint(0, 640),
                    random.randint(0, 480)
                ]
            })
        
        sample_data.append({
            'timestamp': timestamp.isoformat(),
            'model': model,
            'frame_number': i + 1,
            'inference_time_ms': inference_time,
            'detection_count': detection_count,
            'fps': fps,
            'detections': detections
        })
    
    # Save sample data
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sample_file = os.path.join(data_dir, f"sample_performance_data_{timestamp}.json")
    
    with open(sample_file, 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    print(f"✅ Sample data created: {sample_file}")
    return sample_file

def test_visualizer_import():
    """Test importing the visualizer modules"""
    print("🔍 Testing visualizer imports...")
    
    try:
        import visualize_performance
        print("✅ visualize_performance imported successfully")
        
        import visualize_performance_gui
        print("✅ visualize_performance_gui imported successfully")
        
        import launch_visualizer
        print("✅ launch_visualizer imported successfully")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_data_loading():
    """Test data loading functionality"""
    print("📁 Testing data loading...")
    
    try:
        # Create sample data
        sample_file = create_sample_performance_data()
        
        # Test loading
        import visualize_performance
        visualizer = visualize_performance.PerformanceDataVisualizer()
        
        if visualizer.load_data_file(sample_file):
            print("✅ Data loading successful")
            
            # Test summary
            summary = visualizer.get_data_summary()
            print(f"📊 Loaded {summary['total_frames']} frames")
            print(f"🤖 Models: {summary['models_used']}")
            
            return True
        else:
            print("❌ Data loading failed")
            return False
            
    except Exception as e:
        print(f"❌ Error testing data loading: {e}")
        return False

def test_visualization_creation():
    """Test visualization creation (without display)"""
    print("🎨 Testing visualization creation...")
    
    try:
        # Create sample data
        sample_file = create_sample_performance_data()
        
        import visualize_performance
        visualizer = visualize_performance.PerformanceDataVisualizer()
        
        if visualizer.load_data_file(sample_file):
            # Test dashboard creation (but don't open browser)
            fig = visualizer.create_performance_dashboard(save_html=False)
            if fig is not None:
                print("✅ Dashboard creation successful")
                
            # Test report generation
            report = visualizer.generate_summary_report()
            if report:
                print("✅ Report generation successful")
                
            return True
        else:
            print("❌ Could not load data for visualization test")
            return False
            
    except Exception as e:
        print(f"❌ Error testing visualization creation: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Testing Performance Data Visualization System")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_visualizer_import),
        ("Data Loading Test", test_data_loading),
        ("Visualization Creation Test", test_visualization_creation)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔬 Running {test_name}...")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The visualization system is working correctly.")
        print("\n📋 Next steps:")
        print("1. Run your object detection application to generate real data")
        print("2. Use 'python launch_visualizer.py' to access visualization tools")
        print("3. Try 'visualize_data.bat' for quick Windows access")
    else:
        print("⚠️  Some tests failed. Please check the error messages above.")
        print("Make sure all required packages are installed in your virtual environment.")

if __name__ == "__main__":
    main()
