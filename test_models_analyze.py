#!/usr/bin/env python3
"""
Test the ModelsAnalyze folder functionality
"""

from performance_analyzer import ModelPerformanceAnalyzer
import random
from datetime import datetime, timedelta

def test_models_analyze_folder():
    """Test that analysis files are saved to ModelsAnalyze folder"""
    
    print("🧪 Testing ModelsAnalyze folder functionality...")
    
    # Create analyzer and add some test data
    analyzer = ModelPerformanceAnalyzer()
    
    # Add sample data
    for i in range(20):
        model = ['yolo', 'dnn', 'onnx'][i % 3]
        
        detections = [
            {'class_name': 'car', 'confidence': 0.85, 'bbox': [100, 100, 200, 200]},
            {'class_name': 'person', 'confidence': 0.92, 'bbox': [300, 150, 400, 300]}
        ]
        
        analyzer.log_performance(
            model_name=model,
            inference_time=random.uniform(15, 35),
            detection_count=len(detections),
            detections=detections,
            frame_number=i
        )
    
    print("✅ Sample data added")
    
    # Test each analysis function
    try:
        print("📊 Testing model comparison analysis...")
        analyzer.analyze_model_comparison(save_plots=True)
        print("✅ Model comparison saved to ModelsAnalyze/")
        
        print("📈 Testing detection patterns analysis...")
        analyzer.analyze_detection_patterns(save_plots=True)
        print("✅ Detection patterns saved to ModelsAnalyze/")
        
        print("🌐 Testing interactive dashboard...")
        analyzer.create_interactive_dashboard()
        print("✅ Interactive dashboard saved to ModelsAnalyze/")
        
        print("📄 Testing performance report...")
        analyzer.generate_performance_report()
        print("✅ Performance report saved to ModelsAnalyze/")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    success = test_models_analyze_folder()
    
    if success:
        print("\n🎉 All tests passed! ModelsAnalyze folder is working correctly.")
        print("📁 Check the ModelsAnalyze/ folder for generated files.")
    else:
        print("\n❌ Tests failed. Check the error messages above.")
