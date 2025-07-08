#!/usr/bin/env python3
"""
Quick Test Script for Data Science Features
===========================================

This script tests the data science integration without running the full application.
"""

import sys
import os

def test_imports():
    """Test if all required packages are available"""
    print("🧪 Testing Data Science Package Imports...")
    
    required_packages = {
        'pandas': 'Data manipulation and analysis',
        'numpy': 'Numerical computing',
        'matplotlib': 'Static visualizations',
        'seaborn': 'Statistical visualizations',
        'plotly': 'Interactive visualizations',
        'sklearn': 'Machine learning tools',
        'scipy': 'Scientific computing',
        'performance_analyzer': 'Custom analytics module'
    }
    
    results = {}
    
    for package, description in required_packages.items():
        try:
            if package == 'sklearn':
                import sklearn
            elif package == 'performance_analyzer':
                from performance_analyzer import ModelPerformanceAnalyzer
            else:
                __import__(package)
            results[package] = True
            print(f"✅ {package}: {description}")
        except ImportError:
            results[package] = False
            print(f"❌ {package}: {description} - NOT AVAILABLE")
    
    return results

def test_analyzer():
    """Test the performance analyzer functionality"""
    print("\n🔬 Testing Performance Analyzer...")
    
    try:
        from performance_analyzer import ModelPerformanceAnalyzer
        
        # Initialize analyzer
        analyzer = ModelPerformanceAnalyzer()
        print("✅ Analyzer initialized successfully")
        
        # Test logging some sample data
        import time
        from datetime import datetime
        
        sample_detections = [
            {'class_name': 'person', 'confidence': 0.85, 'bbox': [100, 100, 200, 300]},
            {'class_name': 'car', 'confidence': 0.92, 'bbox': [300, 150, 500, 400]}
        ]
        
        analyzer.log_performance(
            model_name='yolo',
            inference_time=25.5,  # ms
            detection_count=2,
            detections=sample_detections,
            frame_number=1
        )
        
        analyzer.log_performance(
            model_name='dnn',
            inference_time=18.3,  # ms
            detection_count=1,
            detections=[sample_detections[0]],
            frame_number=2
        )
        
        print("✅ Sample data logged successfully")
        
        # Test dataframe generation
        df = analyzer.get_performance_dataframe()
        if not df.empty:
            print(f"✅ Generated dataframe with {len(df)} records")
            print(f"   Models: {df['model'].unique().tolist()}")
            print(f"   Avg FPS: {df['fps'].mean():.1f}")
        else:
            print("❌ Dataframe is empty")
        
        return True
        
    except Exception as e:
        print(f"❌ Analyzer test failed: {e}")
        return False

def test_main_integration():
    """Test integration with main application"""
    print("\n🔗 Testing Main Application Integration...")
    
    try:
        from main import ObjectUnderstandingApp
        
        # Check if data science features are available
        app = ObjectUnderstandingApp()
        
        if hasattr(app, 'data_analyzer') and app.data_analyzer:
            print("✅ Data analyzer integrated in main application")
        else:
            print("❌ Data analyzer not available in main application")
        
        if hasattr(app, 'enable_data_logging'):
            print(f"✅ Data logging enabled: {app.enable_data_logging}")
        else:
            print("❌ Data logging not configured")
        
        return True
        
    except Exception as e:
        print(f"❌ Main integration test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🎯 Data Science Features Test Suite")
    print("=" * 50)
    
    # Test imports
    import_results = test_imports()
    
    # Test analyzer if available
    analyzer_ok = False
    if import_results.get('performance_analyzer', False):
        analyzer_ok = test_analyzer()
    
    # Test main integration
    integration_ok = test_main_integration()
    
    # Summary
    print("\n📋 Test Summary:")
    print("=" * 30)
    
    total_packages = len(import_results)
    available_packages = sum(import_results.values())
    
    print(f"📦 Package Availability: {available_packages}/{total_packages}")
    print(f"🔬 Analyzer Test: {'✅ PASS' if analyzer_ok else '❌ FAIL'}")
    print(f"🔗 Integration Test: {'✅ PASS' if integration_ok else '❌ FAIL'}")
    
    if available_packages == total_packages and analyzer_ok and integration_ok:
        print("\n🎉 All tests passed! Data science features are ready to use.")
        print("\nNext steps:")
        print("1. Run: python run.py")
        print("2. Press 'A' during execution to generate analytics")
        print("3. Try: python demo_analytics.py")
    else:
        print("\n⚠️  Some tests failed. Check package installation.")
        print("Install missing packages: pip install -r requirements.txt")

if __name__ == "__main__":
    main()
