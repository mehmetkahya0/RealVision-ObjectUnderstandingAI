#!/usr/bin/env python3
"""
RealVision Object Understanding AI - Main Launcher
==================================================

Professional launcher for the RealVision Object Understanding AI application.
This script provides easy access to all main features from the project root.

Usage:
    python app.py [options]           # Main object detection application
    python app.py --demo             # Run analytics demo
    python app.py --visualize        # Launch visualization tools
    python app.py --test             # Run test suite
    python app.py --help             # Show help

Author: Mehmet Kahya
Date: July 2025
"""

import sys
import os
import argparse
from pathlib import Path

# Add src directory to Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

def main():
    """Main launcher function"""
    
    # Check if arguments look like they should go to the run command
    # This handles cases like: python app.py --input video.mp4
    run_like_args = ['--input', '--camera', '--model', '--confidence', '--output-dir', '--no-gui', '--list-cameras']
    if len(sys.argv) > 1 and any(arg in sys.argv for arg in run_like_args):
        # If we have run-like arguments but no subcommand, inject 'run'
        if sys.argv[1] not in ['run', 'demo', 'visualize', 'test']:
            sys.argv.insert(1, 'run')
    
    parser = argparse.ArgumentParser(
        description="RealVision Object Understanding AI - Professional Computer Vision Application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python app.py                          # Launch main application with webcam
  python app.py --input video.mp4       # Process video file (auto-detects run command)
  python app.py run --input video.mp4   # Process video file (explicit run command)
  python app.py --camera 1              # Use camera index 1
  python app.py --model yolo            # Force YOLO model
  python app.py --confidence 0.7        # Set confidence threshold
  python app.py demo                    # Run analytics demo
  python app.py visualize               # Launch visualization tools
  python app.py test                    # Run test suite
  python app.py run --list-cameras      # List available cameras

For more options, see: python app.py run --help
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Main application command (default)
    run_parser = subparsers.add_parser('run', help='Run the main object detection application')
    run_parser.add_argument('--input', type=str, help='Input video file path')
    run_parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0)')
    run_parser.add_argument('--model', choices=['yolo', 'dnn', 'onnx', 'efficientdet'], help='Detection model to use')
    run_parser.add_argument('--confidence', type=float, default=0.5, help='Confidence threshold (0.0-1.0)')
    run_parser.add_argument('--output-dir', type=str, help='Output directory for screenshots')
    run_parser.add_argument('--no-gui', action='store_true', help='Run without GUI display')
    run_parser.add_argument('--list-cameras', action='store_true', help='List available cameras and exit')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Run analytics demo')
    demo_parser.add_argument('--video', type=str, help='Video file for demo')
    demo_parser.add_argument('--sample', action='store_true', help='Generate sample data')
    
    # Visualization command
    viz_parser = subparsers.add_parser('visualize', help='Launch visualization tools')
    viz_parser.add_argument('--gui', action='store_true', help='Launch GUI visualizer')
    viz_parser.add_argument('--cli', action='store_true', help='Launch CLI visualizer')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Run test suite')
    test_parser.add_argument('--imports', action='store_true', help='Test imports only')
    test_parser.add_argument('--data-science', action='store_true', help='Test data science features')
    test_parser.add_argument('--visualization', action='store_true', help='Test visualization system')
    
    # Parse arguments
    args = parser.parse_args()
    
    # If no command specified, default to run
    if args.command is None:
        args.command = 'run'
    
    # Execute appropriate command
    if args.command == 'run':
        from src.run import main as run_main
        # Convert args back to sys.argv format for run.py
        sys.argv = ['run.py']
        if hasattr(args, 'input') and args.input:
            sys.argv.extend(['--input', args.input])
        if hasattr(args, 'camera') and args.camera != 0:
            sys.argv.extend(['--camera', str(args.camera)])
        if hasattr(args, 'model') and args.model:
            sys.argv.extend(['--model', args.model])
        if hasattr(args, 'confidence') and args.confidence != 0.5:
            sys.argv.extend(['--confidence', str(args.confidence)])
        if hasattr(args, 'output_dir') and args.output_dir:
            sys.argv.extend(['--output-dir', args.output_dir])
        if hasattr(args, 'no_gui') and args.no_gui:
            sys.argv.append('--no-gui')
        if hasattr(args, 'list_cameras') and args.list_cameras:
            sys.argv.append('--list-cameras')
        
        run_main()
        
    elif args.command == 'demo':
        if args.sample:
            from src.demo_sample_analytics import main as demo_sample_main
            demo_sample_main()
        else:
            from src.demo_analytics import main as demo_main
            if args.video:
                sys.argv = ['demo_analytics.py', '--video', args.video]
            demo_main()
            
    elif args.command == 'visualize':
        # Add visualization directory to path
        viz_path = project_root / "visualization"
        sys.path.insert(0, str(viz_path))
        
        if args.gui:
            from visualize_performance_gui import main as viz_gui_main
            viz_gui_main()
        elif args.cli:
            from visualize_performance import main as viz_cli_main
            viz_cli_main()
        else:
            from launch_visualizer import main as launch_viz_main
            launch_viz_main()
            
    elif args.command == 'test':
        # Add tests directory to path
        tests_path = project_root / "tests"
        sys.path.insert(0, str(tests_path))
        
        if args.imports:
            from test_imports import main as test_imports_main
            test_imports_main()
        elif args.data_science:
            from test_data_science import main as test_ds_main
            test_ds_main()
        elif args.visualization:
            from test_visualization_system import main as test_viz_main
            test_viz_main()
        else:
            # Run all tests
            print("🧪 Running comprehensive test suite...")
            try:
                from test_imports import main as test_imports_main
                print("\\n1. Testing imports...")
                test_imports_main()
                
                from test_data_science import main as test_ds_main
                print("\\n2. Testing data science features...")
                test_ds_main()
                
                from test_visualization_system import main as test_viz_main
                print("\\n3. Testing visualization system...")
                test_viz_main()
                
                print("\\n✅ All tests completed successfully!")
            except Exception as e:
                print(f"\\n❌ Test suite failed: {e}")
                sys.exit(1)

if __name__ == "__main__":
    try:
        # Check if we're in the right directory
        if not (Path.cwd() / "src" / "main.py").exists():
            print("❌ Error: Please run this script from the project root directory")
            print("   Expected: RealVision-ObjectUnderstandingAI/")
            print(f"   Current:  {Path.cwd()}")
            sys.exit(1)
            
        main()
    except KeyboardInterrupt:
        print("\\n\\n👋 Application interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\\n❌ Application failed: {e}")
        sys.exit(1)
