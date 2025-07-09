#!/usr/bin/env python3
"""
Quick launcher for Performance Data Visualizer
==============================================

Simple script to launch the performance data visualization tools.
"""

import sys
import os

def main():
    print("🤖 RealVision Performance Data Visualizer")
    print("=" * 50)
    print()
    print("Available visualization tools:")
    print("1. 🖥️  GUI Version (Easy to use)")
    print("2. 💻 Command Line Version (Advanced)")
    print("3. 📊 Quick Dashboard (Auto-detect latest data)")
    print("4. 📁 Open Data Folder")
    print()
    
    while True:
        try:
            choice = input("Choose an option (1-4) or 'q' to quit: ").strip().lower()
            
            if choice == 'q' or choice == 'quit':
                print("👋 Goodbye!")
                break
                
            elif choice == '1':
                print("🚀 Launching GUI version...")
                try:
                    import visualize_performance_gui
                    visualize_performance_gui.main()
                except ImportError as e:
                    print(f"❌ Error importing GUI: {e}")
                    print("Make sure all required packages are installed.")
                except Exception as e:
                    print(f"❌ Error running GUI: {e}")
                break
                
            elif choice == '2':
                print("🚀 Launching command line version...")
                try:
                    import visualize_performance
                    visualize_performance.interactive_visualizer()
                except ImportError as e:
                    print(f"❌ Error importing visualizer: {e}")
                    print("Make sure all required packages are installed.")
                except Exception as e:
                    print(f"❌ Error running visualizer: {e}")
                break
                
            elif choice == '3':
                print("🚀 Creating quick dashboard...")
                try:
                    import visualize_performance
                    visualizer = visualize_performance.PerformanceDataVisualizer()
                    
                    # Find latest data file
                    data_dir = os.path.join(os.path.dirname(__file__), 'data')
                    if os.path.exists(data_dir):
                        import glob
                        json_files = glob.glob(os.path.join(data_dir, "*.json"))
                        if json_files:
                            # Get most recent file
                            latest_file = max(json_files, key=os.path.getmtime)
                            print(f"📁 Using latest data file: {os.path.basename(latest_file)}")
                            
                            if visualizer.load_data_file(latest_file):
                                visualizer.create_performance_dashboard()
                                print("✅ Dashboard created and opened in browser!")
                            else:
                                print("❌ Failed to load data file.")
                        else:
                            print("❌ No data files found in data folder.")
                    else:
                        print("❌ Data folder not found.")
                        
                except Exception as e:
                    print(f"❌ Error creating dashboard: {e}")
                break
                
            elif choice == '4':
                print("📁 Opening data folder...")
                data_dir = os.path.join(os.path.dirname(__file__), 'data')
                if os.path.exists(data_dir):
                    if sys.platform == "win32":
                        os.startfile(data_dir)
                    elif sys.platform == "darwin":  # macOS
                        os.system(f"open '{data_dir}'")
                    else:  # Linux
                        os.system(f"xdg-open '{data_dir}'")
                    print("✅ Data folder opened.")
                else:
                    print("❌ Data folder not found.")
                break
                
            else:
                print("❌ Invalid choice. Please select 1-4 or 'q' to quit.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
