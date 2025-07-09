"""
Setup script for creating .exe file using PyInstaller
Run this script to build a standalone executable for RealVision AI
"""

import os
import sys
import subprocess
from pathlib import Path

def install_pyinstaller():
    """Install PyInstaller if not available"""
    try:
        import PyInstaller
        print("✓ PyInstaller is already installed")
    except ImportError:
        print("Installing PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
        print("✓ PyInstaller installed successfully")

def create_exe():
    """Create the executable file"""
    project_root = Path(__file__).parent
    
    # Files to include
    main_script = project_root / "launcher.py"
    icon_file = project_root / "media" / "icon.ico"  # Optional: create an icon
    
    # Directories to include
    data_dirs = [
        (str(project_root / "src"), "src"),
        (str(project_root / "models"), "models"),
        (str(project_root / "media"), "media"),
        (str(project_root / "output"), "output"),
        (str(project_root / "data"), "data"),
    ]
    
    # Build PyInstaller command
    cmd = [
        "pyinstaller",
        "--onefile",  # Create a single executable file
        "--windowed",  # Don't show console window for GUI
        "--name", "RealVision-AI",
        "--distpath", str(project_root / "dist"),
        "--workpath", str(project_root / "build"),
        "--specpath", str(project_root),
        "--clean",  # Clean PyInstaller cache
        "--noconfirm",  # Replace output directory without asking
        "--log-level", "INFO",  # Set logging level
    ]
    
    # Add icon if it exists
    if icon_file.exists():
        cmd.extend(["--icon", str(icon_file)])
    
    # Add data directories
    for src, dst in data_dirs:
        if Path(src).exists():
            cmd.extend(["--add-data", f"{src};{dst}"])
    
    # Add hidden imports for modules that might not be detected
    hidden_imports = [
        "cv2",
        "numpy",
        "PyQt6.QtCore",
        "PyQt6.QtGui", 
        "PyQt6.QtWidgets",
        "ultralytics",
        "torch",
        "torchvision",
        "PIL",
        "matplotlib",
        "pandas",
        "seaborn",
        "sklearn",
        "scipy",
        "requests",
        "plotly",
        "bokeh",
        "json",
        "pathlib",
        "datetime",
        "time",
        "threading",
        "queue",
        "collections",
        "typing",
        "argparse",
        "os",
        "sys"
    ]
    
    # Optional imports (don't fail if not available)
    optional_imports = [
        "onnxruntime",
        "tensorflow",
        "mediapipe"
    ]
    
    for module in hidden_imports:
        cmd.extend(["--hidden-import", module])
    
    # Add optional imports (don't fail if not available)
    for module in optional_imports:
        try:
            __import__(module)
            cmd.extend(["--hidden-import", module])
        except ImportError:
            print(f"⚠️  Optional module {module} not available, skipping...")
    
    # Add the main script
    cmd.append(str(main_script))
    
    print("Building executable...")
    print("Command:", " ".join(cmd))
    print("-" * 50)
    
    # Run PyInstaller
    try:
        subprocess.run(cmd, check=True, cwd=project_root)
        print("\n✓ Executable created successfully!")
        print(f"Location: {project_root / 'dist' / 'RealVision-AI.exe'}")
        
        # Create a simple batch file to run the exe
        batch_content = f"""@echo off
cd /d "{project_root}"
dist\\RealVision-AI.exe
pause
"""
        
        batch_file = project_root / "RealVision-AI.bat"
        with open(batch_file, "w") as f:
            f.write(batch_content)
        
        print(f"✓ Batch file created: {batch_file}")
        print("\nYou can now run the application by:")
        print("1. Double-clicking RealVision-AI.bat")
        print("2. Or running dist/RealVision-AI.exe directly")
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Error building executable: {e}")
        return False
    
    return True

def create_console_exe():
    """Create a console version for debugging"""
    project_root = Path(__file__).parent
    
    # Files to include
    main_script = project_root / "launcher.py"
    
    # Directories to include
    data_dirs = [
        (str(project_root / "src"), "src"),
        (str(project_root / "models"), "models"),
        (str(project_root / "media"), "media"),
    ]
    
    # Build PyInstaller command for console version
    cmd = [
        "pyinstaller",
        "--onefile",  # Create a single executable file
        "--console",  # Show console window for debugging
        "--name", "RealVision-AI-Console",
        "--distpath", str(project_root / "dist"),
        "--workpath", str(project_root / "build"),
        "--specpath", str(project_root),
        "--clean",  # Clean PyInstaller cache
        "--noconfirm",  # Replace output directory without asking
    ]
    
    # Add data directories
    for src, dst in data_dirs:
        if Path(src).exists():
            cmd.extend(["--add-data", f"{src};{dst}"])
    
    # Add hidden imports
    essential_imports = [
        "cv2",
        "numpy", 
        "PyQt6.QtCore",
        "PyQt6.QtGui", 
        "PyQt6.QtWidgets",
        "pathlib",
        "argparse"
    ]
    
    for module in essential_imports:
        cmd.extend(["--hidden-import", module])
    
    # Add the main script
    cmd.append(str(main_script))
    
    print("Building console executable for debugging...")
    print("Command:", " ".join(cmd))
    print("-" * 50)
    
    # Run PyInstaller
    try:
        subprocess.run(cmd, check=True, cwd=project_root)
        print("\n✓ Console executable created successfully!")
        print(f"Location: {project_root / 'dist' / 'RealVision-AI-Console.exe'}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error building console executable: {e}")
        return False

def create_installer():
    """Create an NSIS installer (optional)"""
    try:
        # This would require NSIS to be installed
        # For now, just provide instructions
        print("\n📦 To create an installer:")
        print("1. Install NSIS (Nullsoft Scriptable Install System)")
        print("2. Create an installer script (.nsi file)")
        print("3. Use makensis to build the installer")
        print("\nFor now, you can distribute the .exe file directly")
    except Exception as e:
        print(f"Installer creation not available: {e}")

def main():
    print("🚀 RealVision AI - Executable Builder")
    print("=" * 50)
    
    # Install PyInstaller
    install_pyinstaller()
    
    # Ask user which version to build
    print("\nChoose build option:")
    print("1. GUI version (windowed)")
    print("2. Console version (for debugging)")
    print("3. Both versions")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    success = False
    
    if choice in ['1', '3']:
        # Create GUI executable
        print("\n📱 Building GUI version...")
        success = create_exe()
    
    if choice in ['2', '3']:
        # Create console executable  
        print("\n💻 Building console version...")
        console_success = create_console_exe()
        success = success or console_success
    
    if choice not in ['1', '2', '3']:
        print("❌ Invalid choice. Building GUI version by default...")
        success = create_exe()
    
    if success:
        # Optional: Create installer
        create_installer()
        
        print("\n🎉 Build completed successfully!")
        print("\nNext steps:")
        print("1. Test the executable(s) in the dist/ folder")
        print("2. Make sure all required model files are in the models/ directory")
        print("3. Distribute the executable(s) as needed")
    else:
        print("\n❌ Build failed. Please check the error messages above.")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()
