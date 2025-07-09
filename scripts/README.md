# 🛠️ Scripts Directory

This directory contains utility scripts, launchers, and setup tools.

## 📁 Script Files

### Environment Setup
- `activate_env.bat` - Windows Command Prompt activation script
- `activate_env.ps1` - PowerShell activation script with enhanced features
- `setup.py` - Python package installation script

### Quick Launchers
- `visualize_data.bat` - Windows batch launcher for visualization tools

## 🚀 Windows Setup Scripts

### PowerShell Script (Recommended)
```powershell
# Run from project root
scripts/activate_env.ps1
```

**Features:**
- Automatic virtual environment creation
- Dependency installation
- Python version verification
- Command reference display
- Enhanced error handling

### Batch File
```cmd
# Run from project root  
scripts/activate_env.bat
```

**Features:**
- Simple Command Prompt compatibility
- Basic environment activation
- Essential command listing

## ⚡ Quick Launchers

### Visualization Launcher
```cmd
# Launch performance data visualizer
scripts/visualize_data.bat
```

Automatically:
- Activates virtual environment
- Launches visualization tool selector
- Handles path resolution

## 🔧 Usage

### First-Time Setup
```bash
# Navigate to project root
cd RealVision-ObjectUnderstandingAI

# Run setup script (Windows)
scripts/activate_env.ps1
```

### Daily Usage
```bash
# Quick environment activation
scripts/activate_env.bat

# Quick visualization launch
scripts/visualize_data.bat
```

## 🐍 Python Environment

The setup scripts handle:
- Virtual environment creation (`venv/`)
- Package installation from `requirements.txt`
- Python path configuration
- Library verification

## 💡 Tips

- **First Run**: Use PowerShell script for complete setup
- **Daily Use**: Batch file is faster for quick activation
- **Troubleshooting**: Scripts display helpful error messages
- **Path Issues**: Always run from project root directory

## 🔄 Adding Custom Scripts

To add new utility scripts:
1. Place `.bat`, `.ps1`, or `.py` files in this directory
2. Update script documentation
3. Test from project root directory
4. Consider adding to main README.md

*Note: All scripts are designed to be run from the project root directory.*
