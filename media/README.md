# 🎬 Media Directory

This directory contains demo videos, sample data, and screenshot outputs.

## 📁 Contents

### Demo Videos
- `demo.mov` - Application demonstration video
- `traffic.mp4` - Sample traffic footage for testing object detection

### Screenshots
- `screenshots/` - Captured screenshots with detection overlays
- Automatically generated during application use (Press 'S' to capture)

## 🎥 Video Formats Supported

The application supports various video formats:
- **MP4** - Most common, excellent compatibility
- **AVI** - Windows standard format  
- **MOV** - QuickTime format, ideal for macOS
- **MKV** - Matroska container format
- **WebM** - Web-optimized format

## 📸 Screenshot Features

Screenshots include:
- High-quality images with detection overlays
- JSON metadata with detection details
- Timestamp-based automatic naming
- Bounding box and confidence information

### Screenshot Structure
```
screenshots/
├── screenshot_20250709_143022.jpg     # Image with overlays
├── screenshot_20250709_143022.json    # Detection metadata
└── ... (additional screenshots)
```

## 🎯 Using Sample Videos

### Basic Usage
```bash
# Process the included traffic video
python run.py --input media/traffic.mp4

# Process with specific model
python run.py --input media/traffic.mp4 --model yolo

# Save processed output
python run.py --input media/traffic.mp4 --output-dir output/
```

### Demo Video
The `demo.mov` file shows the application in action:
- Real-time object detection
- Multiple object tracking  
- Performance metrics display
- Model switching capabilities

## 📁 Adding Your Own Media

To add your own videos:
1. Copy video files to this directory
2. Use: `python run.py --input media/your_video.mp4`
3. Supported formats: MP4, AVI, MOV, MKV, WebM

*Note: Large video files should be compressed for better performance.*
