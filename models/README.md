# 🧠 AI Models Directory

This directory contains all the AI model files used for object detection.

## 📁 Model Files

### YOLOv8 Models
- `yolov8n.pt` - YOLOv8 Nano model (fastest, smaller accuracy)
- Additional YOLO models can be placed here

### ONNX Models  
- `yolov5s.onnx` - YOLOv5 Small in ONNX format for cross-platform compatibility

### EfficientDet Models
- `efficientdet_model/` - Google's EfficientDet model directory (auto-downloaded)
- Contains TensorFlow SavedModel format files

### MobileNet-SSD Models
- `MobileNetSSD_deploy.prototxt` - Network architecture definition
- `MobileNetSSD_deploy.caffemodel` - Pre-trained weights

## 🚀 Model Performance

| Model | Size | Speed | Accuracy | Best Use Case |
|-------|------|-------|----------|---------------|
| YOLOv8n | ~6MB | Fast | High | Real-time applications |
| EfficientDet-D0 | ~24MB | Medium | Very High | High accuracy requirements |
| YOLOv5s ONNX | ~14MB | Medium | High | Cross-platform deployment |
| MobileNet-SSD | ~23MB | Very Fast | Medium | Mobile/embedded devices |

## 🔄 Adding New Models

To add new models:
1. Place model files in this directory
2. Update the application code to recognize new models
3. Test thoroughly with your specific use case

## 📖 Model Details

**YOLOv8**: State-of-the-art object detection with excellent speed/accuracy balance
**EfficientDet**: Google's model with superior accuracy and efficiency trade-offs
**MobileNet-SSD**: Designed for mobile and embedded applications
**ONNX**: Open Neural Network Exchange format for cross-platform compatibility

*Note: Model files are downloaded automatically on first run if not present.*
