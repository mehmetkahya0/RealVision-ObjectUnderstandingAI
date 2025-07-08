"""
Utility functions for the Real-Time Object Understanding Application
"""

import cv2
import numpy as np
import time
import os
import json
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import urllib.request
import hashlib

def download_file_with_progress(url: str, filename: str, description: str = None):
    """Download a file with progress indication"""
    if description is None:
        description = f"Downloading {filename}"
    
    def progress_hook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, (downloaded * 100) // total_size)
            print(f"\r{description}: {percent}%", end='', flush=True)
    
    try:
        urllib.request.urlretrieve(url, filename, progress_hook)
        print(f"\n✓ {description} completed")
        return True
    except Exception as e:
        print(f"\n✗ {description} failed: {e}")
        return False

def verify_file_integrity(filepath: str, expected_hash: str = None) -> bool:
    """Verify file integrity using MD5 hash"""
    if not os.path.exists(filepath):
        return False
    
    if expected_hash is None:
        return True
    
    try:
        with open(filepath, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        return file_hash == expected_hash
    except Exception:
        return False

def get_available_cameras() -> List[int]:
    """Get list of available camera indices"""
    available_cameras = []
    
    for i in range(10):  # Check first 10 camera indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available_cameras.append(i)
        cap.release()
    
    return available_cameras

def calculate_iou(box1: List[int], box2: List[int]) -> float:
    """Calculate Intersection over Union (IoU) of two bounding boxes"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection area
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection_area = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union area
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - intersection_area
    
    return intersection_area / union_area if union_area > 0 else 0.0

def non_max_suppression(detections: List[Dict], iou_threshold: float = 0.4) -> List[Dict]:
    """Apply Non-Maximum Suppression to remove overlapping detections"""
    if not detections:
        return []
    
    # Sort detections by confidence score (descending)
    detections_sorted = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    
    keep = []
    
    while detections_sorted:
        # Take the detection with highest confidence
        best = detections_sorted.pop(0)
        keep.append(best)
        
        # Remove detections with high IoU with the best detection
        detections_sorted = [
            det for det in detections_sorted
            if calculate_iou(best['bbox'], det['bbox']) < iou_threshold
        ]
    
    return keep

def resize_frame_with_aspect_ratio(frame: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    """Resize frame while maintaining aspect ratio"""
    h, w = frame.shape[:2]
    
    # Calculate scaling factor
    scale = min(target_width / w, target_height / h)
    
    # Calculate new dimensions
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize frame
    resized = cv2.resize(frame, (new_w, new_h))
    
    # Create canvas with target dimensions
    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)
    
    # Calculate position to center the resized frame
    y_offset = (target_height - new_h) // 2
    x_offset = (target_width - new_w) // 2
    
    # Place resized frame on canvas
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    
    return canvas

def create_detection_heatmap(frame_shape: Tuple[int, int], detections: List[Dict]) -> np.ndarray:
    """Create a heatmap showing detection density"""
    h, w = frame_shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float32)
    
    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        confidence = detection['confidence']
        
        # Add gaussian blob at detection center
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        
        # Create gaussian kernel
        kernel_size = min(abs(x2 - x1), abs(y2 - y1)) // 2
        if kernel_size > 0:
            y_grid, x_grid = np.ogrid[-center_y:h-center_y, -center_x:w-center_x]
            mask = x_grid*x_grid + y_grid*y_grid <= kernel_size*kernel_size
            heatmap[mask] += confidence
    
    # Normalize heatmap
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    
    # Convert to color heatmap
    heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    return heatmap_colored

def log_detection_data(detections: List[Dict], timestamp: str, log_file: str = "detection_log.json"):
    """Log detection data to file"""
    log_entry = {
        'timestamp': timestamp,
        'detection_count': len(detections),
        'detections': detections
    }
    
    # Load existing log data
    log_data = []
    if os.path.exists(log_file):
        try:
            with open(log_file, 'r') as f:
                log_data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            log_data = []
    
    # Append new entry
    log_data.append(log_entry)
    
    # Keep only last 1000 entries to prevent file from growing too large
    if len(log_data) > 1000:
        log_data = log_data[-1000:]
    
    # Save updated log data
    try:
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)
    except Exception as e:
        print(f"Warning: Failed to save log data: {e}")

def calculate_detection_statistics(detections: List[Dict]) -> Dict:
    """Calculate statistics from detection data"""
    if not detections:
        return {
            'total_objects': 0,
            'unique_classes': 0,
            'class_counts': {},
            'avg_confidence': 0.0,
            'max_confidence': 0.0,
            'min_confidence': 0.0
        }
    
    class_counts = {}
    confidences = []
    
    for detection in detections:
        class_name = detection['class_name']
        confidence = detection['confidence']
        
        class_counts[class_name] = class_counts.get(class_name, 0) + 1
        confidences.append(confidence)
    
    return {
        'total_objects': len(detections),
        'unique_classes': len(class_counts),
        'class_counts': class_counts,
        'avg_confidence': np.mean(confidences),
        'max_confidence': max(confidences),
        'min_confidence': min(confidences)
    }

def create_detection_report(session_data: Dict, output_file: str = None) -> str:
    """Create a detailed detection report"""
    report_lines = [
        "=== Object Detection Session Report ===",
        "",
        f"Session Duration: {session_data.get('duration', 0):.1f} seconds",
        f"Total Frames: {session_data.get('frame_count', 0)}",
        f"Average FPS: {session_data.get('avg_fps', 0):.1f}",
        f"Total Detections: {session_data.get('total_detections', 0)}",
        "",
        "=== Performance Metrics ===",
        f"Average Processing Time: {session_data.get('avg_processing_time', 0)*1000:.1f}ms",
        f"Max FPS: {session_data.get('max_fps', 0):.1f}",
        f"Min FPS: {session_data.get('min_fps', 0):.1f}",
        "",
        "=== Detection Statistics ===",
    ]
    
    # Add class statistics if available
    if 'class_statistics' in session_data:
        class_stats = session_data['class_statistics']
        report_lines.extend([
            f"Unique Classes Detected: {class_stats.get('unique_classes', 0)}",
            f"Average Confidence: {class_stats.get('avg_confidence', 0):.2f}",
            "",
            "Class Distribution:"
        ])
        
        for class_name, count in class_stats.get('class_counts', {}).items():
            report_lines.append(f"  {class_name}: {count}")
    
    report_lines.extend([
        "",
        f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 50
    ])
    
    report_text = "\n".join(report_lines)
    
    if output_file:
        try:
            with open(output_file, 'w') as f:
                f.write(report_text)
            print(f"Report saved to: {output_file}")
        except Exception as e:
            print(f"Warning: Failed to save report: {e}")
    
    return report_text

def optimize_frame_for_detection(frame: np.ndarray, target_size: Tuple[int, int] = (640, 640)) -> Tuple[np.ndarray, float]:
    """Optimize frame for object detection (resize and normalize)"""
    original_h, original_w = frame.shape[:2]
    target_w, target_h = target_size
    
    # Calculate scale factor
    scale = min(target_w / original_w, target_h / original_h)
    
    # Resize frame
    new_w = int(original_w * scale)
    new_h = int(original_h * scale)
    resized_frame = cv2.resize(frame, (new_w, new_h))
    
    # Create padded frame
    padded_frame = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
    
    # Calculate padding offsets
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2
    
    # Place resized frame in center
    padded_frame[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized_frame
    
    return padded_frame, scale

def scale_detections_to_original(detections: List[Dict], scale: float, 
                                offset_x: int = 0, offset_y: int = 0) -> List[Dict]:
    """Scale detection coordinates back to original frame size"""
    scaled_detections = []
    
    for detection in detections:
        x1, y1, x2, y2 = detection['bbox']
        
        # Scale back to original size
        x1 = int((x1 - offset_x) / scale)
        y1 = int((y1 - offset_y) / scale)
        x2 = int((x2 - offset_x) / scale)
        y2 = int((y2 - offset_y) / scale)
        
        scaled_detection = detection.copy()
        scaled_detection['bbox'] = [x1, y1, x2, y2]
        scaled_detections.append(scaled_detection)
    
    return scaled_detections

def create_color_legend(class_names: List[str], colors: List[Tuple[int, int, int]]) -> np.ndarray:
    """Create a color legend for object classes"""
    legend_width = 300
    legend_height = max(400, len(class_names) * 25 + 50)
    
    legend = np.zeros((legend_height, legend_width, 3), dtype=np.uint8)
    
    # Title
    cv2.putText(legend, "Object Classes", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Draw legend entries
    for i, (class_name, color) in enumerate(zip(class_names, colors)):
        y_pos = 60 + i * 25
        
        # Draw color box
        cv2.rectangle(legend, (10, y_pos - 10), (30, y_pos + 10), color, -1)
        cv2.rectangle(legend, (10, y_pos - 10), (30, y_pos + 10), (255, 255, 255), 1)
        
        # Draw class name
        cv2.putText(legend, class_name, (40, y_pos + 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return legend
