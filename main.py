"""
Real-Time Object Understanding Application
==========================================

A powerful computer vision application for real-time object detection,
classification, and understanding using webcam input.

Features:
- Multiple object detection models (YOLOv8, MobileNet)
- Real-time performance optimization
- Advanced object tracking
- Confidence threshold adjustment
- Multiple detection modes
- Performance metrics display
- Screenshot capture functionality
- Modern GUI interface

Author: Mehmet Kahya
Date: July 2025
"""

import cv2
import numpy as np
import time
import threading
import queue
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional
import json
import os
from datetime import datetime

# Global variables to track model availability
YOLO_AVAILABLE = False
MEDIAPIPE_AVAILABLE = False
ONNX_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    pass

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    pass

try:
    import onnxruntime
    ONNX_AVAILABLE = True
except ImportError:
    pass

# Print available models
available_models = []
if YOLO_AVAILABLE:
    available_models.append("YOLO v8")
if MEDIAPIPE_AVAILABLE:
    available_models.append("MediaPipe")
if ONNX_AVAILABLE:
    available_models.append("ONNX Runtime")
    
print(f"Available AI models: {', '.join(available_models) if available_models else 'OpenCV DNN only'}")

class ObjectTracker:
    """Advanced object tracking with ID assignment"""
    
    def __init__(self, max_disappeared: int = 30):
        self.next_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        
    def register(self, centroid):
        """Register a new object"""
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.next_id += 1
        
    def deregister(self, object_id):
        """Deregister an object"""
        del self.objects[object_id]
        del self.disappeared[object_id]
        
    def update(self, rects):
        """Update object tracking"""
        if len(rects) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return {}
            
        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (start_x, start_y, end_x, end_y)) in enumerate(rects):
            cx = int((start_x + end_x) / 2.0)
            cy = int((start_y + end_y) / 2.0)
            input_centroids[i] = (cx, cy)
            
        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                self.register(input_centroids[i])
        else:
            object_centroids = list(self.objects.values())
            object_ids = list(self.objects.keys())
            
            D = np.linalg.norm(np.array(object_centroids)[:, np.newaxis] - input_centroids, axis=2)
            
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]
            
            used_row_indices = set()
            used_col_indices = set()
            
            for (row, col) in zip(rows, cols):
                if row in used_row_indices or col in used_col_indices:
                    continue
                    
                object_id = object_ids[row]
                self.objects[object_id] = input_centroids[col]
                self.disappeared[object_id] = 0
                
                used_row_indices.add(row)
                used_col_indices.add(col)
                
            unused_row_indices = set(range(0, D.shape[0])).difference(used_row_indices)
            unused_col_indices = set(range(0, D.shape[1])).difference(used_col_indices)
            
            if D.shape[0] >= D.shape[1]:
                for row in unused_row_indices:
                    object_id = object_ids[row]
                    self.disappeared[object_id] += 1
                    if self.disappeared[object_id] > self.max_disappeared:
                        self.deregister(object_id)
            else:
                for col in unused_col_indices:
                    self.register(input_centroids[col])
                    
        return self.objects

class PerformanceMonitor:
    """Monitor application performance metrics"""
    
    def __init__(self, window_size: int = 30):
        self.fps_history = deque(maxlen=window_size)
        self.processing_times = deque(maxlen=window_size)
        self.detection_counts = deque(maxlen=window_size)
        
    def update(self, fps: float, processing_time: float, detection_count: int):
        """Update performance metrics"""
        self.fps_history.append(fps)
        self.processing_times.append(processing_time)
        self.detection_counts.append(detection_count)
        
    def get_stats(self) -> Dict[str, float]:
        """Get performance statistics"""
        return {
            'avg_fps': np.mean(self.fps_history) if self.fps_history else 0,
            'avg_processing_time': np.mean(self.processing_times) if self.processing_times else 0,
            'avg_detections': np.mean(self.detection_counts) if self.detection_counts else 0,
            'max_fps': max(self.fps_history) if self.fps_history else 0,
            'min_fps': min(self.fps_history) if self.fps_history else 0
        }

class ObjectUnderstandingApp:
    """Main application class for real-time object understanding"""
    def __init__(self, preferred_model="auto"):
        self.cap = None
        self.running = False
        self.paused = False

        # Models
        self.yolo_model = None
        self.dnn_net = None
        self.mediapipe_objectron = None
        self.mediapipe_hands = None
        self.onnx_session = None
        self.preferred_model = preferred_model
        self.current_model = "auto"  # "yolo", "dnn", "mediapipe", "onnx", "auto"
        self.available_models = []
        
        # Detection settings
        self.confidence_threshold = 0.5
        self.nms_threshold = 0.4
        self.detection_size = (640, 640)
        
        # Tracking and performance
        self.tracker = ObjectTracker()
        self.performance_monitor = PerformanceMonitor()
        
        # UI settings
        self.show_fps = True
        self.show_confidence = True
        self.show_tracking_ids = True
        self.show_performance_stats = True
        
        # Statistics
        self.total_detections = 0
        self.session_start_time = time.time()
        self.frame_count = 0
        
        # Screenshot settings
        self.screenshots_dir = "screenshots"
        os.makedirs(self.screenshots_dir, exist_ok=True)
        
        # Initialize models
        self.initialize_models()
        
    def initialize_models(self):
        """Initialize all available object detection models"""
        global YOLO_AVAILABLE, MEDIAPIPE_AVAILABLE, ONNX_AVAILABLE
        print("Initializing object detection models...")
        
        # Initialize YOLO model
        if YOLO_AVAILABLE:
            try:
                print("Loading YOLOv8 model...")
                self.yolo_model = YOLO('yolov8n.pt')  # nano version for speed
                self.available_models.append("yolo")
                print("✓ YOLOv8 model loaded successfully")
            except Exception as e:
                print(f"✗ Failed to load YOLOv8: {e}")
                YOLO_AVAILABLE = False
        
        # Initialize OpenCV DNN model (MobileNet-SSD)
        try:
            print("Loading MobileNet-SSD model...")
            self.download_dnn_model()
            
            config_path = "MobileNetSSD_deploy.prototxt"
            weights_path = "MobileNetSSD_deploy.caffemodel"
            
            if os.path.exists(config_path) and os.path.exists(weights_path):
                self.dnn_net = cv2.dnn.readNetFromCaffe(config_path, weights_path)
                self.available_models.append("dnn")
                print("✓ MobileNet-SSD model loaded successfully")
            else:
                print("✗ MobileNet-SSD model files not found")
                
        except Exception as e:
            print(f"✗ Failed to load MobileNet-SSD: {e}")
        
        # Initialize MediaPipe models
        if MEDIAPIPE_AVAILABLE:
            try:
                print("Loading MediaPipe models...")
                import mediapipe as mp
                
                # Initialize object detection
                mp_objectron = mp.solutions.objectron
                self.mediapipe_objectron = mp_objectron.Objectron(
                    static_image_mode=False,
                    max_num_objects=5,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5,
                    model_name='Cup'  # Can be 'Cup', 'Chair', 'Camera', 'Shoe'
                )
                
                # Initialize hand detection
                mp_hands = mp.solutions.hands
                self.mediapipe_hands = mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                
                self.available_models.append("mediapipe")
                print("✓ MediaPipe models loaded successfully")
                
            except Exception as e:
                print(f"✗ Failed to load MediaPipe: {e}")
                MEDIAPIPE_AVAILABLE = False
        
        # Initialize ONNX models
        if ONNX_AVAILABLE:
            try:
                print("Loading ONNX models...")
                self.download_onnx_model()
                
                import onnxruntime
                onnx_model_path = "yolov5s.onnx"
                
                if os.path.exists(onnx_model_path):
                    self.onnx_session = onnxruntime.InferenceSession(onnx_model_path)
                    self.available_models.append("onnx")
                    print("✓ ONNX YOLOv5 model loaded successfully")
                else:
                    print("✗ ONNX model file not found")
                    
            except Exception as e:
                print(f"✗ Failed to load ONNX model: {e}")
                ONNX_AVAILABLE = False
        
        # Set model based on preference and availability
        if not self.available_models:
            raise RuntimeError("No object detection models available!")
        
        # Use preferred model if available
        if self.preferred_model != "auto" and self.preferred_model in self.available_models:
            self.current_model = self.preferred_model
        else:
            # Prioritize models by performance and accuracy
            model_priority = ["yolo", "onnx", "dnn", "mediapipe"]
            for model in model_priority:
                if model in self.available_models:
                    self.current_model = model
                    break
                
        print(f"Available models: {', '.join(self.available_models)}")
        print(f"Using model: {self.current_model}")
    
    def download_dnn_model(self):
        """Download MobileNet-SSD model files if not present"""
        import urllib.request
        
        files_to_download = [
            {
                'url': 'https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master/deploy.prototxt',
                'filename': 'MobileNetSSD_deploy.prototxt'
            },
            {
                'url': 'https://drive.google.com/uc?export=download&id=0B3gersZ2cHIxRm5PMWRoTkdHdHc',
                'filename': 'MobileNetSSD_deploy.caffemodel'
            }
        ]
        
        for file_info in files_to_download:
            if not os.path.exists(file_info['filename']):
                try:
                    print(f"Downloading {file_info['filename']}...")
                    urllib.request.urlretrieve(file_info['url'], file_info['filename'])
                    print(f"✓ Downloaded {file_info['filename']}")
                except Exception as e:
                    print(f"✗ Failed to download {file_info['filename']}: {e}")
    
    def download_onnx_model(self):
        """Download ONNX model files if not present"""
        import urllib.request
        
        onnx_model_info = {
            'url': 'https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.onnx',
            'filename': 'yolov5s.onnx'
        }
        
        if not os.path.exists(onnx_model_info['filename']):
            try:
                print(f"Downloading {onnx_model_info['filename']}...")
                urllib.request.urlretrieve(onnx_model_info['url'], onnx_model_info['filename'])
                print(f"✓ Downloaded {onnx_model_info['filename']}")
            except Exception as e:
                print(f"✗ Failed to download {onnx_model_info['filename']}: {e}")
    
    def detect_objects_yolo(self, frame: np.ndarray) -> List[Dict]:
        """Detect objects using YOLO model"""
        if not YOLO_AVAILABLE or self.yolo_model is None:
            return []
            
        try:
            results = self.yolo_model(frame, conf=self.confidence_threshold)
            detections = []
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = box.conf[0].cpu().numpy()
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        # Get class name
                        class_name = self.yolo_model.names[class_id]
                        
                        detections.append({
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': float(confidence),
                            'class_name': class_name,
                            'class_id': class_id
                        })
            
            return detections
            
        except Exception as e:
            print(f"Error in YOLO detection: {e}")
            return []
    
    def detect_objects_dnn(self, frame: np.ndarray) -> List[Dict]:
        """Detect objects using OpenCV DNN model"""
        if self.dnn_net is None:
            return []
            
        try:
            # COCO class names for MobileNet-SSD
            class_names = [
                "background", "aeroplane", "bicycle", "bird", "boat",
                "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                "sofa", "train", "tvmonitor"
            ]
            
            h, w = frame.shape[:2]
            
            # Create blob from image
            blob = cv2.dnn.blobFromImage(
                frame, 0.007843, (300, 300), 127.5, swapRB=True, crop=False
            )
            
            self.dnn_net.setInput(blob)
            detections_dnn = self.dnn_net.forward()
            
            detections = []
            for i in range(detections_dnn.shape[2]):
                confidence = detections_dnn[0, 0, i, 2]
                
                if confidence > self.confidence_threshold:
                    class_id = int(detections_dnn[0, 0, i, 1])
                    
                    if class_id < len(class_names):
                        x1 = int(detections_dnn[0, 0, i, 3] * w)
                        y1 = int(detections_dnn[0, 0, i, 4] * h)
                        x2 = int(detections_dnn[0, 0, i, 5] * w)
                        y2 = int(detections_dnn[0, 0, i, 6] * h)
                        
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': float(confidence),
                            'class_name': class_names[class_id],
                            'class_id': class_id
                        })
            
            return detections
            
        except Exception as e:
            print(f"Error in DNN detection: {e}")
            return []
    
    def detect_objects_mediapipe(self, frame: np.ndarray) -> List[Dict]:
        """Detect objects using MediaPipe models"""
        if not MEDIAPIPE_AVAILABLE or (self.mediapipe_objectron is None and self.mediapipe_hands is None):
            return []
            
        try:
            detections = []
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Hand detection
            if self.mediapipe_hands:
                hands_results = self.mediapipe_hands.process(rgb_frame)
                if hands_results.multi_hand_landmarks:
                    for hand_landmarks in hands_results.multi_hand_landmarks:
                        # Get bounding box for hand
                        h, w, _ = frame.shape
                        x_coords = [landmark.x * w for landmark in hand_landmarks.landmark]
                        y_coords = [landmark.y * h for landmark in hand_landmarks.landmark]
                        
                        x1, x2 = int(min(x_coords)), int(max(x_coords))
                        y1, y2 = int(min(y_coords)), int(max(y_coords))
                        
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': 0.8,  # MediaPipe doesn't provide confidence scores
                            'class_name': 'hand',
                            'class_id': 100  # Custom ID for hand
                        })
            
            # Object detection (3D objects like cups, chairs)
            if self.mediapipe_objectron:
                objectron_results = self.mediapipe_objectron.process(rgb_frame)
                if objectron_results.detected_objects:
                    for detected_object in objectron_results.detected_objects:
                        # Get 2D bounding box from 3D landmarks
                        h, w, _ = frame.shape
                        landmarks_2d = detected_object.landmarks_2d
                        
                        x_coords = [landmark.x * w for landmark in landmarks_2d.landmark]
                        y_coords = [landmark.y * h for landmark in landmarks_2d.landmark]
                        
                        x1, x2 = int(min(x_coords)), int(max(x_coords))
                        y1, y2 = int(min(y_coords)), int(max(y_coords))
                        
                        detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': 0.8,
                            'class_name': 'cup',  # Based on current model
                            'class_id': 101
                        })
            
            return detections
            
        except Exception as e:
            print(f"Error in MediaPipe detection: {e}")
            return []
    
    def detect_objects_onnx(self, frame: np.ndarray) -> List[Dict]:
        """Detect objects using ONNX Runtime with YOLOv5 - Optimized with NMS"""
        if not ONNX_AVAILABLE or self.onnx_session is None:
            return []
            
        try:
            import onnxruntime
            
            # COCO class names (cached as class variable to avoid recreation)
            if not hasattr(self, '_onnx_class_names'):
                self._onnx_class_names = [
                    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
                    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
                ]
            
            # Preprocess image - Optimized with minimal operations
            input_size = 640
            original_shape = frame.shape[:2]
            scale_x = original_shape[1] / input_size
            scale_y = original_shape[0] / input_size
            
            # Efficient resize with pre-allocated array
            resized = cv2.resize(frame, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
            
            # Optimized normalization and transposition
            input_image = resized.astype(np.float32)
            input_image *= (1.0 / 255.0)  # In-place normalization
            input_image = np.transpose(input_image, (2, 0, 1))  # HWC to CHW
            input_image = np.expand_dims(input_image, axis=0)   # Add batch dimension
            
            # Check the expected input data type and convert if necessary
            if not hasattr(self, '_onnx_input_name'):
                self._onnx_input_name = self.onnx_session.get_inputs()[0].name
                expected_type = self.onnx_session.get_inputs()[0].type
                self._onnx_use_float16 = 'float16' in expected_type
            
            if self._onnx_use_float16:
                input_image = input_image.astype(np.float16)
            
            # Run inference
            outputs = self.onnx_session.run(None, {self._onnx_input_name: input_image})
            
            # Post-process results - Optimized with vectorized operations
            predictions = outputs[0]  # Shape: [1, 25200, 85] for YOLOv5
            
            # Reshape if needed
            if len(predictions.shape) == 3:
                predictions = predictions[0]  # Remove batch dimension: [25200, 85]
            
            # Early exit if no predictions
            if len(predictions) == 0:
                return []
            
            # Vectorized confidence filtering (much faster than loops)
            conf_threshold = self.confidence_threshold
            objectness_scores = predictions[:, 4]
            conf_mask = objectness_scores > conf_threshold
            
            if not np.any(conf_mask):
                return []
            
            # Filter predictions early
            filtered_predictions = predictions[conf_mask]
            
            # Extract components (vectorized)
            boxes = filtered_predictions[:, :4]  # x_center, y_center, width, height
            confidences = filtered_predictions[:, 4]  # objectness confidence
            class_scores = filtered_predictions[:, 5:]  # class probabilities
            
            # Get class predictions (vectorized)
            class_ids = np.argmax(class_scores, axis=1)
            class_confidences = np.max(class_scores, axis=1)
            
            # Final confidence = objectness * class_confidence
            final_confidences = confidences * class_confidences
            
            # Second confidence filter
            final_conf_mask = final_confidences > conf_threshold
            if not np.any(final_conf_mask):
                return []
            
            boxes = boxes[final_conf_mask]
            final_confidences = final_confidences[final_conf_mask]
            class_ids = class_ids[final_conf_mask]
            
            # Convert to x1, y1, x2, y2 format for NMS (vectorized)
            half_w = boxes[:, 2] / 2
            half_h = boxes[:, 3] / 2
            boxes_xyxy = np.column_stack([
                boxes[:, 0] - half_w,  # x1
                boxes[:, 1] - half_h,  # y1
                boxes[:, 0] + half_w,  # x2
                boxes[:, 1] + half_h   # y2
            ])
            
            # Apply Non-Maximum Suppression using OpenCV
            indices = cv2.dnn.NMSBoxes(
                boxes_xyxy.tolist(),
                final_confidences.tolist(),
                conf_threshold,
                0.4  # NMS threshold
            )
            
            detections = []
            if len(indices) > 0:
                if isinstance(indices[0], list):
                    indices = [i[0] for i in indices]
                
                # Vectorized coordinate conversion
                selected_boxes = boxes_xyxy[indices]
                selected_confidences = final_confidences[indices]
                selected_class_ids = class_ids[indices]
                
                # Convert back to original image coordinates (vectorized)
                selected_boxes[:, [0, 2]] *= scale_x  # x coordinates
                selected_boxes[:, [1, 3]] *= scale_y  # y coordinates
                
                # Clamp coordinates to image bounds (vectorized)
                selected_boxes[:, [0, 2]] = np.clip(selected_boxes[:, [0, 2]], 0, original_shape[1])
                selected_boxes[:, [1, 3]] = np.clip(selected_boxes[:, [1, 3]], 0, original_shape[0])
                
                # Build detection list
                for i in range(len(selected_boxes)):
                    x1, y1, x2, y2 = selected_boxes[i].astype(int)
                    class_id = int(selected_class_ids[i])
                    
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': float(selected_confidences[i]),
                        'class_name': self._onnx_class_names[class_id] if class_id < len(self._onnx_class_names) else 'unknown',
                        'class_id': class_id
                    })
            
            return detections
            
        except Exception as e:
            print(f"Error in ONNX detection: {e}")
            return []
    
    def detect_objects(self, frame: np.ndarray) -> List[Dict]:
        """Detect objects using the current model"""
        if self.current_model == "yolo":
            return self.detect_objects_yolo(frame)
        elif self.current_model == "dnn":
            return self.detect_objects_dnn(frame)
        elif self.current_model == "mediapipe":
            return self.detect_objects_mediapipe(frame)
        elif self.current_model == "onnx":
            return self.detect_objects_onnx(frame)
        else:
            return []
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict], 
                       tracked_objects: Dict) -> np.ndarray:
        """Draw detection results on frame"""
        frame_copy = frame.copy()
        
        # Color map for different classes
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0),
            (0, 128, 255), (128, 255, 0), (255, 20, 147), (0, 191, 255)
        ]
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            class_id = detection['class_id']
            
            # Get color for this class
            color = colors[class_id % len(colors)]
            
            # Draw bounding box
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
            
            # Prepare label
            label = f"{class_name}"
            if self.show_confidence:
                label += f" {confidence:.2f}"
            
            # Find matching tracked object
            centroid = ((x1 + x2) // 2, (y1 + y2) // 2)
            tracking_id = None
            min_distance = float('inf')
            
            for obj_id, obj_centroid in tracked_objects.items():
                distance = np.sqrt((centroid[0] - obj_centroid[0])**2 + 
                                 (centroid[1] - obj_centroid[1])**2)
                if distance < min_distance:
                    min_distance = distance
                    tracking_id = obj_id
            
            if self.show_tracking_ids and tracking_id is not None and min_distance < 50:
                label += f" ID:{tracking_id}"
            
            # Draw label background
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame_copy, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(frame_copy, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw center point
            cv2.circle(frame_copy, centroid, 4, color, -1)
        
        return frame_copy
    
    def draw_ui_overlay(self, frame: np.ndarray, fps: float, 
                       detection_count: int) -> np.ndarray:
        """Draw UI overlay with statistics and controls"""
        overlay = frame.copy()
        
        # Performance stats
        if self.show_performance_stats:
            stats = self.performance_monitor.get_stats()
            
            # Background for stats
            cv2.rectangle(overlay, (10, 10), (350, 150), (0, 0, 0), -1)
            cv2.rectangle(overlay, (10, 10), (350, 150), (255, 255, 255), 2)
            
            # Stats text
            stats_text = [
                f"Model: {self.current_model.upper()}",
                f"FPS: {fps:.1f} (Avg: {stats['avg_fps']:.1f})",
                f"Detections: {detection_count}",
                f"Processing: {stats['avg_processing_time']*1000:.1f}ms",
                f"Session: {int(time.time() - self.session_start_time)}s"
            ]
            
            for i, text in enumerate(stats_text):
                cv2.putText(overlay, text, (20, 35 + i * 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Controls help
        controls_text = [
            "Controls:",
            "SPACE - Pause/Resume",
            "S - Screenshot",
            "M - Switch Model",
            "C - Toggle Confidence",
            "T - Toggle Tracking IDs",
            "P - Toggle Performance",
            "+/- - Confidence Threshold",
            "Q - Quit"
        ]
        
        # Background for controls
        y_start = frame.shape[0] - len(controls_text) * 25 - 20
        cv2.rectangle(overlay, (10, y_start), (300, frame.shape[0] - 10), (0, 0, 0), -1)
        cv2.rectangle(overlay, (10, y_start), (300, frame.shape[0] - 10), (255, 255, 255), 2)
        
        for i, text in enumerate(controls_text):
            color = (0, 255, 255) if i == 0 else (255, 255, 255)
            cv2.putText(overlay, text, (20, y_start + 20 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Confidence threshold indicator
        threshold_text = f"Confidence Threshold: {self.confidence_threshold:.2f}"
        cv2.putText(overlay, threshold_text, (frame.shape[1] - 350, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Pause indicator
        if self.paused:
            cv2.putText(overlay, "PAUSED", (frame.shape[1] // 2 - 50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        
        return overlay
    
    def save_screenshot(self, frame: np.ndarray, detections: List[Dict]):
        """Save screenshot with detection data"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save image
        img_filename = f"{self.screenshots_dir}/screenshot_{timestamp}.jpg"
        cv2.imwrite(img_filename, frame)
        
        # Save detection data
        data_filename = f"{self.screenshots_dir}/detections_{timestamp}.json"
        detection_data = {
            'timestamp': timestamp,
            'model': self.current_model,
            'confidence_threshold': self.confidence_threshold,
            'detections': detections,
            'total_detections': len(detections)
        }
        
        with open(data_filename, 'w') as f:
            json.dump(detection_data, f, indent=2)
        
        print(f"Screenshot saved: {img_filename}")
        print(f"Detection data saved: {data_filename}")
    
    def initialize_camera(self, source=0):
        """Initialize camera or video file capture"""
        if isinstance(source, str):
            print(f"Initializing video file: {source}...")
        else:
            print(f"Initializing camera {source}...")
        
        self.cap = cv2.VideoCapture(source)
        
        if not self.cap.isOpened():
            if isinstance(source, str):
                raise RuntimeError(f"Could not open video file: {source}")
            else:
                raise RuntimeError(f"Could not open camera {source}")
        
        # Set properties (mainly for cameras, videos will use their native settings)
        if isinstance(source, int):  # Camera
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Get actual properties
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
        if isinstance(source, str):
            # For video files, also get total frame count and duration
            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            print(f"✓ Video file loaded: {width}x{height} @ {fps}fps")
            print(f"   Duration: {duration:.1f}s ({total_frames} frames)")
        else:
            print(f"✓ Camera initialized: {width}x{height} @ {fps}fps")
    
    def run(self, camera_index=0):
        """Run the main application loop"""
        try:
            self.initialize_camera(camera_index)
            self.running = True
            
            print("\n🚀 Starting Real-Time Object Understanding Application")
            print("=" * 60)
            print("Press 'Q' to quit, 'SPACE' to pause/resume")
            print("Use '+'/'-' to adjust confidence threshold")
            print("Press 'S' to take screenshot")
            print("=" * 60)
            
            prev_time = time.time()
            
            while self.running:
                if not self.paused:
                    ret, frame = self.cap.read()
                    if not ret:
                        # Check if this is a video file that reached the end
                        if hasattr(self, 'cap') and self.cap.get(cv2.CAP_PROP_POS_FRAMES) >= self.cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1:
                            print("Video ended. Restarting from beginning...")
                            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            continue
                        else:
                            print("Failed to capture frame or camera disconnected")
                            break
                    
                    self.frame_count += 1
                    start_time = time.time()
                    
                    # Detect objects
                    detections = self.detect_objects(frame)
                    self.total_detections += len(detections)
                    
                    # Update object tracking
                    rects = [det['bbox'] for det in detections]
                    tracked_objects = self.tracker.update(rects)
                    
                    # Calculate FPS
                    current_time = time.time()
                    fps = 1.0 / (current_time - prev_time)
                    prev_time = current_time
                    
                    processing_time = current_time - start_time
                    
                    # Update performance monitor
                    self.performance_monitor.update(fps, processing_time, len(detections))
                    
                    # Draw results
                    frame_with_detections = self.draw_detections(frame, detections, tracked_objects)
                    frame_with_ui = self.draw_ui_overlay(frame_with_detections, fps, len(detections))
                    
                    # Display frame
                    cv2.imshow('Real-Time Object Understanding', frame_with_ui)
                else:
                    # Just show the current frame when paused
                    cv2.waitKey(1)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == 27:  # 'q' or ESC
                    break
                elif key == ord(' '):  # SPACE
                    self.paused = not self.paused
                    print(f"{'Paused' if self.paused else 'Resumed'}")
                elif key == ord('s'):  # Screenshot
                    if not self.paused:
                        self.save_screenshot(frame_with_ui, detections)
                elif key == ord('m'):  # Switch model
                    if len(self.available_models) > 1:
                        current_index = self.available_models.index(self.current_model)
                        next_index = (current_index + 1) % len(self.available_models)
                        self.current_model = self.available_models[next_index]
                        print(f"Switched to {self.current_model.upper()} model")
                elif key == ord('c'):  # Toggle confidence display
                    self.show_confidence = not self.show_confidence
                elif key == ord('t'):  # Toggle tracking IDs
                    self.show_tracking_ids = not self.show_tracking_ids
                elif key == ord('p'):  # Toggle performance stats
                    self.show_performance_stats = not self.show_performance_stats
                elif key == ord('=') or key == ord('+'):  # Increase confidence
                    self.confidence_threshold = min(0.95, self.confidence_threshold + 0.05)
                    print(f"Confidence threshold: {self.confidence_threshold:.2f}")
                elif key == ord('-'):  # Decrease confidence
                    self.confidence_threshold = max(0.05, self.confidence_threshold - 0.05)
                    print(f"Confidence threshold: {self.confidence_threshold:.2f}")
        
        except KeyboardInterrupt:
            print("\n⚠️  Application interrupted by user")
        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("\n🧹 Cleaning up...")
        
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        
        # Print session statistics
        session_time = time.time() - self.session_start_time
        stats = self.performance_monitor.get_stats()
        
        print("\n📊 Session Statistics:")
        print("=" * 40)
        print(f"Duration: {session_time:.1f} seconds")
        print(f"Frames processed: {self.frame_count}")
        print(f"Total detections: {self.total_detections}")
        print(f"Average FPS: {stats['avg_fps']:.1f}")
        print(f"Average processing time: {stats['avg_processing_time']*1000:.1f}ms")
        print(f"Screenshots saved in: {self.screenshots_dir}/")
        print("=" * 40)
        print("✅ Application closed successfully")

def main():
    """Main function"""
    print("🎯 Real-Time Object Understanding Application")
    print("=" * 50)
    
    try:
        app = ObjectUnderstandingApp()
        app.run()
    except Exception as e:
        print(f"❌ Failed to start application: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
