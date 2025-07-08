"""
Configuration file for Real-Time Object Understanding Application
"""

# Camera settings
CAMERA_CONFIG = {
    'default_camera_index': 0,
    'frame_width': 1280,
    'frame_height': 720,
    'fps': 30,
    'buffer_size': 1
}

# Detection settings
DETECTION_CONFIG = {
    'default_confidence_threshold': 0.5,
    'nms_threshold': 0.4,
    'detection_size': (640, 640),
    'max_detections_per_frame': 100
}

# Tracking settings
TRACKING_CONFIG = {
    'max_disappeared_frames': 30,
    'max_tracking_distance': 50
}

# UI settings
UI_CONFIG = {
    'show_fps': True,
    'show_confidence': True,
    'show_tracking_ids': True,
    'show_performance_stats': True,
    'font_scale': 0.6,
    'font_thickness': 2
}

# Performance settings
PERFORMANCE_CONFIG = {
    'monitor_window_size': 30,
    'target_fps': 30
}

# Model settings
MODEL_CONFIG = {
    'yolo_model_size': 'n',  # n, s, m, l, x (nano, small, medium, large, extra-large)
    'prefer_gpu': True,
    'dnn_input_size': (300, 300),
    'dnn_scale_factor': 0.007843,
    'dnn_mean': 127.5
}

# Color scheme for different object classes
COLOR_PALETTE = [
    (255, 0, 0),     # Red
    (0, 255, 0),     # Green
    (0, 0, 255),     # Blue
    (255, 255, 0),   # Yellow
    (255, 0, 255),   # Magenta
    (0, 255, 255),   # Cyan
    (128, 0, 128),   # Purple
    (255, 165, 0),   # Orange
    (0, 128, 255),   # Light Blue
    (128, 255, 0),   # Lime
    (255, 20, 147),  # Deep Pink
    (0, 191, 255),   # Deep Sky Blue
    (255, 69, 0),    # Red Orange
    (50, 205, 50),   # Lime Green
    (138, 43, 226),  # Blue Violet
    (255, 140, 0)    # Dark Orange
]

# COCO class names (for models trained on COCO dataset)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
    'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

# MobileNet-SSD class names
MOBILENET_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

# Keyboard shortcuts
KEYBOARD_SHORTCUTS = {
    'quit': ['q', 27],  # 'q' or ESC
    'pause': [32],      # SPACE
    'screenshot': ['s'],
    'switch_model': ['m'],
    'toggle_confidence': ['c'],
    'toggle_tracking': ['t'],
    'toggle_performance': ['p'],
    'increase_confidence': ['=', '+'],
    'decrease_confidence': ['-'],
    'toggle_fullscreen': ['f'],
    'reset_settings': ['r']
}

# File paths
FILE_PATHS = {
    'screenshots_dir': 'screenshots',
    'models_dir': 'models',
    'logs_dir': 'logs',
    'config_file': 'config.json'
}
