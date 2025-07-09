
import sys
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                             QMessageBox, QComboBox, QSlider, QCheckBox, 
                             QStatusBar, QProgressBar)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QUrl, QTimer
from PyQt6.QtGui import QImage, QPixmap, QDesktopServices, QFont
import cv2
import numpy as np
import os
from pathlib import Path
import time

# Add src directory to Python path
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Import the core application logic
try:
    from src.main import ObjectUnderstandingApp
    MAIN_APP_AVAILABLE = True
except ImportError as e:
    print(f"Failed to import ObjectUnderstandingApp: {e}")
    ObjectUnderstandingApp = None
    MAIN_APP_AVAILABLE = False

class VideoThread(QThread):
    """
    Thread for processing video to prevent GUI from freezing.
    """
    change_pixmap_signal = pyqtSignal(np.ndarray)
    processing_finished_signal = pyqtSignal()
    stats_signal = pyqtSignal(dict)

    def __init__(self, input_source, app_logic=None):
        super().__init__()
        self._run_flag = True
        self.input_source = input_source
        self.app_logic = app_logic
        self.fps_counter = 0
        self.start_time = time.time()

    def run(self):
        """
        Capture video frames, process them, and emit the result.
        """
        cap = cv2.VideoCapture(self.input_source)
        if not cap.isOpened():
            print(f"Error: Could not open video source: {self.input_source}")
            self.processing_finished_signal.emit()
            return

        while self._run_flag and cap.isOpened():
            ret, frame = cap.read()
            if ret:
                # Process frame with object detection if app_logic is available
                if self.app_logic and MAIN_APP_AVAILABLE:
                    try:
                        # Get detections
                        detections = self.app_logic.detect_objects(frame)
                        
                        # Update tracker
                        rects = [det['bbox'] for det in detections]
                        tracked_objects = self.app_logic.tracker.update(rects)
                        
                        # Draw detections and tracking
                        processed_frame = self.app_logic.draw_detections(frame, detections, tracked_objects)
                        
                        # Update statistics
                        self.fps_counter += 1
                        elapsed_time = time.time() - self.start_time
                        if elapsed_time > 1.0:  # Update stats every second
                            fps = self.fps_counter / elapsed_time
                            stats = {
                                'fps': fps,
                                'detections': len(detections),
                                'tracked_objects': len(tracked_objects)
                            }
                            self.stats_signal.emit(stats)
                            self.fps_counter = 0
                            self.start_time = time.time()
                        
                        self.change_pixmap_signal.emit(processed_frame)
                    except Exception as e:
                        print(f"Error processing frame: {e}")
                        self.change_pixmap_signal.emit(frame)
                else:
                    # Just show raw frame if no object detection available
                    self.change_pixmap_signal.emit(frame)
            else:
                # End of video file
                self._run_flag = False
        
        cap.release()
        self.processing_finished_signal.emit()

    def stop(self):
        """Sets a flag to stop the thread."""
        self._run_flag = False
        self.wait()


class RealVisionGUI(QMainWindow):
    """
    Main GUI window for the RealVision application.
    """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RealVision - Object Understanding AI")
        self.setGeometry(100, 100, 1400, 900)
        
        # Initialize app logic
        self.app_logic = None
        if MAIN_APP_AVAILABLE:
            try:
                self.app_logic = ObjectUnderstandingApp()
                print("✓ ObjectUnderstandingApp initialized successfully")
            except Exception as e:
                print(f"✗ Failed to initialize ObjectUnderstandingApp: {e}")
                QMessageBox.warning(self, "Initialization Warning", 
                                  f"Failed to initialize object detection models:\n{e}\n\nYou can still view videos without object detection.")
        else:
            QMessageBox.warning(self, "Import Warning", 
                              "Could not import ObjectUnderstandingApp.\nObject detection will not be available.")

        self.setup_ui()
        self.video_thread = None
        
        # Status tracking
        self.current_fps = 0
        self.current_detections = 0
        self.current_tracked_objects = 0

    def setup_ui(self):
        """Setup the user interface"""
        # Central widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)

        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

        # Control panel at the top
        self.create_control_panel()
        
        # Video display in the center
        self.create_video_display()
        
        # Settings panel at the bottom
        self.create_settings_panel()

    def create_control_panel(self):
        """Create the main control panel"""
        control_frame = QWidget()
        control_layout = QHBoxLayout(control_frame)
        
        # Video controls
        video_group = QWidget()
        video_layout = QHBoxLayout(video_group)
        
        self.btn_open_camera = QPushButton("📹 Open Camera")
        self.btn_open_camera.clicked.connect(self.open_camera)
        self.btn_open_camera.setStyleSheet("QPushButton { font-size: 14px; padding: 10px; }")
        video_layout.addWidget(self.btn_open_camera)

        self.btn_upload_video = QPushButton("📁 Upload Video")
        self.btn_upload_video.clicked.connect(self.upload_video)
        self.btn_upload_video.setStyleSheet("QPushButton { font-size: 14px; padding: 10px; }")
        video_layout.addWidget(self.btn_upload_video)

        self.btn_stop_video = QPushButton("⏹️ Stop")
        self.btn_stop_video.clicked.connect(self.stop_video)
        self.btn_stop_video.setEnabled(False)
        self.btn_stop_video.setStyleSheet("QPushButton { font-size: 14px; padding: 10px; }")
        video_layout.addWidget(self.btn_stop_video)
        
        control_layout.addWidget(video_group)
        
        # Analytics controls
        analytics_group = QWidget()
        analytics_layout = QHBoxLayout(analytics_group)
        
        self.btn_show_analytics = QPushButton("📊 Show Analytics")
        self.btn_show_analytics.clicked.connect(self.show_analytics)
        self.btn_show_analytics.setStyleSheet("QPushButton { font-size: 14px; padding: 10px; }")
        analytics_layout.addWidget(self.btn_show_analytics)
        
        self.btn_generate_report = QPushButton("📋 Generate Report")
        self.btn_generate_report.clicked.connect(self.generate_report)
        self.btn_generate_report.setStyleSheet("QPushButton { font-size: 14px; padding: 10px; }")
        analytics_layout.addWidget(self.btn_generate_report)
        
        control_layout.addWidget(analytics_group)
        
        self.main_layout.addWidget(control_frame)

    def create_video_display(self):
        """Create the video display area"""
        # Video display
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setText("🎯 Welcome to RealVision AI\n\nClick 'Open Camera' or 'Upload Video' to start object detection")
        self.image_label.setStyleSheet("""
            QLabel { 
                background-color: #2b2b2b; 
                color: white; 
                font-size: 20px; 
                border: 2px solid #555555;
                border-radius: 10px;
                padding: 20px;
            }
        """)
        self.image_label.setMinimumHeight(500)
        self.main_layout.addWidget(self.image_label, stretch=1)
        
        # Stats display
        stats_frame = QWidget()
        stats_layout = QHBoxLayout(stats_frame)
        
        self.fps_label = QLabel("FPS: 0")
        self.fps_label.setStyleSheet("QLabel { font-size: 14px; color: #00ff00; font-weight: bold; }")
        stats_layout.addWidget(self.fps_label)
        
        self.detections_label = QLabel("Detections: 0")
        self.detections_label.setStyleSheet("QLabel { font-size: 14px; color: #ffaa00; font-weight: bold; }")
        stats_layout.addWidget(self.detections_label)
        
        self.tracked_label = QLabel("Tracked Objects: 0")
        self.tracked_label.setStyleSheet("QLabel { font-size: 14px; color: #0088ff; font-weight: bold; }")
        stats_layout.addWidget(self.tracked_label)
        
        stats_layout.addStretch()
        self.main_layout.addWidget(stats_frame)

    def create_settings_panel(self):
        """Create the settings panel"""
        settings_frame = QWidget()
        settings_layout = QHBoxLayout(settings_frame)
        
        # Model selection
        model_label = QLabel("Model:")
        settings_layout.addWidget(model_label)
        
        self.model_combo = QComboBox()
        if self.app_logic:
            self.model_combo.addItems(self.app_logic.available_models)
            if self.app_logic.current_model in self.app_logic.available_models:
                self.model_combo.setCurrentText(self.app_logic.current_model)
        else:
            self.model_combo.addItems(["No models available"])
        self.model_combo.currentTextChanged.connect(self.change_model)
        settings_layout.addWidget(self.model_combo)
        
        # Confidence threshold
        conf_label = QLabel("Confidence:")
        settings_layout.addWidget(conf_label)
        
        self.confidence_slider = QSlider(Qt.Orientation.Horizontal)
        self.confidence_slider.setRange(1, 100)
        self.confidence_slider.setValue(50)
        self.confidence_slider.valueChanged.connect(self.change_confidence)
        settings_layout.addWidget(self.confidence_slider)
        
        self.confidence_value_label = QLabel("0.50")
        settings_layout.addWidget(self.confidence_value_label)
        
        # Show options
        self.show_fps_check = QCheckBox("Show FPS")
        self.show_fps_check.setChecked(True)
        settings_layout.addWidget(self.show_fps_check)
        
        self.show_confidence_check = QCheckBox("Show Confidence")
        self.show_confidence_check.setChecked(True)
        settings_layout.addWidget(self.show_confidence_check)
        
        settings_layout.addStretch()
        self.main_layout.addWidget(settings_frame)

    def open_camera(self):
        """Start video capture from the default camera."""
        self.start_video_thread(0)

    def upload_video(self):
        """Open a file dialog to select a video file."""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "", 
            "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm)")
        if filepath:
            self.start_video_thread(filepath)

    def start_video_thread(self, input_source):
        """Starts the video processing thread."""
        if self.video_thread:
            self.video_thread.stop()
        
        self.video_thread = VideoThread(input_source, self.app_logic)
        self.video_thread.change_pixmap_signal.connect(self.update_image)
        self.video_thread.processing_finished_signal.connect(self.on_processing_finished)
        self.video_thread.stats_signal.connect(self.update_stats)
        self.video_thread.start()
        
        # Update UI state
        self.btn_stop_video.setEnabled(True)
        self.btn_open_camera.setEnabled(False)
        self.btn_upload_video.setEnabled(False)
        
        # Update status
        source_text = "Camera" if input_source == 0 else f"Video: {Path(input_source).name}"
        self.status_bar.showMessage(f"Processing {source_text}...")

    def stop_video(self):
        """Stop video processing."""
        if self.video_thread:
            self.video_thread.stop()
            self.status_bar.showMessage("Stopping...")

    def on_processing_finished(self):
        """Handle video processing completion."""
        self.btn_stop_video.setEnabled(False)
        self.btn_open_camera.setEnabled(True)
        self.btn_upload_video.setEnabled(True)
        self.image_label.setText("🎯 Video finished or stopped.\n\nClick 'Open Camera' or 'Upload Video' to start again.")
        self.image_label.setStyleSheet("""
            QLabel { 
                background-color: #2b2b2b; 
                color: white; 
                font-size: 20px; 
                border: 2px solid #555555;
                border-radius: 10px;
                padding: 20px;
            }
        """)
        self.status_bar.showMessage("Ready")
        
        # Reset stats
        self.update_stats({'fps': 0, 'detections': 0, 'tracked_objects': 0})

    def update_image(self, cv_img):
        """Updates the image_label with a new frame from the video thread."""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to a QPixmap."""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        p = QPixmap.fromImage(convert_to_Qt_format)
        return p.scaled(self.image_label.width(), self.image_label.height(), Qt.AspectRatioMode.KeepAspectRatio)

    def update_stats(self, stats):
        """Update the statistics display."""
        self.current_fps = stats.get('fps', 0)
        self.current_detections = stats.get('detections', 0)
        self.current_tracked_objects = stats.get('tracked_objects', 0)
        
        self.fps_label.setText(f"FPS: {self.current_fps:.1f}")
        self.detections_label.setText(f"Detections: {self.current_detections}")
        self.tracked_label.setText(f"Tracked Objects: {self.current_tracked_objects}")

    def change_model(self, model_name):
        """Change the object detection model."""
        if self.app_logic and model_name in self.app_logic.available_models:
            self.app_logic.current_model = model_name
            self.status_bar.showMessage(f"Switched to {model_name} model", 2000)

    def change_confidence(self, value):
        """Change the confidence threshold."""
        confidence = value / 100.0
        self.confidence_value_label.setText(f"{confidence:.2f}")
        if self.app_logic:
            self.app_logic.confidence_threshold = confidence

    def show_analytics(self):
        """Finds the latest HTML dashboard and opens it in the default web browser."""
        output_dir = project_root / "output"
        if not output_dir.exists():
            QMessageBox.warning(self, "Analytics Not Found", 
                              f"The output directory does not exist: {output_dir}")
            return

        dashboards = sorted(output_dir.glob("dashboard_*.html"), key=os.path.getmtime, reverse=True)
        if dashboards:
            latest_dashboard = dashboards[0]
            url = QUrl.fromLocalFile(str(latest_dashboard.resolve()))
            QDesktopServices.openUrl(url)
        else:
            QMessageBox.information(self, "Analytics Not Found", 
                                  "No analytics dashboard found in the 'output' directory.\n\n"
                                  "Process some video first to generate analytics data.")

    def generate_report(self):
        """Generate a new analytics report."""
        try:
            # Import the analytics modules
            from src.demo_analytics import generate_analytics_demo
            from src.performance_analyzer import ModelPerformanceAnalyzer
            
            # Generate analytics
            self.status_bar.showMessage("Generating analytics report...")
            generate_analytics_demo()
            self.status_bar.showMessage("Analytics report generated successfully", 3000)
            
            QMessageBox.information(self, "Report Generated", 
                                  "Analytics report has been generated successfully!\n\n"
                                  "Check the 'output' directory for the latest dashboard.")
        except Exception as e:
            QMessageBox.warning(self, "Report Generation Failed", 
                              f"Failed to generate analytics report:\n{e}")
            self.status_bar.showMessage("Report generation failed", 3000)

    def closeEvent(self, event):
        """Handle the window close event."""
        self.stop_video()
        event.accept()

def install_requirements():
    """Install required packages if not available."""
    required_packages = [
        'PyQt6',
        'opencv-python',
        'numpy',
        'ultralytics',
        'pandas',
        'matplotlib',
        'seaborn'
    ]
    
    import subprocess
    import sys
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])

if __name__ == '__main__':
    # Check and install requirements
    try:
        from PyQt6 import QtWidgets
    except ImportError:
        print("PyQt6 not found. Installing required packages...")
        install_requirements()
        print("Packages installed successfully. Please run the script again.")
        sys.exit(0)

    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("RealVision Object Understanding AI")
    app.setApplicationVersion("1.0")
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show main window
    main_window = RealVisionGUI()
    main_window.show()
    
    # Run application
    sys.exit(app.exec())
