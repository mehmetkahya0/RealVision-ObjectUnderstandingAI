#!/usr/bin/env python3
"""
Demo script for testing the Real-Time Object Understanding Application
with a sample video file or webcam (with proper permissions).
"""

import os
import sys
import cv2
import numpy as np
import urllib.request
from pathlib import Path

def download_sample_video():
    """Download a sample video for testing"""
    video_url = "https://sample-videos.com/zip/10/mp4/SampleVideo_1280x720_1mb.mp4"
    video_filename = "sample_video.mp4"
    
    if not os.path.exists(video_filename):
        print("📥 Downloading sample video for testing...")
        try:
            urllib.request.urlretrieve(video_url, video_filename)
            print(f"✅ Sample video downloaded: {video_filename}")
            return video_filename
        except Exception as e:
            print(f"❌ Failed to download sample video: {e}")
            return None
    else:
        print(f"✅ Using existing sample video: {video_filename}")
        return video_filename

def create_synthetic_test_video():
    """Create a synthetic test video with moving objects"""
    print("🎬 Creating synthetic test video...")
    
    video_filename = "test_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_filename, fourcc, 20.0, (640, 480))
    
    # Create 200 frames of synthetic content
    for frame_num in range(200):
        # Create a frame with moving objects
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Background gradient
        for y in range(480):
            for x in range(640):
                frame[y, x] = [50 + (x + frame_num) % 100, 
                              30 + (y + frame_num) % 80, 
                              70 + (frame_num * 2) % 120]
        
        # Moving circle (simulating a ball)
        center_x = int(320 + 200 * np.sin(frame_num * 0.1))
        center_y = int(240 + 100 * np.cos(frame_num * 0.05))
        cv2.circle(frame, (center_x, center_y), 30, (0, 255, 255), -1)
        
        # Moving rectangle (simulating a car)
        rect_x = int(50 + (frame_num * 3) % 500)
        rect_y = int(350)
        cv2.rectangle(frame, (rect_x, rect_y), (rect_x + 80, rect_y + 40), (255, 0, 0), -1)
        
        # Static objects
        cv2.rectangle(frame, (500, 50), (580, 150), (0, 255, 0), -1)  # Building
        cv2.circle(frame, (100, 100), 25, (255, 255, 0), -1)  # Sun
        
        out.write(frame)
    
    out.release()
    print(f"✅ Synthetic test video created: {video_filename}")
    return video_filename

def test_object_detection_with_video(video_path):
    """Test object detection with a video file"""
    print(f"🎯 Testing object detection with video: {video_path}")
    
    # Import the main application
    try:
        from main import ObjectUnderstandingApp
    except ImportError as e:
        print(f"❌ Failed to import main application: {e}")
        return False
    
    try:
        # Create a modified version of the app that works with video files
        app = ObjectUnderstandingApp()
        
        # Open video file instead of camera
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ Failed to open video: {video_path}")
            return False
        
        print("🚀 Starting object detection test...")
        print("Press 'Q' to quit, 'SPACE' to pause/resume")
        print("Use '+'/'-' to adjust confidence threshold")
        
        frame_count = 0
        paused = False
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    # Loop the video
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                
                frame_count += 1
                
                # Detect objects
                detections = app.detect_objects(frame)
                
                # Update tracking
                rects = [det['bbox'] for det in detections]
                tracked_objects = app.tracker.update(rects)
                
                # Draw results
                frame_with_detections = app.draw_detections(frame, detections, tracked_objects)
                
                # Add simple info overlay
                info_text = f"Frame: {frame_count} | Detections: {len(detections)} | Model: {app.current_model}"
                cv2.putText(frame_with_detections, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display frame
                cv2.imshow('Object Detection Test', frame_with_detections)
            
            # Handle keyboard input
            key = cv2.waitKey(30) & 0xFF
            
            if key == ord('q') or key == 27:  # 'q' or ESC
                break
            elif key == ord(' '):  # SPACE
                paused = not paused
                print(f"{'Paused' if paused else 'Resumed'}")
            elif key == ord('=') or key == ord('+'):  # Increase confidence
                app.confidence_threshold = min(0.95, app.confidence_threshold + 0.05)
                print(f"Confidence threshold: {app.confidence_threshold:.2f}")
            elif key == ord('-'):  # Decrease confidence
                app.confidence_threshold = max(0.05, app.confidence_threshold - 0.05)
                print(f"Confidence threshold: {app.confidence_threshold:.2f}")
            elif key == ord('m'):  # Switch model
                app.current_model = "dnn" if app.current_model == "yolo" else "yolo"
                print(f"Switched to {app.current_model.upper()} model")
        
        cap.release()
        cv2.destroyAllWindows()
        print("✅ Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Main demo function"""
    print("🎯 Real-Time Object Understanding - Demo Mode")
    print("=" * 50)
    
    # Check if we can access camera
    print("📹 Checking camera access...")
    cap = cv2.VideoCapture(0)
    camera_available = cap.isOpened()
    if camera_available:
        ret, _ = cap.read()
        camera_available = ret
    cap.release()
    
    if camera_available:
        print("✅ Camera is accessible!")
        print("You can run the full application with: python run.py")
        
        response = input("Do you want to test with camera (c) or video file (v)? [v]: ").lower()
        if response == 'c':
            print("Starting camera mode...")
            from main import ObjectUnderstandingApp
            app = ObjectUnderstandingApp()
            app.run()
            return
    else:
        print("⚠️  Camera not accessible (permission or hardware issue)")
        print("Testing with video file instead...")
    
    # Try to get a test video
    video_path = None
    
    # First try to create a synthetic video
    if not video_path:
        try:
            video_path = create_synthetic_test_video()
        except Exception as e:
            print(f"Failed to create synthetic video: {e}")
    
    # Try to download a sample video
    if not video_path:
        try:
            video_path = download_sample_video()
        except Exception as e:
            print(f"Failed to download sample video: {e}")
    
    if video_path and os.path.exists(video_path):
        # Test with video file
        success = test_object_detection_with_video(video_path)
        if success:
            print("\n🎉 Demo completed successfully!")
            print("\nNext steps:")
            print("1. Grant camera permissions if you want to use webcam")
            print("2. Run 'python run.py' for the full interactive application")
            print("3. Check the README.md for more features and options")
        else:
            print("\n❌ Demo failed. Check the error messages above.")
    else:
        print("❌ Could not obtain test video. Please check your internet connection.")
        print("\nAlternative: You can run 'python run.py' if camera permissions are granted.")

if __name__ == "__main__":
    main()
