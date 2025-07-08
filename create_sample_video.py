#!/usr/bin/env python3
"""
Generate a sample video for testing the object detection application
"""

import cv2
import numpy as np
import os
from math import sin, cos, pi

def create_sample_video():
    """Create a sample video with moving objects"""
    
    # Video parameters
    width, height = 1280, 720
    fps = 30
    duration = 10  # seconds
    total_frames = fps * duration
    
    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = 'sample_video.mp4'
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"🎬 Creating sample video: {output_path}")
    print(f"   Resolution: {width}x{height}")
    print(f"   Duration: {duration} seconds")
    print(f"   FPS: {fps}")
    
    for frame_num in range(total_frames):
        # Create a blank frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add gradient background
        for y in range(height):
            intensity = int(50 + 30 * sin(2 * pi * y / height + frame_num * 0.1))
            frame[y, :] = (intensity // 3, intensity // 2, intensity)
        
        # Add moving objects that look like real objects
        t = frame_num / fps  # time in seconds
        
        # Moving car-like rectangle
        car_x = int(50 + 200 * sin(t * 0.5))
        car_y = int(height // 2 + 50 * cos(t * 0.3))
        car_w, car_h = 120, 60
        cv2.rectangle(frame, (car_x, car_y), (car_x + car_w, car_y + car_h), (0, 0, 255), -1)
        cv2.rectangle(frame, (car_x + 10, car_y + 40), (car_x + 30, car_y + 60), (0, 0, 0), -1)  # wheel
        cv2.rectangle(frame, (car_x + 90, car_y + 40), (car_x + 110, car_y + 60), (0, 0, 0), -1)  # wheel
        
        # Moving person-like figure
        person_x = int(width - 100 - 300 * sin(t * 0.7))
        person_y = int(height // 2 + 100 * cos(t * 0.4))
        # Head
        cv2.circle(frame, (person_x, person_y), 20, (255, 220, 180), -1)
        # Body
        cv2.rectangle(frame, (person_x - 15, person_y + 20), (person_x + 15, person_y + 80), (100, 100, 255), -1)
        # Arms
        cv2.rectangle(frame, (person_x - 35, person_y + 30), (person_x - 15, person_y + 60), (255, 220, 180), -1)
        cv2.rectangle(frame, (person_x + 15, person_y + 30), (person_x + 35, person_y + 60), (255, 220, 180), -1)
        # Legs
        cv2.rectangle(frame, (person_x - 10, person_y + 80), (person_x, person_y + 120), (0, 0, 100), -1)
        cv2.rectangle(frame, (person_x, person_y + 80), (person_x + 10, person_y + 120), (0, 0, 100), -1)
        
        # Moving bottle-like object
        bottle_x = int(width // 2 + 150 * cos(t * 0.8))
        bottle_y = int(100 + 50 * sin(t * 0.6))
        cv2.rectangle(frame, (bottle_x, bottle_y), (bottle_x + 20, bottle_y + 60), (0, 255, 0), -1)
        cv2.rectangle(frame, (bottle_x + 5, bottle_y - 10), (bottle_x + 15, bottle_y + 5), (0, 200, 0), -1)
        
        # Add some text info
        cv2.putText(frame, f"Frame: {frame_num + 1}/{total_frames}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, f"Time: {t:.1f}s", (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(frame, "Sample Objects: Car, Person, Bottle", (10, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Write frame to video
        out.write(frame)
        
        # Show progress
        if frame_num % 30 == 0:
            progress = (frame_num + 1) / total_frames * 100
            print(f"   Progress: {progress:.1f}%")
    
    # Release everything
    out.release()
    
    print(f"✅ Sample video created successfully: {output_path}")
    print(f"   File size: {os.path.getsize(output_path) / (1024*1024):.1f} MB")
    return output_path

if __name__ == "__main__":
    video_path = create_sample_video()
    print(f"\n💡 You can now test the application with:")
    print(f"   python run.py --input {video_path}")
