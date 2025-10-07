# test_complete_system_fixed.py
import cv2
import sys
import os
import time
import numpy as np

# Add the core directory to path
sys.path.append('core')

from face_detector import OptimizedFaceDetector, CameraStream
from config import RTSP_URL

def test_complete_system():
    print("🚀 Testing Complete Face Recognition System")
    print("=" * 50)
    
    # Initialize components
    print("1. Initializing face detector...")
    face_detector = OptimizedFaceDetector()
    
    print("2. Starting camera stream...")
    camera_stream = CameraStream(RTSP_URL)
    camera_stream.start()
    
    # Wait for camera to start and get first frame
    print("3. Waiting for camera frames...")
    frame_wait_attempts = 0
    max_wait_attempts = 50  # Wait up to 5 seconds
    
    while frame_wait_attempts < max_wait_attempts:
        frame = camera_stream.get_frame()
        # Check if frame is valid (not None and has proper shape)
        if frame is not None and hasattr(frame, 'shape') and len(frame.shape) == 3:
            print("✅ Camera is providing frames!")
            break
        frame_wait_attempts += 1
        time.sleep(0.1)
    
    if frame_wait_attempts >= max_wait_attempts:
        print("❌ Camera not providing valid frames after waiting")
        camera_stream.stop()
        return False
    
    print("4. Starting face recognition test...")
    print("   Press 'q' to quit, 's' to save current frame")
    
    frame_count = 0
    no_frame_count = 0
    max_no_frame = 30  # Stop if 30 consecutive frames fail
    
    try:
        while True:
            frame = camera_stream.get_frame()
            if frame is not None and hasattr(frame, 'shape') and len(frame.shape) == 3:
                no_frame_count = 0  # Reset counter
                
                # Process frame
                processed_frame, names = face_detector.recognize_faces(frame)
                
                # Display frame
                display_frame = cv2.resize(processed_frame, (960, 540))  # Resize for display
                cv2.imshow('Hikvision Face Recognition - Press Q to quit', display_frame)
                
                # Print detection info
                if names:
                    print(f"Frame {frame_count}: Detected {len(names)} faces: {names}")
                elif frame_count % 30 == 0:  # Print status every 30 frames
                    print(f"Frame {frame_count}: No faces detected")
                
                frame_count += 1
                
            else:
                no_frame_count += 1
                if no_frame_count >= max_no_frame:
                    print("❌ Too many consecutive frame failures, stopping...")
                    break
                elif no_frame_count % 10 == 0:
                    print(f"⚠️ No valid frame for {no_frame_count} attempts")
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and frame is not None:
                # Save current frame
                filename = f"debug_frame_{frame_count}.jpg"
                cv2.imwrite(filename, frame)
                print(f"💾 Saved frame to {filename}")
    
    except KeyboardInterrupt:
        print("⏹️ Stopped by user")
    except Exception as e:
        print(f"❌ Error during face recognition: {e}")
    
    finally:
        # Cleanup
        camera_stream.stop()
        cv2.destroyAllWindows()
        print(f"✅ Test completed. Processed {frame_count} frames.")
        return True

if __name__ == "__main__":
    test_complete_system()