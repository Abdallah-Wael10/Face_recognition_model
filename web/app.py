# app.py - UPDATED
from flask import Flask, render_template, Response, jsonify
import cv2
import threading
import time
from datetime import datetime
import os
import sys
import numpy as np

RTSP_URL = "rtsp://admin:Attya%402023@192.168.1.123:554/Streaming/Channels/101"

# Add the parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.face_detector import OptimizedFaceDetector, CameraStream
from core.database import DetectionDB
import config
app = Flask(__name__)

# Global variables
camera_stream = None
face_detector = None
database = None
current_detections = []
system_initialized = False

def initialize_system():
    global camera_stream, face_detector, database, system_initialized
    
    print("🚀 Initializing face recognition system...")
    
    try:
        # Initialize components
        database = DetectionDB()
        
        # Load your existing encodings
        face_detector = OptimizedFaceDetector()
        
        # Start camera stream - Use 0 for webcam or RTSP URL
        camera_stream = CameraStream(rtsp_url=RTSP_URL)
        success = camera_stream.start()
        
        if not success:
            print("❌ Camera stream failed to initialize")
            return False
        
        system_initialized = True
        print("✅ System initialized successfully")
        return True
        
    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        return False

def generate_frames():
    """Generate frames with face detection for streaming"""
    last_detection_log = {}
    frame_count = 0
    
    while True:
        if not system_initialized or camera_stream is None:
            time.sleep(0.1)
            continue
            
        frame = camera_stream.get_frame()
        if frame is not None:
            try:
                # Process frame with face detection
                processed_frame, detected_names = face_detector.recognize_faces(frame)
                
                # Log new detections
                current_time = time.time()
                for name in detected_names:
                    if name != "Unknown":
                        # Log only once per minute per person
                        if (name not in last_detection_log or 
                            current_time - last_detection_log[name] > 60):
                            database.log_detection(name, confidence=0.95)
                            last_detection_log[name] = current_time
                            print(f"👤 Detected: {name}")
                
                # Update current detections for API
                global current_detections
                current_detections = detected_names
                
                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', processed_frame, 
                                         [cv2.IMWRITE_JPEG_QUALITY, 80])
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                       
            except Exception as e:
                print(f"❌ Error processing frame: {e}")
                # Return original frame on error
                ret, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
            # No frame available
            time.sleep(0.1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/detections')
def get_detections():
    if database is None:
        return jsonify({'error': 'Database not initialized'})
    
    recent = database.get_recent_detections(limit=20)
    stats = database.get_detection_stats()
    
    return jsonify({
        'recent_detections': recent,
        'statistics': stats,
        'current_detections': current_detections,
        'system_initialized': system_initialized,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/stats')
def get_stats():
    if database is None:
        return jsonify({'error': 'Database not initialized'})
    
    stats = database.get_detection_stats()
    return jsonify(stats)

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/debug')
def debug():
    """Debug endpoint to check system status"""
    camera_status = "Active" if (camera_stream and camera_stream.get_frame() is not None) else "Inactive"
    encodings_loaded = len(face_detector.known_face_encodings) if face_detector else 0
    
    return jsonify({
        'system_initialized': system_initialized,
        'camera_status': camera_status,
        'encodings_loaded': encodings_loaded,
        'current_detections': current_detections,
        'database_connected': database is not None
    })

if __name__ == '__main__':
    if initialize_system():
        print("🌐 Starting Flask server...")
        app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
    else:
        print("❌ Failed to initialize system")