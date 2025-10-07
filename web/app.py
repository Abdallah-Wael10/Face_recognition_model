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
from core.backend_client import BackendClient
from core.gesture_detector import GestureDetector
import config
app = Flask(__name__)

# Global variables
camera_stream = None
face_detector = None
database = None
backend_client = None
gesture_detector = None
current_detections = []
system_initialized = False

# Debounce and correlation state
_last_clock_in = {}
_last_clock_out = {}
_pending_face = {}

def initialize_system():
    global camera_stream, face_detector, database, backend_client, gesture_detector, system_initialized
    
    print("🚀 Initializing face recognition system...")
    
    try:
        # Initialize components
        database = DetectionDB()
        backend_client = BackendClient()
        gesture_detector = GestureDetector()
        
        # Load your existing encodings
        face_detector = OptimizedFaceDetector()
        
        # Start camera stream - Use 0 for webcam or RTSP URL
        camera_stream = CameraStream(rtsp_url=config.RTSP_URL)
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

def _can_fire(name: str, last_map: dict, cooldown_sec: int) -> bool:
    now = time.time()
    return (name not in last_map) or (now - last_map[name] > cooldown_sec)

def _mark_fired(name: str, last_map: dict):
    last_map[name] = time.time()

def generate_frames():
    """Generate frames with face detection, gesture detection, and backend events"""
    while True:
        if not system_initialized or camera_stream is None:
            time.sleep(0.1)
            continue
        
        frame = camera_stream.get_frame()
        if frame is None:
            time.sleep(0.1)
            continue
        
        try:
            # Face recognition
            processed_frame, faces = face_detector.recognize_faces(frame)
            # Gesture detection
            gesture = gesture_detector.detect(processed_frame) if gesture_detector else {'is_open_palm': False}

            # Update current detections for API
            global current_detections
            current_detections = [f['name'] for f in faces]

            # Pick the first known face as primary
            primary = next((f for f in faces if f['name'] != "Unknown"), None)
            now = time.time()
            if primary:
                name = primary['name']
                conf = float(primary.get('confidence', 0.0))

                # Remember last seen time for correlation window
                _pending_face[name] = now

                # CLOCK IN (debounced)
                if _can_fire(name, _last_clock_in, config.EVENTS['clock_in_cooldown_sec']):
                    database.log_detection(name, confidence=conf)
                    backend_client.send_event(
                        employee_id=None,  # map face name to ID if needed
                        event_type="CLOCK_IN",
                        confidence=conf,
                        context={
                            'frameId': f"{int(now*1000)}-{name}",
                            'gesture': None
                        },
                        image_bgr=None,
                        face_name=name
                    )
                    _mark_fired(name, _last_clock_in)

                # CLOCK OUT when open palm within gesture window (debounced)
                if gesture.get('is_open_palm'):
                    seen_at = _pending_face.get(name)
                    if seen_at and (now - seen_at) <= config.EVENTS['gesture_window_sec']:
                        if _can_fire(name, _last_clock_out, config.EVENTS['clock_out_cooldown_sec']):
                            backend_client.send_event(
                                employee_id=None,
                                event_type="CLOCK_OUT",
                                confidence=conf,
                                context={
                                    'frameId': f"{int(now*1000)}-{name}-out",
                                    'gesture': 'OPEN_PALM'
                                },
                                image_bgr=None,
                                face_name=name
                            )
                            _mark_fired(name, _last_clock_out)

            # Encode frame and stream
            ret, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        except Exception as e:
            print(f"❌ Error processing frame: {e}")
            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

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