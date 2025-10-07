# run.py
import argparse
import sys
from pathlib import Path

from core.face_detector import OptimizedFaceDetector, CameraStream
from core.database import DetectionDB
from web.app import app, initialize_system

def start_live_stream():
    """Start live stream with face detection"""
    print("🎥 Starting live stream...")
    from core.face_detector import main as stream_main
    stream_main()

def start_virtual_camera():
    """Start virtual camera stream"""
    print("🖥️ Starting virtual camera...")
    from core.camera_stream import main as virtual_cam_main
    virtual_cam_main()

def start_web_server(port=5000):
    """Start the web server"""
    print(f"🌐 Starting web server on port {port}...")
    initialize_system()
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)

def main():
    parser = argparse.ArgumentParser(description='Hikvision Face Recognition System')
    parser.add_argument('--mode', 
                       choices=['stream', 'server', 'virtual-cam'], 
                       default='server',
                       help='Operation mode: stream, server, virtual-cam')
    parser.add_argument('--port', type=int, default=5000, help='Web server port')
    
    args = parser.parse_args()
    
    # Create necessary directories
    Path("models").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    
    # Execute selected mode
    if args.mode == 'stream':
        start_live_stream()
    elif args.mode == 'virtual-cam':
        start_virtual_camera()
    elif args.mode == 'server':
        start_web_server(args.port)

if __name__ == "__main__":
    main()