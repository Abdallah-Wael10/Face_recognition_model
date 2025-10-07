# config.py
import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
EMPLOYEES_DIR = BASE_DIR / "Employees"
LOGS_DIR = BASE_DIR / "logs"

# Camera Configuration
RTSP_URL = "rtsp://admin:Attya%402023@192.168.1.123:554/Streaming/Channels/101"
CAMERA_CONFIG = {
    'width': 1280,
    'height': 720,
    'fps': 25,
    'buffer_size': 1,
    'timeout': 10,
    'reconnect_delay': 2
}

# Model Configuration
MODEL_CONFIG = {
    'model_path': MODELS_DIR / 'trained_model.pkl',
    'dataset_path': EMPLOYEES_DIR,
    'tolerance': 0.5,
    'upscale_factor': 0.25
}

# Streaming Optimization
STREAM_CONFIG = {
    'virtual_cam_width': 1280,
    'virtual_cam_height': 720,
    'virtual_cam_fps': 25,
    'preview_scale': 0.5
}

# Detection Settings
DETECTION_CONFIG = {
    'process_every_n_frames': 2,
    'min_face_size': 50,
    'confidence_threshold': 0.7,
    'log_interval': 60  # Log each person once per minute
}

# Server Configuration
SERVER_CONFIG = {
    'host': '0.0.0.0',
    'port': 5000,
    'debug': False,
    'threaded': True
}

# Database Configuration
DATABASE_CONFIG = {
    'detection_log': 'detections.db',
    'backup_interval': 3600  # Backup every hour
}

# Performance Settings
PERFORMANCE_CONFIG = {
    'max_frame_queue': 10,
    'frame_timeout': 5,
    'gstreamer_enabled': True
}


# Backend Integration
BACKEND = {
    'base_url': 'http://localhost:5001',  # TODO
    'token': 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI2OGMwMzMxZTMzMjQzM2JmMWQ0NDMwNGYiLCJlbWFpbCI6ImJlZG93YWVsMzY1QGdtYWlsLmNvbSIsInJvbGUiOiJhZG1pbiIsImZpcnN0TmFtZSI6ImJlZG8iLCJsYXN0TmFtZSI6IndhZWwiLCJ0eXBlIjoiYWNjZXNzIiwiaWF0IjoxNzU5ODMxNjg5LCJleHAiOjE3NjAwMDQ0ODl9.OjbdtzwcljHP-3W044zxafDnau-vtr7hZ78Ui3TpgZY',                             # TODO
    'camera_id': 'Hikvision_001',
    'timeout': 5,
    'retries': 3
}

# Encodings path
ENCODINGS_PATH = str(BASE_DIR / 'model' / 'encodings_fixed.pickle')

# Attendance/event logic
EVENTS = {
    'clock_in_cooldown_sec': 60,
    'clock_out_cooldown_sec': 60,
    'gesture_window_sec': 3,
    'attach_image': False  # set True to send frame snapshots
}