# utils/logger.py
import logging
import os
from datetime import datetime
from config import LOGS_DIR

def setup_logger(name, log_file, level=logging.INFO):
    """Setup a logger with file and console handlers"""
    
    # Create logs directory if it doesn't exist
    LOGS_DIR.mkdir(exist_ok=True)
    
    log_file_path = LOGS_DIR / log_file
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    handler = logging.FileHandler(log_file_path)
    handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    logger.addHandler(console_handler)
    
    return logger

# Create loggers
app_logger = setup_logger('app', 'app.log')
detection_logger = setup_logger('detection', 'detections.log')