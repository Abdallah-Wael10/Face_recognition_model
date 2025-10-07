# utils/helpers.py
import cv2
import numpy as np
from datetime import datetime
import json

def current_timestamp():
    """Get current timestamp in readable format"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def draw_text_with_background(img, text, position, 
                             font_face=cv2.FONT_HERSHEY_SIMPLEX, 
                             font_scale=0.7, color=(255, 255, 255), 
                             bg_color=(0, 0, 0), thickness=2):
    """Draw text with background for better visibility"""
    x, y = position
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font_face, font_scale, thickness
    )
    
    # Draw background rectangle
    cv2.rectangle(img, 
                 (x, y - text_height - baseline), 
                 (x + text_width, y + baseline), 
                 bg_color, 
                 -1)
    
    # Draw text
    cv2.putText(img, text, (x, y), 
               font_face, font_scale, color, thickness)
    
    return img

def calculate_iou(box1, box2):
    """Calculate Intersection over Union for two boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0