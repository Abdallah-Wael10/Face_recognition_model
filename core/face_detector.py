# core/face_detector.py - COMPLETE FIXED VERSION
import os
import face_recognition
import cv2
import pickle
import numpy as np
import time
import threading
import config

class OptimizedFaceDetector:
    def __init__(self, encodings_path=None):
        encodings_path = encodings_path or config.ENCODINGS_PATH
        print("🔄 Loading face recognition model...")
        print(f"🔍 Trying to load encodings from: {encodings_path}")
        print(f"🔍 File exists: {os.path.exists(encodings_path)}")
        
        # Initialize attributes first
        self.known_face_encodings = []
        self.known_face_names = []
        self.frame_count = 0
        
        if not os.path.exists(encodings_path):
            print("❌ Encodings file not found! System will run in face detection mode only.")
            return
        
        try:
            # Try to load encodings
            with open(encodings_path, 'rb') as f:
                encodings_data = pickle.load(f)
            
            print(f"✅ Successfully loaded encodings file!")
            print(f"🔍 Data type: {type(encodings_data)}")
            
            if isinstance(encodings_data, dict):
                self.known_face_encodings = encodings_data.get('encodings', [])
                self.known_face_names = encodings_data.get('names', [])
                print(f"📊 Loaded {len(self.known_face_encodings)} encodings")
                print(f"👥 Names found: {list(set(self.known_face_names))}")
                
            elif isinstance(encodings_data, list):
                self.known_face_encodings = encodings_data
                self.known_face_names = [f"Person_{i}" for i in range(len(encodings_data))]
                print(f"📊 List format - Loaded {len(self.known_face_encodings)} encodings")
            else:
                print(f"❌ Unknown data format: {type(encodings_data)}")
            
        except Exception as e:
            print(f"❌ Error loading encodings: {e}")
            import traceback
            traceback.print_exc()
        
    def recognize_faces(self, frame):
        self.frame_count += 1
        
        # If no encodings loaded, just detect faces without recognition
        if not self.known_face_encodings:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            results = []
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(frame, "Unknown (No encodings)", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                results.append({'name': 'Unknown', 'confidence': 0.0, 'box': (x, y, x+w, y+h)})
            
            cv2.putText(frame, f"Faces: {len(results)}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "NO ENCODINGS LOADED", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            return frame, results
        
        # If encodings are loaded, use face_recognition
        try:
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            
            out = []
            for (face_encoding, (top, right, bottom, left)) in zip(face_encodings, face_locations):
                name = "Unknown"
                confidence = 0.0
                
                matches = face_recognition.compare_faces(
                    self.known_face_encodings, 
                    face_encoding, 
                    tolerance=0.6
                )
                
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                
                if len(face_distances) > 0:
                    best_match_index = int(np.argmin(face_distances))
                    best_distance = float(face_distances[best_match_index])
                    confidence = max(0.0, 1.0 - best_distance)
                    if matches[best_match_index] and confidence > 0.5:
                        name = self.known_face_names[best_match_index]
                
                # scale back up to original frame size
                top *= 2; right *= 2; bottom *= 2; left *= 2
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6), 
                           cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
                out.append({'name': name, 'confidence': confidence, 'box': (left, top, right, bottom)})
            
            cv2.putText(frame, f"Faces: {len(out)}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            return frame, out
            
        except Exception as e:
            print(f"❌ Error in face recognition: {e}")
            import traceback
            traceback.print_exc()
            return frame, []

class CameraStream:
    def __init__(self, rtsp_url=0):
        self.rtsp_url = rtsp_url
        self.cap = None
        self.frame = None
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        
    def start(self):
        """Start the camera stream"""
        print(f"📡 Starting camera stream: {self.rtsp_url}")
        
        try:
            self.cap = cv2.VideoCapture(self.rtsp_url)
            if not self.cap.isOpened():
                print(f"❌ Failed to open camera: {self.rtsp_url}")
                return False
                
            self.running = True
            self.thread = threading.Thread(target=self._update_frame, daemon=True)
            self.thread.start()
            
            # Wait for first frame
            for i in range(50):
                if self.frame is not None:
                    print("✅ Camera stream ready")
                    return True
                time.sleep(0.1)
                
            print("❌ Camera stream timeout")
            return False
            
        except Exception as e:
            print(f"❌ Camera start error: {e}")
            return False
        
    def _update_frame(self):
        """Continuously capture frames"""
        while self.running:
            try:
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    with self.lock:
                        self.frame = frame
                else:
                    print("⚠️ Failed to read frame")
                    time.sleep(0.1)
            except Exception as e:
                print(f"⚠️ Frame read error: {e}")
                time.sleep(0.1)
                
    def get_frame(self):
        """Get current frame"""
        with self.lock:
            return self.frame.copy() if self.frame is not None else None
            
    def stop(self):
        """Stop camera stream"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
        print("✅ Camera stream stopped")             