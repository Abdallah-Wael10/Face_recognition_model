# debug_system.py
import cv2
import face_recognition
import pickle
import numpy as np
import os
import sys

def debug_encodings():
    print("=" * 60)
    print("🔍 DEBUGGING ENCODINGS FILE")
    print("=" * 60)
    
    encodings_path = r'D:\Bureau_Tutorials\Hikvision-Face-Recognition\model\encodings.pickle'
    
    if not os.path.exists(encodings_path):
        print("❌ ENCODINGS FILE NOT FOUND!")
        return False
    
    try:
        with open(encodings_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"✅ Encodings file loaded successfully")
        
        if isinstance(data, dict):
            encodings = data.get('encodings', [])
            names = data.get('names', [])
            print(f"📊 Format: Dictionary")
            print(f"👤 Encodings loaded: {len(encodings)}")
            print(f"🏷️ Names loaded: {len(names)}")
            print(f"👥 Unique names: {set(names)}")
            
        elif isinstance(data, list):
            print(f"📊 Format: List with {len(data)} items")
            encodings = data
            names = [f"Person_{i}" for i in range(len(data))]
        else:
            print(f"❓ Unknown format: {type(data)}")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Error loading encodings: {e}")
        return False

def debug_camera():
    print("\n" + "=" * 60)
    print("📷 DEBUGGING CAMERA")
    print("=" * 60)
    
    # Test webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ WEBCAM NOT DETECTED!")
        return False
    
    print("✅ Webcam detected")
    
    # Try to read a frame
    ret, frame = cap.read()
    if not ret:
        print("❌ Cannot read frame from webcam")
        cap.release()
        return False
    
    print(f"✅ Frame captured: {frame.shape}")
    cap.release()
    return True

def debug_face_detection():
    print("\n" + "=" * 60)
    print("👤 DEBUGGING FACE DETECTION")
    print("=" * 60)
    
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    
    if not ret:
        print("❌ Cannot capture frame for face detection test")
        return False
    
    # Test face detection
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame, model="hog")
    
    print(f"🔍 Faces detected: {len(face_locations)}")
    
    if len(face_locations) > 0:
        print("✅ Face detection is working!")
        for i, (top, right, bottom, left) in enumerate(face_locations):
            print(f"   Face {i+1}: Top={top}, Right={right}, Bottom={bottom}, Left={left}")
    else:
        print("❌ No faces detected in frame")
        print("💡 Tips:")
        print("   - Make sure you're facing the camera")
        print("   - Ensure good lighting")
        print("   - Check if camera lens is covered")
    
    cap.release()
    return len(face_locations) > 0

def debug_face_recognition():
    print("\n" + "=" * 60)
    print("🎯 DEBUGGING FACE RECOGNITION")
    print("=" * 60)
    
    # Load encodings
    encodings_path = r'D:\Bureau_Tutorials\Hikvision-Face-Recognition\model\encodings.pickle'
    
    try:
        with open(encodings_path, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, dict):
            known_encodings = data.get('encodings', [])
            known_names = data.get('names', [])
        else:
            known_encodings = data
            known_names = [f"Person_{i}" for i in range(len(data))]
        
        if len(known_encodings) == 0:
            print("❌ No encodings available for recognition")
            return False
        
        print(f"✅ Loaded {len(known_encodings)} encodings for recognition test")
        
        # Test with camera
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        
        if ret:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            print(f"🔍 Found {len(face_encodings)} face encodings to compare")
            
            for i, face_encoding in enumerate(face_encodings):
                matches = face_recognition.compare_faces(known_encodings, face_encoding)
                face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                
                if len(face_distances) > 0:
                    best_match_index = np.argmin(face_distances)
                    best_distance = face_distances[best_match_index]
                    is_match = matches[best_match_index]
                    
                    print(f"   Face {i+1}: Best match index: {best_match_index}")
                    print(f"   Face {i+1}: Best distance: {best_distance:.4f}")
                    print(f"   Face {i+1}: Is match: {is_match}")
                    
                    if is_match:
                        name = known_names[best_match_index]
                        print(f"   ✅ RECOGNIZED: {name} (confidence: {1-best_distance:.2f})")
                    else:
                        print(f"   ❌ Unknown person (closest distance: {best_distance:.4f})")
        
        cap.release()
        return True
        
    except Exception as e:
        print(f"❌ Face recognition test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 STARTING COMPREHENSIVE SYSTEM DEBUG")
    print("=" * 60)
    
    # Test each component
    encodings_ok = debug_encodings()
    camera_ok = debug_camera()
    detection_ok = debug_face_detection()
    recognition_ok = debug_face_recognition()
    
    print("\n" + "=" * 60)
    print("📋 DEBUG SUMMARY")
    print("=" * 60)
    print(f"📁 Encodings: {'✅ OK' if encodings_ok else '❌ FAILED'}")
    print(f"📷 Camera: {'✅ OK' if camera_ok else '❌ FAILED'}")
    print(f"👤 Face Detection: {'✅ OK' if detection_ok else '❌ FAILED'}")
    print(f"🎯 Face Recognition: {'✅ OK' if recognition_ok else '❌ FAILED'}")
    
    if all([encodings_ok, camera_ok, detection_ok, recognition_ok]):
        print("\n🎉 ALL SYSTEMS GO! Your face recognition should work.")
    else:
        print("\n🔧 Some components need attention. Check the errors above.")