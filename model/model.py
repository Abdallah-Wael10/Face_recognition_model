import face_recognition
#import cv2
import pickle
import numpy as np
from pathlib import Path

class EnhancedFaceTrainer:
    def __init__(self, dataset_path=r'D:\Bureau_Tutorials\Hikvision-Face-Recognition\Bureau Employees\Bureau Employees', model_path='models'):
        self.dataset_path = Path(dataset_path)
        self.model_path = Path(model_path)
        self.model_path.mkdir(exist_ok=True)
        
    def preprocess_image(self, image_path):
        """Enhance image for better face detection"""
        try:
            # Load image
            image = face_recognition.load_image_file(str(image_path))
            
            # Try different face detection models
            face_locations = face_recognition.face_locations(image, model="hog")  # Faster
            if not face_locations:
                face_locations = face_recognition.face_locations(image, model="cnn")  # More accurate but slower
            
            return image, face_locations
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            return None, []
        
    def load_dataset(self):
        known_face_encodings = []
        known_face_names = []
        processed_count = 0
        failed_count = 0
        
        print("🔍 Scanning for faces in dataset...")
        
        for person_dir in self.dataset_path.iterdir():
            if not person_dir.is_dir():
                continue
                
            person_name = person_dir.name
            person_face_count = 0
            
            print(f"\n👤 Processing: {person_name}")
            
            for image_path in person_dir.glob('*.*'):
                if image_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                    image, face_locations = self.preprocess_image(image_path)
                    
                    if image is not None and face_locations:
                        # Get face encodings for all faces found
                        face_encodings = face_recognition.face_encodings(image, face_locations)
                        
                        for encoding in face_encodings:
                            known_face_encodings.append(encoding)
                            known_face_names.append(person_name)
                            person_face_count += 1
                            processed_count += 1
                    else:
                        print(f"  ⚠️ No face detected: {image_path.name}")
                        failed_count += 1
            
            print(f"  ✅ {person_name}: {person_face_count} faces added")
        
        print(f"\n📊 Training Summary:")
        print(f"  ✅ Successful: {processed_count} faces")
        print(f"  ⚠️ Failed: {failed_count} images")
        print(f"  👥 Unique people: {len(set(known_face_names))}")
        
        return known_face_encodings, known_face_names
    
    def train_and_save(self):
        print("🚀 Starting enhanced face training...")
        face_encodings, face_names = self.load_dataset()
        
        if len(face_encodings) == 0:
            print("❌ No faces found in the dataset!")
            return False
        
        print(f"🎉 Training completed! Found {len(face_encodings)} faces from {len(set(face_names))} people.")
        
        # Save the trained model with metadata
        model_data = {
            'encodings': face_encodings,
            'names': face_names,
            'metadata': {
                'total_faces': len(face_encodings),
                'total_people': len(set(face_names)),
                'training_date': np.datetime64('now')
            }
        }
        
        model_file = self.model_path / 'encodings.pkl'
        with open(model_file, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"💾 Model saved to: {model_file}")
        return True

if __name__ == "__main__":
    trainer = EnhancedFaceTrainer()
    trainer.train_and_save()