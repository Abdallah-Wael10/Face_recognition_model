# test_pickle.py
import pickle
import sys
import os

def test_pickle_file():
    encodings_path = r'D:\Bureau_Tutorials\Hikvision-Face-Recognition\model\encodings.pickle'
    
    print("🔍 Testing pickle file...")
    print(f"File exists: {os.path.exists(encodings_path)}")
    
    if os.path.exists(encodings_path):
        try:
            with open(encodings_path, 'rb') as f:
                data = pickle.load(f)
            
            print(f"✅ Successfully loaded pickle file!")
            print(f"Data type: {type(data)}")
            
            if isinstance(data, dict):
                print("📊 Dictionary contents:")
                for key, value in data.items():
                    print(f"  {key}: {type(value)} - {len(value) if hasattr(value, '__len__') else 'N/A'}")
                    
            elif isinstance(data, list):
                print(f"📊 List with {len(data)} items")
                if len(data) > 0:
                    print(f"First item type: {type(data[0])}")
                    
        except Exception as e:
            print(f"❌ Error loading pickle: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"❌ File not found: {encodings_path}")

if __name__ == "__main__":
    test_pickle_file()