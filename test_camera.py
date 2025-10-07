# test_camera.py
import cv2

def test_cameras():
    print("🔍 Testing available cameras...")
    
    # Test different camera indices
    for i in range(5):
        print(f"Testing camera index {i}...")
        cap = cv2.VideoCapture(i)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"✅ Camera {i} WORKS! Frame shape: {frame.shape}")
                cap.release()
                return i
            else:
                print(f"❌ Camera {i} opens but cannot read frame")
        else:
            print(f"❌ Camera {i} cannot be opened")
        
        cap.release()
    
    print("❌ No working cameras found!")
    return -1

if __name__ == "__main__":
    working_camera = test_cameras()
    if working_camera >= 0:
        print(f"\n🎯 Use camera index: {working_camera}")
    else:
        print("\n💡 Try these solutions:")
        print("1. Check if camera is connected")
        print("2. Try different USB port")
        print("3. Update camera drivers")
        print("4. Restart computer")