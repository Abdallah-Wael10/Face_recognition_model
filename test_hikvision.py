# test_hikvision.py
import cv2
import time
from config import RTSP_URL, CAMERA_CONFIG

def test_hikvision_camera():
    print("🔍 Testing Hikvision Camera Connection")
    print("=" * 50)
    print(f"📡 RTSP URL: {RTSP_URL.split('@')[1].split(':')[0]}")
    print(f"👤 Username: {RTSP_URL.split('://')[1].split(':')[0]}")
    
    # Mask password for security
    masked_url = RTSP_URL.split('://')[0] + "://" + RTSP_URL.split('://')[1].split('@')[0][0] + "***@" + RTSP_URL.split('@')[1]
    print(f"🔒 URL: {masked_url}")
    
    cap = None
    try:
        # Try different OpenCV backends for RTSP
        backends = [
            cv2.CAP_FFMPEG,
            cv2.CAP_ANY,
            cv2.CAP_GSTREAMER
        ]
        
        for backend in backends:
            print(f"\n🔄 Trying backend: {backend}")
            try:
                cap = cv2.VideoCapture(RTSP_URL, backend)
                
                if cap.isOpened():
                    print("✅ Camera opened successfully!")
                    
                    # Set camera properties
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_CONFIG['width'])
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_CONFIG['height'])
                    cap.set(cv2.CAP_PROP_FPS, CAMERA_CONFIG['fps'])
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, CAMERA_CONFIG['buffer_size'])
                    
                    # Try to read frames
                    frames_received = 0
                    start_time = time.time()
                    
                    while frames_received < 10 and (time.time() - start_time) < 10:
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            frames_received += 1
                            print(f"✅ Frame {frames_received}: {frame.shape}")
                            
                            # Show first frame
                            if frames_received == 1:
                                # Resize for display
                                display_frame = cv2.resize(frame, (640, 360))
                                cv2.imshow('Hikvision Camera Test - Press any key', display_frame)
                                cv2.waitKey(1000)  # Show for 1 second
                            
                        else:
                            print("❌ Failed to read frame")
                            break
                    
                    cv2.destroyAllWindows()
                    
                    if frames_received > 0:
                        print(f"\n🎉 SUCCESS! Received {frames_received} frames")
                        print(f"📐 Frame size: {frame.shape}")
                        print(f"🎯 Backend that worked: {backend}")
                        return True, backend
                    else:
                        print("❌ Camera opened but no frames received")
                
                else:
                    print("❌ Failed to open camera with this backend")
                    
            except Exception as e:
                print(f"❌ Backend {backend} error: {e}")
            finally:
                if cap:
                    cap.release()
                    
    except Exception as e:
        print(f"❌ General error: {e}")
    
    print("\n❌ All connection attempts failed")
    return False, None

def check_network_connectivity():
    """Check if camera is reachable on network"""
    import socket
    from config import RTSP_URL
    
    # Extract IP from RTSP URL
    camera_ip = RTSP_URL.split('@')[1].split(':')[0]
    camera_port = 554
    
    print(f"\n🌐 Testing network connectivity to {camera_ip}:{camera_port}")
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((camera_ip, camera_port))
        sock.close()
        
        if result == 0:
            print("✅ Camera is reachable on network")
            return True
        else:
            print("❌ Camera is not reachable on network")
            return False
            
    except Exception as e:
        print(f"❌ Network test error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Hikvision Camera Diagnostic Tool")
    print("=" * 50)
    
    # Test network connectivity first
    network_ok = check_network_connectivity()
    
    if network_ok:
        # Test camera connection
        success, backend = test_hikvision_camera()
        
        if success:
            print(f"\n💡 Use this backend in your code: cv2.VideoCapture(RTSP_URL, {backend})")
        else:
            print("\n🔧 Troubleshooting tips:")
            print("1. Verify camera IP address in config.py")
            print("2. Check username and password")
            print("3. Ensure RTSP service is enabled on camera")
            print("4. Try different RTSP URL format")
            print("5. Check firewall settings")
    else:
        print("\n🔧 Network issues detected:")
        print("1. Verify camera IP address")
        print("2. Check if camera and PC are on same network")
        print("3. Try pinging the camera IP")
        print("4. Check network cables and connections")