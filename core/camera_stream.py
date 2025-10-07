# core/camera_stream.py
import cv2
import pyvirtualcam
import numpy as np
import time
from config import RTSP_URL, STREAM_CONFIG

def open_rtsp(url, timeout_sec=10):
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    t0 = time.time()
    while not cap.isOpened() and time.time() - t0 < timeout_sec:
        time.sleep(0.5)
    if not cap.isOpened():
        raise RuntimeError("Could not open RTSP stream.")
    return cap

def main():
    cap = open_rtsp(RTSP_URL)

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Unable to read from stream. Check RTSP URL.")

    h, w = frame.shape[:2]
    print(f"Camera source resolution: {w}x{h}")

    with pyvirtualcam.Camera(
        width=STREAM_CONFIG['virtual_cam_width'], 
        height=STREAM_CONFIG['virtual_cam_height'], 
        fps=STREAM_CONFIG['virtual_cam_fps'], 
        backend="obs"
    ) as cam:
        print(f"✅ Virtual camera started: {cam.device} "
              f"({STREAM_CONFIG['virtual_cam_width']}x{STREAM_CONFIG['virtual_cam_height']} "
              f"@ {STREAM_CONFIG['virtual_cam_fps']}fps)")
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Frame read failed — reconnecting...")
                    cap.release()
                    time.sleep(1)
                    cap = open_rtsp(RTSP_URL)
                    continue

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(
                    frame_rgb, 
                    (STREAM_CONFIG['virtual_cam_width'], STREAM_CONFIG['virtual_cam_height'])
                )

                cam.send(frame_resized)
                cam.sleep_until_next_frame()

                preview = cv2.resize(frame, (
                    STREAM_CONFIG['virtual_cam_width'], 
                    STREAM_CONFIG['virtual_cam_height']
                ))
                cv2.imshow("Hikvision Stream", preview)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print("❌ Stopped by user.")
        finally:
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()