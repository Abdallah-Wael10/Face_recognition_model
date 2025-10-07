import time
import json
import threading
import queue
import base64
import requests
from datetime import datetime
from typing import Optional, Dict
import config

class BackendClient:
    def __init__(self):
        self.base_url = config.BACKEND['base_url'].rstrip('/')
        self.token = config.BACKEND['token']
        self.camera_id = config.BACKEND['camera_id']
        self.timeout = config.BACKEND.get('timeout', 5)
        self.session = requests.Session()
        self.q = queue.Queue(maxsize=128)
        self.worker = threading.Thread(target=self._worker, daemon=True)
        self.worker.start()

    def _headers(self, idempotency_key: Optional[str] = None) -> Dict[str, str]:
        h = {
            'Authorization': f"Bearer {self.token}",
            'Content-Type': 'application/json'
        }
        if idempotency_key:
            h['Idempotency-Key'] = idempotency_key
        return h

    def _post(self, path: str, payload: Dict, idempotency_key: Optional[str] = None):
        url = f"{self.base_url}{path}"
        for attempt in range(config.BACKEND.get('retries', 3)):
            try:
                r = self.session.post(url, headers=self._headers(idempotency_key), json=payload, timeout=self.timeout)
                if r.status_code < 500:
                    return r
            except requests.RequestException:
                pass
            time.sleep(min(2 ** attempt, 5))
        return None

    def send_event(self, employee_id: Optional[str], event_type: str, confidence: float,
                   context: Dict, image_bgr=None, face_name: Optional[str] = None):
        ts = datetime.utcnow().isoformat()
        frame_id = context.get('frameId') or f"{ts}-{face_name or employee_id or 'unknown'}-{event_type}"
        image_b64 = None
        if image_bgr is not None and config.EVENTS.get('attach_image', False):
            try:
                import cv2
                _, buf = cv2.imencode('.jpg', image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                image_b64 = base64.b64encode(buf.tobytes()).decode('utf-8')
            except Exception:
                image_b64 = None

        payload = {
            'employeeId': employee_id,
            'eventType': event_type,
            'timestamp': ts,
            'cameraId': self.camera_id,
            'confidence': confidence,
            'context': {
                'faceName': face_name,
                **context
            },
            'image': image_b64
        }
        self.q.put((payload, frame_id))

    def _worker(self):
        while True:
            payload, frame_id = self.q.get()
            try:
                self._post('/api/attendance/events', payload, idempotency_key=frame_id)
            finally:
                self.q.task_done()