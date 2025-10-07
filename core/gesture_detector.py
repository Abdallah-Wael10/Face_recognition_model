import mediapipe as mp
import cv2

class GestureDetector:
    def __init__(self, min_detection_confidence=0.6, min_tracking_confidence=0.5):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def detect(self, frame_bgr):
        # Returns dict: {'is_open_palm': bool, 'hand_bbox': (l,t,r,b) or None}
        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        if not results.multi_hand_landmarks:
            return {'is_open_palm': False, 'hand_bbox': None}

        # Simple heuristic: open palm => most fingers extended (>=4)
        for hand_landmarks in results.multi_hand_landmarks:
            lm = hand_landmarks.landmark
            # Finger tips and PIP joints indices in Mediapipe
            tips = [4, 8, 12, 16, 20]
            pips = [3, 6, 10, 14, 18]

            image_h, image_w = frame_bgr.shape[:2]
            extended = 0
            for tip, pip in zip(tips[1:], pips[1:]):  # skip thumb for simplicity
                if lm[tip].y < lm[pip].y:
                    extended += 1

            # Thumb heuristic (x orientation-based, rough)
            if lm[4].x < lm[3].x:
                extended += 1

            xs = [int(l.x * image_w) for l in lm]
            ys = [int(l.y * image_h) for l in lm]
            l, t, r, b = min(xs), min(ys), max(xs), max(ys)
            is_open = extended >= 4
            if is_open:
                return {'is_open_palm': True, 'hand_bbox': (l, t, r, b)}

        return {'is_open_palm': False, 'hand_bbox': None}