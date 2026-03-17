import mediapipe as mp
import cv2

class FaceAnalyzer:
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh()

    def analyze(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb)
        return result

    def get_eye_gaze(self, lm, w):
        left = lm.landmark[33].x * w
        right = lm.landmark[133].x * w
        iris = lm.landmark[468].x * w if len(lm.landmark) > 468 else lm.landmark[1].x * w

        r = (iris - left) / (right - left + 1e-6)

        if r < 0.35:
            return "LEFT"
        elif r > 0.65:
            return "RIGHT"
        return "CENTER"

    def detect_face_states(self, lm, h):
        lip = abs((lm.landmark[14].y - lm.landmark[13].y) * h)
        eye = abs((lm.landmark[145].y - lm.landmark[159].y) * h)
        return lip, eye