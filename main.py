import cv2
import time
from collections import deque

# Vision
from vision.yolo_detector import YOLODetector
from vision.face_analyzer import FaceAnalyzer

# Logic
from logic.temporal import stable
from logic.behaviour import compute_score, classify_behavior

# Analytics
from analytics.tracker import Tracker

# ==============================
# SETTINGS
# ==============================
STUDENT_NAME = "John Doe"

FRAME_SKIP = 2
TALKING_THRESHOLD = 6
SILENT_THRESHOLD = 20
EYE_CLOSE_TIME = 10

# ==============================
# INIT
# ==============================
detector = YOLODetector("yolov8m.pt")
face_analyzer = FaceAnalyzer()
tracker = Tracker()

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

history = deque(maxlen=15)

frame_count = 0
talking_frames = 0
silent_frames = 0
eye_closed_start = None
last_phone_state = False

# ==============================
# MAIN LOOP
# ==============================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    h, w, _ = frame.shape

    phone = False
    talking = False
    eyes_closed = False
    gaze = "CENTER"

    # ==============================
    # FACE ANALYSIS
    # ==============================
    res = face_analyzer.analyze(frame)

    if res.multi_face_landmarks:
        for lm in res.multi_face_landmarks:

            gaze = face_analyzer.get_eye_gaze(lm, w)
            lip, eye = face_analyzer.detect_face_states(lm, h)

            # Talking
            if lip > 8:
                talking_frames += 1
                silent_frames = 0
            else:
                silent_frames += 1
                talking_frames = 0

            talking = talking_frames > TALKING_THRESHOLD
            silent = silent_frames > SILENT_THRESHOLD

            # Eyes
            if eye < 3:
                if eye_closed_start is None:
                    eye_closed_start = time.time()
            else:
                eye_closed_start = None

            if eye_closed_start and time.time() - eye_closed_start > EYE_CLOSE_TIME:
                eyes_closed = True

    # ==============================
    # YOLO (PHONE DETECTION)
    # ==============================
    if frame_count % FRAME_SKIP == 0:
        phone, persons, phones = detector.detect(frame)
    else:
        phone = last_phone_state

    last_phone_state = phone

    # ==============================
    # TEMPORAL SMOOTHING
    # ==============================
    history.append({
        "phone": phone,
        "talking": talking,
        "eyes": eyes_closed,
        "gaze": gaze == "CENTER"
    })

    s_phone = stable(history, "phone")
    s_talk = stable(history, "talking")
    s_eye = stable(history, "eyes")
    s_gaze = stable(history, "gaze")

    # ==============================
    # ANALYTICS
    # ==============================
    tracker.update_phone(s_phone)

    score = compute_score(s_phone, s_eye, s_talk, s_gaze)
    tracker.update_attention(score)

    # ==============================
    # BEHAVIOR
    # ==============================
    behavior = classify_behavior(s_eye, s_phone, s_talk, silent, s_gaze)

    # ==============================
    # DISPLAY
    # ==============================
    cv2.putText(frame, f"{STUDENT_NAME}", (20, 30), 0, 0.7, (0, 0, 0), 2)
    cv2.putText(frame, f"Behavior: {behavior}", (20, 60), 0, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, f"Score: {score}%", (20, 90), 0, 0.7, (255, 0, 0), 2)
    cv2.putText(frame, f"Gaze: {gaze}", (20, 120), 0, 0.6, (0, 0, 0), 1)

    cv2.imshow("AI Behavior Monitoring", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# ==============================
# FINAL REPORT
# ==============================
report = tracker.report(STUDENT_NAME)

print("\n===== FINAL REPORT =====")
print(f"Student Name      : {report['name']}")
print(f"Phone Usage       : {report['phone_time']} sec")
print(f"Attention %       : {report['attention']:.2f}%")
print("========================")

cap.release()
cv2.destroyAllWindows()