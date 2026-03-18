from backend.server import send_update
import cv2
import time
from collections import deque
MAX_DURATION = 30  # ⏱️ total webcam time in seconds
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


start_time = time.time()
# ==============================
# MAIN LOOP
# ==============================
while cap.isOpened():
    # ⏱️ TIMER CHECK
    elapsed_time = time.time() - start_time
    remaining_time = int(MAX_DURATION - elapsed_time)

    if remaining_time <= 0:
        print("Time limit reached. Stopping...")
        break
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    h, w, _ = frame.shape

    phone = False
    talking = False
    eyes_closed = False
    gaze = "CENTER"
    silent = False

    # ==============================
    # FACE ANALYSIS
    # ==============================
    res = face_analyzer.analyze(frame)

    if res.multi_face_landmarks:
        for lm in res.multi_face_landmarks:

            gaze = face_analyzer.get_eye_gaze(lm, w)
            lip, eye = face_analyzer.detect_face_states(lm, h)

            # Talking detection
            if lip > 8:
                talking_frames += 1
                silent_frames = 0
            else:
                silent_frames += 1
                talking_frames = 0

            talking = talking_frames > TALKING_THRESHOLD
            silent = silent_frames > SILENT_THRESHOLD

            # Eye closure detection
            if eye < 3:
                if eye_closed_start is None:
                    eye_closed_start = time.time()
            else:
                eye_closed_start = None

            if eye_closed_start and (time.time() - eye_closed_start > EYE_CLOSE_TIME):
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
    # SEND DATA TO FRONTEND
    # ==============================
    report_data = tracker.report(STUDENT_NAME)

    analytics = {
        "student_name": STUDENT_NAME,
        "total_phone_usage": report_data["phone_time"],
        "current_phone_session_duration": 0,
        "phone_orientation": "N/A",
        "head_direction": gaze,
        "is_talking": talking,
        "eyes_closed_long": eyes_closed,
        "attention_status": behavior
    }

    send_update(frame, analytics)

    # ==============================
    # DISPLAY
    # ==============================
    cv2.putText(frame, f"{STUDENT_NAME}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    cv2.putText(frame, f"Behavior: {behavior}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.putText(frame, f"Score: {score}%", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.putText(frame, f"Gaze: {gaze}", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
    
    cv2.putText(frame, f"Time Left: {remaining_time}s", (20, 150),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 255), 2)

    cv2.imshow("AI Powered Smart Classrom Monitoring System", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# ==============================
# SEND DATA TO FRONTEND (LIVE)
# ==============================
report_data = tracker.report(STUDENT_NAME)

analytics = {
    "type": "live",  # 👈 IMPORTANT
    "student_name": STUDENT_NAME,
    "phone_usage": report_data["phone_time"],
    "attention": score,
    "attention_status": behavior,
    "is_talking": s_talk,  # 👈 use smoothed value
    "head_direction": gaze
}

send_update(frame, analytics)
# ==============================
# FINAL REPORT
# ==============================
report = tracker.report(STUDENT_NAME)

print("\n===== FINAL REPORT =====")
print(f"Student Name      : {report['name']}")
print(f"Phone Usage       : {report['phone_time']} sec")
print(f"Attention %       : {report['attention']:.2f}%")
print("========================")

# SEND FINAL REPORT TO FRONTEND
final_report = {
    "type": "final",  # 👈 VERY IMPORTANT
    "student_name": report["name"],
    "phone_usage": report["phone_time"],
    "attention": report["attention"]
}

send_update(None, final_report)

# ⏳ Prevent socket from closing too fast
time.sleep(1)

cap.release()
cv2.destroyAllWindows()

cap.release()
cv2.destroyAllWindows()