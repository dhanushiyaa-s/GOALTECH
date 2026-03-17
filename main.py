import cv2

from face_module.face import load_known_faces, recognize_faces
from behaviour_module.behaviour import classify

# KEEP THESE IMPORTS SAME (your files)
from attention_module.attention import get_attention
from behaviour_module.detect import detect_phone


# -------------------------------
# LOAD FACES (dummy here)
# -------------------------------
known_face_encodings, known_face_names = load_known_faces()

# -------------------------------
# START CAMERA
# -------------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not working ❌")
    exit()

print("System Started ✅ Press 'q' to quit")

# -------------------------------
# LOOP
# -------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # -------------------------------
    # MODULE CALLS
    # -------------------------------
    face_locations, names = recognize_faces(frame)

    # ⚠️ SAFE CALLS (in case your functions differ)
    try:
        attention_score = get_attention(frame)
    except:
        attention_score = 1.0  # fallback

    try:
        phone_detected = detect_phone(frame)
    except:
        phone_detected = False

    # -------------------------------
    # DISPLAY
    # -------------------------------
    for (top, right, bottom, left), name in zip(face_locations, names):

        status = classify(attention_score, phone_detected)

        # Color coding
        color = (0, 255, 0)

        if status == "Using Phone":
            color = (0, 0, 255)
        elif status == "Not Attentive":
            color = (0, 165, 255)

        # Draw box
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        # Label
        label = f"{name} - {status}"
        cv2.putText(frame, label, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        print(label)

    # Show frame
    cv2.imshow("Smart Classroom Monitoring", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()