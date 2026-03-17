import cv2
import time
from ultralytics import YOLO
import numpy as np

# ==============================
# 👤 STUDENT NAME
# ==============================
STUDENT_NAME = "John Doe"

# ==============================
# SETTINGS
# ==============================
SESSION_LIMIT = 120  # 2 minutes
CONFIDENCE_THRESHOLD = 0.45 # Slightly reduced to catch more detections
IOU_THRESHOLD = 0.25  # Minimum IoU for phone-person overlap to count as "use"
ASPECT_RATIO_THRESHOLD = 1.3 # If width/height > this, consider it sideways (e.g., 1.3 for 4:3 or 16:9 like aspect ratios)

# Load model
model = YOLO("yolov8n.pt") 

cap = cv2.VideoCapture(0)

session_start = time.time()
phone_start = None
total_phone_usage = 0
current_session_phone_usage = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame, exiting...")
        break

    current_time = time.time()
    elapsed = current_time - session_start
    remaining_time = int(SESSION_LIMIT - elapsed)

    if remaining_time <= 0:
        print("Session limit reached, exiting...")
        break

    results = model(frame, verbose=False)

    phone_being_used_in_frame = False
    person_boxes_in_frame = []
    phone_boxes_in_frame = []

    # Process all detections
    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if conf > CONFIDENCE_THRESHOLD:
                if label == "person":
                    person_boxes_in_frame.append((x1, y1, x2, y2))
                    # Draw ALL person boxes
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1) # Green for all persons
                    cv2.putText(frame, f"Person {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                elif label == "cell phone":
                    phone_boxes_in_frame.append((x1, y1, x2, y2))
                    # Draw ALL phone boxes
                    # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 165, 255), 1) # Orange for all phones
                    # cv2.putText(frame, f"Phone {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)

    # Determine if any phone is being used by any person and its orientation
    detected_used_phones = [] # To store bounding boxes and orientation of phones actively being used
    
    for p_box in person_boxes_in_frame:
        for ph_box in phone_boxes_in_frame:
            # Calculate Intersection over Union (IoU)
            ix1 = max(p_box[0], ph_box[0])
            iy1 = max(p_box[1], ph_box[1])
            ix2 = min(p_box[2], ph_box[2])
            iy2 = min(p_box[3], ph_box[3])

            inter_area = max(0, ix2 - ix1 + 1) * max(0, iy2 - iy1 + 1)
            p_area = (p_box[2] - p_box[0] + 1) * (p_box[3] - p_box[1] + 1)
            ph_area = (ph_box[2] - ph_box[0] + 1) * (ph_box[3] - ph_box[1] + 1)
            union_area = float(p_area + ph_area - inter_area)

            if union_area > 0:
                iou = inter_area / union_area
                if iou > IOU_THRESHOLD:
                    phone_being_used_in_frame = True
                    
                    # Determine phone orientation
                    phone_width = ph_box[2] - ph_box[0]
                    phone_height = ph_box[3] - ph_box[1]
                    orientation_status = "VERTICAL"
                    if phone_height > 0 and (phone_width / phone_height) > ASPECT_RATIO_THRESHOLD:
                        orientation_status = "HORIZONTAL (Sideways)"
                    elif phone_width > 0 and (phone_height / phone_width) > ASPECT_RATIO_THRESHOLD:
                         orientation_status = "VERTICAL" # Explicitly state vertical
                    else:
                        orientation_status = "UNKNOWN/SQUARE-ISH" # For cases where aspect ratio isn't distinct

                    detected_used_phones.append((ph_box, orientation_status)) # Store box and orientation
                    
                    # Highlight the person using the phone
                    cv2.rectangle(frame, (p_box[0], p_box[1]), (p_box[2], p_box[3]), (0, 255, 255), 2) # Yellow for person using phone
                    cv2.putText(frame, f"Using Phone", (p_box[0], p_box[1] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    break # Assuming one phone per person for simplicity of person highlight, but still check all phones for usage
        # No 'break' here, as one person might pick up another phone, or a second person might pick up a phone.
        # The phone_being_used_in_frame will correctly reflect if AT LEAST ONE phone is in use.

    # Draw specific highlights for phones determined to be "in use"
    current_frame_orientation_display = "N/A" # Default for display
    if detected_used_phones: # If any phone is being used
        # We'll just display the orientation of the *first* detected used phone for simplicity in the overlay text
        # If you need to display orientation for ALL used phones, you'd iterate and draw labels per phone
        _ph_box, _orientation = detected_used_phones[0] 
        current_frame_orientation_display = _orientation

        for ph_box, orientation in detected_used_phones:
            cv2.rectangle(frame, (ph_box[0], ph_box[1]), (ph_box[2], ph_box[3]), (255, 0, 0), 3) # Blue for used phone
            cv2.putText(frame, f"Phone: {orientation}", (ph_box[0], ph_box[1] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    # ==============================
    # PHONE TIME ACCUMULATION
    # ==============================
    if phone_being_used_in_frame:
        if phone_start is None:
            phone_start = current_time
        current_session_phone_usage = int(current_time - phone_start)
    else:
        if phone_start is not None:
            total_phone_usage += int(current_time - phone_start)
            phone_start = None
            current_session_phone_usage = 0

    # ==============================
    # CCTV-STYLE TEXT (Simple & Clean)
    # ==============================

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    font_thickness = 1
    text_color = (0, 0, 0) # Black color for text

    text_lines = []
    text_lines.append(f"Student: {STUDENT_NAME}")
    text_lines.append(f"Total Usage: {total_phone_usage} sec")
    if phone_being_used_in_frame:
        text_lines.append(f"Current Session: {current_session_phone_usage} sec")
    text_lines.append(f"Time Remaining: {remaining_time} sec")
    
    status_text = "Phone: IN USE" if phone_being_used_in_frame else "Phone: IDLE"
    status_color = (0, 0, 255) if phone_being_used_in_frame else (0, 165, 255)

    y_start_pos = 40
    line_height = 30

    for i, line in enumerate(text_lines):
        cv2.putText(frame,
                    line,
                    (30, y_start_pos + i * line_height),
                    font,
                    font_scale,
                    text_color,
                    font_thickness)
    
    # Display the general usage status
    cv2.putText(frame,
                status_text,
                (30, y_start_pos + len(text_lines) * line_height),
                font,
                font_scale,
                status_color,
                font_thickness + 1)
    
    # Display the orientation of the first detected used phone
    if phone_being_used_in_frame:
        cv2.putText(frame,
                    f"Orientation: {current_frame_orientation_display}",
                    (30, y_start_pos + (len(text_lines) + 1) * line_height), # Below general status
                    font,
                    font_scale,
                    (128, 0, 128), # Purple for orientation
                    font_thickness)


    cv2.imshow("CCTV Behavior Monitoring", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

# Finalize last phone session if still active when loop breaks
if phone_start is not None:
    total_phone_usage += int(time.time() - phone_start)

cap.release()
cv2.destroyAllWindows()

# ==============================
# FINAL REPORT
# ==============================

print("\n===== FINAL REPORT =====")
print(f"Student Name      : {STUDENT_NAME}")
print(f"Total Phone Usage : {total_phone_usage} seconds")
print(f"Session Duration  : {int(time.time() - session_start)} seconds")
print("========================")