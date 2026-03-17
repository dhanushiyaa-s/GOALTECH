import cv2

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def load_known_faces():
    # Not needed for OpenCV version
    return [], []

def recognize_faces(frame, *args):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    face_locations = []
    names = []

    for (x, y, w, h) in faces:
        face_locations.append((y, x+w, y+h, x))
        names.append("Student")  # generic label

    return face_locations, names