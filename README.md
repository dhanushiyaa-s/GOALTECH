# 🎓 AI-Powered Smart Classroom Monitoring System

An intelligent computer vision system that monitors classroom activity in real-time using AI.

---

## 🚀 Features

- 👤 Face Detection (OpenCV)
- 👁️ Attention Monitoring (MediaPipe / fallback logic)
- 📱 Phone Usage Detection (YOLO - Ultralytics)
- 🧠 Behavior Classification (Attentive / Not Attentive / Using Phone)
- 🎥 Real-time Webcam Processing

---

## 🧠 Problem Statement

Manual classroom monitoring is inefficient for:
- Taking attendance
- Tracking student attention
- Detecting distractions

This system automates all of the above using AI.

---

## ⚙️ Tech Stack

- Python
- OpenCV
- MediaPipe (with fallback support)
- YOLO (Ultralytics)
- NumPy

---

## 📁 Project Structure
GOALTECH/
│── main.py
│
├── face_module/
│ ├── face.py
│ └── init.py
│
├── attention_module/
│ ├── attention.py
│ └── init.py
│
├── behaviour_module/
│ ├── detect.py
│ ├── behaviour.py
│ └── init.py
│
└── data/

---

▶️ How to Run

1. Install dependencies:

pip install opencv-python ultralytics mediapipe

2. Run the project:
python main.py