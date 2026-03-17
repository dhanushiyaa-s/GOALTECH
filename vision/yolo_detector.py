from ultralytics import YOLO
from utils.geometry import compute_iou

class YOLODetector:
    def __init__(self, model_path, conf=0.25):
        self.model = YOLO(model_path)
        self.conf = conf

    def detect(self, frame):
        results = self.model(frame, imgsz=960, conf=self.conf, verbose=False)
        persons, phones = [], []

        for r in results:
            for b in r.boxes:
                label = self.model.names[int(b.cls[0])]
                box = tuple(map(int, b.xyxy[0]))

                if label == "person":
                    persons.append(box)
                elif label == "cell phone":
                    phones.append(box)

        phone = False
        for p in persons:
            for ph in phones:
                if compute_iou(p, ph) > 0.01:
                    phone = True

        return phone, persons, phones