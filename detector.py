# detector.py
from ultralytics import YOLO
import cv2
import pyttsx3
import time
import torch

class ObjectDetector:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLO('yolov8s.pt').to(self.device)
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 160)

        self.important_objects = [
            "bottle", "chair", "cup", "dog", "cat", "person",
            "tv", "laptop", "keyboard", "drawer", "remote"
        ]
        self.last_alert_time = 0
        self.cooldown = 2
        self.prev_time = 0

    def detect(self, update_frame, log_callback=None):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            if log_callback:
                log_callback("❌ Could not open webcam.")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.model(frame, verbose=False)[0]
            detections = results.boxes

            current_time = time.time()
            fps = 1 / (current_time - self.prev_time) if self.prev_time else 0
            self.prev_time = current_time

            for box in detections:
                cls_id = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                width = x2 - x1
                class_name = self.model.names[cls_id]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, class_name, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                if class_name in self.important_objects and width > 200 and time.time() - self.last_alert_time > self.cooldown:
                    msg = f"⚠️ {class_name} ahead!"
                    if log_callback:
                        log_callback(msg)
                    self.engine.say(msg)
                    self.engine.runAndWait()
                    self.last_alert_time = time.time()
                    break

            update_frame(frame, fps)

        cap.release()
