from ultralytics import YOLO
import cv2
import pyttsx3
import time
import torch

# Device setup
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[INFO] Using device: {device}")

# Load YOLOv8s model
model = YOLO('yolov8s.pt')
model.to(device)

# Initialize TTS engine
engine = pyttsx3.init()
engine.setProperty('rate', 160)

# Important objects list
important_objects = [
    "bottle", "chair", "cup", "dog", "cat", "person",
    "tv", "laptop", "keyboard", "drawer", "remote"
]

# Start webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Could not open webcam.")
    exit()

last_alert_time = 0
cooldown = 2
prev_time = 0

print("[INFO] YOLOv8s Detection Running... Press 'q' to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8s inference
    results = model(frame, verbose=False)[0]  # Get first result
    detections = results.boxes

    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time else 0
    prev_time = current_time

    # Process detections
    for box in detections:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        width = x2 - x1

        class_name = model.names[cls_id]

        # Draw box and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, class_name, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Trigger alert if risky object detected
        if class_name in important_objects and width > 200 and time.time() - last_alert_time > cooldown:
            msg = f"Warning! {class_name} ahead."
            print("[ALERT]:", msg)
            engine.say(msg)
            engine.runAndWait()
            last_alert_time = time.time()
            break

    # Show FPS on frame
    cv2.putText(frame, f"FPS: {int(fps)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Show webcam output
    cv2.imshow("🛡️ SafeNav AI - YOLOv8s", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("[INFO] Exiting...")
        break

cap.release()
cv2.destroyAllWindows()
