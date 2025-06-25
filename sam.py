import torch
import cv2

# Load the pretrained YOLOv5s model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv5 on the current frame (BGR to RGB conversion)
    results = model(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Render results on frame
    annotated_frame = results.render()[0]  # Rendered frame is returned as a list

    # Show the result
    cv2.imshow('SafeNav AI - Object Detection', annotated_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
