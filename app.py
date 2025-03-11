import cv2
import torch
import numpy as np
from ultralytics import YOLO
from playsound import play_alert
import threading
device = "mps" if torch.backends.mps.is_available() else "cpu"

#model selection
model = YOLO("yolo11n.pt") 

# PHONE ID IS 67 IN DATASET
PHONE_CLASS_ID = 67  
PERSON_CLASS_ID = 0  


cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0] 

    people = []
    phones = []

    for det in results.boxes:
        class_id = int(det.cls)
        x1, y1, x2, y2 = map(int, det.xyxy[0])  
        
        if class_id == PERSON_CLASS_ID:
            people.append((x1, y1, x2, y2)) 
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  
            cv2.putText(frame, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if class_id == PHONE_CLASS_ID:
            phones.append((x1, y1, x2, y2)) 
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  
            cv2.putText(frame, "Phone", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    for px1, py1, px2, py2 in people:
        for fx1, fy1, fx2, fy2 in phones:
            if px1 < fx1 < px2 and py1 < fy1 < py2:  
                cv2.putText(frame, "ALERT! Phone detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                
                
                threading.Thread(target=play_alert, daemon=True).start()

    cv2.imshow("Mobile Phone Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
