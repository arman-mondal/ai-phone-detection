import cv2
import torch
import numpy as np
from ultralytics import YOLO
import threading
from pydub import AudioSegment
from pydub.playback import play
import time

def play_alert():
    sound = AudioSegment.from_file("alert.mp3")
    play(sound)

device = "mps" if torch.backends.mps.is_available() else "cpu"

# Model selection
model = YOLO("yolo12n.pt") 

# PHONE ID IS 67 IN DATASET
PHONE_CLASS_ID = 67  
PERSON_CLASS_ID = 0  

cap = cv2.VideoCapture(0)

phone_detected_start_time = None  # To track when the phone was first detected
ALERT_THRESHOLD = 10  # Seconds

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

    phone_in_person = False
    for px1, py1, px2, py2 in people:
        for fx1, fy1, fx2, fy2 in phones:
            if px1 < fx1 < px2 and py1 < fy1 < py2:  
                phone_in_person = True
                cv2.putText(frame, "ALERT! Phone detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                break

    if phone_in_person:
        if phone_detected_start_time is None:
            phone_detected_start_time = time.time()  # Start the timer
        elif time.time() - phone_detected_start_time >= ALERT_THRESHOLD:
            threading.Thread(target=play_alert, daemon=True).start()
            phone_detected_start_time = None  # Reset the timer after playing the alert
    else:
        phone_detected_start_time = None  # Reset the timer if no phone is detected

    cv2.imshow("Mobile Phone Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
