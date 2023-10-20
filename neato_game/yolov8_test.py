import cv2 as cv
from ultralytics import YOLO
from helpers import is_moving, get_com, player_out, person_bboxes
import time
import numpy as np
from sort import Sort
import threading

# Load YOLO model
model = YOLO('yolov8n-pose.pt')
red = 10
radius = 5  # Radius of the circle
thickness = -1  # Thickness -1 fills the circle
colors = np.random.uniform(0,255,size=(999,3))

all_person = None
prev_boxes = [] # This would be person_bboxes object
sort = Sort()

# Start camera
cap = cv.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Unable to access the camera.")
    exit()

time.sleep(0.5)
while True:
    xsum = 0
    ysum = 0
    # Read a frame from the camera
    ret, frame = cap.read()

    # Check if the frame was read successfully
    if not ret:
        print("Error: Unable to read frame.")
        break


    # Feed frame into YOLO model
    results = model(frame,verbose=False)
    # Get results
    attr = results[0]
    boxes = attr.boxes.cpu().numpy()
    xyxys = boxes.xyxy
    confs = boxes.conf

    # Store all results as a `person_bboxes` object
    all_person = person_bboxes(xyxys,confs,sort)
    all_person.update()
   
    # Draw bounding box (SORT)
    for track in all_person.tracks:
        x1,y1,x2,y2,track_id = int(track[0]),int(track[1]),int(track[2]),int(track[3]),int(track[4])
        cv.rectangle(frame,(x1,y1),(x2,y2),colors[track_id],2)
        cv.putText(frame,str(track_id),(x1+10,y1+40), cv.FONT_HERSHEY_PLAIN,3,colors[track_id],2)

    cv.imshow('Webcam Feed', frame)

    if len(prev_boxes) != 0:
        if is_moving(prev_boxes[0],xyxys[0]):
            t1 = threading.Thread(target=player_out)
            t1.start()



    prev_boxes = xyxys

    # Exit the loop if 'q' is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the OpenCV window
cap.release()
cv.destroyAllWindows()