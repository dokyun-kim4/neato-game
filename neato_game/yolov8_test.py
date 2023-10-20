import cv2 as cv
from ultralytics import YOLO
from helpers import is_moving, get_com, player_out
import time
import numpy as np
from sort import Sort
import threading

# Load YOLO model
model = YOLO('yolov8n-pose.pt')
red = 10
radius = 5  # Radius of the circle
thickness = -1  # Thickness -1 fills the circle
colors = np.random.uniform(0,255,size=(100,3))
prev_boxes = []

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
    attr = results[0]
    boxes = attr.boxes.cpu().numpy()
    xyxys = list(boxes.xyxy)
    confs = boxes.conf

    for i,xyxy in enumerate(xyxys):
        xyxy = list(xyxy)
        xyxy.append(confs[i])
        xyxys[i] = xyxy
    
    if len(xyxys) == 0:
        tracks = sort.update()
    else:
        tracks = sort.update(np.array(xyxys))


    # # Draw bounding box (SORT)
    for track in tracks:
        x1,y1,x2,y2,track_id = int(track[0]),int(track[1]),int(track[2]),int(track[3]),int(track[4])
        cv.rectangle(frame,(x1,y1),(x2,y2),colors[track_id],2)
        cv.putText(frame,str(track_id),(x1+10,y1+40), cv.FONT_HERSHEY_PLAIN,3,colors[track_id],2)


    keypoints = attr.keypoints.xy.cpu().numpy()[0]
    # Draw keypoints
    for i,point in enumerate(keypoints):
        cv.circle(frame, (int(point[0]), int(point[1])), radius, (0,0,red), thickness)
        cv.putText(frame,str(i),(int(point[0]),int(point[1])),cv.FONT_HERSHEY_PLAIN, 2, (255,180,180), 2)
        xsum += int(point[0])
        ysum += int(point[1])
        red += 30
    


    cv.circle(frame,get_com(keypoints),radius,(0,0,0), thickness)
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