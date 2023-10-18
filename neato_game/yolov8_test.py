import cv2 as cv
from ultralytics import YOLO
from is_moving import is_moving
import time
from playsound import playsound

# Load YOLO model
model = YOLO('yolov8n-pose.pt')
red = 10
radius = 5  # Radius of the circle
thickness = -1  # Thickness -1 fills the circle

prev_boxes = []

# Start camera
cap = cv.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Unable to access the camera.")
    exit()

time.sleep(1)
while True:
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
    xyxys = boxes.xyxy

    keypoints = attr.keypoints.xy.cpu().numpy()[0]

    i = 0
    color = [(0,0,255),(255,0,0),(0,255,0)]
    # Draw bounding box
    for xyxy in xyxys:
        cv.rectangle(frame,(int(xyxy[0]),int(xyxy[1])),(int(xyxy[2]),int(xyxy[3])),color[i])
        cv.putText(frame,str((int(xyxy[0]),int(xyxy[1]))),(int(xyxy[0]),int(xyxy[1])),cv.FONT_HERSHEY_PLAIN, 1, (255,200,180), 2)
        i += 1

    # Draw keypoints
    for i,point in enumerate(keypoints):
        cv.circle(frame, (int(point[0]), int(point[1])), radius, (0,0,red), thickness)
        cv.putText(frame,str(i),(int(point[0]),int(point[1])),cv.FONT_HERSHEY_PLAIN, 2, (255,180,180), 2)
        red += 30
    cv.imshow('Webcam Feed', frame)
    if len(prev_boxes) != 0:
        if is_moving(prev_boxes[0],xyxys[0]):
            # time.sleep(0.1)
            print("MOVEMENT DETECTED")
            playsound('playerOut.mp3')

    prev_boxes = xyxys

    # Exit the loop if 'q' is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the OpenCV window
cap.release()
cv.destroyAllWindows()