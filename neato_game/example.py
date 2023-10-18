import cv2 as cv
from ultralytics import YOLO

model = YOLO('yolov8n-pose.pt')

results = model('img/test1.png')

image = cv.imread('img/test1.png')

red = 10
radius = 5  # Radius of the circle
thickness = -1  # Thickness -1 fills the circle


attr = results[0]

boxes = attr.boxes.cpu().numpy()

xyxys = boxes.xyxy
print(len(xyxys))


keypoints = attr.keypoints.xy.cpu().numpy()[0]


for xyxy in xyxys:
    cv.rectangle(image,(int(xyxy[0]),int(xyxy[1])),(int(xyxy[2]),int(xyxy[3])),(0,255,0))
    cv.putText(image,str((int(xyxy[0]),int(xyxy[1]))),(int(xyxy[0]),int(xyxy[1])),cv.FONT_HERSHEY_PLAIN, 1, (255,200,180), 2)
    

for i,point in enumerate(keypoints):
    cv.circle(image, (int(point[0]), int(point[1])), radius, (0,0,red), thickness)
    cv.putText(image,str(i),(int(point[0]),int(point[1])),cv.FONT_HERSHEY_PLAIN, 2, (255,180,180), 2)
    red += 30

cv.imshow('person test',image)
cv.waitKey()
cv.destroyAllWindows()
