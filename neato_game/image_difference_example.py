import numpy as np
import cv2 as cv
from image_difference import Diff

# start video capture
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()    

img_queue_len = 1 # The number of images to keep, not including the current one
img_queue:list[np.ndarray] = [] # The image queue, from oldest image to newest
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Our operations on the frame come here
    img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # if the image queue isn't long enough (like when the loop starts), fill it
    if len(img_queue) <= img_queue_len:
        img_queue.append(img)
        continue
    # otherwise get rid of the oldest image
    else:
        img_queue.pop(0)

    # Display the resulting frame
    cv.imshow('frame', Diff(img, img_queue[0]))

    # press Q key to quit
    if cv.waitKey(1) == ord('q'):
        break

    # add the latest image to the image queue
    img_queue.append(img)

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()