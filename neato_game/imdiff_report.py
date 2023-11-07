import numpy as np
import cv2 as cv
from PIL import Image 

show = False
save = True 

threshold = 15
kernel = np.array([
    [0, 1, 1, 1, 0],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1],
    [0, 1, 1, 1, 0]
], dtype=np.uint8)

def Show(img, frame_name=""):
    cv.imshow(frame_name, img)
    cv.waitKey(10000)
    pass

def Save(img, file_name):
    file_name = f"img/{file_name}.png"
    cv.imwrite(file_name, img)
    print(f"saved image {file_name}")
    pass

frame1 = cv.imread('img/SG_frame_1.png', cv.IMREAD_COLOR)
if show:
    Show(frame1)
if save:
    Save(frame1, "frame1")

frame2 = cv.imread('img/SG_frame_2.png', cv.IMREAD_COLOR)
if show:
    Show(frame2)
if save:
    Save(frame2, "frame2")

if save:
    image1 = Image.fromarray(cv.cvtColor(frame1, cv.COLOR_BGR2RGB))
    image2 = Image.fromarray(cv.cvtColor(frame2, cv.COLOR_BGR2RGB))

    image1.save("img/frames.gif", format="GIF", append_images=[image2], save_all=True, duration=200, loop=0)

difference = cv.absdiff(frame1, frame2)
if show:
    Show(difference)
if save:
    Save(difference, "difference")

gray = cv.cvtColor(difference, cv.COLOR_BGR2GRAY)
if show:
    Show(gray)
if save:
    Save(gray, "gray")

_, filtered = cv.threshold(gray, threshold, 255, cv.THRESH_BINARY)
if show:
    Show(filtered)
if save:
    Save(filtered, "filtered")

opened = cv.morphologyEx(filtered, cv.MORPH_OPEN, kernel)
if show:
    Show(opened)
if save:
    Save(opened, "opened")
