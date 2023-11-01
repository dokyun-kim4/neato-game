import numpy as np
import cv2 as cv


# set the default kernel to a cross with the proper dtype
default_kernel = np.array([
    [0, 1, 0],
    [1, 1, 1],
    [0, 1, 0]
], dtype=np.uint8)

def Diff(img1: np.ndarray, img2: np.ndarray, threshold:int=10, kernel=default_kernel):
    """
    Finds the pixels that have changed between two images

    Args:
        img1 (MatLike): First image to be compared (order doesn't matter).
        img2 (MatLike): Second image to be compared (order doesn't matter).
        threshold (int): Pixels who's value changed by more than this are considered to have changed. Defaults to 50.
        kernel (NDArray[uint8]): The kernel for the opening operation. Defaults to a cross.

    Returns:
        opened (MatLike): A binary image where white pixels have changed
    """

    # Find the absolute difference between the two images
    difference = cv.absdiff(img1, img2)

    # convert the image to grayscale if necessary
    if len(difference.shape) > 2:
        gray = cv.cvtColor(difference, cv.COLOR_BGR2GRAY)
    else:
        gray = difference   

    # threshold the pixel values to convert to a binary image
    _, filtered = cv.threshold(gray, threshold, 255, cv.THRESH_BINARY)

    # open the image to reduce background noise
    opened = cv.morphologyEx(filtered, cv.MORPH_OPEN, kernel)

    # return the processed image
    return opened