import numpy as np
import math

def is_moving(prev_box_coord,crnt_box_coord):
    """
    Given 2 bounding boxes containing a person, (previous frame & current frame), determine if the person moved

    Args:
        prev_box: A list containing 2 corner coordinates of the bounding box from previous frame [x1,y1,x2,y2]
        crnt_box: A list containing 2 corner coordinates of the bounding box from current frame [x3,y3,x4,y4]

    Returns:
        A boolean representing if a person moved between the two frames
    """
#------------------------------- Declare Variables ----------------------------------------------------------------------------#
    # Prev_box
    xy1 = [prev_box_coord[0],prev_box_coord[1]]
    xy2 = [prev_box_coord[2],prev_box_coord[3]]

    # Crnt_box
    xy3 = [crnt_box_coord[0],crnt_box_coord[1]]
    xy4 = [crnt_box_coord[2],crnt_box_coord[3]]

#-------------------------------- Implement Function-----------------------------------------------------------------------------#
    # What will be considered "moving"?

    # 1. bounding box is in a different location; calculate distance traveled from previous location to current
    dist1 = math.dist(xy1,xy3)
    dist2 = math.dist(xy2,xy4)

    if dist1 >= 5 or dist2 >= 5:
        return True

    # 2. bounding box is different size
    length1 = abs(xy1[0]-xy2[0])
    height1 = abs(xy1[1]-xy2[1])

    length2 = abs(xy3[0]-xy4[0])
    height2 = abs(xy3[1]-xy4[1])

    if abs(length2 - length1) >= 5 or abs(height2 - height1) >= 5:
        return True


    return False

def get_com(points):
    """
    Calculate center of mass given a list of points.

    Args:
        points: list of points represented as a list

    Returns:
        center of mass
    """
    points = list(points)
    for i,point in enumerate((points)):
        if point[0] == 0 and point[1] == 0:
            points.pop(i)
    xsum = 0
    ysum = 0
    count = len(points)

    for point in points:
        xsum += point[0]
        ysum += point[1]

    return (int(xsum//count),int(ysum//count))

