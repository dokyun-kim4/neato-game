import numpy as np
import math
from playsound import playsound
from sort import Sort
from image_difference import Diff
import cv2 as cv

class person_bboxes:
    def __init__(self,xyxys: np.ndarray, confs: np.ndarray,sort: Sort)->None:
        self.sort = sort
        self.xyxys = xyxys
        self.confs = confs
        self.conf_boxes = np.array([[xyxy[0],xyxy[1],xyxy[2],xyxy[3],confs[i]] for i,xyxy in enumerate(xyxys)])
        self.tracks = np.array([])

    def update(self)->None:
        """
        Update bbox ids using SORT
        """
        if len(self.xyxys) != 0:
            self.tracks = self.sort.update(self.conf_boxes)
        else:
            self.tracks = self.sort.update()
                  

def getMovedPeopleIDs(prev_boxes: person_bboxes, crnt_boxes: person_bboxes, prev_img: np.ndarray, crnt_img: np.ndarray)-> list:
    """
    Given 2 `person_bboxes` objects, return a list containing ids of people who moved
    Assuming people do not leave or enter the frame once game starts

    Args:
        prev_bboxes (person_bbox): person_bboxes object representing bboxes from the previous frame
        crnt_bboxes (person_bbox): person_bboxes object representing bboxes from the current frame
        prev_img (MatLike): the previous image for detecting movement
        crnt_img (MatLike): the current image for detecting movement
    
    Returns:
        movedPeopleIds (list): list containing ids of people who moved
    """
    movedPeopleIds: list = []
    prev_tracks: np.ndarray = prev_boxes.tracks
    crnt_tracks: np.ndarray  = crnt_boxes.tracks

    
    for i in range(len(prev_tracks)):
        if isMoving(prev_tracks[i],crnt_tracks[i], prev_img, crnt_img):
            movedPeopleIds.append(int(prev_tracks[i][4]))
    
    return movedPeopleIds


def isMoving (prev_box_coord: np.ndarray,crnt_box_coord: np.ndarray, prev_img: np.ndarray, crnt_img: np.ndarray)->bool:
    """
    Given 2 bounding boxes containing a person, (previous frame & current frame), determine if the person moved

    Args:
        prev_box (np.ndarray): NumPy array containing 2 corner coordinates of the bounding box from previous frame [x1,y1,x2,y2,..]
        crnt_box (np.ndarray): NumPy array containing 2 corner coordinates of the bounding box from current frame [x3,y3,x4,y4,..]

    Returns:
        moving (bool): A boolean representing if a person moved between the two frames
    """
#------------------------------- Declare Variables ----------------------------------------------------------------------------#
    BBOX_MOVEMENT_THRESH = 5
    IMAGE_DIFFERENCE_THRESH = 1000
    
    # Prev_box
    xy1 = [prev_box_coord[0],prev_box_coord[1]]
    xy2 = [prev_box_coord[2],prev_box_coord[3]]

    # Crnt_box
    xy3 = [crnt_box_coord[0],crnt_box_coord[1]]
    xy4 = [crnt_box_coord[2],crnt_box_coord[3]]

    moving = False

#-------------------------------- Implement Function-----------------------------------------------------------------------------#

    # What will be considered "moving"?

    # 1. bounding box is in a different location; calculate distance traveled from previous location to current
    dist1 = math.dist(xy1,xy3)
    dist2 = math.dist(xy2,xy4)

    if dist1 >= BBOX_MOVEMENT_THRESH or dist2 >= BBOX_MOVEMENT_THRESH:
        moving = True

    # 2. bounding box is different size
    length1 = abs(xy1[0]-xy2[0])
    height1 = abs(xy1[1]-xy2[1])

    length2 = abs(xy3[0]-xy4[0])
    height2 = abs(xy3[1]-xy4[1])

    if abs(length2 - length1) >= BBOX_MOVEMENT_THRESH or abs(height2 - height1) >= BBOX_MOVEMENT_THRESH:
        moving = True

    # 3. the pixels in the bounding box changed enough
    diff = Diff(prev_img, crnt_img, threshold=25)
    crnt_x1_pixel = round(xy3[0])
    crnt_y1_pixel = round(xy3[1])
    crnt_x2_pixel = round(xy4[0])
    crnt_y2_pixel = round(xy4[1])
    moved_in_bbox = int(np.sum(diff[crnt_y1_pixel:crnt_y2_pixel, crnt_x1_pixel:crnt_x2_pixel]) / 255)

    if moved_in_bbox >= IMAGE_DIFFERENCE_THRESH:
        moving = True

    return moving
    

def getCOM(points: np.ndarray)->tuple:
    """
    Calculate center of mass given a list of points.

    Args:
        points (np.ndarray): list of points represented as a list

    Returns:
        com (tuple): xy coordinates for center of mass represented as a tuple
    """
    com = None
    points_list = points.tolist()
    for i,point in enumerate((points_list)):
        if point[0] == 0 and point[1] == 0:
            points_list.pop(i)
    xsum = 0
    ysum = 0
    count = len(points_list)

    for point in points_list:
        xsum += point[0]
        ysum += point[1]

    com = (int(xsum//count),int(ysum//count))
    return com

def playerOut()->None:
    """
    Play a sound indicating that the player is out from the game
    """
    playsound('playerOut.mp3')