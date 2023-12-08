# ROS imports
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry as Odom
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from neato2_interfaces.msg import Bump

# Computer vision imports
import cv2 as cv
from ultralytics import YOLO
from .sort import Sort

# Misc.
import time
from threading import Thread
import numpy as np
import math
from typing import Literal
import random

# Custom
from .helpers import *



States = Literal["wait", "turn_to", "scan_and_elim", "turn_away", "game_end"]

class neatoGame(Node):
    def __init__(self, image_topic):
        super().__init__('run_neato_game') # type: ignore

        # Load YOLO model
        self.model = YOLO('yolov8n-pose.pt')
        self.colors = np.random.uniform(0,255,size=(999,3))
        self.outHistory = []

        self.crnt_people = None # This would be person_bboxes object
        self.prev_people = None # This would be person_bboxes object
        self.prev_frame = None # This would be the previous image
        self.sort = Sort()
        self.initial_angle: float | None = None
        self.angle: float = 0.0
        self.bumped = False

        self.crnt_frame =  None  # the latest image from the camera
        self.bridge = CvBridge() # used to convert ROS messages to OpenCV
        self.waitTime = 5 # countdown delay before turning towards players (sec)
        self.state: States
        self.initial_player_count = None

        self.create_subscription(Image, image_topic, self.process_image, 10)
        self.create_subscription(Odom,'odom', self.update_angle, 10)
        self.create_subscription(Bump, 'bump', self.update_bumped, 10)
        self.pub = self.create_publisher(Twist, 'cmd_vel', 10)
        thread = Thread(target=self.loop_wrapper)
        thread.start()

    def update_angle(self, msg: Odom):
        """Update the angle of the Neato every time an Odom message is recieved"""
        # get the angle
        angle: float = get_angle(msg.pose.pose.orientation)
        # set the anlge the Neato started at if it hasnt been set already (self.angle would be 0, which is why it isn't set)
        if self.initial_angle is None:
            self.initial_angle = angle
        # otherwise, set the current angle, and make sure it is between 0 and 360
        else:
            angle = self.initial_angle - angle
            if angle < 0:
                angle += 360
            self.angle = angle

    def update_bumped(self, msg):
        self.bumped=   msg.left_front == 1 or \
                            msg.left_side == 1 or \
                            msg.right_front == 1 or \
                            msg.right_side == 1

    def process_image(self, msg):
        """ Process image messages from ROS and stash them in an attribute
            called cv_image for subsequent processing """
        if self.crnt_frame is not None:
            self.prev_frame = self.crnt_frame.copy() # type: ignore
        self.crnt_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")

        # # Feed frame into YOLO model
        # results = self.model(self.crnt_frame,verbose=False)
        # # Get results
        # attr = results[0]
        # boxes = attr.boxes.cpu().numpy()
        # xyxys = boxes.xyxy
        # confs = boxes.conf

        # # Store all results as a `person_bboxes` object
        # if self.crnt_people is not None:
        #     self.prev_people = self.crnt_people
        # self.crnt_people = person_bboxes(xyxys,confs,self.sort)
        # self.crnt_people.update()

    def scan_and_eliminate(self):
        
        #Feed frame into YOLO model
        results = self.model(self.crnt_frame,verbose=False)
        # Get results
        attr = results[0]
        boxes = attr.boxes.cpu().numpy()
        xyxys = boxes.xyxy
        confs = boxes.conf
        # Store all results as a `person_bboxes` object
        if self.crnt_people is not None:
            self.prev_people = self.crnt_people
        self.crnt_people = person_bboxes(xyxys,confs,self.sort)
        self.crnt_people.update()

        if self.prev_people is not None:
            if self.initial_player_count is None:
                self.initial_player_count = len(self.prev_people.tracks)


            ids = getMovedPeopleIDs(self.prev_people,self.crnt_people,self.prev_frame,self.crnt_frame) # type: ignore
            for id in ids:
                if id not in self.outHistory:
                    self.outHistory.append(id)
                    print(f"Person #{int(id)} has moved!")
                    # Robot will turn to person here
                    # Need a function to calculate where to turn to
                    t1 = Thread(target = playerOut, args=(int(id),))
                    t1.start()

        # Draw bounding box (SORT)
        bbox_frame = self.crnt_frame
        for track in self.crnt_people.tracks: # type: ignore
            x1,y1,x2,y2,track_id = int(track[0]),int(track[1]),int(track[2]),int(track[3]),int(track[4])
            cv.rectangle(bbox_frame,(x1,y1),(x2,y2),self.colors[track_id],2) # type: ignore
            cv.putText(bbox_frame,str(track_id),(x1+10,y1+40), cv.FONT_HERSHEY_PLAIN,3,self.colors[track_id],2) # type: ignore
        cv.imshow('video_window', bbox_frame) # type: ignore

    def turn_towards(self, deg: float, max_speed=120.0, thresh=5.0) -> bool:
        """
        Command the Neato to turn towards a specific angle, with an optional maximum speed
        
        Args: 
            deg (float): the angle to turn to in degrees
            max_speed (float): the maximum angular speed of the Neato in degrees per second, defaults to 120
            thresh (float): the theshold for the target angle in degrees, defaults to 5
        
        Returns:
            (bool): True if the Neato is done turning
        """
        # create the message to publish to the Neato to make it move
        msg = Twist()

        # find how much the Neato needs to turn to get to the target angle
        delta_angle = get_delta_angle(self.angle, deg)
            
        # if the Neato has reached the desired angle...
        if abs(delta_angle) < thresh:
            # tell the Neato to stop turning
            msg.angular.z = 0.0
            self.pub.publish(msg)

            # return True to go to the next state
            return True
        # otherwise, the Neato needs to turn

        # get the direction and speed
        direction = np.sign(delta_angle)
        speed = min(abs(delta_angle), max_speed)

        # tell the Neato to move
        msg.angular.z = np.deg2rad(speed) * -direction
        self.pub.publish(msg)

        # return False to stay in this state
        return False
    
    def reached_time(self, next_state: States):
        Thread(target=self.wait_random_thread_target, args=(next_state))

    def wait_random_thread_target(self, next_state: States):
        min = 2
        max = 5
        wait_time = random.random() * (max-min) + min
        time.sleep(wait_time)
        self.state = next_state

    def loop_wrapper(self):
        """ This function takes care of calling the run_loop function repeatedly.
            We are using a separate thread to run the loop_wrapper to work around
            issues with single threaded executors in ROS2 """
        cv.namedWindow('video_window')

        self.state = "wait"
        target_time = time.time() + random.uniform(2, 5)
        t2 = Thread(target = countdown)
        t2.start()


        while True:
            
            print(self.prev_people,self.crnt_people)

            if self.bumped or (len(self.outHistory) == self.initial_player_count):
                self.state = "game_end"

            match self.state:
                case "wait":
                    if time.time() > target_time:
                        self.state = "turn_to"
                case "turn_to":
                    done_turning = self.turn_towards(180)
                    if done_turning:
                        self.state = "scan_and_elim"
                        target_time = time.time() + random.uniform(2, 5)
                        time.sleep(0.3)
                        
                case "scan_and_elim":
                    self.scan_and_eliminate()
                    if time.time() > target_time:
                        self.state = "turn_away"
                        self.prev_people, self.crnt_people,self.prev_frame = None, None, None
                case "turn_away":
                    done_turning = self.turn_towards(0)
                    if done_turning:
                        self.state = "wait"
                        target_time = time.time() + self.waitTime
                        t2 = Thread(target = countdown)
                        t2.start()
                        
                case _:
                    break

            self.run_loop()
            time.sleep(0.1)

    
    def run_loop(self):
        # NOTE: only do cv2.imshow and cv2.waitKey in this function 
        if not self.crnt_frame is None:
            waitKey = cv.waitKey(1) & 0xFF

if __name__ == '__main__':
    node = neatoGame("/camera/image_raw")
    node.run() # type: ignore

def main(args=None):
    rclpy.init()
    n = neatoGame("camera/image_raw")
    rclpy.spin(n)
    rclpy.shutdown()
