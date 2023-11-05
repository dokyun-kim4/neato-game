import rclpy
from threading import Thread
from rclpy.node import Node
import time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry as Odom
import cv2 as cv
from ultralytics import YOLO
from .helpers import *
import time
import numpy as np
import math
from .sort import Sort

class neatoGame(Node):
    def __init__(self, image_topic):
        super().__init__('run_neato_game') # type: ignore

        # Load YOLO model
        self.model = YOLO('yolov8n-pose.pt')
        self.colors = np.random.uniform(0,255,size=(999,3))
        self.outHistory = []

        self.prev_people = None # This would be person_bboxes object
        self.prev_frame = None # This would be the previous image
        self.sort = Sort()
        self.detect_movement = False
        self.initial_angle: float | None = None
        self.angle: float = 0.0

        self.crnt_frame =  None                        # the latest image from the camera
        self.bridge = CvBridge()                    # used to convert ROS messages to OpenCV

        self.create_subscription(Image, image_topic, self.process_image, 10)
        self.odom_sub = self.create_subscription(Odom,'odom',callback=self.update_angle,qos_profile=10)
        self.pub = self.create_publisher(Twist, 'cmd_vel', 10)
        thread = Thread(target=self.loop_wrapper)
        thread.start()

    def update_angle(self, msg: Odom):
        """Update the angle of the Neato every time an Odom message is recieved"""
        # get the angle
        angle: float = get_angle(msg.pose.pose.orientation)
        # set the anlge the Neato started at if it hasnt been set already (self.angle would be 0, which is why it isn't set)
        if self.initial_angle == None:
            self.initial_angle = angle
        # otherwise, set the current angle, and make sure it is between 0 and 360
        else:
            angle = self.initial_angle - angle
            if angle < 0:
                angle += 360
            self.angle = angle

    def process_image(self, msg):
        """ Process image messages from ROS and stash them in an attribute
            called cv_image for subsequent processing """
        self.crnt_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            # Feed frame into YOLO model
        results = self.model(self.crnt_frame,verbose=False)
        # Get results
        attr = results[0]
        boxes = attr.boxes.cpu().numpy()
        xyxys = boxes.xyxy
        confs = boxes.conf

        # Store all results as a `person_bboxes` object
        crnt_people = person_bboxes(xyxys,confs,self.sort)
        crnt_people.update()

        if self.prev_people is None or self.prev_frame is None and self.crnt_frame:
            self.prev_people = crnt_people
            self.prev_frame = self.crnt_frame.copy() # type: ignore


        if self.detect_movement:
            ids = getMovedPeopleIDs(self.prev_people,crnt_people,self.prev_frame,self.crnt_frame) # type: ignore
            for id in ids:
                if id not in self.outHistory:
                    self.outHistory.append(id)
                    print(f"Person #{int(id)} has moved!")
                    # Robot will turn to person here
                    # Need a function to calculate where to turn to
                    t1 = Thread(target = playerOut)
                    t1.start()

            # update the previous values, must be done before drawing bounding box
            self.prev_people = crnt_people
            self.prev_frame = self.crnt_frame.copy() # type: ignore

            # Draw bounding box (SORT)
            for track in crnt_people.tracks:
                x1,y1,x2,y2,track_id = int(track[0]),int(track[1]),int(track[2]),int(track[3]),int(track[4])
                cv.rectangle(self.crnt_frame,(x1,y1),(x2,y2),self.colors[track_id],2)
                cv.putText(self.crnt_frame,str(track_id),(x1+10,y1+40), cv.FONT_HERSHEY_PLAIN,3,self.colors[track_id],2)
        else:
            # update the previous values, must be done before drawing bounding box, which is why it in in both the if and else
            self.prev_people = crnt_people
            self.prev_frame = self.crnt_frame.copy() # type: ignore

    def turnTo(self, deg, max_speed=120.0, thresh=5.0) -> None:
        """
        Command the Neato to turn to a specific angle, with an optional maximum speed, without blocking the code
        
        Args: 
            deg (float): the angle to turn to in degrees
            max_speed (float): the maximum angular speed of the Neato in degrees per second, defaults to 120
            thresh (float): the theshold for the target angle in degrees, defaults to 5
        """
        # Start a thread so this doesn't block the run loop
        Thread(target=self.turnToThreadTarget, args=(deg, max_speed, thresh)).start()

    def turnToThreadTarget(self, deg, max_speed, thresh) -> None:
        """
        Command the Neato to turn to a specific angle, with an optional maximum speed. Blocks code.
        
        Args: 
            deg (float): the angle to turn to in degrees
            max_speed (float): the maximum angular speed of the Neato in degrees per second
            thresh (float): the theshold for the target angle in degrees
        """
        # create the message to publish to the Neato to make it move
        msg = Twist()

        # find how much the Neato needs to turn to get to the target angle
        delta_angle = get_delta_angle(self.angle, deg)
        # repeat until the Neato is pretty close to the target angle
        while abs(delta_angle) > thresh:
            
            # get the direction and speed
            direction = np.sign(delta_angle)
            speed = min(abs(delta_angle), max_speed)

            # tell the Neato to move
            msg.angular.z = np.deg2rad(speed) * -direction
            self.pub.publish(msg)

            # wait a bit so the Neato actually has time to do something
            time.sleep(0.05)

            # recalculate how much the Neato needs to turn to get to the target angle
            delta_angle = get_delta_angle(self.angle, deg)

        # tell the Neato to stop
        msg.angular.z = 0.0
        self.pub.publish(msg)
        

    def loop_wrapper(self):
        """ This function takes care of calling the run_loop function repeatedly.
            We are using a separate thread to run the loop_wrapper to work around
            issues with single threaded executors in ROS2 """
        cv.namedWindow('video_window')
        while True:
            # Turn towards people
            # Start scanning for t seconds
            # if movement detected, indicate ppl got out
            # Turn back to rest position
            # wait for randomly decided time
            self.run_loop()
            time.sleep(0.1)

    
    def run_loop(self):
        # NOTE: only do cv2.imshow and cv2.waitKey in this function 
        if not self.crnt_frame is None:
            cv.imshow('video_window', self.crnt_frame)
            waitKey = cv.waitKey(1) & 0xFF
            if waitKey == ord('w'):
                self.detect_movement = not self.detect_movement
            if waitKey == ord('t'):
                # tell the Neato to stop so it's easy to stop if something goes wrong. Probably not nescessary any more, but nice to have
                msg = Twist()
                msg.angular.z = 0.0
                self.pub.publish(msg)

                # get an angle from the user and tell the Neato to go to it
                command_angle = float(input("What angle should the neato go to? "))
                self.turnTo(command_angle)

if __name__ == '__main__':
    node = neatoGame("/camera/image_raw")
    node.run() # type: ignore


def main(args=None):
    rclpy.init()
    n = neatoGame("camera/image_raw")
    rclpy.spin(n)
    rclpy.shutdown()
