#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from puppy_control.srv import SetRunActionName

class CupFollower:

    def __init__(self):
        rospy.init_node('cup_follower')

        self.bridge = CvBridge()

        # ----------- Stand at Startup -----------
        rospy.loginfo("Waiting for stand service...")
        rospy.wait_for_service('/puppy_control/runActionGroup')

        try:
            stand_srv = rospy.ServiceProxy(
                '/puppy_control/runActionGroup',
                SetRunActionName
            )
            rospy.loginfo("Standing robot...")
            stand_srv('stand.d6ac', True)
            rospy.sleep(2.0)
        except rospy.ServiceException as e:
            rospy.logerr(e)

        # ----------- ROS Interfaces -----------
        self.image_sub = rospy.Subscriber(
            "/usb_cam/image_raw",
            Image,
            self.image_callback,
            queue_size=1
        )

        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

        # ----------- Control Parameters -----------
        self.linear_speed = 0.10
        self.angular_gain = 0.0025
        self.max_angular = 0.5

        # NORMAL CUP SIZE TUNING (for 320x240 resolution)
        self.min_detect_area = 800        # ignore noise
        self.stop_area = 12000            # stop distance
        self.dead_zone = 15               # pixels

        # smoothing
        self.last_area = 0
        self.smooth_factor = 0.6

        rospy.loginfo("Optimized OpenCV Cup Follower Started.")

    def image_callback(self, msg):

        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        frame = cv2.resize(frame, (320, 240))

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # -------- White Cup HSV --------
        lower = np.array([0, 0, 190])
        upper = np.array([180, 50, 255])

        mask = cv2.inRange(hsv, lower, upper)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        twist = Twist()

        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)

            # Ignore very small objects (noise)
            if area < self.min_detect_area:
                self.search(twist)
                self.cmd_pub.publish(twist)
                return

            # Smooth area (avoid jitter)
            area = self.smooth_factor * area + (1 - self.smooth_factor) * self.last_area
            self.last_area = area

            x, y, w, h = cv2.boundingRect(largest)
            center_x = x + w // 2
            image_center = frame.shape[1] // 2
            error_x = center_x - image_center

            # Apply dead zone (avoid shaking)
            if abs(error_x) < self.dead_zone:
                error_x = 0

            # Steering
            angular = -error_x * self.angular_gain
            angular = max(min(angular, self.max_angular), -self.max_angular)

            # Forward control
            if area < self.stop_area:
                twist.linear.x = self.linear_speed
                twist.angular.z = angular
            else:
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                rospy.loginfo("Cup reached. Stopping.")

        else:
            self.search(twist)

        self.cmd_pub.publish(twist)

    def search(self, twist):
        twist.linear.x = 0.0
        twist.angular.z = 0.25


if __name__ == '__main__':
    try:
        CupFollower()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
