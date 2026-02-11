#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

class CupFollower:

    def __init__(self):
        rospy.init_node('cup_follower')

        self.bridge = CvBridge()

        self.image_sub = rospy.Subscriber(
            "/image_raw",
            Image,
            self.image_callback
        )

        self.cmd_pub = rospy.Publisher(
            "/cmd_vel",
            Twist,
            queue_size=10
        )

        self.target_distance_area = 15000  # stop threshold
        self.linear_speed = 0.15
        self.angular_gain = 0.002

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Example: detect white cup (adjust range!)
        lower = np.array([0, 0, 200])
        upper = np.array([180, 40, 255])

        mask = cv2.inRange(hsv, lower, upper)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        twist = Twist()

        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)

            x, y, w, h = cv2.boundingRect(largest)
            center_x = x + w // 2

            image_center = frame.shape[1] // 2
            error_x = center_x - image_center

            # Rotate toward cup
            twist.angular.z = -error_x * self.angular_gain

            # Move forward until close enough
            if area < self.target_distance_area:
                twist.linear.x = self.linear_speed
            else:
                twist.linear.x = 0.0
                rospy.loginfo("Cup reached. Stopping.")

        else:
            twist.linear.x = 0.0
            twist.angular.z = 0.0

        self.cmd_pub.publish(twist)

if __name__ == '__main__':
    CupFollower()
    rospy.spin()
