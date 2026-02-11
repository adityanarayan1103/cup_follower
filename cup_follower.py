#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from ultralytics import YOLO

class YoloCupFollower:

    def __init__(self):
        rospy.init_node('yolo_cup_follower')

        self.bridge = CvBridge()

        # Load YOLO model
        self.model = YOLO('/home/ubuntu/yolov8n.pt')

        self.image_sub = rospy.Subscriber(
            "/usb_cam/image_raw",
            Image,
            self.image_callback
        )

        self.cmd_pub = rospy.Publisher(
            "/cmd_vel",
            Twist,
            queue_size=10
        )

        # Parameters
        self.linear_speed = 0.15
        self.angular_gain = 0.003
        self.max_angular_speed = 0.6
        self.stop_area_threshold = 50000

        self.state = "SEARCH"

        rospy.loginfo("YOLO Cup Follower Started.")

    def image_callback(self, msg):

        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        results = self.model(frame, verbose=False)[0]

        twist = Twist()

        cup_detected = False

        for box in results.boxes:
            cls_id = int(box.cls[0])
            class_name = self.model.names[cls_id]

            if class_name == "cup":
                cup_detected = True

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                area = (x2 - x1) * (y2 - y1)

                center_x = (x1 + x2) / 2
                image_center = frame.shape[1] / 2
                error_x = center_x - image_center

                if area < self.stop_area_threshold:
                    self.state = "APPROACH"
                    twist.linear.x = self.linear_speed

                    angular = -error_x * self.angular_gain
                    angular = max(min(angular, self.max_angular_speed),
                                  -self.max_angular_speed)
                    twist.angular.z = angular
                else:
                    if self.state != "STOP":
                        rospy.loginfo("Cup reached. Stopping.")
                        self.state = "STOP"

                    twist.linear.x = 0.0
                    twist.angular.z = 0.0

                break

        if not cup_detected:
            # SEARCH behavior
            self.state = "SEARCH"
            twist.linear.x = 0.0
            twist.angular.z = 0.3

        self.cmd_pub.publish(twist)


if __name__ == '__main__':
    try:
        YoloCupFollower()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
