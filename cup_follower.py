#!/usr/bin/env python3

import rospy
import cv2
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
from ultralytics import YOLO
from puppy_control_msgs.srv import SetRunActionName

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
            rospy.loginfo("Sending stand action...")
            stand_srv('stand.d6ac', True)
            rospy.sleep(2.0)
            rospy.loginfo("Robot standing.")
        except rospy.ServiceException as e:
            rospy.logerr(f"Stand service failed: {e}")

        # ----------- Load YOLO Model -----------
        self.model = YOLO('/home/ubuntu/puppypi/yolov8n.pt')

        # ----------- ROS Interfaces -----------
        self.image_sub = rospy.Subscriber(
            "/usb_cam/image_raw",
            Image,
            self.image_callback,
            queue_size=1
        )

        self.cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

        # ----------- Control Parameters -----------
        self.linear_speed = 0.12
        self.angular_gain = 0.004
        self.max_angular = 0.6
        self.stop_area = 40000
        self.conf_threshold = 0.4

        # ----------- Performance Control -----------
        self.frame_count = 0
        self.inference_interval = 5  # Run YOLO every 5 frames

        rospy.loginfo("YOLO Cup Follower Started.")

    def image_callback(self, msg):

        self.frame_count += 1

        # Frame skipping
        if self.frame_count % self.inference_interval != 0:
            return

        frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")

        # Reduce resolution for Raspberry Pi
        frame = cv2.resize(frame, (224, 224))

        results = self.model(frame, verbose=False)[0]

        twist = Twist()
        cup_found = False

        for box in results.boxes:
            cls_id = int(box.cls[0])
            class_name = self.model.names[cls_id]
            confidence = float(box.conf[0])

            if class_name == "cup" and confidence > self.conf_threshold:
                cup_found = True

                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                area = (x2 - x1) * (y2 - y1)

                center_x = (x1 + x2) / 2
                image_center = frame.shape[1] / 2
                error_x = center_x - image_center

                if area < self.stop_area:
                    twist.linear.x = self.linear_speed

                    angular = -error_x * self.angular_gain
                    angular = max(min(angular, self.max_angular),
                                  -self.max_angular)

                    twist.angular.z = angular
                else:
                    twist.linear.x = 0.0
                    twist.angular.z = 0.0
                    rospy.loginfo("Cup reached. Stopping.")

                break

        if not cup_found:
            # Search behavior
            twist.linear.x = 0.0
            twist.angular.z = 0.3

        self.cmd_pub.publish(twist)


if __name__ == '__main__':
    try:
        CupFollower()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
