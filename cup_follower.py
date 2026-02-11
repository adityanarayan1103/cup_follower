import rospy
import cv2
import numpy as np
import threading
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from puppy_control.srv import SetRunActionName
from puppy_control.msg import Velocity, Pose
from common import PID  # Use the PID from your robot

class CupFollower:
    def __init__(self):
        rospy.init_node('cup_follower')
        
        self.bridge = CvBridge()
        self.lock = threading.RLock()
        
        # Publishers
        self.velocity_pub = rospy.Publisher('/puppy_control/velocity', Velocity, queue_size=1)
        self.pose_pub = rospy.Publisher('/puppy_control/pose', Pose, queue_size=1)
        
        # State
        self.target_color = None  # Will be set to white
        self.tracker = None
        self.is_running = False
        
        # PID for steering
        self.x_pid = PID.PID(P=0.001, I=0.0001, D=0.00005)
        
        # Stand up first
        rospy.wait_for_service('/puppy_control/runActionGroup')
        stand_srv = rospy.ServiceProxy('/puppy_control/runActionGroup', SetRunActionName)
        stand_srv('stand.d6ac', True)
        rospy.sleep(2.0)
        
        # Subscribe to camera
        self.image_sub = rospy.Subscriber('/usb_cam/image_raw', Image, self.image_callback, queue_size=1)
        
        # Initialize tracker directly (skip color picker for white cup)
        self.target_color = ([255, 255, 255], [255, 255, 255])  # LAB and RGB for white
        self.tracker = ObjectTracker(self.target_color)  # Use your robot's ObjectTracker class
        self.is_running = True  # Start immediately
        
        rospy.loginfo("Cup Follower Started")

    def image_callback(self, msg):
        try:
            # Convert ROS to OpenCV
            rgb_image = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                (msg.height, msg.width, 3)
            )
            
            if self.tracker and self.is_running:
                # Process tracking
                result = self.tracker(rgb_image, rgb_image.copy(), 0.2)
                
                # Control robot
                self.control_movement()
                
        except Exception as e:
            rospy.logerr(f"Error: {e}")

    def control_movement(self):
        with self.lock:
            if self.tracker and self.tracker.last_color_circle:
                (x, y), r = self.tracker.last_color_circle
                
                # Image dimensions
                img_w, img_h = 320, 240
                
                # PID for X (steering)
                self.x_pid.SetPoint = img_w / 2.0
                self.x_pid.update(x)
                yaw_rate = self.x_pid.output
                yaw_rate = max(min(yaw_rate, 0.5), -0.5)
                
                # Area-based forward speed
                area = 3.14159 * r * r
                if area < 5000:
                    forward = 10.0  # Move forward
                elif area > 15000:
                    forward = 0.0   # Stop (close enough)
                else:
                    forward = 5.0   # Slow approach
                
                # Publish velocity
                vel = Velocity()
                vel.x = forward
                vel.y = 0.0
                vel.yaw_rate = -yaw_rate  # Negative to correct direction
                
                self.velocity_pub.publish(vel)
                rospy.loginfo(f"Moving: x={vel.x}, yaw={vel.yaw_rate:.3f}, area={area:.0f}")
            else:
                # Search - rotate in place
                vel = Velocity()
                vel.x = 0.0
                vel.y = 0.0
                vel.yaw_rate = 5.0  # Rotate to search
                self.velocity_pub.publish(vel)
                rospy.loginfo("Searching...")

if __name__ == '__main__':
    try:
        CupFollower()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
