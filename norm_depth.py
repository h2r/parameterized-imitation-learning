import numpy as np
import sys
import cv2
from cv_bridge import CvBridge
import rospy
from  sensor_msgs.msg import Image 

rospy.init_node("check_depth")
sub = rospy.wait_for_message("/kinect2/qhd/image_depth_rect", Image)
bridge = CvBridge()
image = bridge.imgmsg_to_cv2(sub, desired_encoding="passthrough")
min_depth = np.amin(image)
max_depth = np.amax(image)

depth_map = np.uint8(255.0*(image - min_depth)/(max_depth-min_depth))
cv2.imwrite("normalized_depth.png", depth_map)
