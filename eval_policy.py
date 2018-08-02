#!/usr/bin/python
import rospy
import itertools
import message_filters
import numpy as np
import time
import cv2
from cv_bridge import CvBridge
from std_msgs.msg import Int64
from geometry_msgs.msg import Point, Quaternion
from sensor_msgs.msg import CompressedImage, JointState
