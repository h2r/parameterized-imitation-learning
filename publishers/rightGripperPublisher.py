#!/usr/bin/env python  
import roslib
import rospy
import tf
from tf.transformations import euler_from_quaternion
import numpy as np
import message_filters
from std_msgs.msg import Int64, Header
from sensor_msgs.msg import JointState
from imitate_msgs.msg import GripperStamped


is_open = None

def callback(data):
	global is_open
	position = np.array(data.position) - 0.0016998404159380444
	if position.sum() < 0.05:
		is_open = 1
	else:
		is_open = 0

# Publisher From Here
rospy.init_node('movo_right_gripper')
listener = rospy.Subscriber("/movo/right_gripper/joint_states", JointState, callback)

# Give time for initialization
rospy.Rate(1).sleep()

robot_in_map = rospy.Publisher('/movo/right_gripper/gripper_is_open', GripperStamped, queue_size=1)

rate = rospy.Rate(30)
while not rospy.is_shutdown():
	h = Header()
	h.stamp = rospy.Time.now()
	msg = GripperStamped()
	msg.header = h
	msg.data = Int64(is_open)
	robot_in_map.publish(msg)
	rate.sleep()
