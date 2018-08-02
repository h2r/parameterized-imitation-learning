#!/usr/bin/env python  
import roslib
import rospy
import tf
from tf.transformations import euler_from_quaternion
import numpy as np
import message_filters
from std_msgs.msg import Int64
from sensor_msgs.msg import JointState

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

robot_in_map = rospy.Publisher('gripper_is_open', Int64, queue_size=1)

rate = rospy.Rate(10)
while not rospy.is_shutdown():
	
	# try:
	#     (trans, rot) = listener.lookupTransform('/base_link', '/right_gripper_base_link', rospy.Time(0))
	#     msg = Pose(Point(*trans), Quaternion(*rot))

	# except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
	#     continue
	# # Publish the message
	robot_in_map.publish(is_open)
	rate.sleep()
