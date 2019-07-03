#!/usr/bin/env python  
import roslib
import rospy
from std_msgs.msg import Int64

# Publisher From Here
rospy.init_node('toggle_test')

robot_in_map = rospy.Publisher('/toggle', Int64, queue_size=1)

rate = rospy.Rate(100)
while not rospy.is_shutdown():
	rate.sleep()