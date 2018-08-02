#!/usr/bin/env python  
import numpy as np
import roslib
import rospy
import tf
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import Twist, Vector3

rospy.init_node('movo_right_tf_listener')

listener = tf.TransformListener()

robot_in_map = rospy.Publisher('tf/right_arm_vels', Twist, queue_size=1)

prev_time = rospy.get_time()
curr_time = rospy.get_time()
prev_rot, prev_trans = None, None

rate = rospy.Rate(10)
while not rospy.is_shutdown():
    try:
        curr_time = rospy.get_time()
        (curr_trans, curr_rot) = listener.lookupTransform('/base_link', '/right_gripper_base_link', rospy.Time(0))
        # First Iteration
        if not prev_rot and not prev_trans:
            prev_trans, prev_rot = curr_trans, curr_rot
            continue
        delta = float(curr_time)-float(prev_time)
        lin_vel = (np.array(curr_trans) - np.array(prev_trans))/delta
        ang_vel = (np.array(euler_from_quaternion(curr_rot)) - np.array(euler_from_quaternion(prev_rot)))/delta
        msg = Twist(Vector3(*lin_vel), Vector3(*ang_vel))

        # Update
        prev_time = curr_time
        prev_trans, prev_rot = curr_trans, curr_rot

    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        continue
    # Publish the message
    robot_in_map.publish(msg)
    rate.sleep()
