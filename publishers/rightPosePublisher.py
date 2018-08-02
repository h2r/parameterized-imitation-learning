#!/usr/bin/env python  
import roslib
import rospy
import tf
from tf.transformations import euler_from_quaternion
from geometry_msgs.msg import Pose, Point, Quaternion

rospy.init_node('movo_right_tf_listener_pose')

listener = tf.TransformListener()

robot_in_map = rospy.Publisher('tf/right_arm_pose', Pose, queue_size=1)

rate = rospy.Rate(10)
while not rospy.is_shutdown():
    try:
        (trans, rot) = listener.lookupTransform('/base_link', '/right_gripper_base_link', rospy.Time(0))
        msg = Pose(Point(*trans), Quaternion(*rot))

    except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
        continue
    # Publish the message
    robot_in_map.publish(msg)
    rate.sleep()
