#!/usr/bin/python
import rospy
import moveit_python
from moveit_msgs.msg import CollisionObject
from geometry_msgs.msg import Pose, Point, Quaternion
from shape_msgs.msg import SolidPrimitive
from std_msgs.msg import Header

rospy.init_node("moveit_table_scene")
scene = moveit_python.PlanningSceneInterface("base_link")
#pub = rospy.Publisher("/move_group/monitored_planning_scene", CollisionObject, queue_size=1)
table = CollisionObject()
# Header
h = Header()
h.stamp = rospy.Time.now()
# Shape
box = SolidPrimitive()
box.type = SolidPrimitive.BOX
box.dimensions = [0.808, 1.616, 2.424]
# Pose
position = Point(*[1.44, 0.0, 0.03])
orient = Quaternion(*[-0.510232, 0.49503, 0.515101, 0.478832])
# Create Collision Object
scene.addSolidPrimitive("table", box, Pose(*[position, orient]))
