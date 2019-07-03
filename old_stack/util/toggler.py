#!/usr/bin/python
import sys
import rospy
import subprocess
from std_msgs.msg import String

p = None
is_recording = False
task = sys.argv[1]

def toggle(value):
	global p
	global is_recording
	global task
	
	if value.data is "0":
		if is_recording:
			p.terminate()
			is_recording = False
	elif value.data is "1":
		if not is_recording:
			p = subprocess.Popen(["./record.py", task])
			is_recording = True

rospy.init_node('imitate_toggler', log_level=rospy.DEBUG)
toggle_sub = rospy.Subscriber('/unity_learning_record', String, toggle)
rospy.spin()
