#!/usr/bin/python
import cv2 
import time
import csv
import os
import sys
import rospy
import itertools
import numpy as np
from cv_bridge import CvBridge
from namedlist import namedlist
from std_msgs.msg import Int64, String
from sensor_msgs.msg import CompressedImage, Image, JointState
from geometry_msgs.msg import Twist, Pose, TwistStamped, PoseStamped
from imitate_msgs.msg import GripperStamped

class ImitateRecorder():
  def __init__(self, task):
    rospy.init_node('imitate_recorder2', log_level=rospy.DEBUG)
    # Create the appropriate directory in the datas for the task we are training
    if not os.path.exists('datas/' + task + '/'):
      os.mkdir('datas/' + task + '/')
    self.save_folder = None # The specific folder
    self.writer = None # The writer to create our txt files
    self.text_file = None # The file that we are writing to currently
    self.is_recording = False # Toggling recording
    self.counter = 0 # The unique key for each datapoint we save
    self.bridge = CvBridge()
    # Initialize current values
    self.Data = namedlist('Data', ['pose', 'twist', 'grip', 'rgb', 'depth'])
    self.data = self.Data(pose=None, twist=None, grip=None, rgb=None, depth=None)

  def toggle_collection(self, toggle):
    if toggle is "0":
      self.counter = 0
      self.is_recording = False
      self.unsubscribe()
      time.sleep(1)
      if self.text_file != None:
        self.text_file.close()
        self.text_file = None
      print("-----Stop Recording-----")
    else:
      save_folder = 'datas/' + task + '/' + str(time.time()) + '/'
      os.mkdir(save_folder)
      self.save_folder = save_folder
      self.text_file = open(save_folder + 'vectors.txt', 'w')
      self.writer = csv.writer(self.text_file)
      self.is_recording = True
      print("=====Start Recording=====")
      self.collect_data()

  def collect_data(self):
    # Initialize Listeners
    self.init_listeners()
    rospy.Rate(5).sleep()
    # Define the rate at which we will collect data
    rate = rospy.Rate(5)
    while not rospy.is_shutdown():
      if None not in self.data:
        print("Data Collected!!")
        rgb_image = self.bridge.imgmsg_to_cv2(self.data.rgb, desired_encoding="passthrough")
        depth_image = self.bridge.imgmsg_to_cv2(self.data.depth, desired_encoding="passthrough")
        cv2.imwrite(self.save_folder + str(self.counter) + '_rgb.png', rgb_image)
        cv2.imwrite(self.save_folder + str(self.counter) + '_depth.png', depth_image)
        posit = self.data.pose.position
        orient = self.data.pose.orientation
        lin = self.data.twist.linear
        ang = self.data.twist.angular
        arr = [self.counter, posit.x, posit.y, posit.z, orient.w, orient.x, orient.y, orient.z, lin.x, lin.y, lin.z, ang.x, ang.y, ang.z, self.data.grip, time.time()]
        self.writer.writerow(arr)
        self.data = self.Data(pose=None, twist=None, grip=None, rgb=None, depth=None)
        self.counter += 1
      rate.sleep()

  def init_listeners(self):
    # The Topics we are Subscribing to for data
    self.right_arm_pose = rospy.Subscriber('/tf/right_arm_pose', PoseStamped, self.pose_callback)
    self.right_arm_vel = rospy.Subscriber('/tf/right_arm_vels', TwistStamped, self.vel_callback)
    self.rgb_state_sub = rospy.Subscriber('/kinect2/qhd/image_color_rect', Image, self.rgb_callback)
    self.depth_state_sub = rospy.Subscriber('/kinect2/qhd/image_depth_rect', Image, self.depth_callback)
    self.gripper_state_sub = rospy.Subscriber('/movo/right_gripper/gripper_is_open', GripperStamped, self.gripper_callback)

  def unsubscribe(self):
    self.right_arm_pose.unregister()
    self.right_arm_vel.unregister()
    self.rgb_state_sub.unregister()
    self.depth_state_sub.unregister()
    self.gripper_state_sub.unregister()

  def pose_callback(self, pose):
    if None in self.data:
      self.data.pose = pose.pose

  def vel_callback(self, twist):
    if None in self.data:
      self.data.twist = twist.twist

  def rgb_callback(self, rgb):
    if None in self.data:
      self.data.rgb = rgb

  def depth_callback(self, depth):
    if None in self.data:
      self.data.depth = depth

  def gripper_callback(self, gripper):
    if None in self.data:
      self.data.grip = gripper.data.data

if __name__ == '__main__':
  task = sys.argv[1]
  recorder = ImitateRecorder(task)
  recorder.toggle_collection("1")
  recorder.toggle_collection("0")
