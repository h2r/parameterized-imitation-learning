#!/usr/bin/python
import sys
import rospy
import itertools
import message_filters
import numpy as np
import time
import cv2
import moveit_commander
from cv_bridge import CvBridge
from std_msgs.msg import Int64
from geometry_msgs.msg import Point, Quaternion, Pose, Twist
from sensor_msgs.msg import CompressedImage, JointState

class ApproxTimeSync(message_filters.ApproximateTimeSynchronizer):
  def add(self, msg, my_queue, my_queue_index=None):
        self.allow_headerless = True
        if hasattr(msg, 'timestamp'):
            stamp = msg.timestamp
        elif not hasattr(msg, 'header') or not hasattr(msg.header, 'stamp'):
            if not self.allow_headerless:
                rospy.logwarn("Cannot use message filters with non-stamped messages. "
                              "Use the 'allow_headerless' constructor option to "
                              "auto-assign ROS time to headerless messages.")
                return
            stamp = rospy.Time.now()
	else:
            stamp = msg.header.stamp
	
	#TODO ADD HEADER TO ALLOW HEADERLESS
	# http://book2code.com/ros_kinetic/source/ros_comm/message_filters/src/message_filters/__init__.y
        #setattr(msg, 'header', a)
        #msg.header.stamp = stamp
	#super(message_filters.ApproximateTimeSynchronizer, self).add(msg, my_queue)
        self.lock.acquire()
        my_queue[stamp] = msg
        while len(my_queue) > self.queue_size:
            del my_queue[min(my_queue)]
        # self.queues = [topic_0 {stamp: msg}, topic_1 {stamp: msg}, ...]
        if my_queue_index is None:
            search_queues = self.queues
        else:
            search_queues = self.queues[:my_queue_index] + \
                self.queues[my_queue_index+1:]
        # sort and leave only reasonable stamps for synchronization
        stamps = []
        for queue in search_queues:
            topic_stamps = []
            for s in queue:
                stamp_delta = abs(s - stamp)
                if stamp_delta > self.slop:
                    continue  # far over the slop
                topic_stamps.append((s, stamp_delta))
            if not topic_stamps:
                self.lock.release()
                return
            topic_stamps = sorted(topic_stamps, key=lambda x: x[1])
            stamps.append(topic_stamps)
        for vv in itertools.product(*[zip(*s)[0] for s in stamps]):
            vv = list(vv)
            # insert the new message
            if my_queue_index is not None:
                vv.insert(my_queue_index, stamp)
            qt = list(zip(self.queues, vv))
            if ( ((max(vv) - min(vv)) < self.slop) and
                (len([1 for q,t in qt if t not in q]) == 0) ):
                msgs = [q[t] for q,t in qt]
                self.signalMessage(*msgs)
                for q,t in qt:
                    del q[t]
                break  # fast finish after the synchronization
        self.lock.release()

class ImitateLearner():
	"""
	This class will evaluate the outputs of the net.
	"""
	def __init__(self, arm='right'):
		rospy.init_node("{}_arm_eval".format(arm))
		self.queue = []
		self.rgb = None
		self.depth = None
		self.pos = None
		self.orient = None
		self.prevTime = None
		self.time = None
		self.arm = arm

		# Initialize Subscribers
		self.listener()

		# Enable Robot
		moveit_commander.roscpp_initialize(sys.argv)
		robot = moveit_commander.RobotCommander()
		self.group_arms = moveit_commander.MoveGroupCommander('upper_body')
		self.group_arms.set_pose_reference_frame('/base_link')
		self.left_ee_link = 'left_ee_link'
		self.right_ee_link = 'right_ee_link'

		# Set the rate of our evaluation
		rate = rospy.Rate(0.5)

		# Give time for initialization
		rospy.Rate(1).sleep()

		# This is to the get the time delta
		first_time = True
		while not rospy.is_shutdown():
			# Todo: Connect Net
			if first_time:
				self.prevTime = time.time()
				first_time = False
			# Given the output, solve for limb joints
			#limb_joints = self.get_limb_joints(output)

			# If valid joints then move to joint
			#if limb_joints is not -1:
			#	right.move_to_joint_positions(limb_joints)
			#	self.prevTime = self.time
			#else:
			#	print 'ERROR: IK solver returned -1'
			print(self.pos)
			rate.sleep()

	def listener(self):
		"""
		Listener for all of the topics
		"""
		print("Listener Initialized")
		pose_sub = message_filters.Subscriber('/tf/{}_arm_pose'.format(self.arm), Pose)
		twist_sub = message_filters.Subscriber('tf/{}_arm_vels'.format(self.arm), Twist)
		rgb_sub = message_filters.Subscriber('/kinect2/sd/image_color_rect/compressed', CompressedImage)
		depth_sub = message_filters.Subscriber('/kinect2/sd/image_depth_rect/compressed', CompressedImage)
		gripper_sub = message_filters.Subscriber('/movo/{}_gripper/gripper_is_open'.format(self.arm), Int64)
		ts = ApproxTimeSync([pose_sub, twist_sub, rgb_sub, depth_sub, gripper_sub], 1, 0.1)
		ts.registerCallback(self.listener_callback)

	def listener_callback(self, pose, twist, rgb, depth, gripper):
		"""
		This method updates the variables.
		"""
		bridge = CvBridge()
		self.time = time.time()
		self.rgb = bridge.compressed_imgmsg_to_cv2(rgb)
		self.depth = bridge.compressed_imgmsg_to_cv2(depth)
		self.pos = pose.position
		self.orient = pose.orientation
		# Create input for net. x, y, z
		queue_input = np.array([self.pos.x, self.pos.y, self.pos.z])
		if len(self.queue) == 0:
			self.queue = [queue_input for i in range(5)]
		else:
			self.queue.pop(0)
			self.queue.append(queue_input)

	def get_next_pose(self, output):
		"""
		This method gets the ik_solver solution for the arm joints.
		"""
		[goal_pos, goal_orient] = self.calculate_move(np.reshape(output[0, :3], (3,)), np.reshape(output[0, 3:], (3,)))
		return Point(*goal_pos), Quaternion(*goal_orient)
		
	def calculate_move(self, lin, ang):
		"""
		This calculates the position and orientation (in quaterion) of the next pose given
		the linear and angular velocities outputted by the net.
		"""
		delta = self.time - self.prevTime
		print("------------")
		print(delta)
		print("------------")
		#delta = 1/30
		# Position Update
		curr_pos = np.array([self.pos.x, self.pos.y, self.pos.z])
		goal_pos = np.add(curr_pos, delta*np.array(lin))
		# Orientation Update
		curr_orient = np.array([self.orient.x, self.orient.y, self.orient.z, self.orient.w])
		w_ang = np.concatenate([[0], ang])
		goal_orient = np.add(curr_orient, 0.5*delta*np.matmul(w_ang, np.transpose(curr_orient)))
		# Update the prevTime
		return goal_pos, goal_orient

if __name__ == '__main__':
	learner = ImitateLearner()