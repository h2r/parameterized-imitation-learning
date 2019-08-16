#!/usr/bin/python
import cv2
import time
import csv
import os
import sys
import rospy
import itertools
import numpy as np
#from tf.transformations import euler_from_quaternion
from cv_bridge import CvBridge
from namedlist import namedlist
from std_msgs.msg import Int64, String
from sensor_msgs.msg import CompressedImage, Image, JointState
from geometry_msgs.msg import Twist, Pose, TwistStamped, PoseStamped, Vector3
import torch
from model import Model
import argparse
import copy
from matplotlib import pyplot as plt
import signal

class ImitateEval:
    def __init__(self, weights):
        self.bridge = CvBridge()
        self.Data = namedlist('Data', ['pose', 'rgb', 'depth'])
        self.data = self.Data(pose=None, rgb=None, depth=None)
        self.is_start = True

        checkpoint = torch.load(weights, map_location="cpu")
        self.model = Model(**checkpoint['kwargs'])
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def change_start(self):
        radius = 0.07
        # Publisher for the movement and the starting pose
        self.movement_publisher = rospy.Publisher('/iiwa/CollisionAwareMotion', Pose, queue_size=10)
        self.target_start = Pose()
        self.target_start.position.x = -0.15 + np.random.rand()*2*radius - radius # -0.10757
        self.target_start.position.y = 0.455 + np.random.rand()*2*radius - radius # 0.4103
        self.target_start.position.z = 1.015
        self.target_start.orientation.x = 0.0
        self.target_start.orientation.y = 0.0
        self.target_start.orientation.z = 0.7071068
        self.target_start.orientation.w = 0.7071068


    def move_to_button(self, tau, tolerance):
        self.init_listeners()
        rospy.Rate(5).sleep()
        rate = rospy.Rate(15)

        pose_to_move = copy.deepcopy(self.target_start)
        eof = []
        while not rospy.is_shutdown():
            if None not in self.data:
                # Position from the CartesianPose Topic!!
                pos = self.data.pose.position
                pos = [pos.x, pos.y, pos.z]
                if self.is_start:
                    for _ in range(5):
                        eof += pos
                    self.is_start = False
                else:
                    eof = pos + eof[:-3]

                eof_input = torch.from_numpy(np.array(eof)).type(torch.FloatTensor)
                eof_input = eof_input.unsqueeze(0)#.zero_()

                tau_input = torch.Tensor(tau).to(eof_input).view(1, -1)

                rgb = self.process_images(self.data.rgb, True)
                depth = self.process_images(self.data.depth, False)

                # print("RGB min: {}, RGB max: {}".format(np.amin(rgb), np.amax(rgb)))
                # print("Depth min: {}, Depth max: {}".format(np.amin(depth), np.amax(depth)))
                print("EOF: {}".format(eof_input))
                print("Tau: {}".format(tau_input))

                torch.save(rgb, "/home/amazon/Desktop/rgb_tensor.pt")
                torch.save(depth, "/home/amazon/Desktop/depth_tensor.pt")
                torch.save(eof, "/home/amazon/Desktop/eof_tensor.pt")
                torch.save(tau_input, "/home/amazon/Desktop/tau.pt")

                with torch.no_grad():
                    out, aux = self.model(rgb, depth, eof_input, tau_input)
                    torch.save(out, "/home/amazon/Desktop/out.pt")
                    torch.save(aux, "/home/amazon/Desktop/aux.pt")
                    out = out.squeeze()
                    x_cartesian = out[0].item()
                    y_cartesian = out[1].item()
                    z_cartesian = out[2].item()
                print("X:{}, Y:{}, Z:{}".format(x_cartesian, y_cartesian, z_cartesian))
                print("Aux: {}".format(aux))
                # This new pose is the previous pose + the deltas output by the net, adjusted for discrepancy in frame
                # It used to be:
                # pose_to_move.position.x += -y_cartesian
                # pose_to_move.position.y += x_cartesian
                # pose_to_move.position.z += z_cartesian

                pose_to_move.position.x -= y_cartesian
                pose_to_move.position.y += x_cartesian
                pose_to_move.position.z += z_cartesian
                #print(pose_to_move)

                # Publish to Kuka!!!!
                for i in range(10):
                    self.movement_publisher.publish(pose_to_move)
                    rospy.Rate(10).sleep()

                rospy.wait_for_message("/iiwa/CollisionAwareExecutionStatus", String)
                # End publisher

                self.data = self.Data(pose=None,rgb=None,depth=None)
                rate.sleep()

                print(pose_to_move.position.y, -pose_to_move.position.x, pose_to_move.position.z)
                print(distance(tau, (pose_to_move.position.y, -pose_to_move.position.x, pose_to_move.position.z)))
                if distance(tau, (pose_to_move.position.y, -pose_to_move.position.x, pose_to_move.position.z)) < tolerance:
                    break

    def process_images(self, img_msg, is_it_rgb):
        crop_right=586
        crop_lower=386
        img = self.bridge.compressed_imgmsg_to_cv2(img_msg, desired_encoding="passthrough")
        if(is_it_rgb):
            img = img[:,:,::-1]
        # Does this crop work?
        #rgb = img[0:386, 0:586]
        #rgb = img.crop((0, 0, crop_right, crop_lower))
        rgb = cv2.resize(img, (160,120))
        rgb = np.array(rgb).astype(np.float32)

        rgb = 2*((rgb - np.amin(rgb))/(np.amax(rgb)-np.amin(rgb)))-1


        rgb = torch.from_numpy(rgb).type(torch.FloatTensor)
        if is_it_rgb:
            rgb = rgb.view(1, rgb.shape[0], rgb.shape[1], rgb.shape[2]).permute(0, 3, 1, 2)
        else:
            rgb = rgb.view(1, 1, rgb.shape[0], rgb.shape[1])
            #plt.imshow(rgb[0,0] / 2 + .5)
            # plt.show()

        return rgb

    def move_to_start(self):
        # Publish starting position to Kuka!!!!
        for i in range(10):
            self.movement_publisher.publish(self.target_start)
            rospy.Rate(10).sleep()

        rospy.wait_for_message("/iiwa/CollisionAwareExecutionStatus", String)
        # End publisher

    def init_listeners(self):
        # The Topics we are Subscribing to for data
        self.right_arm_pose = rospy.Subscriber('/iiwa/state/CartesianPose', PoseStamped, self.pose_callback)
        self.rgb_state_sub = rospy.Subscriber('/camera3/camera/color/image_rect_color/compressed', CompressedImage, self.rgb_callback)
        self.depth_state_sub = rospy.Subscriber('/camera3/camera/depth/image_rect_raw/compressed', CompressedImage, self.depth_callback)

    def unsubscribe(self):
        self.right_arm_pose.unregister()
        self.rgb_state_sub.unregister()
        self.depth_state_sub.unregister()

    def pose_callback(self, pose):
        if None in self.data:
            self.data.pose = pose.pose

    def rgb_callback(self, rgb):
        if None in self.data:
            self.data.rgb = rgb

    def depth_callback(self, depth):
        if None in self.data:
            self.data.depth = depth


def translate_tau(button):
    b_0 = int(button[0])
    b_1 = int(button[1])
    tau = [.558-.069*b_0, .22-.063*b_1, .22]
    return tau


def distance(a, b):
    a = [a[0]-.035, a[1], .94]
    print(a)
    return np.sqrt(np.sum([np.abs(aa - bb) for aa, bb in zip(a,b)]))


def get_tau():
    button = input('Please enter a button for the robot to try to press (e.g. "0,0", "1,2"): ')
    return button + (0,)
    tau = translate_tau(button)
    return tau


def main(weights, tolerance):
    agent = ImitateEval(weights)
    agent.change_start()
    agent.move_to_start()
    tau = get_tau()
    agent.move_to_button(tau, tolerance)


def sighandler(signal, frame):
    raise Exception('op')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for evaluating imitation net")
    parser.add_argument('-w', '--weights', required=True, help='Filepath for model weightsself.')
    parser.add_argument('-t', '--tolerance', default=.1, type=float, help='Tolerance for button presses.')
    args = parser.parse_args()
    rospy.init_node('eval_imitation', log_level=rospy.DEBUG)

    signal.signal(signal.SIGINT, sighandler)

    cont = False
    while True:
        try:
            main(args.weights, args.tolerance)
        except Exception:
            if cont:
                break
            cont = True
            continue
        cont = False
