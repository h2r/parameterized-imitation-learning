from __future__ import print_function, division
#import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import csv

from PIL import Image
from random import shuffle

import time
import math
import sys

from tqdm import tqdm

splits = {}

def parse_param_(root_dir, mode, cases, tau, directions):
    global splits
    file = open(root_dir+"/"+ mode + "_data.csv", "w+")
    writer = csv.writer(file, delimiter='\t')
    #cases = ["/test_00", "/test_01", "/test_02", "/test_10", "/test_11", "/test_12", "/test_20", "/test_21", "/test_22"]
    #tau = [[200, 150], [400, 150], [600, 150], [200, 300], [400, 300], [600, 300], [200, 450], [400, 450], [600 ,450]]
    for case, tau, direction in tqdm(zip(cases, tau, directions)):
        if case not in splits.keys():
            dirs = [x[0] for x in os.walk(root_dir + case)][1:]
            shuffle(dirs)
            split_idx = int(math.ceil(len(dirs)*float(sys.argv[2])))
            splits[case] = {"train": dirs[:split_idx], "test": dirs[split_idx:]}
        sub_dirs = splits[case]
        for sub_dir in sub_dirs[mode]:
            for root, _, file in os.walk(sub_dir):
                file = sorted(file)
                vectors = pd.read_csv(root+"/"+file[-1], header=-1)
                for i in range(0, len(file)-1, 2):
                    prevs = []
                    eof = []
                    if i < 10 :
                        repeat = int(5-(0.5*i))
                        prevs += [float(file[0][:-10]) for j in range(repeat)]
                        prevs += [float(file[i-j][:-10]) for j in range(i, 1, -2)]
                    else:
                        prevs += [float(file[i-j][:-10]) for j in range(10, 1, -2)]
                    for prev in prevs:
                        pos = np.array([float(vectors[vectors[0]==prev][j]) for j in range(1,4)])
                        eof.append(pos)
                    eof = repr(list(np.array(eof).flatten()))
                    label = repr([float(vectors[vectors[0]==float(file[i][:-10])][j]) for j in range(8,14)])
                    gripper = repr(int(vectors[vectors[0]==float(file[i][:-10])][14]))
                    writer.writerow([root+"/"+file[i], root+"/"+file[i+1], eof, label, gripper, repr(tau), direction])
    print("{} Data Creation Done".format(mode))

def parse_param_2(root_dir, mode, cases, tau, directions):
    global splits
    file = open(root_dir+"/"+ mode + "_data.csv", "w+")
    writer = csv.writer(file, delimiter='\t')
    # For every folder, tau, and direction
    for case, tau, direction in tqdm(zip(cases, tau, directions)):
        # Create the splits
        if case not in splits.keys():
            dirs = [x[0] for x in os.walk(root_dir + case)][1:]
            shuffle(dirs)
            split_idx = int(math.ceil(len(dirs)*float(sys.argv[2])))
            splits[case] = {"train": dirs[:split_idx], "test": dirs[split_idx:]}
        # Go into every subdirectory
        sub_dirs = splits[case]
        for sub_dir in sub_dirs[mode]:
            for root, _, file in os.walk(sub_dir):
                file = sorted(file)
                vectors = pd.read_csv(root+"/"+file[-1], header=-1)
                # We will start from 10 to start creating a sequential dataset (rgb, depth)
                # The length of the history will be 5 
                for i in range(8, len(file)-1, 2):
                    depth = repr([root+"/"+file[i-j] for j in range(8,-1,-2)])
                    rgb = repr([root+"/"+file[i-j+1] for j in range(8,-1,-2)])
                    tau = repr([tau for _ in range(5)])
                    prevs = [float(file[i-j][:-10]) for j in range(8, -1, -2)]
                    eof = []
                    for prev in prevs:
                        pos = np.array([float(vectors[vectors[0]==prev][j]) for j in range(1,4)])
                        eof.append(pos)
                    # Label and Gripper still stay the same
                    label = repr([float(vectors[vectors[0]==float(file[i][:-10])][j]) for j in range(8,14)])
                    gripper = repr(int(vectors[vectors[0]==float(file[i][:-10])][14]))
                    # This is how we will represent the data
                    writer.writerow([depth, rgb, eof, label, gripper, tau, direction])
    print("{} Data Creation Done".format(mode))

if __name__ == '__main__':
    #cases = ["/case_00_down", "/case_00_left", "/case_00_right", "/case_00_up", "/case_01_down", "/case_01_left", "/case_01_right", "/case_01_up", "/case_02_down", "/case_02_left", "/case_02_right", "/case_02_up"]
    #tau = [[200, 150], [200, 150], [200, 150], [200, 150], [400, 150], [400, 150], [400, 150], [400, 150], [600, 150], [600, 150], [600, 150], [600, 150]]
    cases = ['/new_down', '/new_left', '/new_right', '/new_up']
    tau = [[200, 150], [200, 150], [200, 150], [200, 150]]
    directions = [0, 1, 2, 3]
    root_dir = sys.argv[1]
    modes = ["train", "test"]
    datasets = {mode: parse_param_2(root_dir, mode, cases, tau, directions) for mode in modes}
