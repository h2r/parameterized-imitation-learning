from __future__ import print_function, division
#import cv2
import numpy as np
import os
import pandas as pd
import csv

from PIL import Image
from random import shuffle

import re
import time
import math
import sys
import copy
import shutil
from functools import cmp_to_key
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
                print(file)
                #vectors = pd.read_csv(root+"/"+file[-1], header=-1)
                #for i in range(0, len(file)-1, 2):
                #    prevs = []
                #    eof = []
                #    if i < seq_len:
                #        repeat = int(5-(0.5*i))
                #        prevs += [float(file[0][:-10]) for j in range(repeat)]
                #        prevs += [float(file[i-j][:-10]) for j in range(i, 1, -2)]
                #    else:
                #        prevs += [float(file[i-j][:-10]) for j in range(10, 1, -2)]
                #    for prev in prevs:
                #        pos = np.array([float(vectors[vectors[0]==prev][j]) for j in range(1,4)])
                #        eof.append(pos)
                #    eof = repr(list(np.array(eof).flatten()))
                #    label = repr([float(vectors[vectors[0]==float(file[i][:-10])][j]) for j in range(8,14)])
                #    gripper = repr(int(vectors[vectors[0]==float(file[i][:-10])][14]))
                #    writer.writerow([root+"/"+file[i], root+"/"+file[i+1], eof, label, gripper, repr(tau), direction])
    print("{} Data Creation Done".format(mode))

def preprocess_images(root_diri, cases):
    for case in cases:
         dirs = [x[0] for x in os.walk(root_dir + case)]
         for sub_dir in dirs:
             for root, _, files in os.walk(sub_dir):
                 files = sorted(files)
                 files = files[:-1]
                 for i in range(0, len(files), 2):
                     depth = Image.open(root+"/"+files[i])
                     rgb = Image.open(root+"/"+files[i+1])
                     rgb = rgb.resize((160,120))
                     depth = depth.resize((160,120))
                     depth.save(root+"/"+files[i])
                     rgb.save(root+"/"+files[i+1])
                     

def parse_param_2(root_dir, mode, cases, tau, seq_len, split_percen, dest, directions=None):
    global splits
    file = open(dest+"/"+ mode + "_data.csv", "w+")
    writer = csv.writer(file, delimiter=',')
    # For every folder, tau, and direction
    for case, tau in tqdm(zip(cases, tau)):
        #print(case)
        #print(tau)
        # Create the splits
        if case not in splits.keys():
            dirs = [x[0] for x in os.walk(root_dir + case)][1:]
            shuffle(dirs)
            split_idx = int(math.ceil(len(dirs)*float(split_percen)))
            splits[case] = {"train": dirs[:split_idx], "test": dirs[split_idx:]}
        # Go into every subdirectory
        sub_dirs = splits[case]
        #dirs = [x[0] for x in os.walk(root_dir + case)][1:]
        for sub_dir in sub_dirs[mode]:
            for root, _, file in os.walk(sub_dir):
                file = sorted(file)
                pics = file[:-1]
                pics = sorted(pics, key=cmp_to_key(compare_names))         
                vectors = pd.read_csv(root+"/"+file[-1], header=-1)
                # We will start from 10 to start creating a sequential dataset (rgb, depth)
                # The length of the history will be 5 
                for i in range((seq_len*2)-2, len(pics), 2):
                    # First get the current directory
                    row = [root, int(pics[i][:-10]), tau[0], tau[1]]
                    #depth = [root+"/"+pics[i-j] for j in range(seq_len,-1,-2)]
                    #rgb = [root+"/"+pics[i-j+1] for j in range(seq_len,-1,-2)]
                    #tau = [tau for _ in range(seq_len)]
                    #print(pics)
		    #print(i)
                    #print([pics[i-j] for j in range(seq_len*2, -1, -2)])
                    prevs = [int(pics[i-j][:-10]) for j in range((seq_len*2)-2, -1, -2)]
                    eof = []
                    for prev in prevs:
                        pos = [float(vectors[vectors[0]==prev][j]) for j in range(1,4)]
                        eof += pos
                    ## Label and Gripper still stay the same
                    label = [float(vectors[vectors[0]==float(pics[i][:-10])][j]) for j in range(8,14)]
                    aux_label = [float(vectors[vectors[0]==float(pics[-2][:-10])][j]) for j in range(1,8)]
                    #print(label)
                    row += prevs
                    row += label
                    row += aux_label
                    row += eof
                    #gripper = repr(int(vectors[vectors[0]==float(pics[i][:-8])][14]))
                    ## This is how we will represent the data
                    #writer.writerow([depth, rgb, eof, label, gripper, tau])
                    writer.writerow(row)
                    #print(prevs)
    print("{} Dta Creation Done".format(mode))

def compare_names(name1, name2):
    num1 = extract_number(name1)
    num2 = extract_number(name2)
    if num1 == num2:
        if name1 > name2:
            return 1
        else:
            return -1
    else:
        return num1 - num2

def extract_number(name):
    numbers = re.findall('\d+', name)
    return int(numbers[0])

def clean_data(root_dir, cases):
    for case in cases:
        # Create the splits
        dirs = [x[0] for x in os.walk(root_dir + case)][1:]
        # Go into every subdirectory
        for sub_dir in dirs:
            for root, _, file in os.walk(sub_dir):
                print(root)
                file = sorted(file)
                vector_file = root+"/"+file[-1]
                vectors = csv.read_csv(vector_file, header=-1)
                delete_rows = []
                for i in range(1, len(file), 2):
                    label = [round(float(vectors[vectors[0]==float(file[i][:-8])][j])*10) for j in range(8,14)]
                    if sum(label) == 0:
                       #print(float(file[i][:-8]))
                       #print(root+"/"+file[i]) #rgb
                       #print(root+"/"+file[i-1]) #depth
                       delete_rws.append(file[i][:-8])
                       os.remove(root+"/"+file[i])
                       os.remove(root+"/"+file[i-1])
                if (len(file)-1)/2 != len(delete_rows):                
                    last = None
                    last_num = 0  
                    clean_file = root+"/vector2.txt"
                    with open(vector_file, "rb") as input, open(clean_file, "wb") as out:
                        writer = csv.writer(out)
                        for row in csv.reader(input):
                            if row[0] not in delete_rows:
                                writer.writerow(row)
                    with open(clean_file, "r") as fd:
                        last = [l for l in fd][-1]
                        last = last.strip().split(',')
                        last_num = int(last[0])
                        last[8:14] = ['0.0' for _ in range(8,14)]
                    counter = last_num + 1
                    with open(clean_file, "a") as fd:
                        for _ in range(10):
                            shutil.copy(root+"/"+str(last_num) + "_depth.png", root+ "/" + str(counter) + "_depth.png")
                            shutil.copy(root+"/"+str(last_num) + "_rgb.png", root+"/" + str(counter) + "_rgb.png")
                            row = copy.deepcopy(last)
                            row[0] = str(counter)
                            row = ",".join(row) + "\n"
                            counter += 1
                            fd.write(row)
                os.remove(vector_file)
            if len(os.listdir(sub_dir)) == 0:
                os.rmdir(sub_dir)

if __name__ == '__main__':
    #cases = ["/case_00_down", "/case_00_left", "/case_00_right", "/case_00_up", "/case_01_down", "/case_01_left", "/case_01_right", "/case_01_up", "/case_02_down", "/case_02_left", "/case_02_right", "/case_02_up"]
    #tau = [[200, 150], [200, 150], [200, 150], [200, 150], [400, 150], [400, 150], [400, 150], [400, 150], [600, 150], [600, 150], [600, 150], [600, 150]]
    #cases = ['/new_down', '/new_left', '/new_right', '/new_up']
    #tau = [[200, 150], [200, 150], [200, 150], [200, 150]]
    #cases = ['/33_buttons']
    #cases = ['/00_buttons', '/03_buttons', '/30_buttons', '/33_buttons', '/11_buttons', '/22_buttons']
    #cases = ['/11_buttons']
    cases = ['/33_buttons','/11_buttons', '/00_buttons', '/22_buttons', '/30_buttons', '/03_buttons','/21_buttons', '/12_buttons']
    #tau = [[3,3], [1,1], [0,0], [2,2], [3,0], [0,3]]
    tau = [[0,0],[0,3],[3,0],[3,3],[1,1],[2,2],[2,1],[1,2]]
    #tau = [[1,1]]
    #directions = [0, 1, 2, 3]
    root_dir = sys.argv[1]
    modes = ["train", "test"]
    seq_len = 10
    split_percen = sys.argv[2]
    dest = sys.argv[3]
    #clean_data(root_dir, cases)
    #preprocess_images(root_dir, cases)
    datasets = {mode: parse_param_2(root_dir, mode, cases, tau, seq_len, split_percen, dest=dest) for mode in modes}
