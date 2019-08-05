from __future__ import print_function, division
import numpy as np
import os
import os.path as osp
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

import argparse

#import h5py

# NEW IMPORTS
import lmdb
import pyarrow as pa
import six
import msgpack

splits = {}
# num_of_train = 103253
# num_of_test = 5372

num_of_train = 0
num_of_test = 0


def preprocess_images(root_dir, cases, crop_right=586, crop_lower=386):
    for case in cases:
         dirs = [x[0] for x in os.walk(root_dir + case)][1:]
         for sub_dir in dirs:
             for root, _, files in os.walk(sub_dir):
                 files = sorted(files)
                 files = files[:-1]
                 if '.DS_Store' in files:
                 	files.remove('.DS_Store')
                 for i in range(0, len(files), 2):
                     depth = Image.open(root+"/"+files[i])
                     rgb = Image.open(root+"/"+files[i+1])
                     # Crop
                     #rgb = rgb.crop((0, 0, crop_right, crop_lower))
                     #depth = depth.crop((0, 0, crop_right, crop_lower))
                     rgb = rgb.resize((160,120))
                     depth = depth.resize((160,120))
                     depth.save(root+"/"+files[i])
                     rgb.save(root+"/"+files[i+1])

def create_h5(dest, mode):
    global num_of_train, num_of_test
    filename = dest+"/data"+mode+".hdf5"
    # Create h5 datasets
    with h5py.File(filename, "w") as out:
        # Training Data
        if mode == "train":
            out.create_dataset("rgb", (num_of_train, 3, 120, 160), maxshape=(None, 3, 120, 160))
            out.create_dataset("depth", (num_of_train, 1, 120, 160), maxshape=(None, 1, 120, 160))
            out.create_dataset("eof", (num_of_train, 15), maxshape=(None, 15))
            out.create_dataset("tau", (num_of_train, 3), maxshape=(None, 3))
            out.create_dataset("aux", (num_of_train, 6), maxshape=(None, 6))
            out.create_dataset("target", (num_of_train, 6), maxshape=(None, 6))
        else:
            # Testing Data
            out.create_dataset("rgb", (num_of_test, 3, 120, 160), maxshape=(None, 3, 120, 160))
            out.create_dataset("depth", (num_of_test, 1, 120, 160), maxshape=(None, 1, 120, 160))
            out.create_dataset("eof", (num_of_test, 15), maxshape=(None, 15))
            out.create_dataset("tau", (num_of_test, 3), maxshape=(None, 3))
            out.create_dataset("aux", (num_of_test, 6), maxshape=(None, 6))
            out.create_dataset("target", (num_of_test, 6), maxshape=(None, 6))

    with open(dest+"/"+mode+"_data.csv", "r") as info, h5py.File(filename, "a") as out:
        for i, row in enumerate(csv.reader(info)):
            # print(i+1)
            rgb = np.array(Image.open(root_dir + "/" + row[0]).convert("RGB"))
            depth = np.array(Image.open(root_dir + "/" + row[1]))
            # Normalize
            rgb = (2*(rgb - np.amin(rgb))/(np.amax(rgb)-np.amin(rgb)))-1
            # Reshape
            rgb = np.reshape(rgb, (3, rgb.shape[0], -1))
            depth = np.reshape(depth, (1, depth.shape[0], -1))
            eof = np.array([float(x) for x in row[2:17]])
            tau = np.array([float(x) for x in row[17:20]])
            aux = np.array([float(x) for x in row[20:26]])
            target = [float(x) for x in row[26:32]]
            target = np.array(target)

            out['rgb'][i,:,:,:] = rgb
            out['depth'][i,:,:,:] = depth
            out['eof'][i,:] = eof
            out['tau'][i,:] = tau
            out['aux'][i,:] = aux
            out['target'][i,:] = target

def parse_raw_data_subsampler(root_dir, mode, cases, split_percen, dest, skip_amt=4):
    global splits, num_of_train, num_of_test
    # Output File
    with open(dest+"/"+ mode + "_data.csv", "w+") as file:
        writer = csv.writer(file, delimiter=',')
        # Go through all of the cases
        for case in cases:
            # Create the splits
            if case not in splits.keys():
                dirs = [x[0] for x in os.walk(root_dir + case)][1:]
                shuffle(dirs)
                split_idx = int(math.ceil(len(dirs)*float(split_percen)))
                splits[case] = {"train": dirs[:split_idx], "test": dirs[split_idx:]}
            # Go into every subdirectory
            sub_dirs = splits[case]
            for sub_dir in sub_dirs[mode]:
                for root, _, element in os.walk(sub_dir):
                    element = sorted(element)
                    pics = element[:-1]
                    if '.DS_Store' in pics:
                        pics.remove('.DS_Store')
                    pics = sorted(pics, key=cmp_to_key(compare_names))
                    vectors = pd.read_csv(root+"/vectors.txt", header=-1)
                    last = vectors.iloc[-1]
                    # FOR KUKA
                    tau = [last.iloc[1], last.iloc[2], last.iloc[3]]
                    # FOR SIM
                    #tau = [last.iloc[15], last.iloc[16]]
                    aux_target = [last.iloc[1], last.iloc[2], last.iloc[3],
                                  last.iloc[4], last.iloc[5], last.iloc[6]]
                    # The length of the history will be 5
                    prev_pose = None
                    for i in range(0, len(pics), skip_amt*2):
                        # rgb, depth
                        row = [root+"/"+pics[i+1], root+"/"+pics[i]]
                        curr_idx = int(pics[i][:-10])
                        data = vectors[curr_idx == vectors[0]]
                        if i == 0:
                            prevs = [curr_idx for _ in range(5)]
                            prev_pose = [data.iloc[0,1], data.iloc[0,2], data.iloc[0,3], data.iloc[0,4], data.iloc[0,5], data.iloc[0,6]]
                        else:
                            prevs.pop(0)
                            prevs.append(curr_idx)
                        eof = []
                        for prev in prevs:
                            pos = [float(vectors[vectors[0]==prev][j]) for j in range(1,4)]
                            eof += pos
                        # EOF: 2:17
                        row += eof
                        # Tau: 17:20
                        row += tau
                        # Auxiliary Target: 20:26
                        row += aux_target
                        # Output Target: Linear Vel and Angular Vel (Eular): 26:32
                        if i == 0:
                            output = [data.iloc[0,7], data.iloc[0,8], data.iloc[0,9],
                                      data.iloc[0,10], data.iloc[0,11], data.iloc[0,12]]
                        else:
                            output = [data.iloc[0,1] - prev_pose[0], data.iloc[0,2] - prev_pose[1], data.iloc[0,3] - prev_pose[2],
                                      data.iloc[0,4] - prev_pose[3], data.iloc[0,5] - prev_pose[4], data.iloc[0,6] - prev_pose[5]]
                            prev_pose = [data.iloc[0,1], data.iloc[0,2], data.iloc[0,3],
                                         data.iloc[0,4], data.iloc[0,5], data.iloc[0,6]]
                        row += output
                        writer.writerow(row)
                        if mode == 'train':
                            num_of_train += 1
                        else:
                            num_of_test += 1
    print("{} data creation done".format(mode))

def parse_raw_data(root_dir, mode, cases, split_percen, dest):
    global splits, num_of_train, num_of_test
    # Output File
    with open(dest+"/"+ mode + "_data.csv", "w+") as file:
        writer = csv.writer(file, delimiter=',')
        # Go through all of the cases
        for case in cases:
            # Create the splits
            if case not in splits.keys():
                dirs = [x[0] for x in os.walk(root_dir + case)][1:]
                shuffle(dirs)
                split_idx = int(math.ceil(len(dirs)*float(split_percen)))
                splits[case] = {"train": dirs[:split_idx], "test": dirs[split_idx:]}
            # Go into every subdirectory
            sub_dirs = splits[case]
            for sub_dir in sub_dirs[mode]:
                for root, _, element in os.walk(sub_dir):
                    element = sorted(element)
                    pics = element[:-1]
                    if '.DS_Store' in pics:
                        pics.remove('.DS_Store')
                    pics = sorted(pics, key=cmp_to_key(compare_names))
                    # IMPORTANT: change the string to /vectors.txt for Kuka data!
                    vectors = pd.read_csv(root+"/vector.txt", header=-1)
                    last = vectors.iloc[-1]
                    # FOR KUKA
                    # tau = [last.iloc[1], last.iloc[2], last.iloc[3]]
                    # aux_target = [last.iloc[1], last.iloc[2], last.iloc[3],
                    #               last.iloc[4], last.iloc[5], last.iloc[6]]

                    # FOR SIM
                    tau = [last.iloc[15], last.iloc[16], last.iloc[17]]
                    aux_target = [last.iloc[18], last.iloc[19]]

                    # The length of the history will be 5
                    for i in range(0, len(pics), 2):
                        # rgb, depth
                        row = [root+"/"+pics[i+1], root+"/"+pics[i]]
                        curr_idx = int(pics[i][:-10])
                        data = vectors[curr_idx == vectors[0]]
                        if i == 0:
                            prevs = [curr_idx for _ in range(5)]
                        else:
                            prevs.pop(0)
                            prevs.append(curr_idx)
                        eof = []
                        for prev in prevs:
                            pos = [float(vectors[vectors[0]==prev][j]) for j in range(1,4)]
                            eof += pos
                        # EOF: 2:17
                        row += eof
                        # Tau: 17:20
                        row += tau
                        # Auxiliary Target: 20:26
                        row += aux_target
                        # Output for Kuka Target: Linear Vel and Angular Vel (Eular): 26:32
                        # output = [data.iloc[0,7], data.iloc[0,8], data.iloc[0,9],
                        #           data.iloc[0,10], data.iloc[0,11], data.iloc[0,12]]

                        # Output for 2D Sim Target: Linear and Angular Vel (Quaternion)
                        output = [data.iloc[0,8], data.iloc[0,9], data.iloc[0,10],
                                  data.iloc[0,11], data.iloc[0,12], data.iloc[0,13], data.iloc[0,14]]
                        row += output
                        writer.writerow(row)
                        if mode == 'train':
                            num_of_train += 1
                        else:
                            num_of_test += 1
    print("{} data creation done".format(mode))

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

# IMPORTANT: The file is called /vectors.txt for Kuka data!
def clean_kuka_data(root_dir, cases, data_file_name='/vector.txt', clean_file_name='/clean_vector.txt'):
    preprocess_images(root_dir, cases)
    print("Cleaning Kuka Data")
    for case in cases:
        dirs = [x[0] for x in os.walk(root_dir + case)][1:]
        # Go into every subdirectory
        for sub_dir in dirs:
            for root, _, file in os.walk(sub_dir):
                file = sorted(file)
                for element in file:
                    if element[0:2] == "0_" or element[0:2] == "1_" or element[0:2] == "2_" or element[0:2] == "3_" or element[0:2] == "4_":
                        os.remove(root+"/"+element)
                vector_file = root+data_file_name
                clean_file = root+clean_file_name
                with open(vector_file, 'r+') as info, open(clean_file, 'w+') as out:
                    writer = csv.writer(out)
                    for row in csv.reader(info):
                        if int(row[0]) > 4:
                            writer.writerow(row)
                os.remove(vector_file)
                os.rename(root+clean_file_name, root+data_file_name)

def serialize_pyarrow(obj):
    return pa.serialize(obj).to_buffer()

def norm_img(img):
    return 2*((img - np.amin(img))/(np.amax(img)-np.amin(img)))-1

def create_lmdb(dest, mode, write_frequency=5000):
    global num_of_train, num_of_test

    if mode == 'train':
        data_len = num_of_train
    else:
        data_len = num_of_test

    data_file = osp.join(dest, mode+"_data.csv")
    print("Loading Data from {}...".format(data_file))
    with open(data_file, "r") as info:

        # Create LMDB
        lmdb_path = osp.join(dest, "{}.lmdb".format(mode))
        isdir = osp.isdir(lmdb_path)

        print("Generate LMDB to {}".format(lmdb_path))

        db = lmdb.open(lmdb_path, subdir=isdir,
                       map_size=1099511627776 * 2, readonly=False,
                       meminit=False, map_async=True)

        # Begin iterating through to write
        txn = db.begin(write=True)
        for idx, row in enumerate(csv.reader(info)):
            # Open Images and normalize [-1,1]
            rgb = norm_img(np.array(Image.open(row[0]).convert("RGB")))
            depth = norm_img(np.array(Image.open(row[1])))

            # Reshape Images to have channel first
            rgb = np.reshape(rgb, (3, rgb.shape[0], -1))
            depth = np.reshape(depth, (1, depth.shape[0], -1))
            eof = np.array([float(x) for x in row[2:17]])
            tau = np.array([float(x) for x in row[17:20]])
            aux = np.array([float(x) for x in row[20:22]])
            target = [float(x) for x in row[22:29]]
            target = np.array(target)
            txn.put(u'{}'.format(idx).encode('ascii'), serialize_pyarrow((rgb, depth, eof, tau, aux, target)))
            if idx % write_frequency == 0:
                print("[{}/{}]".format(idx, data_len))
                txn.commit()
                txn = db.begin(write=True)

        # Once all the data is gone through
        txn.commit()
        keys = [u'{}'.format(k).encode('ascii') for k in range(idx + 1)]
        with db.begin(write=True) as txn:
            txn.put(b'__keys__', serialize_pyarrow(keys))
            txn.put(b'__len__', serialize_pyarrow(len(keys)))

        print("Flushing database ...")
        db.sync()
        db.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Input to data cleaner')
    parser.add_argument('-r', '--root_dir', required=True, help='Root directory of data i.e. data/buttons')
    parser.add_argument('-d', '--dest_dir', required=True, help='Destination directory for train_data.csv and test_data.csv i.e. data/buttons/all_buttons')
    parser.add_argument('-c', '--cases', nargs='+', required=True, help='Name of the specific cases that we want to include i.e. -c /11_button /12_button /13_button')
    parser.add_argument('-s', '--split_percen', required=False, default=0.95, type=float, help='The Train/Test data split proportion')
    parser.add_argument('-cd', '--clean_data', required=False, default=False, type=bool, help='Flag to turn on image preprocessing')
    args = parser.parse_args()

    root_dir = args.root_dir
    dest_dir = args.dest_dir
    cases = args.cases
    split_percen = args.split_percen
    # if args.clean_data:
    #     print("IMAGE PREPROCESSING STARTING...")
    #     clean_kuka_data(root_dir, cases)
    #     print("IMAGE PREPROCESSING DONE")
    # print("LMDB CREATION STARTING...")
    for mode in ['train', 'test']:
        parse_raw_data(root_dir, mode, cases, split_percen, dest_dir)
        create_lmdb(dest_dir, mode)
        print(mode+" dataset creation done")
