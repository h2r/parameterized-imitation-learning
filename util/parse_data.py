import numpy as np
import os
import os.path as osp
import pandas as pd
import csv
from PIL import Image
from random import shuffle
import re
from functools import cmp_to_key
import argparse
import lmdb
import pyarrow as pa
import tqdm
from multiprocessing import Pool

splits = {}

num_of_train = 0
num_of_test = 0


def preprocess_image(path, crop_right=586, crop_lower=386):
    with Image.open(path) as img:
        # Crop
        #img = img.crop((0, 0, crop_right, crop_lower))
        img = img.resize((160,120))
        img.save(path)


def preprocess_images(root_dir, cases, crop_right=586, crop_lower=386):
    with Pool(processes=8) as pool:
        all_imgs = []
        for i, case in enumerate(cases):
            dirs = [x[0] for x in os.walk(root_dir + case)][1:]
            for sub_dir in dirs:
                for root, _, files in os.walk(sub_dir):
                    all_imgs += [root + '/' + file for file in files if not file == 'vectors.txt']
        for result in tqdm.tqdm(pool.imap_unordered(preprocess_image, all_imgs, chunksize=128), desc='Preprocessing images', total=len(all_imgs)):
            pass


def parse_raw_data(mode, config):
    global splits, num_of_train, num_of_test
    # Output File
    with open(config.dest_dir+"/"+ mode + "_data.csv", "w+") as file:
        writer = csv.writer(file, delimiter=',')
        # Go through all of the cases
        sub_dirs = []
        for i, case in enumerate(config.train_cases + config.test_cases):
            # Create the splits
            if case not in splits.keys():
                dirs = [x[0] for x in os.walk(config.root_dir + case)][1:]
                if len(dirs) == 0:
                    raise Exception('Case %s not found at %s.' % (case, config.root_dir + case))
                shuffle(dirs)

                if config.max_traj > 0:
                    dirs = dirs[:config.max_traj]

                if len(config.test_cases) == 0:
                    split_idx = int(len(dirs)*float(config.split_percen))
                else:
                    split_idx = len(dirs) if (case in config.train_cases) else 0
                splits[case] = {"train": dirs[:split_idx], "test": dirs[split_idx:]}
            # Go into every subdirectory
            sub_dirs += splits[case][mode]
        with Pool(processes=8) as pool:
            for result in tqdm.tqdm(pool.imap_unordered(parse_trajectory, zip(sub_dirs, [config]*len(sub_dirs)), chunksize=8), desc='Parsing data', total=len(sub_dirs)):
                for row in result:
                    writer.writerow(row)
                    if mode == 'train':
                        num_of_train += 1
                    else:
                        num_of_test += 1

def parse_trajectory(ins):
    sub_dir, config = ins
    rows = []
    for root, _, element in os.walk(sub_dir):
        element = sorted(element)
        pics = element[:-1]
        if '.DS_Store' in pics:
            pics.remove('.DS_Store')
        pics = sorted(pics, key=cmp_to_key(compare_names))

        vectors = pd.read_csv(root+"/vectors.txt", header=-1)
        last = vectors.iloc[-1]
        if config.simulation:
            # FOR SIM
            tau = [last.iloc[i] for i in range(15, len(last) - 2)]
            aux_target = [last.iloc[i] for i in range(len(last) - 2, len(last))]
        else:
            # FOR KUKA
            tau = [last.iloc[1], last.iloc[2], last.iloc[3]]
            aux_target = [last.iloc[1], last.iloc[2], last.iloc[3],
                          last.iloc[4], last.iloc[5], last.iloc[6]]
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
                eof = pos + eof
            # EOF: 2:17
            row += eof
            # Tau: 17:20
            row += tau
            # Auxiliary Target: 20:26
            row += aux_target
            if config.simulation:
                # Output for 2D Sim Target: Linear and Angular Vel (Quaternion)
                output = [data.iloc[0,8], data.iloc[0,9], data.iloc[0,10],
                          data.iloc[0,11], data.iloc[0,12], data.iloc[0,13], data.iloc[0,14]]
            else:
                # Output for Kuka Target: Linear Vel and Angular Vel (Eular): 26:32
                output = [data.iloc[0,7], data.iloc[0,8], data.iloc[0,9],
                          data.iloc[0,10], data.iloc[0,11], data.iloc[0,12]]

            row += output
            rows += [row]
    return rows

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

def clean_datapoint(sub_dir, data_file_name='/vectors.txt', clean_file_name='/clean_vector.txt'):
    for root, _, file in os.walk(sub_dir):
        file = sorted(file)
        dirty_flag = False
        for element in file:
            if element[0:2] == "0_" or element[0:2] == "1_" or element[0:2] == "2_" or element[0:2] == "3_" or element[0:2] == "4_":
                os.remove(root+"/"+element)
                dirty_flag = True
        if dirty_flag:
            vector_file = root+data_file_name
            clean_file = root+clean_file_name
            with open(vector_file, 'r+') as info, open(clean_file, 'w+') as out:
                writer = csv.writer(out)
                for row in csv.reader(info):
                    if int(row[0]) > 4:
                        writer.writerow(row)
            os.remove(vector_file)
            os.rename(root+clean_file_name, root+data_file_name)

def clean_kuka_data(root_dir, cases, data_file_name='/vectors.txt', clean_file_name='/clean_vector.txt'):
    print("Cleaning Kuka Data")
    with Pool(processes=8) as pool:
        dirs = []
        for i, case in enumerate(cases):
            dirs += [x[0] for x in os.walk(root_dir + case)][1:]
        for result in tqdm.tqdm(pool.imap_unordered(clean_datapoint, dirs, chunksize=16), desc='Cleaning sub-directories', total=len(dirs)):
            pass
    preprocess_images(root_dir, cases)

def serialize_pyarrow(obj):
    return pa.serialize(obj).to_buffer()

def norm_img(img):
    if np.amin(img) == np.amax(img):
        return img
    return 2*((img - np.amin(img))/(np.amax(img)-np.amin(img)))-1

def get_row(ins):
    row, config = ins
    # Open Images and normalize [-1,1]
    rgb, depth = None, None
    with Image.open(row[0]).convert("RGB") as img:
        rgb = norm_img(np.array(img))
    with Image.open(row[1]) as img:
        depth = norm_img(np.array(img))

    # Reshape Images to have channel first
    rgb = np.transpose(rgb, (2, 0, 1))
    depth = np.reshape(depth, (1, depth.shape[0], depth.shape[1]))
    eof = np.array([float(x) for x in row[2:17]])
    if config.simulation:
        rl = len(row)
        tau = np.array([float(x) for x in row[17:rl-9]])
        aux = np.array([float(x) for x in row[rl-9:rl-7]])
        target = np.array([float(x) for x in row[rl-7:rl]])
    else:
        tau = np.array([float(x) for x in row[17:20]])
        aux = np.array([float(x) for x in row[20:26]])
        target = np.array([float(x) for x in row[26:32]])
    return (rgb, depth, eof, tau, aux, target)

def create_lmdb(mode, config, write_frequency=5000):
    global num_of_train, num_of_test

    if mode == 'train':
        data_len = num_of_train
    else:
        data_len = num_of_test

    data_file = osp.join(config.dest_dir, mode+"_data.csv")
    print("Loading Data from {}...".format(data_file))
    with open(data_file, "r") as info:
        # Create LMDB
        lmdb_path = osp.join(config.dest_dir, "{}.lmdb".format(mode))
        isdir = osp.isdir(lmdb_path)

        print("Generate LMDB to {}".format(lmdb_path))

        with lmdb.open(lmdb_path, subdir=isdir,
                       map_size=1099511627776 * 2, readonly=False,
                       meminit=False, map_async=True) as db:

            # Begin iterating through to write
            txn = db.begin(write=True)
            total = sum(1 for line in csv.reader(info))
            info.seek(0)
            with Pool(processes=8) as pool:
                for idx, result in enumerate(tqdm.tqdm(pool.imap_unordered(get_row, zip(csv.reader(info), [config]*total), chunksize=16), total=total, desc='Writing LMDB')):
                    txn.put(u'{}'.format(idx).encode('ascii'), serialize_pyarrow(result))
                    if idx % write_frequency == 0:
                        txn.commit()
                        txn = db.begin(write=True)

            # Once all the data is gone through
            txn.commit()
            keys = [u'{}'.format(k).encode('ascii') for k in range(total)]
            with db.begin(write=True) as txn:
                txn.put(b'__keys__', serialize_pyarrow(keys))
                txn.put(b'__len__', serialize_pyarrow(len(keys)))

            print("Flushing database ...")
            db.sync()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Input to data cleaner')
    parser.add_argument('-r', '--root_dir', required=True, help='Root directory of data i.e. data/buttons')
    parser.add_argument('-d', '--dest_dir', required=True, help='Destination directory for train_data.csv and test_data.csv i.e. data/buttons/all_buttons')
    parser.add_argument('-tr', '--train_cases', nargs='+', required=True, help='Name of the specific train cases that we want to include i.e. -c /11_button /12_button /13_button')
    parser.add_argument('-te', '--test_cases', nargs='*', default = [], required=False, help='Name of the specific test cases that we want to include i.e. -c /11_button /12_button /13_button')
    parser.add_argument('-s', '--split_percen', required=False, default=0.95, type=float, help='The Train/Test data split proportion')
    parser.add_argument('-cd', '--clean_data', default=False, dest='clean_data', action='store_true', help='Flag to turn on image preprocessing')
    parser.add_argument('-sim', '--simulation', default=False, dest='simulation', action='store_true', help='Flag to label data as simulated')
    parser.add_argument('-m', '--max_traj', default=0, type=int, help='Maximum number of trajectories to use per case')
    args = parser.parse_args()

    os.makedirs(args.dest_dir)

    if args.clean_data:
        print("IMAGE PREPROCESSING STARTING...")
        clean_kuka_data(args.root_dir, args.train_cases + args.test_cases)
        print("IMAGE PREPROCESSING DONE")
    print("LMDB CREATION STARTING...")
    for mode in ['train', 'test']:
        print(mode + ' dataset creation begun.')
        parse_raw_data(mode, args)
        create_lmdb(mode, args)
        print(mode + ' dataset creation done.')
