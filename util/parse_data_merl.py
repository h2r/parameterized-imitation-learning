import lmdb
import os.path as osp
import numpy as np
import pyarrow as pa
from PIL import Image
import csv

import argparse

def serialize_pyarrow(obj):
    return pa.serialize(obj).to_buffer()

def norm_img(img):
    return 2*((img - np.amin(img))/(np.amax(img)-np.amin(img)))-1

def create_cases_csv(csv_path, dest_path, cases, mode, episode_len=40, num_pos=8):
    """
    Note that cases is a list of buttons. will be handled and detailed by argparse
    """
    file_path = osp.join(dest_path, mode+"_data.csv")
    with open(file_path, 'w+') as out:
        writer = csv.writer(out, delimiter=',')
        for i in range(2,num_pos+1):
            input_file = None
            if mode == "train":
                input_file = osp.join(csv_path, "data_training"+str(i))
            elif mode == "test":
                input_file = osp.join(csv_path, "data_test"+str(i))
            print(input_file)
            with open(input_file, 'r') as info:
                arr = [line.strip().split(',') for line in info]
                for line_idx in range(0, len(arr), episode_len):
                    for case in cases:
                        if case in arr[line_idx][0]:
                            for ep_idx in range(0,episode_len):
                                writer.writerow(arr[line_idx+ep_idx])

def parse_raw_data_merl(dest_path, mode, episode_len=40, write_frequency=5000):
    lmdb_path = osp.join(dest_path, "{}.lmdb".format(mode))
    isdir = osp.isdir(lmdb_path)
    episode_counter = 1
    csv_path = osp.join(dest_path, "{}_data.csv".format(mode))

    print("Loading Data from {}...".format(csv_path))
    with open(csv_path, 'r') as datafile:
        arr = [x.strip().split(',') for x in datafile]
        data_len = len(arr)

        print("Generate LMDB to {}".format(lmdb_path))
        db = lmdb.open(lmdb_path, subdir=isdir,
                      map_size=1099511627776 * 2, readonly=False,
                      meminit=False, map_async=True)
        txn = db.begin(write=True)
        aux = np.array([float(x) for x in arr[episode_len-1][20:26]])

        for idx, row in enumerate(arr):
            # Open Images and normalize [-1,1]
            rgb = norm_img(np.array(Image.open(row[0]).convert("RGB")))
            depth = norm_img(np.array(Image.open(row[1])))

            # Reshape Images to have channel first
            rgb = np.reshape(rgb, (3, rgb.shape[0], -1))
            depth = np.reshape(depth, (1, depth.shape[0], -1))
            eof = np.array([float(x) for x in row[2:17]])
            tau = np.array([float(x) for x in row[17:20]])
            if idx >= episode_len*episode_counter:
                episode_counter += 1
                aux = np.array([float(x) for x in arr[(episode_len*episode_counter)-1][20:26]])
            target = [float(x) for x in row[26:32]]
            # Add the dummy value
            target.append(1.0)
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
    parser.add_argument('-pth', '--csv_path', required=True, help='path to the directory where the csv files are i.e. ./data')
    parser.add_argument('-d', '--dest_path', required=True, help='Destination directory for train.lmdb or test.lmdb i.e. ./data')
    parser.add_argument('-m', '--mode', required=True, help='train or test i.e. -m train')
    parser.add_argument('-el', '--episode_len', required=False, default=40, type=int, help='Length of each episode. Default set to 28')
    parser.add_argument('-c', '--cases', required=True, nargs='+', help='the number of holes that you would want to include in the training i.e. -c 0_hole 1_hole 8_hole')
    args = parser.parse_args()

    print("Creating CSV for Cases...")
    create_cases_csv(args.csv_path,
                     args.dest_path,
                     args.cases,
                     args.mode,
                     episode_len=args.episode_len,
                     num_pos=8)
    parse_raw_data_merl(args.dest_path,
                        args.mode,
                        episode_len=args.episode_len)
