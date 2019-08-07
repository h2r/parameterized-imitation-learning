import torch
from torch.utils.data import Dataset

import lmdb
#import h5py

import os.path as osp
import pyarrow as pa
import random

class ImitationLMDB(Dataset):
    def __init__(self, dest, mode):
        super(ImitationLMDB, self).__init__()
        lmdb_file = osp.join(dest, mode+".lmdb")
        # Open the LMDB file
        self.env = lmdb.open(lmdb_file, subdir=osp.isdir(lmdb_file),
                             readonly=True, lock=False,
                             readahead=False, meminit=False)

        with self.env.begin(write=False) as txn:
            self.length = self.loads_pyarrow(txn.get(b'__len__'))
            self.keys = self.loads_pyarrow(txn.get(b'__keys__'))

        self.shuffled = [i for i in range(self.length)]
        #random.shuffle(self.shuffled)

    def loads_pyarrow(self, buf):
        return pa.deserialize(buf)

    def __getitem__(self, idx):
        rgb, depth, eof, tau, aux, target = None, None, None, None, None, None
        index = self.shuffled[idx]
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])

        # RGB, Depth, EOF, Tau, Aux, Target
        unpacked = self.loads_pyarrow(byteflow)

        # load data
        rgb = torch.from_numpy(unpacked[0]).type(torch.FloatTensor)
        depth = torch.from_numpy(unpacked[1]).type(torch.FloatTensor)
        eof = torch.from_numpy(unpacked[2]).type(torch.FloatTensor)
        tau = torch.from_numpy(unpacked[3]).type(torch.FloatTensor)
        aux = torch.from_numpy(unpacked[4]).type(torch.FloatTensor)
        target = torch.from_numpy(unpacked[5]).type(torch.FloatTensor)

        return [rgb, depth, eof, tau, target, aux]

    def __len__(self):
        return self.length
