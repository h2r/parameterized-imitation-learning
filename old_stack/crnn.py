from __future__ import print_function, division
#import cv2
import matplotlib.pyplot as plt
from array import array
import numpy as np
import os
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from PIL import Image
from random import shuffle
from tqdm import tqdm
tqdm.monitor_interval = 0

import time
import math
import csv
import sys

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class SpatialCRNN(nn.Module):
    def __init__(self, hidden_dim=62, num_layers=1, rtype='LSTM', device="cuda:0"):
        super(SpatialCRNN, self).__init__()
        self.device = device
        # For the LSTM/GRU
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # Note that all of the layers have valid padding
        self.layer1_rgb = nn.Conv2d(3, 64, kernel_size=7, stride=2)
        self.layer1_depth = nn.Conv2d(1, 16, kernel_size=7, stride=2)
        self.bn1 = nn.BatchNorm2d(80)
        self.layer2 = nn.Conv2d(80, 32, kernel_size=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.layer3 = nn.Conv2d(32, 32, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(32)
        self.layer4 = nn.Conv2d(32, 32, kernel_size=3)
        self.bn4 = nn.BatchNorm2d(32)
        self.spatial_softmax = SpatialSoftmax(53, 73, 32)

        # Aux Network
        self.aux1 = nn.Linear(64, 40)
        self.aux2 = nn.Linear(40, 7)

        # 64 + 4(aux) + 3(eof)
        self.fl1 = nn.Linear(64, 50)
        # This is where we will enter the features along with the EOF history and tau history to the Recurrent Network
        # 50 + 3 (EOF [x,y,z]) + 2 (tau [pixel_x, pixel_y])
        if rtype == 'LSTM':
            self.r_layer = nn.LSTM(self.hidden_dim, 50, num_layers=self.num_layers, batch_first=True)
            self.hidden = self.init_hidden_lstm()
        elif rtype == 'GRU':
            self.r_layer = nn.GRU(self.hidden_dim, 50, num_layers=self.num_layers, batch_first=True, bidirectional=True)
            self.hidden = self.init_hidden_gru()
        # Take the output from the LSTM and then put through fully connected layers
        self.fl2 = nn.Linear(100, 50)
        self.output = nn.Linear(50, 6)
        #self.gripper_out = nn.Linear(30, 2)

    def forward(self, inputs):
        rgb = inputs[0]
        depth = inputs[1]
        eofs = inputs[2]
        taus = inputs[3]
        bs, seq_len, _ = taus.shape
        seq = torch.ones((bs, seq_len, self.hidden_dim), dtype=torch.float32, device=self.device)
        for i in range(seq_len):
            x_rgb = self.layer1_rgb(rgb[:,i,:,:,:])
            x_depth = self.layer1_depth(depth[:,i,:,:,:])
            x = F.relu(self.bn1(torch.cat([x_rgb, x_depth], dim=1)))
            x = F.relu(self.bn2(self.layer2(x)))
            x = F.relu(self.bn3(self.layer3(x)))
            x = F.relu(self.bn4(self.layer4(x)))
            x = self.spatial_softmax(x)

            aux = F.relu(self.aux1(x))
            aux = self.aux2(aux)

            x = F.relu(self.fl1(x))
            x = torch.cat([aux, x, eofs[:,i,:], taus[:,i,:]], 1)
            seq[:,i,:] = x
        output, self.hidden = self.r_layer(seq)
        #print(self.hidden.shape)
        output = output.view(bs,seq_len,2,50)
        output = F.tanh(torch.cat([output[:,seq_len-1,0,:], output[:,seq_len-1,1,:]], dim=1))
        output = F.relu(self.fl2(output))
        return self.output(output), aux

    def init_hidden_lstm(self, bs=64):
        return (torch.zeros(self.num_layers,bs,self.hidden_dim), torch.zeros(1,bs,self.hidden_dim))

    def init_hidden_gru(self, bs=64):
        return torch.zeros(2*self.num_layers,bs,self.hidden_dim)


class AttentionCRNN(nn.Module):
    """
    Using the ideas of attention from "Show, Attend, and Tell"
    """
    def __init__(self, hidden_dim=53, num_layers=1, rtype='LSTM'):
        super(AttentionCRNN, self).__init__()

        # For the LSTM/GRU
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        # Note that all of the layers have valid padding
        self.layer1_rgb = nn.Conv2d(3, 64, kernel_size=7, stride=2)
        self.layer1_depth = nn.Conv2d(1, 16, kernel_size=7, stride=2)
        self.layer2 = nn.Conv2d(80, 32, kernel_size=1)
        self.layer3 = nn.Conv2d(32, 32, kernel_size=3)
        self.layer4 = nn.Conv2d(32, 32, kernel_size=3)
        self.spatial_softmax = SpatialSoftmax(53, 73, 32)
        self.fl1 = nn.Linear(64, 50)
        # This is where we will enter the features along with the EOF history and tau history to the Recurrent Network
        # 50 + 3 (EOF [x,y,z]) + 2 (tau [pixel_x, pixel_y])
        if rtype == 'LSTM':
            self.r_layer = nn.LSTM(self.hidden_dim, 50, num_layers=self.num_layers, batch_first=True)
            self.hidden = self.init_hidden_lstm()
        elif rtype == 'GRU':
            self.r_layer = nn.GRU(self.hidden_dim, 50, num_layers=self.num_layers, batch_first=True, bidirectional=True)
            self.hidden = self.init_hidden_gru()
        # Take the output from the LSTM and then put through fully connected layers
        self.fl2 = nn.Linear(100, 50)
        self.output = nn.Linear(50, 6)
        #self.gripper_out = nn.Linear(30, 2)

    def forward(self, inputs):
        rgb = inputs[0]
        depth = inputs[1]
        eofs = inputs[2]
        taus = inputs[3]
        bs, seq_len, _ = taus.shape
        seq = torch.ones((bs, seq_len, 53), dtype=torch.float32, device="cuda:0")
        for i in range(seq_len):
            x_rgb = self.layer1_rgb(rgb[:,i,:,:,:])
            x_depth = self.layer1_depth(depth[:,i,:,:,:])
            x = F.relu(torch.cat([x_rgb, x_depth], dim=1))
            x = F.relu(self.layer2(x))
            x = F.relu(self.layer3(x))
            x = F.relu(self.layer4(x))
            print(x.shape)
            break
            #x = self.spatial_softmax(x)
        #     x = F.relu(self.fl1(x))
        #     x = torch.cat([x, eofs[:,i,:]], 1)
        #     seq[:,i,:] = x
        # output, self.hidden = self.r_layer(seq)
        # print(self.hidden.shape)
        # output = output.view(bs,5,2,50)
        # output = F.relu(torch.cat([output[:,4,0,:], output[:,4,1,:]], dim=1))
        # output = F.relu(self.fl2(output))
        # return self.output(output)

    def init_hidden_lstm(self, bs=64):
        return (torch.zeros(self.num_layers,bs,self.hidden_dim), torch.zeros(1,bs,self.hidden_dim))

    def init_hidden_gru(self, bs=64):
        return torch.zeros(2*self.num_layers,bs,self.hidden_dim)

class SpatialSoftmax(nn.Module):
    """
    Spatial Softmax Implementation
    """
    def __init__(self, height, width, channel, temperature=None, data_format='NCHW'):
        super(SpatialSoftmax, self).__init__()
        self.data_format = data_format
        self.height = height
        self.width = width
        self.channel = channel

        if temperature:
            self.temperature = Parameter(torch.ones(1)*temperature)
        else:
            self.temperature = 1.

        pos_x, pos_y = np.meshgrid(
            np.linspace(-1., 1., self.height),
            np.linspace(-1., 1., self.width))
        pos_x = torch.from_numpy(pos_x.reshape(self.height*self.width)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self.height*self.width)).float()
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)

    def forward(self, feature):
        if self.data_format == 'NHWC':
            feature = feature.transpose(1, 3).tranpose(2, 3).view(-1, self.height*self.width)
        else:
            feature = feature.view(-1, self.height*self.width)

        softmax_attention = F.softmax(feature/self.temperature, dim=-1)
        expected_x = torch.sum(Variable(self.pos_x)*softmax_attention, dim=1, keepdim=True)
        expected_y = torch.sum(Variable(self.pos_y)*softmax_attention, dim=1, keepdim=True)
        expected_xy = torch.cat([expected_x, expected_y], 1)
        feature_keypoints = expected_xy.view(-1, self.channel*2)

        return feature_keypoints

class BehaviorCloneLoss(nn.Module):
    """
    The Loss function described in the paper
    """
    def __init__(self, lamb_l2=0.01, lamb_l1=1.0, lamb_c=0.005, lamb_g=0.01, lamb_aux=0.0001):
        super(BehaviorCloneLoss, self).__init__()
        self.lamb_l2 = lamb_l2
        self.lamb_l1 = lamb_l1
        self.lamb_c = lamb_c
        self.lamb_g = lamb_g
        self.lamb_aux = lamb_aux
        self.l2 = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.aux = nn.MSELoss()
        #self.g = nn.CrossEntropyLoss()
        #self.aux = nn.CrossEntropyLoss()

    def forward(self, outputs, targets, batch_size=64):
        output = outputs[0]
        #gripper_output = outputs[1]
        target = targets[0]
        #gripper_target = targets[1]
        #g_loss = self.g(gripper_output, gripper_target)
        l2_loss = self.l2(output, target)
        l1_loss = self.l1(output, target)

        aux_output = outputs[1]
        aux_target = targets[1]
        aux_loss = self.aux(aux_output, aux_target)


        return self.lamb_l2*l2_loss + self.lamb_l1*l1_loss + self.lamb_aux*aux_loss
        #return self.lamb_l2*l2_loss + self.lamb_l1*l1_loss + self.lamb_g*g_loss + self.lamb_aux*aux_loss

class CRNNDataset(Dataset):
    def __init__(self, root_dir, mode, seq_len):
        self.seq_len = seq_len
        file = None
        if mode is 'train':
            file = root_dir + "/train_data.csv"
        else:
            file = root_dir + "/test_data.csv"
        self.data = []
        with open(file, "r") as data:
            reader = csv.reader(data, delimiter = ",")
            for line in reader:
                data = {
                    "root": line[0],
                    "curr_idx": int(line[1]),
                    "tau": [int(line[2]), int(line[3])],
                    "seq": [int(line[i]) for i in range(4,4+seq_len)],
                    "label": [float(line[i]) for i in range(4+seq_len, 10+seq_len)],
                    "aux_label": [float(line[i]) for i in range(10+seq_len, 17+seq_len)],
                    "eof": [[float(line[i-2]), float(line[i-1]), float(line[i])] for i in range(19+seq_len, len(line), 3)]
                }
                self.data.append(data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        element = self.data[idx]
        # Sequence of Data
        tau = [element["tau"] for _ in range(self.seq_len)]
        rgb_files = [element["root"]+"/"+str(element["seq"][i])+"_rgb.png" for i in range(self.seq_len)]
        depth_files = [element["root"]+"/"+str(element["seq"][i])+"_depth.png" for i in range(self.seq_len)]
        rgb_images, depth_images = [], []
        for rgb, depth in zip(rgb_files, depth_files):
            rgb_image = Image.open(rgb)
            depth_image = Image.open(depth)
            rgb_image = np.asarray(rgb_image).astype(float)
            depth_image = np.asarray(depth_image).astype(float)
            rgb_image = np.reshape(rgb_image, (3, rgb_image.shape[0], -1))
            depth_image = np.reshape(depth_image, (1, depth_image.shape[0], -1))
            # Normalize
            rgb_image = (2*(rgb_image - np.amin(rgb_image))/(np.amax(rgb_image)-np.amin(rgb_image)))-1
            # Create the sequences
            rgb_images.append(rgb_image)
            depth_images.append(depth_image)
        #return rgb_image, depth_image, np.array(eval(element["eof"])), np.array(eval(element["tau"])), (np.array(eval(element["label"])), eval(element["gripper"]))
        return (np.array(rgb_images), np.array(depth_images), np.array(element["eof"]), np.array(tau)), (np.array(element["label"]), np.array(element["aux_label"]))

def train(root_dir, name, device="cuda:1", num_epochs=1000, bs=64, lr=0.0001, seq_len=5, weight=None, dest=None):
    curr_time = time.time()
    modes = ["train", "test"]
    costs = {mode: [] for mode in modes}
    model = SpatialCRNN(rtype="GRU", num_layers=2, device=device).cuda(device)
    if weight is not None:
        model.load_state_dict(torch.load(weight, map_location=device))
    criterion = BehaviorCloneLoss().cuda(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    datasets = {mode: CRNNDataset(root_dir, mode, seq_len) for mode in modes}
    dataloaders = {mode: DataLoader(datasets[mode], batch_size=bs, shuffle=True, num_workers=6) for mode in modes}
    data_sizes = {mode: len(datasets[mode]) for mode in modes}
    print("DataLoading took {} minutes".format((time.time()-curr_time)/60))
    print("Starting Training...")
    min_test_loss = float('inf')
    for epoch in range(1, num_epochs+1):
        for mode in modes:
            running_loss = 0.0
            for inputs, targets in tqdm(dataloaders[mode], desc='{}:{}/{}'.format(mode, epoch, num_epochs), ascii=True):
                bs = inputs[0].shape[0]
                inputs = [x.float().cuda(device) for x in inputs]
                targets = [x.float().cuda(device) for x in targets]
                if mode == "train":
                    model.train()
                    optimizer.zero_grad()
                    out = model(inputs)
                    loss = criterion(out, targets)
                    loss.backward()
                    optimizer.step()
                    running_loss = running_loss + (loss.data[0]*bs)
                elif mode == "test":
                    model.eval()
                    with torch.no_grad():
                        out = model(inputs)
                        loss = criterion(out, targets)
                        running_loss = running_loss + (loss.data[0]*bs)
            cost = running_loss/data_sizes[mode]
            print("{} Loss: {}".format(mode, cost))
            # Print the cost and accuracy every 10 epoch
            if epoch % 5 == 0:
                costs[mode].append(cost)
            if mode == 'test' and cost <= min_test_loss:
                min_test_loss = cost
                torch.save(model.state_dict(), dest + name+".pt")
                torch.save(optimizer.state_dict(), dest + name + ".pt")
            if epoch % 25 == 0: 
                torch.save(model.state_dict(), dest + name + "_"+str(epoch)+".pt")
                torch.save(optimizer.state_dict(), dest + name + "_optim_" + str(epoch) + ".pt")
    # Save the Model
    # torch.save(model.state_dict(), "results/reach/case_00_"+str(num_epochs))

    # plot the cost
    x = np.arange(0, len(costs["train"]))
    plt.plot(x, np.squeeze(costs["train"]), x, np.squeeze(costs["test"]))
    plt.ylabel("Loss")
    plt.xlabel("Epochs")
    plt.title("Learning rate = " + str(lr))
    plt.legend(["Training", "Testing"])
    plt.savefig(dest + name + ".png")

if __name__ == '__main__':
    root_dir = sys.argv[1]
    device = "cuda:{}".format(sys.argv[2])
    name = sys.argv[3]
    dest = sys.argv[4]
    #dataset = CRNNDataset(root_dir, 'train', 5) 
    #dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=6)
    #for inputs, targets in dataloader:
    #    print(inputs[0].shape)
    #    print(inputs[1].shape)
    #    print(inputs[2].shape)
    #    print(inputs[3].shape)
    #    print(targets[0].shape)
    #    print(targets[1].shape)
    #    break 
    prev = time.time()
    train(root_dir, name, device=device, num_epochs=1000, dest=dest, seq_len=10)
    print("Training Took {} hours".format((time.time() - prev)/3600))

    # model = AttentionCRNN(rtype="GRU", num_layers=2)
    # dataset = CRNNDataset(root_dir, "test")
    # dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=6)
    # for rgb, depth, eof, tau, target in dataloader:
    #     output = model([rgb.float(), depth.float(), eof.float(), tau.float()])
    #     break
