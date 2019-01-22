from __future__ import print_function, division
#import cv2
import matplotlib.pyplot as plt
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

import time
import math
import csv
import sys

import dsae

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class Model(nn.Module):
	"""
	The net implemented from Deep Imitation Learning from Virtual Teleoperation
	"""
	def __init__(self):
		super(Model, self).__init__()
		# Note that all of the layers have valid padding
		self.layer1_rgb = nn.Conv2d(3, 64, kernel_size=7, stride=2)
		self.layer1_depth = nn.Conv2d(1, 16, kernel_size=7, stride=2)
		self.layer2 = nn.Conv2d(80, 32, kernel_size=1)
		self.layer3 = nn.Conv2d(32, 32, kernel_size=3)
		self.layer4 = nn.Conv2d(32, 32, kernel_size=3)
		self.spatial_softmax = SpatialSoftmax(53, 73, 32)
		# Testing the auxiliary for finding quaternion
		#self.aux1 = nn.Linear(64, 40)
		#self.aux2 = nn.Linear(40, 7)

		self.fl1 = nn.Linear(64, 50)
		# This is where the concatenation of the output from spatialsoftmax
		# and the end effector information of the last 5 movements
		# are added. We use 7 points(x, y, z, a, b, c, d) so 50 + 35 = 85
		#self.fl2 = nn.Linear(102, 50)
		# self.fl2 = nn.Linear(95, 50)
		self.fl2 = nn.Linear(65, 50)
		#self.fl2 = nn.Linear(110, 50)
		#self.fltest = nn.Linear(80, 50)
		#self.fl2 = nn.Linear(85, 50)
		# We use 7 instead of 6 in the paper for (linear velocity, angular velocity, grip {0, 1})
		self.output = nn.Linear(50, 6)
		self.gripper_out = nn.Linear(50, 2)

	def forward(self, input_layer):
		rgb = input_layer[0]
		depth = input_layer[1]
		#7x5 quaternion + position 35 + 50 = 85 
		eof = input_layer[2]
		x_rgb = self.layer1_rgb(rgb)
		x_depth = self.layer1_depth(depth)
		x = F.relu(torch.cat([x_rgb, x_depth], dim=1))
		x = F.relu(self.layer2(x))
		x = F.relu(self.layer3(x))
		x = F.relu(self.layer4(x))
		x = self.spatial_softmax(x)
		#aux = F.relu(self.aux1(x))
		#aux = F.relu(self.aux2(aux))
		x = F.relu(self.fl1(x))
		#x = F.relu(self.fl2(torch.cat([aux, x, eof], dim=1)))
		x = F.relu(self.fl2(torch.cat([x, eof], dim=1)))
		#x = F.relu(self.fltest(x))
		gripper = self.gripper_out(x)
		x = self.output(x)
		#return x, aux, gripper
		return x, gripper

class Model2(nn.Module):
	"""
	The net implemented from Deep Imitation Learning from Virtual Teleoperation
	"""
	def __init__(self):
		super(Model2, self).__init__()
		# Note that all of the layers have valid padding

		self.fl1 = nn.Linear(64, 50)
		# This is where the concatenation of the output from spatialsoftmax
		# and the end effector information of the last 5 movements
		# are added. We use 7 points(x, y, z, a, b, c, d) so 50 + 35 = 85
		self.fl2 = nn.Linear(65, 50)
		# We use 7 instead of 6 in the paper for (linear velocity, angular velocity, grip {0, 1})
		self.output = nn.Linear(50, 6)
		self.gripper_out = nn.Linear(50, 2)

	def forward(self, input_layer):
		features = input_layer[0]
		#7x5 quaternion + position 35 + 50 = 85 
		eof = input_layer[1]
		x = F.relu(self.fl1(features))
		#x = F.relu(self.fl2(torch.cat([aux, x, eof], dim=1)))
		#print(x.shape)
		#print(eof.shape)
		x = F.relu(self.fl2(torch.cat([x, eof], dim=1)))
		#x = F.relu(self.fltest(x))
		gripper = self.gripper_out(x)
		x = self.output(x)
		#return x, aux, gripper
		return x, gripper

class Model3(nn.Module):
	"""
	The net implemented from Deep Imitation Learning from Virtual Teleoperation
	"""
	def __init__(self):
		super(Model3, self).__init__()
		# Note that all of the layers have valid padding

		self.fl1 = nn.Linear(64, 50)
		# This is where the concatenation of the output from spatialsoftmax
		# and the end effector information of the last 5 movements
		# also add tau
		self.fl2 = nn.Linear(67, 50)
		# We use 7 instead of 6 in the paper for (linear velocity, angular velocity, grip {0, 1})
		self.output = nn.Linear(50, 6)
		self.gripper_out = nn.Linear(50, 2)

	def forward(self, input_layer):
		features = input_layer[0]
		#7x5 quaternion + position 35 + 50 = 85 
		eof = input_layer[1]
		tau = input_layer[2]
		x = F.relu(self.fl1(features))
		#x = F.relu(self.fl2(torch.cat([aux, x, eof], dim=1)))
		#print(x.shape)
		#print(eof.shape)
		x = F.relu(self.fl2(torch.cat([tau, x, eof], dim=1)))
		#x = F.relu(self.fltest(x))
		gripper = self.gripper_out(x)
		x = self.output(x)
		#return x, aux, gripper
		return x, gripper
		
class Model4(nn.Module):
	"""
	The net implemented from Deep Imitation Learning from Virtual Teleoperation
	"""
	def __init__(self):
		super(Model4, self).__init__()
		# Note that all of the layers have valid padding
		self.layer1_rgb = nn.Conv2d(3, 64, kernel_size=7, stride=2)
		self.layer1_depth = nn.Conv2d(1, 16, kernel_size=7, stride=2)
		self.layer2 = nn.Conv2d(80, 32, kernel_size=1)
		self.layer3 = nn.Conv2d(32, 32, kernel_size=3)
		self.layer4 = nn.Conv2d(32, 32, kernel_size=3)
		self.spatial_softmax = SpatialSoftmax(53, 73, 32)
		# Testing the auxiliary for finding quaternion
		#self.aux1 = nn.Linear(64, 40)
		#self.aux2 = nn.Linear(40, 7)

		self.fl1 = nn.Linear(64, 50)
		# This is where the concatenation of the output from spatialsoftmax
		# and the end effector information of the last 5 movements
		# are added. We use 7 points(x, y, z, a, b, c, d) so 50 + 35 = 85
		#self.fl2 = nn.Linear(102, 50)
		# self.fl2 = nn.Linear(95, 50)
		self.fl2 = nn.Linear(67, 50)
		#self.fl2 = nn.Linear(110, 50)
		#self.fltest = nn.Linear(80, 50)
		#self.fl2 = nn.Linear(85, 50)
		# We use 7 instead of 6 in the paper for (linear velocity, angular velocity, grip {0, 1})
		self.output = nn.Linear(50, 6)
		self.gripper_out = nn.Linear(50, 2)

	def forward(self, input_layer):
		rgb = input_layer[0]
		depth = input_layer[1]
		#7x5 quaternion + position 35 + 50 = 85 
		eof = input_layer[2]
		tau = input_layer[3]
		x_rgb = self.layer1_rgb(rgb)
		x_depth = self.layer1_depth(depth)
		x = F.relu(torch.cat([x_rgb, x_depth], dim=1))
		x = F.relu(self.layer2(x))
		x = F.relu(self.layer3(x))
		x = F.relu(self.layer4(x))
		x = self.spatial_softmax(x)
		#aux = F.relu(self.aux1(x))
		#aux = F.relu(self.aux2(aux))
		x = F.relu(self.fl1(x))
		#x = F.relu(self.fl2(torch.cat([aux, x, eof], dim=1)))
		x = F.relu(self.fl2(torch.cat([tau, x, eof], dim=1)))
		#x = F.relu(self.fltest(x))
		gripper = self.gripper_out(x)
		x = self.output(x)
		#return x, aux, gripper
		return x, gripper

class SpatialCRNN(nn.Module):
	def __init__(self):
		super(SpatialCRNN, self).__init__()
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
		self.r_layer = nn.LSTM(55, 50)
		# Take the output from the LSTM and then put through fully connected layers
		self.fl2 = nn.Linear(50, 30)
		self.output = nn.Linear(30, 6)
		self.gripper_out = nn.Linear(30, 2)

	def forward(self, inputs):
		rgb = input_layer[0]
		depth = input_layer[1]
		eofs = input_layer[2]
		taus = input_layer[3]
		_, seq_len, _ = taus.shape
		for i in range(seq_len):
			x_rgb = self.layer1_rgb(rgb[:,i,:,:,:])
			x_depth = self.layer1_depth(depth[:,i,:,:,:])
			x = F.relu(torch.cat([x_rgb, x_depth], dim=1))
			x = F.relu(self.layer2(x))
			x = F.relu(self.layer3(x))
			x = F.relu(self.layer4(x))
			x = self.spatial_softmax(x)





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
		self.g = nn.CrossEntropyLoss()
		self.aux = nn.MSELoss()

	def arccos(self, output, target):
		_, n = output.shape
		num = torch.bmm(target.view(1, 1, n), output.view(1, n, 1)).view(1)
		den = torch.bmm(torch.norm(target.view(1,n,1),p=2,dim=1,keepdim=True),torch.norm(output.view(1,n,1),p=2,dim=1,keepdim=True)).view(1)
		if sum(den) == 0:
			return 0
		else:
			return torch.mean(torch.acos(num/den))


	def forward(self, outputs, targets, batch_size=64):
		output = outputs[0]
		#aux_output = outputs[1]
		gripper_output = outputs[1]
		target = targets[0]
		#aux_target = targets[1]
		gripper_target = targets[1]
		g_loss = self.g(gripper_output, gripper_target)
		l2_loss = self.l2(output, target)
		l1_loss = self.l1(output, target)

		print(ouput.shape)

		# For the arccos loss
		# bs, n = output.shape
		# num = torch.bmm(target.view(bs,1,n), output.view(bs,n,1)).view(bs)
		# den = torch.bmm(torch.norm(target.view(bs,n,1),p=2,dim=1,keepdim=True),torch.norm(output.view(bs,n,1),p=2,dim=1,keepdim=True)).view(bs)
		# frac = torch.ones((bs)).cuda()
		# for i, val in enumerate(den):
		# 	if val == 0:
		# 		frac[i] = 0
		# 	else:
		# 		frac[i] = num[i]/den[i]
		# frac = Variable(frac)
		# c_loss = torch.mean(torch.acos(frac))


		# if c_loss == float('nan'):
		# 	print("PROBLEM")
		# 	print(output)
		# 	print(target)
		# 	print(num)
		# 	print(den)
		#	print(frac)
		# For the aux loss
		#aux_loss = self.aux(aux_output, aux_target)

		#return self.lamb_l2*l2_loss + self.lamb_l1*l1_loss + self.lamb_c*c_loss + self.lamb_g*g_loss + self.lamb_aux*aux_loss
		return self.lamb_l2*l2_loss + self.lamb_l1*l1_loss + self.lamb_g*g_loss

class ArcCosLoss(nn.Module):
	"""
	ArcCos Loss which is a part of the behvior clone loss
	"""
	def __init__(self):
		super(ArcCosLoss, self).__init__()

	def forward(self, output, target):



class ImitationDataset(Dataset):
	def __init__(self, root_dir, mode):
		file = None
		if mode is 'train':
			file = root_dir + "/train_data.csv"
		else:
			file = root_dir + "/test_data.csv"
		self.data = []
		with open(file, "r") as data:
			reader = csv.reader(data, delimiter = "\t")
			for line in reader:
				data = {
					"depth": line[0],
					"rgb": line[1],
					"eof": line[2],
					"label": line[3],
					"gripper": line[4]
				}
				self.data.append(data)

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		element = self.data[idx]
		rgb_image = Image.open(element["rgb"])
		depth_image = Image.open(element["depth"])
		rgb_image = np.asarray(rgb_image.resize((160,120))).astype(float)
		depth_image = np.asarray(depth_image.resize((160,120))).astype(float)
		rgb_image = np.reshape(rgb_image, (3, rgb_image.shape[0], -1))
		depth_image = np.reshape(depth_image, (1, depth_image.shape[0], -1))
		# Normalize
		rgb_image = (2*(rgb_image - np.amin(rgb_image))/(np.amax(rgb_image)-np.amin(rgb_image)))-1
		return (rgb_image, depth_image, np.array(eval(element["eof"]))), (np.array(eval(element["label"])), eval(element["gripper"]))

class ImitationDataset2(Dataset):
	def __init__(self, root_dir, mode):
		file = None
		if mode is 'train':
			file = root_dir + "/train_data.csv"
		else:
			file = root_dir + "/test_data.csv"
		self.data = []
		with open(file, "r") as data:
			reader = csv.reader(data, delimiter = "\t")
			for line in reader:
				data = {
					"depth": line[0],
					"rgb": line[1],
					"eof": line[2],
					"label": line[3],
					"gripper": line[4]
				}
				self.data.append(data)

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		element = self.data[idx]
		rgb_image = Image.open(element["rgb"])
		depth_image = Image.open(element["depth"])
		rgb_image = np.asarray(rgb_image.resize((160,120))).astype(float)
		depth_image = np.asarray(depth_image.resize((160,120))).astype(float)
		rgb_image = np.reshape(rgb_image, (3, rgb_image.shape[0], -1))
		depth_image = np.reshape(depth_image, (1, depth_image.shape[0], -1))
		# Normalize
		rgb_image = (2*(rgb_image - np.amin(rgb_image))/(np.amax(rgb_image)-np.amin(rgb_image)))-1
		return (rgb_image, depth_image), np.array(eval(element["eof"])), (np.array(eval(element["label"])), eval(element["gripper"]))

class ImitationDataset3(Dataset):
	def __init__(self, root_dir, mode):
		file = None
		if mode is 'train':
			file = root_dir + "/train_data.csv"
		else:
			file = root_dir + "/test_data.csv"
		self.data = []
		with open(file, "r") as data:
			reader = csv.reader(data, delimiter = "\t")
			for line in reader:
				data = {
					"depth": line[0],
					"rgb": line[1],
					"eof": line[2],
					"label": line[3],
					"gripper": line[4],
					"tau": line[5]
				}
				self.data.append(data)

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		element = self.data[idx]
		rgb_image = Image.open(element["rgb"])
		depth_image = Image.open(element["depth"])
		rgb_image = np.asarray(rgb_image.resize((160,120))).astype(float)
		depth_image = np.asarray(depth_image.resize((160,120))).astype(float)
		rgb_image = np.reshape(rgb_image, (3, rgb_image.shape[0], -1))
		depth_image = np.reshape(depth_image, (1, depth_image.shape[0], -1))
		# Normalize
		rgb_image = (2*(rgb_image - np.amin(rgb_image))/(np.amax(rgb_image)-np.amin(rgb_image)))-1
		return (rgb_image, depth_image), np.array(eval(element["eof"])), np.array(eval(element["tau"])), (np.array(eval(element["label"])), eval(element["gripper"]))

class ImitationDataset4(Dataset):
	def __init__(self, root_dir, mode):
		file = None
		if mode is 'train':
			file = root_dir + "/train_data.csv"
		else:
			file = root_dir + "/test_data.csv"
		self.data = []
		with open(file, "r") as data:
			reader = csv.reader(data, delimiter = "\t")
			for line in reader:
				data = {
					"depth": line[0],
					"rgb": line[1],
					"eof": line[2],
					"label": line[3],
					"gripper": line[4],
					"tau": line[5]
				}
				self.data.append(data)

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		element = self.data[idx]
		rgb_image = Image.open(element["rgb"])
		depth_image = Image.open(element["depth"])
		rgb_image = np.asarray(rgb_image.resize((160,120))).astype(float)
		depth_image = np.asarray(depth_image.resize((160,120))).astype(float)
		rgb_image = np.reshape(rgb_image, (3, rgb_image.shape[0], -1))
		depth_image = np.reshape(depth_image, (1, depth_image.shape[0], -1))
		# Normalize
		rgb_image = (2*(rgb_image - np.amin(rgb_image))/(np.amax(rgb_image)-np.amin(rgb_image)))-1
		return rgb_image, depth_image, np.array(eval(element["eof"])), np.array(eval(element["tau"])), (np.array(eval(element["label"])), eval(element["gripper"]))

def train(root_dir, name, device="cuda:0", num_epochs=1000, bs=64, lr=0.0001, weight=None):
	curr_time = time.time()
	modes = ["train", "test"]
	#modes = ["train"]
	costs = {mode: [] for mode in modes}
	#accuracy = {mode: [] for mode in modes}
	model = Model()
	if weight is not None:
		model.load_state_dict(torch.load(weight, map_location="cuda:0"))
	model.cuda(device)
	criterion = BehaviorCloneLoss().cuda(device)
	optimizer = optim.Adam(model.parameters(), lr=lr)
	datasets = {mode: ImitationDataset(root_dir, mode) for mode in modes}
	dataloaders = {mode: DataLoader(datasets[mode], batch_size=bs, shuffle=True, num_workers=6) for mode in modes}
	data_sizes = {mode: len(datasets[mode]) for mode in modes}
	#print(data_sizes)
	print("DataLoading took {} minutes".format((time.time()-curr_time)/60))
	print("Starting Training...")
	for epoch in range(1, num_epochs+1):
		for mode in modes:
			running_loss = 0.0
			for input_layer, target in tqdm(dataloaders[mode], desc='{}:{}/{}'.format(mode, epoch, num_epochs), ascii=True):
				input_layer = [x.float().cuda(device) for x in input_layer]
				#target = target.cuda()
				target = [x.float().cuda(device) if i == 0 else x.long().cuda(device) for i, x in enumerate(target)]
				if mode == "train":
					model.train()
					optimizer.zero_grad()
					outputs = model(input_layer)
					loss = criterion(outputs, target)
					loss.backward()
					optimizer.step()
					running_loss = running_loss + loss.data[0]
				elif mode == "test":
					model.eval()
					with torch.no_grad():
						outputs = model(input_layer)
						loss = criterion(outputs, target)
						running_loss = running_loss + loss.data[0]
			cost = running_loss/data_sizes[mode]
			print("{} Loss: {}".format(mode, cost))
			# Print the cost and accuracy every 10 epoch
			if epoch % 10 == 0:
				costs[mode].append(cost)
		if epoch % 100 == 0:
			torch.save(model.state_dict(), "results/reach/" + name + "_"+str(epoch))

	# Save the Model
	# torch.save(model.state_dict(), "results/reach/case_00_"+str(num_epochs))

	# plot the cost
	x = np.arange(0, len(costs["train"]))
	plt.plot(x, np.squeeze(costs["train"]), x, np.squeeze(costs["test"]))
	plt.ylabel("Loss")
	plt.xlabel("Epochs (per tens)")
	plt.title("Learning rate =" + str(lr))
	plt.legend(["Training", "Testing"])
	plt.savefig("results/reach/" + name + ".png")

def train2(root_dir, name, device="cuda:0", num_epochs=150, bs=64, lr=0.0001, weight=None, dest="results/reach_encoded3/", dsae_weights="results/dsae/decoding_lowest_loss_407"):
	curr_time = time.time()
	modes = ["train", "test"]
	#modes = ["train"]
	costs = {mode: [] for mode in modes}
	#accuracy = {mode: [] for mode in modes}
	model = Model2()
	if weight is not None:
		model.load_state_dict(torch.load(weight, map_location="cuda:0"))
	model.cuda(device)
	criterion = BehaviorCloneLoss().cuda(device)
	optimizer = optim.Adam(model.parameters(), lr=lr)
	datasets = {mode: ImitationDataset2(root_dir, mode) for mode in modes}
	dataloaders = {mode: DataLoader(datasets[mode], batch_size=bs, shuffle=True, num_workers=6) for mode in modes}
	data_sizes = {mode: len(datasets[mode]) for mode in modes}
	#print(data_sizes)
	print("DataLoading took {} minutes".format((time.time()-curr_time)/60))
	print("Starting Training...")
	for epoch in range(1, num_epochs+1):
		for mode in modes:
			running_loss = 0.0
			for imgs, eof, target in tqdm(dataloaders[mode], desc='{}:{}/{}'.format(mode, epoch, num_epochs), ascii=True):
				img_input = [x.float().cuda(device) for x in imgs]
				eof_input = eof.float().cuda(device)
				features = dsae.get_prediction(img_input, dsae_weights, device=device)
				target = [x.float().cuda(device) if i == 0 else x.long().cuda(device) for i, x in enumerate(target)]
				if mode == "train":
					model.train()
					optimizer.zero_grad()
					outputs = model([features, eof_input])
					loss = criterion(outputs, target)
					loss.backward()
					optimizer.step()
					running_loss = running_loss + loss.data[0]
				elif mode == "test":
					model.eval()
					with torch.no_grad():
						outputs = model([features, eof_input])
						loss = criterion(outputs, target)
						running_loss = running_loss + loss.data[0]
			cost = running_loss/data_sizes[mode]
			print("{} Loss: {}".format(mode, cost))
			# Print the cost and accuracy every 10 epoch
			if epoch % 1 == 0:
				costs[mode].append(cost)
		if epoch % 50 == 0:
			torch.save(model.state_dict(), dest + name + "_"+str(epoch)+".pt")
			torch.save(optimizer.state_dict(), dest + name + "_optim_" + str(epoch) + ".pt")

	# Save the Model
	# torch.save(model.state_dict(), "results/reach/case_00_"+str(num_epochs))

	# plot the cost
	x = np.arange(0, len(costs["train"]))
	plt.plot(x, np.squeeze(costs["train"]), x, np.squeeze(costs["test"]))
	plt.ylabel("Loss")
	plt.xlabel("Epochs ")
	plt.title("Learning rate =" + str(lr))
	plt.legend(["Training", "Testing"])
	plt.savefig(dest + name + ".png")

def train3(root_dir, name, device="cuda:0", num_epochs=1000, bs=64, lr=0.0001, weight=None, dest="results/reach_encoded3/", dsae_weights="results/dsae/decoding_lowest_loss_407"):
	curr_time = time.time()
	modes = ["train", "test"]
	#modes = ["train"]
	costs = {mode: [] for mode in modes}
	#accuracy = {mode: [] for mode in modes}
	model = Model3()
	if weight is not None:
		model.load_state_dict(torch.load(weight, map_location="cuda:0"))
	model.cuda(device)
	criterion = BehaviorCloneLoss().cuda(device)
	optimizer = optim.Adam(model.parameters(), lr=lr)
	datasets = {mode: ImitationDataset3(root_dir, mode) for mode in modes}
	dataloaders = {mode: DataLoader(datasets[mode], batch_size=bs, shuffle=True, num_workers=6) for mode in modes}
	data_sizes = {mode: len(datasets[mode]) for mode in modes}
	#print(data_sizes)
	print("DataLoading took {} minutes".format((time.time()-curr_time)/60))
	print("Starting Training...")
	for epoch in range(1, num_epochs+1):
		for mode in modes:
			running_loss = 0.0
			for imgs, eof, tau, target in tqdm(dataloaders[mode], desc='{}:{}/{}'.format(mode, epoch, num_epochs), ascii=True):
				img_input = [x.float().cuda(device) for x in imgs]
				eof_input = eof.float().cuda(device)
				tau_input = tau.float().cuda(device)
				features = dsae.get_prediction(img_input, dsae_weights, device=device)
				target = [x.float().cuda(device) if i == 0 else x.long().cuda(device) for i, x in enumerate(target)]
				if mode == "train":
					model.train()
					optimizer.zero_grad()
					outputs = model([features, eof_input, tau_input])
					loss = criterion(outputs, target)
					loss.backward()
					optimizer.step()
					running_loss = running_loss + loss.data[0]
				elif mode == "test":
					model.eval()
					with torch.no_grad():
						outputs = model([features, eof_input, tau_input])
						loss = criterion(outputs, target)
						running_loss = running_loss + loss.data[0]
			cost = running_loss/data_sizes[mode]
			print("{} Loss: {}".format(mode, cost))
			# Print the cost and accuracy every 10 epoch
			if epoch % 1 == 0:
				costs[mode].append(cost)
		if epoch % 50 == 0:
			torch.save(model.state_dict(), dest + name + "_"+str(epoch)+".pt")
			torch.save(optimizer.state_dict(), dest + name + "_optim_" + str(epoch) + ".pt")

	# Save the Model
	# torch.save(model.state_dict(), "results/reach/case_00_"+str(num_epochs))

	# plot the cost
	x = np.arange(0, len(costs["train"]))
	plt.plot(x, np.squeeze(costs["train"]), x, np.squeeze(costs["test"]))
	plt.ylabel("Loss")
	plt.xlabel("Epochs ")
	plt.title("Learning rate =" + str(lr))
	plt.legend(["Training", "Testing"])
	plt.savefig(dest + name + ".png")

def train4(root_dir, name, device="cuda:0", num_epochs=1000, bs=64, lr=0.0001, weight=None, dest="results/reach_encoded3/"):
	curr_time = time.time()
	modes = ["train", "test"]
	#modes = ["train"]
	costs = {mode: [] for mode in modes}
	#accuracy = {mode: [] for mode in modes}
	model = Model4()
	if weight is not None:
		model.load_state_dict(torch.load(weight, map_location="cuda:0"))
	model.cuda(device)
	criterion = BehaviorCloneLoss().cuda(device)
	optimizer = optim.Adam(model.parameters(), lr=lr)
	datasets = {mode: ImitationDataset4(root_dir, mode) for mode in modes}
	dataloaders = {mode: DataLoader(datasets[mode], batch_size=bs, shuffle=True, num_workers=6) for mode in modes}
	data_sizes = {mode: len(datasets[mode]) for mode in modes}
	#print(data_sizes)
	print("DataLoading took {} minutes".format((time.time()-curr_time)/60))
	print("Starting Training...")
	for epoch in range(1, num_epochs+1):
		for mode in modes:
			running_loss = 0.0
			for rgb, depth, eof, tau, target in tqdm(dataloaders[mode], desc='{}:{}/{}'.format(mode, epoch, num_epochs), ascii=True):
				rgb_input = rgb.float().cuda(device)
				depth_input = depth.float().cuda(device)
				eof_input = eof.float().cuda(device)
				tau_input = tau.float().cuda(device)
				target = [x.float().cuda(device) if i == 0 else x.long().cuda(device) for i, x in enumerate(target)]
				if mode == "train":
					model.train()
					optimizer.zero_grad()
					outputs = model([rgb_input, depth_input, eof_input, tau_input])
					loss = criterion(outputs, target)
					loss.backward()
					optimizer.step()
					running_loss = running_loss + loss.data[0]
				elif mode == "test":
					model.eval()
					with torch.no_grad():
						outputs = model([rgb_input, depth_input, eof_input, tau_input])
						loss = criterion(outputs, target)
						running_loss = running_loss + loss.data[0]
			cost = running_loss/data_sizes[mode]
			print("{} Loss: {}".format(mode, cost))
			# Print the cost and accuracy every 10 epoch
			if epoch % 1 == 0:
				costs[mode].append(cost)
		if epoch % 50 == 0:
			torch.save(model.state_dict(), dest + name + "_"+str(epoch)+".pt")
			torch.save(optimizer.state_dict(), dest + name + "_optim_" + str(epoch) + ".pt")

	# Save the Model
	# torch.save(model.state_dict(), "results/reach/case_00_"+str(num_epochs))

	# plot the cost
	x = np.arange(0, len(costs["train"]))
	plt.plot(x, np.squeeze(costs["train"]), x, np.squeeze(costs["test"]))
	plt.ylabel("Loss")
	plt.xlabel("Epochs ")
	plt.title("Learning rate =" + str(lr))
	plt.legend(["Training", "Testing"])
	plt.savefig(dest + name + ".png")

def get_prediction2(model, input_layer):
	with torch.no_grad():
		image_input = [input_layer[0], input_layer[1]]
		print(image_input[0].shape)
		eof_input = input_layer[2]
		features = dsae.get_prediction(image_input, "results/dsae/decoding_lowest_loss_407")
		output = model([features, eof_input])
	return output

def get_prediction(rgb, depth, eof, weights="torch_imp"):
	"""
	Gets output from loaded in model. Weights is the saved torch model. 
	"""
	# Initialize Model
	model = Model()
	model.load_state_dict(torch.load(weights, map_location="cuda:0"))
	model.eval()
	model.cuda("cuda:0")

	# Process Input
	depth_image = cv2.resize(depth, (160,120))
	rgb_image = cv2.resize(rgb, (160,120))
	depth_input = Variable(torch.from_numpy(np.reshape(depth_image, (1, 1, depth_image.shape[0], -1))).type(torch.FloatTensor))
	rgb_input = Variable(torch.from_numpy(np.reshape(rgb_image, (1, 3, rgb_image.shape[0], -1))).type(torch.FloatTensor))
	eof_input = Variable(torch.from_numpy(np.reshape(np.array(eof), (1,15))).type(torch.FloatTensor))
	#eof_input = Variable(torch.from_numpy(np.reshape(np.array(eof), (1,60))).type(torch.FloatTensor))
	#eof_input = Variable(torch.from_numpy(np.reshape(np.array(eof), (1,35))).type(torch.FloatTensor))
	input_layer = [rgb_input.cuda("cuda:0"), depth_input.cuda("cuda:0"), eof_input.cuda("cuda:0")]
	output = model(input_layer)
	output = output.cpu()
	return output.data.numpy()

if __name__ == '__main__':
	root_dir = sys.argv[1]
	device = "cuda:{}".format(sys.argv[2])
	name = sys.argv[3]
	dest = sys.argv[4]
	prev = time.time()
	train4(root_dir, name, device=device, num_epochs=1000, dest=dest)
	#train2(root_dir, name, device=device, weight="results/reach_encoded/case_00_encoded_100")
	#train(root_dir, bs=32, weight="results/reach/case_10_1000")
	print("Training Took {} hours".format((time.time() - prev)/3600))
