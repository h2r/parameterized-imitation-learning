from src.model import Model2, Model3

import torch
import numpy as np

import tqdm

def norm_img(img):
    return 2*((img - np.amin(img))/(np.amax(img)-np.amin(img)))-1

def evaluate(file, weights, error=1e-03):
	correct_count = 0
	total = 0
	# Initialize Model
	model = Model2(is_aux=True, nfilm=3)
	checkpoint = torch.load(weights, map_location="cpu")
	model.load_state_dict(checkpoint['model_state_dict'])
	model.eval()
	with open(file, 'r') as f:
		arr = [line.strip.split(',') for line in f]
		total = len(arr)
		for row in tqdm.tqdm(arr):
			rgb = norm_img(np.array(Image.open(row[0]).convert("RGB")))
            depth = norm_img(np.array(Image.open(row[1])))

            # Reshape Images to have channel first
            rgb = np.reshape(rgb, (3, rgb.shape[0], -1))
            depth = np.reshape(depth, (1, depth.shape[0], -1))
            eof = np.array([float(x) for x in row[2:17]])
            tau = np.array([float(x) for x in row[17:20]])
            target = [float(x) for x in row[26:32]]
            # Add the dummy value
            target.append(1.0)
            target = np.array(target)

            rgb = torch.from_numpy(np.expand_dims(rgb, 0)).type(torch.FloatTensor)
            depth = torch.from_numpy(np.expand_dims(depth, 0)).type(torch.FloatTensor)
            eof = torch.from_numpy(np.expand_dims(eof, 0)).type(torch.FloatTensor)
            tau = torch.from_numpy(np.expand_dims(tau, 0)).type(torch.FloatTensor)

            with torch.no_grad():
            	out, _ = model(rgb, depth, eof, tau)
            	out = out.squeeze().item()
            	diff = np.average(np.absolute(target - out))
            	if diff <= error:
            		correct_count += 1
    return correct_count/total


