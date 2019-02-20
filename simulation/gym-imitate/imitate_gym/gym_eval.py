import gym
import imitate_gym
from PIL import Image
import numpy as np
import torch
from crnn import SpatialCRNN
import sys
seq_len = 10
weights = sys.argv[1]
tau = [[int(sys.argv[2]), int(sys.argv[3])] for _ in range(seq_len)]
tau = torch.from_numpy(np.array(tau)).type(torch.FloatTensor)
tau = torch.unsqueeze(tau, dim=0)
env = gym.make('buttons-v0')
env.reset()
model = SpatialCRNN(rtype="GRU", num_layers=2, device="cpu")
model.load_state_dict(torch.load(weights, map_location='cpu'), strict=False)
model.eval()
eof = None
rgbs = []
while True:
    env.render()
    rgb = env.render('rgb_array')
    if eof is None:
        obs, _,_,_ = env.step([0.,0.,0.,0.,0.,0.,0.,0.])
        pos = obs['achieved_goal']
        eof = [pos for _ in range(seq_len)]
        img_rgb = Image.fromarray(rgb, 'RGB')
        img_rgb = np.array(img_rgb.resize((160,120)))
        img_rgb = np.reshape(img_rgb,(3,120,160))
        rgbs = [img_rgb for _ in range(seq_len)]
    else:
        img_rgb = Image.fromarray(rgb, 'RGB')
        img_rgb = np.array(img_rgb.resize((160,120)))
        img_rgb = np.reshape(img_rgb,(3,120,160))
        rgbs.pop(0)
        rgbs.append(img_rgb)
        input_rgb = torch.from_numpy(np.array(rgbs)).type(torch.FloatTensor)
        input_rgb = torch.unsqueeze(input_rgb, dim=0)
        input_eof = torch.from_numpy(np.array(eof)).type(torch.FloatTensor)
        input_eof = torch.unsqueeze(input_eof, dim=0)
        vels, _ = model([input_rgb, input_eof, tau])
        action = [vels[0][0].item(), vels[0][1].item(), vels[0][2].item(), 0., 0., 0., 0., 0.]
        print(action)
        obs, _, _, _ = env.step(action)
        eof.pop(0)
        eof.append(obs['achieved_goal'])

