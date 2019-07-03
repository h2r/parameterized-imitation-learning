import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import h5py
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader
import csv
import tqdm
import argparse
from PIL import Image
from apex import amp

class SpatialSoftmax(nn.Module):
    """
    Spatial Softmax Implementation
    """
    def __init__(self, height, width, channel):
        super(SpatialSoftmax, self).__init__()
        self.height = height
        self.width = width
        self.channel = channel

        pos_x, pos_y = np.meshgrid(
            np.linspace(-1., 1., self.height),
            np.linspace(-1., 1., self.width))
        pos_x = torch.from_numpy(pos_x.reshape(self.height*self.width)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self.height*self.width)).float()
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)

    def forward(self, feature):
        feature = feature.view(-1, self.height*self.width)
        softmax_attention = F.softmax(feature, dim=-1)
        expected_x = torch.sum(self.pos_x*softmax_attention, dim=1, keepdim=True)
        expected_y = torch.sum(self.pos_y*softmax_attention, dim=1, keepdim=True)
        expected_xy = torch.cat([expected_x, expected_y], 1)
        feature_keypoints = expected_xy.view(-1, self.channel*2)

        return feature_keypoints

class Model(nn.Module):
    """
    The net implemented from Deep Imitation Learning from Virtual Teleoperation with parameterization
    """
    def __init__(self, is_aux=False):
        super(Model, self).__init__()
        self.is_aux = is_aux
        # Note that all of the layers have valid padding
        self.layer1_rgb = nn.Conv2d(3, 64, kernel_size=7, stride=2)
        self.layer1_depth = nn.Conv2d(1, 16, kernel_size=7, stride=2)
        self.spatial_film = nn.Sequential(nn.Linear(3,2),
                                          nn.ReLU(),
                                          nn.Linear(2,2),
                                          nn.ReLU(),
                                          nn.Linear(2,2))
        self.conv1 = nn.Sequential(nn.Conv2d(80, 32, kernel_size=1),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3),
                                   nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3),
                                   nn.ReLU())                           
        self.spatial_softmax = SpatialSoftmax(53, 73, 32)
        # Testing the auxiliary for finding final pose. It was shown in many tasks that
        # predicting the final pose was a helpful auxiliary task. EE Pose is <x,y,z,q_x,q_y,q_z,q_w>.
        # Note that the output from the spatial softmax is 32 (x,y) positions and thus 64 variables
        self.aux = nn.Sequential(nn.Linear(64, 40),
                                 nn.ReLU(),
                                 nn.Linear(40, 6))
        # This is where the concatenation of the output from spatialsoftmax
        self.fl1 = nn.Linear(64, 50)
        # Concatenating the Auxiliary Predictions and EE history. Past 5 history of <x,y,z>.
        # This comes out to 50 + 6 (aux) + 15 (ee history) = 71
        if self.is_aux:
        	self.fl2 = nn.Linear(71, 50)
        else:
        	self.fl2 = nn.Linear(65, 50)
        # FiLM Conditioning: Input x,y pixel location to learn alpha and beta
        self.film = nn.Sequential(nn.Linear(3,2),
                                  nn.ReLU(),
                                  nn.Linear(2,2),
                                  nn.ReLU(),
                                  nn.Linear(2,2))
        # We use 7 to incorporate the loss function (linear vel, angular vel, dummy)
        self.output = nn.Linear(50, 7)

        # Initialize the weights
        resnet = torchvision.models.resnet101(pretrained=True)
        resnet = list(resnet.children())[0]
        self.layer1_rgb.weight.data = resnet.weight.data
        nn.init.uniform_(self.layer1_depth.weight,a=-0.01,b=0.01)
        # Convolutional Weight Updates
        nn.init.uniform_(self.conv1[0].weight,a=-0.01,b=0.01)
        nn.init.uniform_(self.conv2[0].weight,a=-0.01,b=0.01)
        nn.init.uniform_(self.conv3[0].weight,a=-0.01,b=0.01)
        for i in range(0,5,2):
            nn.init.uniform_(self.spatial_film[i].weight,a=-0.01,b=0.01)
            nn.init.uniform_(self.film[i].weight,a=-0.01,b=0.01)
            if i < 3:
                nn.init.uniform_(self.aux[i].weight,a=-0.01,b=0.01)
            if i < 5:
                nn.init.uniform_(self.film[i].weight,a=-0.01,b=0.01)
        nn.init.uniform_(self.fl1.weight,a=-0.01,b=0.01)
        nn.init.uniform_(self.fl2.weight,a=-0.01,b=0.01)
        nn.init.uniform_(self.output.weight,a=-0.01,b=0.01)


    def forward(self, rgb, depth, eof, tau):
        x_rgb = self.layer1_rgb(rgb)
        x_depth = self.layer1_depth(depth)
        x = F.relu(torch.cat([x_rgb, x_depth], dim=1))

        # Spatial Film
        spatial_params = self.spatial_film(tau)
        spatial_alpha = spatial_params[:,0].unsqueeze(1)
        spatial_beta = spatial_params[:,1].unsqueeze(1)

        x = self.conv1(x) 
        x = self.conv2(x)
        # Move these 2 lines around for experiments
        original_shape = x.shape
        x = torch.add(torch.mul(spatial_alpha, x.view(original_shape[0], -1)), spatial_beta).view(original_shape)
        
        x = self.conv3(x)
        x = self.spatial_softmax(x)
        aux = self.aux(x)
        x = F.relu(self.fl1(x))

        if self.is_aux:
        	x = F.relu(self.fl2(torch.cat([aux, x, eof], dim=1)))
        else:
        	x = F.relu(self.fl2(torch.cat([x, eof], dim=1)))

        # FiLM Conditioning here
        params = self.film(tau)
        # Unsqueeze to maintain batch size as first dimension
        alpha = params[:,0].unsqueeze(1)
        beta = params[:,1].unsqueeze(1)
        # alpha*x + beta
        x = torch.add(torch.mul(alpha, x), beta)
        x = self.output(x)
        return x, aux


class Model2(nn.Module):
    """
    The net implemented from Deep Imitation Learning from Virtual Teleoperation with parameterization
    """
    def __init__(self, is_aux=True, nfilm=1):
        super(Model, self).__init__()
        self.is_aux = is_aux
        # Note that all of the layers have valid padding
        self.layer1_rgb = nn.Conv2d(3, 64, kernel_size=7, stride=2)
        self.layer1_depth = nn.Conv2d(1, 16, kernel_size=7, stride=2)
        self.spatial_film = nn.Sequential(nn.Linear(3,2),
                                          nn.ReLU(),
                                          nn.Linear(2,2),
                                          nn.ReLU(),
                                          nn.Linear(2,2))
        if nfilm == 2 or nfilm == 3:            
            self.spatial_film2 = nn.Sequential(nn.Linear(3,2),
                                              nn.ReLU(),
                                              nn.Linear(2,2),
                                              nn.ReLU(),
                                              nn.Linear(2,2))

        if nfilm == 3:            
            self.spatial_film3 = nn.Sequential(nn.Linear(3,2),
                                              nn.ReLU(),
                                              nn.Linear(2,2),
                                              nn.ReLU(),
                                              nn.Linear(2,2))

        self.conv1 = nn.Sequential(nn.Conv2d(80, 32, kernel_size=1),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3),
                                   nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(32, 32, kernel_size=3),
                                   nn.ReLU())                           
        self.spatial_softmax = SpatialSoftmax(53, 73, 32)
        # Testing the auxiliary for finding final pose. It was shown in many tasks that
        # predicting the final pose was a helpful auxiliary task. EE Pose is <x,y,z,q_x,q_y,q_z,q_w>.
        # Note that the output from the spatial softmax is 32 (x,y) positions and thus 64 variables
        self.aux = nn.Sequential(nn.Linear(64, 40),
                                 nn.ReLU(),
                                 nn.Linear(40, 6))
        # This is where the concatenation of the output from spatialsoftmax
        self.fl1 = nn.Linear(64, 50)
        # Concatenating the Auxiliary Predictions and EE history. Past 5 history of <x,y,z>.
        # This comes out to 50 + 6 (aux) + 15 (ee history) = 71
        if self.is_aux:
        	self.fl2 = nn.Linear(71, 50)
        else:
        	self.fl2 = nn.Linear(65, 50)
        # FiLM Conditioning: Input x,y pixel location to learn alpha and beta
        self.film = nn.Sequential(nn.Linear(3,2),
                                  nn.ReLU(),
                                  nn.Linear(2,2),
                                  nn.ReLU(),
                                  nn.Linear(2,2))
        # We use 7 to incorporate the loss function (linear vel, angular vel, dummy)
        self.output = nn.Linear(50, 7)

        # Initialize the weights
        resnet = torchvision.models.resnet101(pretrained=True)
        resnet = list(resnet.children())[0]
        self.layer1_rgb.weight.data = resnet.weight.data
        nn.init.uniform_(self.layer1_depth.weight,a=-0.01,b=0.01)
        # Convolutional Weight Updates
        nn.init.uniform_(self.conv1[0].weight,a=-0.01,b=0.01)
        nn.init.uniform_(self.conv2[0].weight,a=-0.01,b=0.01)
        nn.init.uniform_(self.conv3[0].weight,a=-0.01,b=0.01)
        for i in range(0,5,2):
            nn.init.uniform_(self.spatial_film[i].weight,a=-0.01,b=0.01)
            if nfilm == 2 or nfilm == 3:
                nn.init.uniform_(self.spatial_film2[i].weight,a=-0.01,b=0.01)
            if nfilm == 3:
                nn.init.uniform_(self.spatial_film3[i].weight,a=-0.01,b=0.01)
            nn.init.uniform_(self.film[i].weight,a=-0.01,b=0.01)
            if i < 3:
                nn.init.uniform_(self.aux[i].weight,a=-0.01,b=0.01)
            if i < 5:
                nn.init.uniform_(self.film[i].weight,a=-0.01,b=0.01)
        nn.init.uniform_(self.fl1.weight,a=-0.01,b=0.01)
        nn.init.uniform_(self.fl2.weight,a=-0.01,b=0.01)
        nn.init.uniform_(self.output.weight,a=-0.01,b=0.01)
        self.nfilm = nfilm

    def forward(self, rgb, depth, eof, tau):
        x_rgb = self.layer1_rgb(rgb)
        x_depth = self.layer1_depth(depth)
        x = F.relu(torch.cat([x_rgb, x_depth], dim=1))

        # Spatial Film 1
        spatial_params1 = self.spatial_film(tau)
        spatial_alpha1 = spatial_params1[:,0].unsqueeze(1)
        spatial_beta1 = spatial_params1[:,1].unsqueeze(1)
        
        # First FILM
        x = self.conv1(x) 
        original_shape = x.shape
        x = torch.add(torch.mul(spatial_alpha, x.view(original_shape[0], -1)), spatial_beta).view(original_shape)
        
        # Second FILM
        x = self.conv2(x)
        if self.nfilm == 2 or self.nfilm == 3:
            # Spatial Film 2
            spatial_params2 = self.spatial_film2(tau)
            spatial_alpha2 = spatial_params2[:,0].unsqueeze(1)
            spatial_beta2 = spatial_params2[:,1].unsqueeze(1)
            original_shape = x.shape
            x = torch.add(torch.mul(spatial_alpha2, x.view(original_shape[0], -1)), spatial_beta2).view(original_shape)
        
        # Third FILM
        x = self.conv3(x)
        if self.nfilm == 3:
            # Spatial Film 3
            spatial_params3 = self.spatial_film3(tau)
            spatial_alpha3 = spatial_params3[:,0].unsqueeze(1)
            spatial_beta3 = spatial_params3[:,1].unsqueeze(1)
            original_shape = x.shape
            x = torch.add(torch.mul(spatial_alpha3, x.view(original_shape[0], -1)), spatial_beta3).view(original_shape)

        x = self.spatial_softmax(x)
        aux = self.aux(x)
        x = F.relu(self.fl1(x))

        if self.is_aux:
        	x = F.relu(self.fl2(torch.cat([aux, x, eof], dim=1)))
        else:
        	x = F.relu(self.fl2(torch.cat([x, eof], dim=1)))

        # FiLM Conditioning here
        params = self.film(tau)
        # Unsqueeze to maintain batch size as first dimension
        alpha = params[:,0].unsqueeze(1)
        beta = params[:,1].unsqueeze(1)
        # alpha*x + beta
        x = torch.add(torch.mul(alpha, x), beta)
        x = self.output(x)
        return x, aux
class BehaviorCloneLoss(nn.Module):
    """
    The Loss function described in the paper
    """
    def __init__(self, lamb_l2=0.01, lamb_l1=1.0, lamb_c=0.005, lamb_aux=0.0001):
        super(BehaviorCloneLoss, self).__init__()
        self.lamb_l2 = lamb_l2
        self.lamb_l1 = lamb_l1
        self.lamb_c = lamb_c
        self.lamb_aux = lamb_aux
        self.l2 = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.aux = nn.MSELoss()

        self.eps = 1e-7

    def forward(self, out, aux_out, target, aux_target):
        l2_loss = self.l2(out, target)
        l1_loss = self.l1(out, target)

        # For the arccos loss
        bs, n = out.shape
        num = torch.bmm(target.view(bs,1,n), out.view(bs,n,1))
        den = torch.bmm(torch.norm(target.view(bs,n,1),p=2,dim=1,keepdim=True),
                        torch.norm(out.view(bs,n,1),p=2,dim=1,keepdim=True))
        a_cos = torch.squeeze(torch.acos(torch.clamp(torch.div(num, den), 0, 1-self.eps)))
        c_loss = torch.mean(a_cos)
        #print("THIS IS DEN {}".format(den))
        #print("THIS IS ACOS {}".format(torch.acos(torch.div(num, den))))
        #print("THIS IS C_LOSS {}".format(c_loss))
        # For the aux loss
        aux_loss = self.aux(aux_out, aux_target)

        return self.lamb_l2*l2_loss + self.lamb_l1*l1_loss + self.lamb_c*c_loss + self.lamb_aux*aux_loss

class ImitationDataset(Dataset):
    def __init__(self, data_file, mode):
        super(ImitationDataset, self).__init__()

        self.data = h5py.File(data_file + "/data"  + mode + '.hdf5', 'r', driver='core')

    def __len__(self):
        return self.data['rgb'].shape[0]

    def __getitem__(self, idx):
        rgb = torch.from_numpy(self.data['rgb'][idx,:,:,:]).type(torch.FloatTensor)
        depth = torch.from_numpy(self.data['depth'][idx,:,:,:]).type(torch.FloatTensor)
        eof = torch.from_numpy(self.data['eof'][idx,:]).type(torch.FloatTensor)
        tau = torch.from_numpy(self.data['tau'][idx,:]).type(torch.FloatTensor)
        aux = torch.from_numpy(self.data['aux'][idx,:]).type(torch.FloatTensor)
        target = self.data['target'][idx,:]
        target = torch.from_numpy(target).type(torch.FloatTensor)
        return [rgb, depth, eof, tau, target, aux]

def train(data_file, save_path, num_epochs=1000, bs=128, lr=0.002, device='cuda:0', weight=None, is_aux=True, nfilm=1):
    modes = ['train', 'test']
    # Define model, dataset, dataloader, loss, optimizer
    model = Model2(is_aux=is_aux, nfilm=nfilm).cuda(device)
    if weight is not None:
        model.load_state_dict(torch.load(weight, map_location=device))
    criterion = BehaviorCloneLoss().cuda(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model, optimizer = amp.initialize(model, optimizer, opt_level='O0')
    #model = nn.DataParallel(model)
    datasets = {mode: ImitationDataset(data_file, mode) for mode in modes}
    dataloaders = {mode: DataLoader(datasets[mode], batch_size=bs, shuffle=True, num_workers=8, pin_memory=True) for mode in modes}
    data_sizes = {mode: len(datasets[mode]) for mode in modes}
    lowest_test_cost = float('inf')
    cost_file = open(save_path+"/costs.txt", 'w+')
    for epoch in tqdm.trange(1, num_epochs+1, desc='Epochs'):
        for mode in modes:
            running_loss = 0.0
            for data in tqdm.tqdm(dataloaders[mode], desc='{}:{}/{}'.format(mode, epoch, num_epochs),ascii=True):
                inputs = data[:-2]
                targets = data[-2:]
                curr_bs = inputs[0].shape[0]
                inputs = [x.cuda(device, non_blocking=True) for x in inputs]
                targets = [x.cuda(device, non_blocking=True) for x in targets]
                if mode == "train":
                    model.train()
                    optimizer.zero_grad()
                    out, aux_out = model(inputs[0], inputs[1], inputs[2], inputs[3])
                    loss = criterion(out, aux_out, targets[0], targets[1])
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    optimizer.step()
                    running_loss += (loss.item()*curr_bs)
                elif mode == "test":
                    model.eval()
                    with torch.no_grad():
                        out, aux_out = model(inputs[0], inputs[1], inputs[2], inputs[3])
                        loss = criterion(out, aux_out, targets[0], targets[1])
                        running_loss += (loss.item()*curr_bs)
            cost = running_loss/data_sizes[mode]
            print(str(epoch)+","+mode+","+str(cost)+"\n")
            cost_file.write(str(epoch)+","+mode+","+str(cost)+"\n")
            if mode == 'test':
                if lowest_test_cost >= cost:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': cost
                        }, save_path+"/best_checkpoint.tar")
                    lowest_test_cost = cost
            if epoch % 50 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': cost
                    }, save_path+"/checkpoint.tar")
            tqdm.tqdm.write("{} loss: {}".format(mode, cost))
    cost_file.close()

if __name__ == '__main__':
    m = Model()
    m(torch.ones(64,3,120,160), torch.ones(64,1,120,160), torch.ones(64,15), torch.ones(64,3))
    parser = argparse.ArgumentParser(description='Input to data cleaner')
    parser.add_argument('-d', '--data_file', required=True, help='Path to data.hdf5')
    parser.add_argument('-s', '--save_path', required=True, help='Path to save the model weights/checkpoints and results')
    parser.add_argument('-ne', '--num_epochs', required=False, default=1000, type=int, help='Number of epochs')
    parser.add_argument('-bs', '--batch_size', required=False, default=64, type=int, help='Batch Size')
    parser.add_argument('-lr', '--learning_rate', required=False, default=0.001, type=float, help='Learning Rate')
    parser.add_argument('-device', '--device', required=False, default="cuda:0", type=str, help='The cuda device')
    parser.add_argument('-aux', '--aux', required=False, default=True, type=bool, help='Whether or not to connect the auxiliary task')
    parser.add_argument('-nf', '--nfilm', required=False, default=1, type=int, help='Number of film layers')
    args = parser.parse_args()

    train(args.data_file,
          args.save_path,
          num_epochs=args.num_epochs,
          bs=args.batch_size,
          lr=args.learning_rate,
          device=args.device,
          is_aux=args.aux,
          nfilm=args.nfilm)
