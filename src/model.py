import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

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
        super(Model2, self).__init__()
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
        self.nfilm = nfilm

    def forward(self, rgb, depth, eof, tau):
        x_rgb = self.layer1_rgb(rgb)
        x_depth = self.layer1_depth(depth)
        x = F.relu(torch.cat([x_rgb, x_depth], dim=1))

        # Spatial Film
        spatial_params = self.spatial_film(tau)
        spatial_alpha = spatial_params[:,0].unsqueeze(1)
        spatial_beta = spatial_params[:,1].unsqueeze(1)
        
        # First FILM
        x = self.conv1(x)
        if self.nfilm == 1:
            original_shape = x.shape
            x = torch.add(torch.mul(spatial_alpha, x.view(original_shape[0], -1)), spatial_beta).view(original_shape)
        
        # Second FILM
        x = self.conv2(x)
        if self.nfilm == 2:
            # Spatial Film 2
            original_shape = x.shape
            x = torch.add(torch.mul(spatial_alpha, x.view(original_shape[0], -1)), spatial_beta).view(original_shape)
        
        # Third FILM
        x = self.conv3(x)
        if self.nfilm == 3:
            # Spatial Film 3
            original_shape = x.shape
            x = torch.add(torch.mul(spatial_alpha, x.view(original_shape[0], -1)), spatial_beta).view(original_shape)

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
