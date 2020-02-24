import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from src.datasets import ImitationLMDB

import torch
import torch.optim as optim
from torch.optim import Optimizer
import math
from torch.utils.data import DataLoader

import tqdm
import os
import sys



def train():
    modes = ['train', 'test']
    device = torch.device('cuda:0')
    # Define model, dataset, dataloader, loss, optimizer
    model = Model().to(device)
    #optimizer = optim.Adam(model.parameters())
    optimizer = Novograd(model.parameters())
    l2_norm = 0.002

    cost_file = 'w+'

    datasets = {mode: ImitationLMDB('temp_data_color', mode) for mode in modes}
    for epoch in tqdm.trange(1, 500, desc='Epochs'):
        dataloaders = {mode: DataLoader(datasets[mode], batch_size=64, shuffle=True, num_workers=8, pin_memory=True) for mode in modes}
        data_sizes = {mode: len(datasets[mode]) for mode in modes}
        for mode in modes:
            running_loss = 0.0
            for data in tqdm.tqdm(dataloaders[mode], desc='{}:{}/{}'.format(mode, epoch, 100),ascii=True):
                inputs = data[:-2]
                targets = data[-2:]
                curr_bs = inputs[0].shape[0]
                inputs = [x.to(device, non_blocking=False) for x in inputs]
                targets = [x.to(device, non_blocking=False) for x in targets][1]
                '''
                targets = torch.zeros(curr_bs, 9, 3).to(device)
                for r in range(3):
                    for c in range(3):
                        i = 3*r+c
                        targets[:, i] = inputs[0][:, :, (r+1)*inputs[0].size(2)//4, (c+1)*inputs[0].size(3)//4]
                '''
                for input in inputs:
                    if torch.any(torch.isnan(input)):
                        input.zero_()

                if mode == "train":
                    model.train()
                    optimizer.zero_grad()

                    out = model(inputs[0], inputs[1], inputs[3], False)#running_loss == 0 and epoch%10 == 0)
                    loss = torch.nn.MSELoss()(out, targets)
                    if l2_norm != 0:
                        l2_crit = torch.nn.MSELoss(size_average=False)
                        l2_loss = 0
                        for param in model.parameters():
                            l2_loss += l2_crit(param, torch.zeros_like(param))
                        loss += l2_norm * l2_loss

                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()*curr_bs
                elif mode == "test":
                    model.eval()
                    with torch.no_grad():
                        out = model(inputs[0], inputs[1], inputs[3])
                        loss = torch.nn.MSELoss()(out, targets)
                        running_loss += loss.item()*curr_bs
            cost = running_loss/data_sizes[mode]
            tqdm.tqdm.write("Epoch {} {} loss: {}".format(epoch, mode, cost))
    for mode in modes:
        datasets[mode].close()


def apply_film(active, x, params):
    if active:
        original_shape = x.shape
        alpha = params[:,0]
        beta = params[:,1]
        x = torch.add(torch.mul(alpha, x.view(original_shape[0], -1)), beta).view(original_shape)

    return x



class Novograd(Optimizer):
    """
    Implements Novograd algorithm.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.95, 0.98))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        grad_averaging: gradient averaging
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
    """

    def __init__(self, params, lr=1e-3, betas=(0.95, 0.98), eps=1e-8,
                 weight_decay=0, grad_averaging=False, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                      weight_decay=weight_decay,
                      grad_averaging=grad_averaging,
                      amsgrad=amsgrad)

        super(Novograd, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Novograd, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
            and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Sparse gradients are not supported.')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros([]).to(state['exp_avg'].device)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros([]).to(state['exp_avg'].device)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                norm = torch.sum(torch.pow(grad, 2))

                if exp_avg_sq == 0:
                    exp_avg_sq = norm
                else:
                    exp_avg_sq.mul_(beta2).add_(1 - beta2, norm)

                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                grad.div_(denom)
                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)
                if group['grad_averaging']:
                    grad.mul_(1 - beta1)
                exp_avg.mul_(beta1).add_(grad)

                p.data.add_(-group['lr'], exp_avg)

        return loss


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


class CoordConv2d(nn.Module):
    """
    CoordConv implementation (from uber, but really from like the 90s)
    """
    def __init__(self, width, height, *args, **kwargs):
        super(CoordConv2d, self).__init__()
        self.width = width
        self.height = height

        self.conv = nn.Conv2d(args[0], *args[1:], **kwargs)
        coords = torch.zeros(2, width, height)
        coords[0] = torch.stack([torch.arange(height).float()]*width, dim=0) * 2 / height - 1
        coords[1] = torch.stack([torch.arange(width).float()]*height, dim=1) * 2 / width - 1
        self.register_buffer('coords', coords)

    def forward(self, data):
        return self.conv(data)#torch.cat([data, torch.stack([self.coords]*data.size(0), dim=0)], dim=1))


class SpatialAttention2d(nn.Module):
    """
    CoordConv implementation (from uber, but really from like the 90s)
    """
    def __init__(self, channels, k):
        super(SpatialAttention2d, self).__init__()
        self.lin1 = nn.Linear(channels, k)
        self.lin2 = nn.Linear(k, 1)

    def forward(self, data):
        orgnl_shape = list(data.shape)
        temp_shape = [orgnl_shape[0], orgnl_shape[2]*orgnl_shape[3], orgnl_shape[1]]
        orgnl_shape[1] = 1
        atn = data.permute(0, 2, 3, 1).view(temp_shape)
        atn = self.lin2(torch.tanh(self.lin1(atn)))
        soft = torch.softmax(atn, dim=1)
        mask = (soft*temp_shape[1]).permute(0, 2, 1).view(orgnl_shape)
        return data*mask


class ChannelAttention2d(nn.Module):
    """
    CoordConv implementation (from uber, but really from like the 90s)
    """
    def __init__(self, width, height):
        super(ChannelAttention2d, self).__init__()
        self.lin = nn.Linear(width*height, 1)

    def forward(self, data):
        atn = self.lin(data.view(data.shape[0], data.shape[1], 1, -1))
        soft = torch.softmax(atn, dim=1)
        mask = soft*atn.shape[1]
        return data*mask


class Model(nn.Module):
    """
    The net implemented from Deep Imitation Learning from Virtual Teleoperation with parameterization
    """
    def __init__(self, *args, **kwargs):
        super(Model, self).__init__()
        # Note that all of the layers have valid padding
        #                                                               (120, 160)
        self.layer1_rgb = CoordConv2d(120, 160, 3, 64, kernel_size=7, stride=2) #   (57, 77)
        self.layer1_depth = CoordConv2d(120, 160, 1, 16, kernel_size=7, stride=2) # (57, 77)

        self.conv1 = CoordConv2d(57, 77, 64, 32, kernel_size=3, stride=2) # (28, 38)
        self.sa1   = SpatialAttention2d(32, 64)
        self.ca1   = ChannelAttention2d(28, 38)
        self.ss1   = SpatialSoftmax(28, 38, 32)
        self.conv2 = CoordConv2d(28, 38, 32, 32, kernel_size=3, stride=2) # (13, 18)
        self.sa2   = SpatialAttention2d(32, 64)
        self.ca2   = ChannelAttention2d(13, 18)
        self.ss2   = SpatialSoftmax(13, 18, 32)
        self.conv3 = CoordConv2d(13, 18, 32, 32, kernel_size=3, stride=2) # (6, 8)
        self.sa3   = SpatialAttention2d(32, 64)
        self.ca3   = ChannelAttention2d(6, 8)
        self.ss3   = SpatialSoftmax(6, 8, 32)

        self.film = nn.Sequential(nn.Linear(3,32),
                                  nn.ReLU(),
                                  nn.Linear(32,64),
                                  nn.ReLU(),
                                  nn.Linear(64,3*2))

        self.classifier = nn.Sequential(nn.Linear(64*3+3, 256),
                                        nn.ReLU(),
                                        nn.Linear(256, 2))

    def forward(self, rgb, depth, tau, print=False):
        x_rgb = self.layer1_rgb(rgb)
        x_depth = self.layer1_depth(depth)
        x = torch.cat([x_rgb], dim=1)
        # Spatial Film
        spatial_params = self.film(tau).view(-1, 3, 2, 1)

        x = apply_film(True, self.conv1(F.relu(x)), spatial_params[:, 0])

        if print:
            for i in range(x.size(1)):
                plt.subplot(8, 4, i+1)
                try:
                    plt.imshow(x[0,i].detach().cpu().numpy())
                except:
                    y = x[0,i].detach().cpu().numpy()
                    y = y - np.amin(y)
                    y = y / np.amax(y)
                    plt.imshow(y)
            plt.show()
        x = self.ca1(self.sa1(x))
        ss1 = self.ss1(x)

        if print:
            for i in range(x.size(1)):
                plt.subplot(8, 4, i+1)
                try:
                    plt.imshow(x[0,i].detach().cpu().numpy())
                except:
                    y = x[0,i].detach().cpu().numpy()
                    y = y - np.amin(y)
                    y = y / np.amax(y)
                    plt.imshow(y)
            plt.show()
        x = apply_film(True, self.conv2(F.relu(x)), spatial_params[:, 1])

        if print:
            for i in range(x.size(1)):
                plt.subplot(8, 4, i+1)
                try:
                    plt.imshow(x[0,i].detach().cpu().numpy())
                except:
                    y = x[0,i].detach().cpu().numpy()
                    y = y - np.amin(y)
                    y = y / np.amax(y)
                    plt.imshow(y)
            plt.show()
        x = self.ca2(self.sa2(x))
        ss2 = self.ss2(x)

        if print:
            for i in range(x.size(1)):
                plt.subplot(8, 4, i+1)
                try:
                    plt.imshow(x[0,i].detach().cpu().numpy())
                except:
                    y = x[0,i].detach().cpu().numpy()
                    y = y - np.amin(y)
                    y = y / np.amax(y)
                    plt.imshow(y)
            plt.show()
        x = apply_film(True, self.conv3(F.relu(x)), spatial_params[:, 2])

        if print:
            for i in range(x.size(1)):
                plt.subplot(8, 4, i+1)
                try:
                    plt.imshow(x[0,i].detach().cpu().numpy())
                except:
                    y = x[0,i].detach().cpu().numpy()
                    y = y - np.amin(y)
                    y = y / np.amax(y)
                    plt.imshow(y)
            plt.show()
        x = self.ca3(self.sa3(x))
        ss3 = self.ss3(x)

        if print:
            for i in range(x.size(1)):
                plt.subplot(8, 4, i+1)
                try:
                    plt.imshow(x[0,i].detach().cpu().numpy())
                except:
                    y = x[0,i].detach().cpu().numpy()
                    y = y - np.amin(y)
                    y = y / np.amax(y)
                    plt.imshow(y)
            plt.show()

        x = self.classifier(torch.cat([ss1, ss2, ss3, tau], dim=1))#.view(-1, 9, 3)

        return x


if __name__ == '__main__':
    train()
