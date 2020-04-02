import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt



def apply_film(active, x, params):
    if active:
        original_shape = x.shape
        alpha = params[:,0]
        beta = params[:,1]
        x = torch.add(torch.mul(alpha, x.view(original_shape[0], -1)), beta).view(original_shape)

    return x


class CoordConv2d(nn.Module):
    """
    CoordConv implementation (from uber, but really from like the 90s)
    """
    def __init__(self, *args, use_coords=False, attend=False, batch_norm=False, dropout=0, conditioning=0, **kwargs):
        super(CoordConv2d, self).__init__()
        self.use_coords = use_coords
        self.attend = attend
        self.conditioning = conditioning
        self.width = 5
        self.height = 5
        self.dropout = nn.Dropout2d(0)
        self.batch_norm = nn.BatchNorm2d(args[1]) if batch_norm else lambda x: x

        coords = torch.zeros(2, self.width, self.height)
        coords[0] = torch.stack([torch.arange(self.height).float()]*self.width, dim=0) * 2 / self.height - 1
        coords[1] = torch.stack([torch.arange(self.width).float()]*self.height, dim=1) * 2 / self.width - 1
        self.register_buffer('coords', coords)

        if self.use_coords:
            args = list(args)
            args[0] += 2
            args = tuple(args)

        self.conv = nn.Conv2d(*args, **kwargs)

        if self.attend:
            self.attend = SpatialAttention2d(args[1], self.attend, self.conditioning)

    def reset(self):
        self.height = 5
        self.width = 5
        self.setup()

    def setup(self):
        pos_x, pos_y = np.meshgrid(
            np.linspace(-1., 1., self.height),
            np.linspace(-1., 1., self.width))
        self.coords = torch.zeros(2, self.width, self.height).to(self.coords)
        self.coords[0] = torch.from_numpy(pos_x).float().to(self.coords)
        self.coords[1] = torch.from_numpy(pos_y).float().to(self.coords)

    def forward(self, data, cond=None):
        if self.use_coords:
            flag = False
            if not (self.width == data.shape[2]):
                self.width = data.shape[2]
                flag = True
            if not (self.height == data.shape[3]):
                self.height = data.shape[3]
                flag = True

            if flag:
                self.setup()

            data = torch.cat([data, torch.stack([self.coords]*data.size(0), dim=0) * data.mean() * 2], dim=1)


        x = self.conv(data)
        x = F.leaky_relu(x)
        if self.attend:
            x = self.attend(x, cond)
        x = self.batch_norm(x)
        return self.dropout(x)



class SpatialAttention2d(nn.Module):
    """
    CoordConv implementation (from uber, but really from like the 90s)
    """
    def __init__(self, channels, k, conditioning=0):
        super(SpatialAttention2d, self).__init__()
        self.conditioning = conditioning
        self.lin1 = nn.Linear(channels, k)
        self.lin2 = nn.Linear(k+conditioning, 1)

    def forward(self, data, cond=None, b_print=False, print_path=''):
        orgnl_shape = list(data.shape)
        if (cond is None) or (self.conditioning == 0):
            cond = torch.zeros(orgnl_shape[0], self.conditioning)
        temp_shape = [orgnl_shape[0], orgnl_shape[2]*orgnl_shape[3], orgnl_shape[1]]
        orgnl_shape[1] = 1
        atn = data.permute(0, 2, 3, 1).view(temp_shape)
        atn = self.lin1(atn)
        atn = torch.tanh(torch.cat([atn, cond.unsqueeze(1).expand(atn.size(0), atn.size(1), self.conditioning)], dim=2))
        atn = self.lin2(atn)
        soft = torch.softmax(atn, dim=1)
        mask = soft.permute(0, 2, 1).view(orgnl_shape)*temp_shape[1]
        if b_print:
            plt.figure(1)
            plt.imshow(mask[0, 0].detach().cpu().numpy())
            plt.savefig(print_path+'mask.png')
        return data*mask


class ChannelAttention2d(nn.Module):
    """
    CoordConv implementation (from uber, but really from like the 90s)
    """
    def __init__(self, img_size):
        super(ChannelAttention2d, self).__init__()
        self.lin = nn.Linear(img_size, 1)

    def forward(self, data):
        atn = self.lin(data.view(data.shape[0], data.shape[1], 1, -1))
        soft = torch.softmax(atn, dim=1)
        mask = soft*atn.shape[1]
        return data#*mask


class SpatialSoftmax(nn.Module):
    """
    Spatial Softmax Implementation
    """
    def __init__(self):
        super(SpatialSoftmax, self).__init__()
        self.height = 5
        self.width = 5
        self.channel = 5

        pos_x, pos_y = np.meshgrid(
            np.linspace(-1., 1., self.height),
            np.linspace(-1., 1., self.width))
        pos_x = torch.from_numpy(pos_x.reshape(self.height*self.width)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self.height*self.width)).float()
        self.register_buffer('pos_x', pos_x)
        self.register_buffer('pos_y', pos_y)

    def reset(self):
        self.height = 5
        self.width = 5
        self.channel = 5
        self.setup()

    def setup(self):
        pos_x, pos_y = np.meshgrid(
            np.linspace(-1., 1., self.height),
            np.linspace(-1., 1., self.width))
        self.pos_x = torch.from_numpy(pos_x.reshape(self.height*self.width)).float().to(self.pos_x)
        self.pos_y = torch.from_numpy(pos_y.reshape(self.height*self.width)).float().to(self.pos_y)

    def forward(self, feature):
        flag = False
        if not (self.channel == feature.shape[1]):
            self.channel = feature.shape[1]
            flag = True
        if not (self.width == feature.shape[2]):
            self.width = feature.shape[2]
            flag = True
        if not (self.height == feature.shape[3]):
            self.height = feature.shape[3]
            flag = True

        if flag:
            self.setup()

        feature = torch.log(F.relu(feature.view(-1, self.height*self.width)) + 1e-6)
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
    def __init__(self, use_bias=True, use_tau=True, eof_size=15, tau_size=3, aux_size=6, out_size=7):
        super(Model, self).__init__()
        self.use_bias = use_bias
        self.use_tau  = use_tau
        self.eof_size = eof_size
        self.tau_size = tau_size
        self.aux_size = aux_size
        self.out_size = out_size


        self.conv1 = CoordConv2d(4, 32, kernel_size=5, stride=2, batch_norm=True, dropout=.2, bias=use_bias)
        self.conv2 = CoordConv2d(32, 32, kernel_size=3, stride=1, batch_norm=True, dropout=.2, bias=use_bias)
        self.conv3 = CoordConv2d(32, 64, kernel_size=3, stride=2, batch_norm=True, dropout=.2, bias=use_bias)
        self.conv4 = CoordConv2d(64, 64, kernel_size=3, stride=1, batch_norm=True, dropout=.2, bias=use_bias)
        self.conv5 = CoordConv2d(64, 128, kernel_size=3, stride=2, batch_norm=True, dropout=.2, bias=use_bias)#, attend=512, conditioning=32, use_coords=True)
        self.conv6 = CoordConv2d(128, 128, kernel_size=3, stride=1, batch_norm=True, dropout=.2, bias=use_bias)
        self.conv7 = CoordConv2d(128, 256, kernel_size=3, stride=1, batch_norm=True, dropout=.2, bias=use_bias)
        self.conv8 = CoordConv2d(256, 256, kernel_size=3, stride=1, batch_norm=True, dropout=.2, bias=use_bias)

        self.conv_lin1 = nn.Linear(256*(11*6), 512)
        self.conv_lin2 = nn.Linear(512, 512)
        self.dropout   = nn.Dropout(.5)

        self.spatial_softmax = SpatialSoftmax()

        self.resetables = [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv7, self.conv8, self.spatial_softmax]

        if not use_tau:
            self.augment_tau = nn.Embedding(9, tau_size)

        self.film = nn.Sequential(nn.Linear(tau_size, 32, bias=use_bias),
                                  nn.LeakyReLU(),
                                  nn.Linear(32, 32*8, bias=use_bias))

        # Testing the auxiliary for finding final pose. It was shown in many tasks that
        # predicting the final pose was a helpful auxiliary task. EE Pose is <x,y,z,q_x,q_y,q_z,q_w>.
        # Note that the output from the spatial softmax is 32 (x,y) positions and thus 64 variables
        self.aux = nn.Sequential(nn.Linear(512*2 + tau_size, 512, bias=use_bias),
                                 nn.LeakyReLU(),
                                 nn.Linear(512, aux_size, bias=use_bias))
        # This is where the concatenation of the output from spatialsoftmax
        self.fl1 = nn.Linear(512*2 + tau_size, 512, bias=use_bias)
        # Concatenating the Auxiliary Predictions and EE history. Past 5 history of <x,y,z>.
        # This comes out to 50 + 6 (aux) + 15 (ee history) = 71\
        self.fl2 = nn.Linear(512+eof_size+aux_size, 512, bias=use_bias)

	    # We use 6 to incorporate the loss function (linear vel, angular vel)
        self.output = nn.Linear(512, out_size, bias=use_bias)


    def reset(self):
        for res in self.resetables:
            res.reset()


    def forward(self, rgb, depth, eof, tau, b_print=False, print_path='', aux_in=None, use_aux=False):
        cond = self.film(tau).view(-1, 8, 32).permute(1, 0, 2)

        x = torch.cat([rgb, depth], dim=1)
        x = self.conv1(x, cond[0])
        x = self.conv2(x, cond[1])
        if b_print:
            plt.figure(3)
            for i in range(min(64, x.size(1))):
                plt.subplot(8,8,i+1)
                try:
                    plt.imshow(x[0,i].detach().cpu().numpy())
                except:
                    y = x[0,i].detach().cpu().numpy()
                    y = y - np.amin(y)
                    y = y / np.amax(y)
                    plt.imshow(y)
            plt.savefig(print_path+'activations2.png')
        x = self.conv3(x, cond[2])
        x = self.conv4(x, cond[3])
        if b_print:
            plt.figure(3)
            for i in range(min(64, x.size(1))):
                plt.subplot(8,8,i+1)
                try:
                    plt.imshow(x[0,i].detach().cpu().numpy())
                except:
                    y = x[0,i].detach().cpu().numpy()
                    y = y - np.amin(y)
                    y = y / np.amax(y)
                    plt.imshow(y)
            plt.savefig(print_path+'activations4.png')
        x = self.conv5(x, cond[4])
        x = self.conv6(x, cond[5])
        x = self.conv7(x, cond[6])
        x = self.conv8(x, cond[7])
        if b_print:
        #    print(tau[0])
        #    print(eof[0])
            plt.figure(1)
            plt.imshow(rgb[0].permute(1,2,0).detach().cpu().numpy())
            plt.savefig(print_path+'rgb.png')
            plt.figure(2)
            plt.imshow(depth[0,0].detach().cpu().numpy())
            plt.savefig(print_path+'depth.png')
            plt.figure(3)
            for i in range(min(64, x.size(1))):
                plt.subplot(8,8,i+1)
                try:
                    plt.imshow(x[0,i].detach().cpu().numpy())
                except:
                    y = x[0,i].detach().cpu().numpy()
                    y = y - np.amin(y)
                    y = y / np.amax(y)
                    plt.imshow(y)
            plt.savefig(print_path+'activations8.png')

        x2 = self.spatial_softmax(x)
        x = self.dropout(F.leaky_relu(self.conv_lin1(x.view(x.size(0), -1))))
        x = self.dropout(F.leaky_relu(self.conv_lin2(x)))

        if not self.use_tau:
            tau = self.augment_tau(tau.view(-1))
        x = torch.cat([x.view(x.size(0), -1), x2, tau], dim=1)

        aux = self.aux(x)
        if aux_in is not None:
            aux = aux_in
        if b_print:
            print(aux[0])
            plt.figure(4)
            plt.clf()
            plt.scatter(x2[0,::2].detach().cpu().numpy(), x2[0,1::2].detach().cpu().numpy() * -1)
            plt.xlim(-1, 1)
            plt.ylim(-1, 1)
            plt.savefig(print_path+'spatial_softmax.png')

        x = self.dropout(F.leaky_relu(self.fl1(x)))
        x = self.dropout(F.leaky_relu(self.fl2(torch.cat([aux.detach(), x, eof], dim=1))))
        x = self.output(x)

        return x, aux#, x2
