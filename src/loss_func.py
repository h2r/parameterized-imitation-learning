import torch
import torch.nn as nn
from math import pi

class LossException(Exception):
    def __init__(self, *args, **kwargs):
        super(LossException, self).__init__(*args, **kwargs)


def fix_rot(goal_rot, cur_rot):
    true_rot = torch.abs(torch.remainder(goal_rot, 2) - torch.remainder(cur_rot, 2))
    mask = true_rot > 1
    true_rot[mask] = 2 - true_rot[mask]

    return true_rot

class BehaviorCloneLoss(nn.Module):
    """
    Behavior Clone Loss
    """
    def __init__(self, lamb_l2=0.01, lamb_l1=1.0, lamb_c=0.005, lamb_aux=0.0001, eps=1e-7):
        super(BehaviorCloneLoss, self).__init__()
        self.lamb_l2 = lamb_l2
        self.lamb_l1 = lamb_l1
        self.lamb_c = lamb_c
        self.lamb_aux = lamb_aux
        self.l2 = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.aux = nn.MSELoss()

        self.eps = eps

    def forward(self, out, aux_out, target, aux_target, flag=False):
        if torch.any(torch.isnan(out)):
            print(out)
            raise LossException('nan in model outputs!')

        '''
        x = out[:, 0:1] * torch.sin(out[:, 1:2]) * torch.cos(out[:, 2:3])
        y = out[:, 0:1] * torch.sin(out[:, 1:2]) * torch.sin(out[:, 2:3])
        z = out[:, 0:1] * torch.cos(out[:, 1:2])

        out = torch.cat([x, y, z, out[:, 3:]], dim=1)
        '''
        l2_loss = self.l2(out[:, :2], target[:, :2]) * 2 / 3 + self.l2(10 * out[:, 2], 10 * target[:, 2]) / 3
        l1_loss = self.l1(out, target)

        # For the arccos loss
        bs, n = out.shape
        num = torch.bmm(target.view(bs,1,n), out.view(bs,n,1)).squeeze()
        den = torch.norm(target,p=2,dim=1) * torch.norm(out,p=2,dim=1) + self.eps
        div = num / den
        a_cos = torch.acos(torch.clamp(div, -1 + self.eps, 1 - self.eps))
        c_loss = torch.mean(a_cos)
        # For the aux loss
        aux_loss = self.aux(aux_out, aux_target)

        weighted_loss = self.lamb_l2*l2_loss + self.lamb_l1*l1_loss + self.lamb_c*c_loss + self.lamb_aux*aux_loss

        if flag:#torch.isnan(weighted_loss):
            #print(out)
            print('===============')
            print('===============')
            #print(target)

            print(' ')
            print(' ')
            print(' ')

            print('weighted loss: %.2f' % weighted_loss)
            print('l2 loss: %.2f' % l2_loss)
            print('l1 loss: %.2f' % l1_loss)
            print('c loss: %.2f' % c_loss)
            print('aux loss: %.2f' % aux_loss)

            print(' ')

            print('L2 x: %.2f' % self.l2(out[:, 0], target[:, 0]))
            print('L2 y: %.2f' % self.l2(out[:, 1], target[:, 1]))
            print('L2 theta: %.2f' % self.l2(10 * out[:, 2], 10 * target[:, 2]))

            if torch.isnan(c_loss):
                print('num: %s' % str(num))
                print('den: %s' % str(den))
                print('div: %s' % str(div))
                print('acos: %s' % str(a_cos))

            #raise LossException('Loss is nan!')

        return weighted_loss
