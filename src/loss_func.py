import torch
import torch.nn as nn

class LossException(Exception):
    def __init__(self, *args, **kwargs):
        super(LossException, self).__init__(*args, **kwargs)

class BehaviorCloneLoss(nn.Module):
    """
    Behavior Clone Loss
    """
    def __init__(self, lamb_l2=0.01, lamb_l1=1.0, lamb_c=0.005, lamb_aux=0.0001, eps=1e-6):
        super(BehaviorCloneLoss, self).__init__()
        self.lamb_l2 = lamb_l2
        self.lamb_l1 = lamb_l1
        self.lamb_c = lamb_c
        self.lamb_aux = lamb_aux
        self.l2 = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.aux = nn.MSELoss()

        self.eps = eps

    def forward(self, out, aux_out, target, aux_target):
        if torch.any(torch.isnan(out)):
            print(out)
            raise LossException('nan in model outputs!')

        l2_loss = self.l2(out, target)
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

        if torch.isnan(weighted_loss):
            print(out)
            print('===============')
            print('===============')
            print(target)

            print(' ')
            print(' ')
            print(' ')

            print('weighted loss: %.2f' % weighted_loss)
            print('l2 loss: %.2f' % l2_loss)
            print('l1 loss: %.2f' % l1_loss)
            print('c loss: %.2f' % c_loss)
            print('aux loss: %.2f' % aux_loss)

            if torch.isnan(c_loss):
                print('num: %s' % str(num))
                print('den: %s' % str(den))
                print('div: %s' % str(div))
                print('acos: %s' % str(a_cos))

            raise LossException('Loss is nan!')

        return weighted_loss
