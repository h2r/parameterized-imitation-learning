import torch
import torch.nn as nn

class BehaviorCloneLoss(nn.Module):
    """
    Behavior Clone Loss
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
        # For the aux loss
        aux_loss = self.aux(aux_out, aux_target)

        return self.lamb_l2*l2_loss + self.lamb_l1*l1_loss + self.lamb_c*c_loss + self.lamb_aux*aux_loss