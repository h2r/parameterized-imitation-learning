import torch
from torch.nn import functional as F

def mellowmax2(self ,beta):
    c = np.max(self.Q[0])
    mm = c + np.log((1/self.a_size)*np.sum(np.exp(self.omega * (self.Q[0] - c ))))/self.omega
    b = 0
    for a in self.Q[0]:
        b+=np.exp(beta * (a-mm))*(a-mm)
    return b

def mellowmax(x, dim=0, beta=6, omega=3):
    c = torch.max(x, dim=dim, keepdim=True)[0]
    mm = c + torch.log((1/x.size(dim))*torch.sum(torch.exp(omega * (x - c)))) / omega
    return torch.exp(beta * (x - mm)) * (x - mm)
