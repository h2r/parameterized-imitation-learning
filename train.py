from src.model import Model
from src.loss_func import BehaviorCloneLoss, LossException
from src.datasets import ImitationLMDB

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import Optimizer
import math
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt

import tqdm
import argparse
import os
import sys

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

def train(config):
    modes = ['train', 'test']
    # Define model, dataset, dataloader, loss, optimizer
    kwargs = {'use_bias':config.use_bias, 'use_tau':config.use_tau, 'eof_size':config.eof_size, 'tau_size':config.tau_size, 'aux_size':config.aux_size, 'out_size':config.out_size}
    model = Model(**kwargs).to(config.device)
    try:
        os.makedirs(config.save_path)
    except:
        os.makedirs(config.save_path, exist_ok=True)
        checkpoint = torch.load(config.save_path+'/best_checkpoint.tar', map_location=config.device)
        model = Model(**checkpoint['kwargs']).to(config.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print('Using following args from loaded model:')
        print(checkpoint['kwargs'])
    criterion = BehaviorCloneLoss(config.lambda_l2, config.lambda_l1, config.lambda_c, config.lambda_aux).to(config.device)
    if config.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    else:
        optimizer = Novograd(model.parameters(), lr=config.learning_rate)
    #llr = LambdaLR(optimizer, lambda epoch: config.learning_rate ** (epoch + 10))
    l2_norm = config.l2_norm
    lowest_test_cost = float('inf')

    if False: # config.weight is not None:
        cost_file = 'a'
    else:
        cost_file = 'w+'

    with open(config.save_path+"/costs.txt", cost_file) as cost_file:
        gradients = torch.zeros((100, 2))

        datasets = {mode: ImitationLMDB(config.data_file, mode) for mode in modes}
        for epoch in tqdm.trange(1, config.num_epochs+1, desc='Epochs'):

            #if epoch == 1:
            #    print(datasets['train'][0][3])
            dataloaders = {mode: DataLoader(datasets[mode], batch_size=config.batch_size, shuffle=True, num_workers=8, pin_memory=True) for mode in modes}
            data_sizes = {mode: len(datasets[mode]) for mode in modes}
            for mode in modes:
                running_loss = 0.0
                for data in tqdm.tqdm(dataloaders[mode], desc='{}:{}/{}'.format(mode, epoch, config.num_epochs),ascii=True):
                    inputs = data[:-2]
                    targets = data[-2:]
                    curr_bs = inputs[0].shape[0]
                    inputs = [x.to(config.device, non_blocking=False) for x in inputs]
                    targets = [x.to(config.device, non_blocking=False) for x in targets]

                    if config.sim:
                        #targets[1][:, 0] = (targets[1][:, 0] / 400 - 1) * 3
                        #targets[1][:, 1] = (targets[1][:, 1] / 300 - 1) * 3
                        pass
                    else:
                        targets[0][:, 3:] = 0
                        targets[1][:, 3:] = 0


                    #targets += [torch.stack([targets[1][:,0]/400 - 1, targets[1][:,1]/300 - 1],dim=1)]

                    if config.zero_eof:
                        inputs[2][:, :-3] = 0   # No trajectory info from eof

                    if config.abstract_tau:
                        #'''
                        inputs[3][inputs[3][:, 0] < .455, 0] = 2
                        inputs[3][inputs[3][:, 0] < .525, 0] = 1
                        inputs[3][inputs[3][:, 0] < 1, 0] = 0
                        inputs[3][inputs[3][:, 1] < .115, 1] = 2
                        inputs[3][inputs[3][:, 1] < .185, 1] = 1
                        inputs[3][inputs[3][:, 1] < 1, 1] = 0
                        '''
                        inputs[3][inputs[3][:, 0] < .455, 0] = 30
                        inputs[3][inputs[3][:, 0] < .525, 0] = 49
                        inputs[3][inputs[3][:, 0] < 1, 0] = 73
                        inputs[3][inputs[3][:, 1] < .115, 1] = 62
                        inputs[3][inputs[3][:, 1] < .185, 1] = 95
                        inputs[3][inputs[3][:, 1] < 1, 1] = 127
                        '''
                    elif config.sim:
                        inputs[3] = targets[1][:,:2]

                    if not config.use_tau:
                        inputs[3] = inputs[3][:, 0].long()*3 + inputs[3][:, 1].long()

                    #print(inputs[3][0])

                    for input in inputs:
                        if torch.any(torch.isnan(input)):
                            input.zero_()

                    #rangex = 200*inputs[3][:,0].long()
                    #rangey = 150*inputs[3][:,1].long()
                    #inputs[1][:,rangex-20:rangex+20, rangey-20:rangey+20] = 1
                    aux_in = targets[1]
                    #aux_in[:,0] = aux_in[:,0] / 400 - 1
                    #aux_in[:,1] = aux_in[:,1] / 300 - 1

                    with torch.autograd.detect_anomaly():
                        if mode == "train":
                            model.train()
                            optimizer.zero_grad()

                            out, aux_out = model(inputs[0], inputs[1], inputs[2] * config.scale, inputs[3] * config.scale, False and (running_loss == 0), config.save_path+'/')
                            loss = criterion(out, aux_out, targets[0] * config.scale, targets[1] * config.scale)
                            if l2_norm != 0:
                                l2_crit = torch.nn.MSELoss(size_average=False)
                                l2_loss = 0
                                for param in model.parameters():
                                    l2_loss += l2_crit(param, torch.zeros_like(param))
                                loss += l2_norm * l2_loss
                                #loss += l2_norm * torch.sum(l2_crit(param) for param in model.parameters())

                            '''
                            if running_loss == 0:
                                rgb = inputs[0][0].cpu().squeeze().permute(1, 2, 0)
                                print(rgb.min(), rgb.max())

                                depth = inputs[1][0].cpu().squeeze()
                                print(depth.min(), depth.max())

                                print('EOF: %s' % inputs[2][0].squeeze())
                                print('TAU: %s' % inputs[3][0].squeeze())

                                print('Target: %s' % targets[0][0].squeeze())
                                print('Aux: %s' % targets[1][0].squeeze())

                                if model is not None:
                                    print('Model out: %s' % out[0].squeeze())
                                    print('Model aux: %s' % aux_out[0].squeeze())

                                print('==========================')
                            '''

                            loss.backward()
                            optimizer.step()
                            running_loss += loss.item()#*curr_bs)
                        elif mode == "test":
                            model.eval()
                            with torch.no_grad():
                                out, aux_out = model(inputs[0], inputs[1], inputs[2] * config.scale, inputs[3] * config.scale, aux_in=aux_in)
                                loss = criterion(out, aux_out, targets[0] * config.scale, targets[1] * config.scale)
                                running_loss += loss.item()#*curr_bs)
                cost = running_loss/data_sizes[mode]
                cost_file.write(str(epoch)+","+mode+","+str(cost)+"\n")
                if mode == 'test':
                    model.reset()
                    if lowest_test_cost >= cost:
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'kwargs': kwargs,
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': cost
                            }, config.save_path+"/best_checkpoint.tar")
                        lowest_test_cost = cost
                    if epoch % config.save_rate == 0:
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'kwargs': kwargs,
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss': cost
                            }, config.save_path+"/"+str(epoch)+"_checkpoint.tar")
                tqdm.tqdm.write("Epoch {} {} loss: {}".format(epoch, mode, cost))
                #llr.step()
        for mode in modes:
            datasets[mode].close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Input to data cleaner')
    parser.add_argument('-d', '--data_file', required=True, help='Path to data.lmdb')
    parser.add_argument('-s', '--save_path', required=True, help='Path to save the model weights/checkpoints and results')
    parser.add_argument('-ne', '--num_epochs', default=1000, type=int, help='Number of epochs')
    parser.add_argument('-bs', '--batch_size', default=64, type=int, help='Batch Size')
    parser.add_argument('-sc', '--scale', default=1, type=float, help='Scaling factor for non-image data')
    parser.add_argument('-lr', '--learning_rate', default=0.0005, type=float, help='Learning Rate')
    parser.add_argument('-device', '--device', default="cuda:0", type=str, help='The cuda device')
    parser.add_argument('-ub', '--use_bias', default=False, dest='use_bias', action='store_true', help='Flag to include biases in layers')
    parser.add_argument('-zf', '--zero_eof', default=False, dest='zero_eof', action='store_true', help='Flag to only use current position in eof')
    parser.add_argument('-l1', '--lambda_l1', default=1, type=float, help='l1 loss weight')
    parser.add_argument('-l2', '--lambda_l2', default=.01, type=float, help='l2 loss weight')
    parser.add_argument('-lc', '--lambda_c', default=.005, type=float, help='c loss weight')
    parser.add_argument('-la', '--lambda_aux', default=1, type=float, help='aux loss weight')
    parser.add_argument('-eofs', '--eof_size', default=15, type=int, help='EOF Size')
    parser.add_argument('-taus', '--tau_size', default=3, type=int, help='Tau Size')
    parser.add_argument('-auxs', '--aux_size', default=6, type=int, help='Aux Size')
    parser.add_argument('-outs', '--out_size', default=6, type=int, help='Out Size')
    parser.add_argument('-sr', '--save_rate', default=25, type=int, help='Epochs between checkpoints')
    parser.add_argument('-l_two', '--l2_norm', default=0, type=float, help='l2 norm constant')
    parser.add_argument('-opt', '--optimizer', default='adam', help='Optimizer, currently options are "adam" and "novograd"')
    parser.add_argument('-si', '--sim', default=False, dest='sim', action='store_true', help='Flag indicating data is from 2d sim')
    parser.add_argument('-u', '--use_tau', default=True, dest='use_tau', action='store_false', help='Flag indicating not to use tau')
    parser.add_argument('-at', '--abstract_tau', default=False, dest='abstract_tau', action='store_true', help='Flag indicating not to use tau')
    args = parser.parse_args()

    device = None
    if torch.cuda.is_available():
        args.device = torch.device(args.device)
    else:
        args.device = torch.device('cpu')

    old_print = print
    def print2(*kargs, **kwargs):
        old_print(*kargs, **kwargs)
        sys.stdout.flush()
    print = print2

    train(args)
