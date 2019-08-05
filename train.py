from src.model import Model
from src.loss_func import BehaviorCloneLoss, LossException
from src.datasets import ImitationLMDB

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import tqdm
import argparse
import os
import sys

def train(data_file, save_path, num_epochs=1000, bs=64, lr=0.001, device='cuda:0',
          weight=None, is_aux=True, nfilm=1, relu_first=True, use_bias=True,
          lamb_l2=0.01, lamb_l1=1.0, lamb_c=0.005, lamb_aux=0.0001, use_dummy=False,
          eof_size=15, tau_size=3, aux_size=6, out_size=7):
    modes = ['train', 'test']
    # Define model, dataset, dataloader, loss, optimizer
    model = Model(use_bias=use_bias, eof_size=eof_size, tau_size=tau_size, aux_size=aux_size, out_size=out_size).to(device)
    if weight is not None:
        model.load_state_dict(torch.load(weight, map_location=device))
    criterion = BehaviorCloneLoss(lamb_l2, lamb_l1, lamb_c, lamb_aux, use_dummy=use_dummy).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    #model = nn.DataParallel(model)
    lowest_test_cost = float('inf')

    if weight is not None:
        cost_file = open(save_path+"/costs.txt", 'a')
    else:
        cost_file = open(save_path+"/costs.txt", 'w+')

    gradients = torch.zeros((100, 2))

    for epoch in tqdm.trange(1, num_epochs+1, desc='Epochs'):
        datasets = {mode: ImitationLMDB(data_file, mode) for mode in modes}
        dataloaders = {mode: DataLoader(datasets[mode], batch_size=bs, shuffle=True, num_workers=8, pin_memory=True) for mode in modes}
        data_sizes = {mode: len(datasets[mode]) for mode in modes}
        for mode in modes:
            running_loss = 0.0
            for data in tqdm.tqdm(dataloaders[mode], desc='{}:{}/{}'.format(mode, epoch, num_epochs),ascii=True):
                inputs = data[:-2]
                targets = data[-2:]
                curr_bs = inputs[0].shape[0]
                inputs = [x.to(device, non_blocking=False) for x in inputs]
                targets = [x.to(device, non_blocking=False) for x in targets]

                # abstractify tau
                x = inputs[3][:,0]
                y = inputs[3][:,1]

                x[x < 240] = 0
                x[((240 < x) + (x < 500)) == 2] = 1
                x[500 < x] = 2

                y[y < 240] = 0
                y[((240 < y) + (y < 400)) == 2] = 1
                y[400 < y] = 2

                inputs[3][:,0] = x
                inputs[3][:,1] = y

                with torch.autograd.detect_anomaly():
                    if mode == "train":
                        model.train()
                        optimizer.zero_grad()
                        inputs[1] = torch.zeros(inputs[1].size()).to(inputs[1])

                        out, aux_out = model(inputs[0], inputs[1], inputs[2], inputs[3])
                        try:
                            loss = criterion(out, aux_out, targets[0], targets[1])
                        except LossException as le:
                            print('Last 100 gradient magnitudes:')
                            print(gradients)
                            raise le
                        loss.backward()

                        nanc = 0
                        # checking gradient magnitudes
                        grad_mags = torch.zeros((0,))
                        for param in model.parameters():
                            if param.grad is not None:
                                grad_mags = torch.cat([grad_mags.to(param.grad), torch.abs(param.grad).view(-1)])
                        mean_mags = torch.mean(grad_mags)
                        max_mags = torch.max(grad_mags)

                        gradients[1:] = gradients[:-1] # shift all entries right by one
                        gradients[0, 0] = mean_mags
                        gradients[0, 1] = max_mags


                        optimizer.step()
                        running_loss += (loss.item()*curr_bs)
                    elif mode == "test":
                        model.eval()
                        with torch.no_grad():
                            out, aux_out = model(inputs[0], inputs[1], inputs[2], inputs[3])
                            try:
                                loss = criterion(out, aux_out, targets[0], targets[1])
                            except LossException as le:
                                print('Last 100 gradient magnitudes:')
                                print(gradients)
                                raise le
                            running_loss += (loss.item()*curr_bs)
            cost = running_loss/data_sizes[mode]
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
            if epoch % 5 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': cost
                    }, save_path+"/checkpoint_"+str(epoch)+".tar")
            tqdm.tqdm.write("{} loss: {}".format(mode, cost))
    cost_file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Input to data cleaner')
    parser.add_argument('-d', '--data_file', required=True, help='Path to data.hdf5')
    parser.add_argument('-s', '--save_path', required=True, help='Path to save the model weights/checkpoints and results')
    parser.add_argument('-ne', '--num_epochs', required=False, default=1000, type=int, help='Number of epochs')
    parser.add_argument('-bs', '--batch_size', required=False, default=64, type=int, help='Batch Size')
    parser.add_argument('-lr', '--learning_rate', required=False, default=0.001, type=float, help='Learning Rate')
    parser.add_argument('-device', '--device', required=False, default="cuda:0", type=str, help='The cuda device')
    parser.add_argument('-aux', '--aux', required=False, default=True, type=bool, help='Whether or not to connect the auxiliary task')
    parser.add_argument('-nf', '--nfilm', required=False, default=0, type=int, help='Number of film layers')
    parser.add_argument('-rf', '--relu_first', required=False, default=True, type=bool, help='Film after relu')
    parser.add_argument('-ub', '--use_bias', required=False, default=False, type=bool, help='Include biases in layers')
    parser.add_argument('-l1', '--lambda_l1', required=False, default=1, type=float, help='l1 loss weight')
    parser.add_argument('-l2', '--lambda_l2', required=False, default=.01, type=float, help='l2 loss weight')
    parser.add_argument('-lc', '--lambda_c', required=False, default=.005, type=float, help='c loss weight')
    parser.add_argument('-la', '--lambda_aux', required=False, default=.0001, type=float, help='aux loss weight')
    parser.add_argument('-du', '--dummy', required=False, default=False, type=bool, help='Wether to use the loss dummy')
    parser.add_argument('-eofs', '--eof_size', required=False, default=15, type=int, help='EOF Size')
    parser.add_argument('-taus', '--tau_size', required=False, default=3, type=int, help='Tau Size')
    parser.add_argument('-auxs', '--aux_size', required=False, default=2, type=int, help='Aux Size')
    parser.add_argument('-outs', '--out_size', required=False, default=7, type=int, help='Out Size')
    args = parser.parse_args()

    device = None
    if torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        device = torch.device('cpu')

    os.makedirs(args.save_path, exist_ok=True)

    old_print = print
    def print2(*kargs, **kwargs):
        old_print(*kargs, **kwargs)
        sys.stdout.flush()
    print = print2

    train(args.data_file,
          args.save_path,
          num_epochs=args.num_epochs,
          bs=args.batch_size,
          lr=args.learning_rate,
          device=device,
          is_aux=args.aux,
          nfilm=args.nfilm,
          use_bias=args.use_bias,
          lamb_l1=args.lambda_l1,
          lamb_l2=args.lambda_l2,
          lamb_c=args.lambda_c,
          lamb_aux=args.lambda_aux,
          use_dummy=args.dummy,
          eof_size=args.eof_size,
          tau_size=args.tau_size,
          aux_size=args.aux_size,
          out_size=args.out_size)
