from src.model import Model2
from src.loss_func import BehaviorCloneLoss
from src.datasets import ImitationLMDB

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import tqdm
import argparse

def train(data_file, save_path, num_epochs=1000, bs=64, lr=0.001, device='cuda:0', weight=None, is_aux=True, nfilm=1):
    modes = ['train', 'test']
    # Define model, dataset, dataloader, loss, optimizer
    model = Model2(is_aux=is_aux, nfilm=nfilm).to(device)
    if weight is not None:
        model.load_state_dict(torch.load(weight, map_location=device))
    criterion = BehaviorCloneLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    #model, optimizer = amp.initialize(model, optimizer, opt_level='O0')
    #model = nn.DataParallel(model)
    datasets = {mode: ImitationLMDB(data_file, mode) for mode in modes}
    dataloaders = {mode: DataLoader(datasets[mode], batch_size=bs, shuffle=True, num_workers=8, pin_memory=True) for mode in modes}
    data_sizes = {mode: len(datasets[mode]) for mode in modes}
    lowest_test_cost = float('inf')
    
    if weight is not None:
        cost_file = open(save_path+"/costs.txt", 'a')
    else:
        cost_file = open(save_path+"/costs.txt", 'w+')

    for epoch in tqdm.trange(1, num_epochs+1, desc='Epochs'):
        for mode in modes:
            running_loss = 0.0
            for data in tqdm.tqdm(dataloaders[mode], desc='{}:{}/{}'.format(mode, epoch, num_epochs),ascii=True):
                inputs = data[:-2]
                targets = data[-2:]
                curr_bs = inputs[0].shape[0]
                inputs = [x.to(device, non_blocking=True) for x in inputs]
                targets = [x.to(device, non_blocking=True) for x in targets]
                if mode == "train":
                    model.train()
                    optimizer.zero_grad()
                    out, aux_out = model(inputs[0], inputs[1], inputs[2], inputs[3])
                    loss = criterion(out, aux_out, targets[0], targets[1])
                    # with amp.scale_loss(loss, optimizer) as scaled_loss:
                    #     scaled_loss.backward()
                    loss.backward()
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

    device = None
    if torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        device = torch.device('cpu')

    train(args.data_file,
          args.save_path,
          num_epochs=args.num_epochs,
          bs=args.batch_size,
          lr=args.learning_rate,
          device=device,
          is_aux=args.aux,
          nfilm=args.nfilm)