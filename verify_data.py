from src.datasets import ImitationLMDB
from src.model import Model
from matplotlib import pyplot as plt
import torch
import argparse


def verify_data(config):
    model = None
    if config.weights is not None:
        checkpoint = torch.load(config.weights, map_location='cpu')
        model = Model(**checkpoint['kwargs'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

    dataset = ImitationLMDB(config.data_dir, config.mode)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=200, shuffle=config.shuffle)

    max_ = None
    min_ = None
    for data in dataloader:
        if max_ is not None:
            for i in range(len(data) - 2):
                data[2 + i] = torch.cat([data[2+i], max_[i], min_[i]], dim=0)
        min_ = [torch.min(d, dim=0).values.unsqueeze(0) for d in data[2:]]
        max_ = [torch.max(d, dim=0).values.unsqueeze(0) for d in data[2:]]

        print([m.squeeze() for m in min_])
        print([m.squeeze() for m in max_])
        print('============')
        print('============')

        '''
        rgb = data[0].squeeze().permute(1, 2, 0)
        print(rgb.min(), rgb.max())

        depth = data[1].squeeze()
        print(depth.min(), depth.max())

        print('EOF: %s' % data[2].squeeze())
        print('TAU: %s' % data[3].squeeze())

        print('Target: %s' % data[4].squeeze())
        print('Aux: %s' % data[5].squeeze())

        if model is not None:
            out, aux = model(data[0], data[1], data[2], data[3])
            print('Model out: %s' % out.squeeze())
            print('Model aux: %s' % aux.squeeze())

        if config.show:
            plt.imshow((rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6))
            plt.show()

            plt.imshow((depth - depth.min()) / (depth.max() - depth.min() + 1e-6))
            plt.show()

        print('==========================')
        '''





if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Qualitative data verification')
    parser.add_argument('-d', '--data_dir', required=True, help='Location of lmdb to pull data from.')
    parser.add_argument('-w', '--weights', help='Weights for model to evaluate on data.')
    parser.add_argument('-m', '--mode', default='test', help='Mode to evaluate data of.')
    parser.add_argument('-s', '--shuffle', default=False, dest='shuffle', action='store_true', help='Weights for model to evaluate on data.')
    parser.add_argument('-sh', '--show', default=False, dest='show', action='store_true', help='Flag to show visual inputs.')
    args = parser.parse_args()

    verify_data(args)
