from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

import argparse

def load_file(filename):
    data = pd.read_csv(filename, sep=',', header=None)
    data.columns = ["epoch", "train/test", "loss"]

    train_inds = data['train/test'].values == 'train'

    losses = np.stack([data['epoch'], data['loss']])
    train_losses = losses[(slice(None),) + np.where(train_inds)]
    test_losses = losses[(slice(None),) + np.where(1 - train_inds)]

    return train_losses, test_losses, filename[:-4]


def plot_file(train_losses, test_losses, identifier=None):
    if identifier is None:
        label = 'Loss'
    else:
        label = 'Loss(%s)' % identifier
    plt.plot(train_losses[0], train_losses[1], label='%s %s'%('Train', label))
    plt.plot(test_losses[0], test_losses[1], label='%s %s'%('Test', label))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('loss_files',
                        help='The path to the loss file you want to plot.',
                        nargs='*')
    args = parser.parse_args()

    print('Loading data...')
    losses = [load_file(loss) for loss in args.loss_files]
    print('Data loaded.')

    for loss in losses:
        plot_file(*loss)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train/Test Loss')
    plt.legend()
    plt.show()
