from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

import argparse

def load_file(filename):
    data = pd.read_csv(filename, sep=',', header=None)
    data.columns = ["epoch", "loss_category", "loss"]

    categories = data['loss_category'].unique()
    
    losses = [(category, filter_loss(data, category)) for category in categories]

    return losses, filename[:-4]
    
    
def filter_loss(data, category):
    cat_inds = data['loss_category'].values == category
    
    losses = np.stack([data['epoch'], data['loss']])
    cat_losses = losses[(slice(None),) + np.where(cat_inds)]
    
    return cat_losses


def plot_file(losses, identifier=None):
    if identifier is None:
        label = 'Loss'
    else:
        label = 'Loss(%s)' % identifier
        
    for cat, loss in losses:
        plt.plot(loss[0], loss[1], label='%s %s'%(cat, label))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('loss_files',
                        help='The path(s) to the loss file(s) you want to plot.',
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
