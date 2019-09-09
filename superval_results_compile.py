import torch
from glob import glob
import argparse
from matplotlib import pyplot as plt
import numpy as np


def parse_path(path):
    buttons = path[path.rfind('(')+1:path.rfind(')')].split(',')
    buttons = [(int(button[-2]), int(button[-3])) for button in buttons if len(button) > 0]
    return buttons


def get_pcts(button_pcts, train_btns):
    train_inds = list(zip(*train_btns))
    train_pct = torch.mean(button_pcts[train_inds])
    total_pct = torch.mean(button_pcts)
    test_pct  = (9 * total_pct - len(train_btns) * train_pct) / (9 - len(train_btns))
    return torch.stack([train_pct, test_pct, total_pct])

def errorfill(x, y, yerr, color=None, alpha_fill=0.3, ax=None, plt_label=None):
    ax = ax if ax is not None else plt.gca()
    # if color is None:
    #     color = ax._get_lines.color_cycle.next()
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = y - yerr
        ymax = y + yerr
    elif len(yerr) == 2:
        ymin, ymax = yerr
    ax.plot(x, y, color=color, label=plt_label)
    ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill)


def plot(percentages):
    plt.figure(1)
#    plt.errorbar(range(1, 9), percentages[0, :-1, 1, 0], yerr=percentages[0, :-1, 1, 2], label='tau')
#    plt.errorbar(range(1, 9), percentages[1, :-1, 1, 0], yerr=percentages[1, :-1, 1, 2], label='onehot')
    tauerr = [percentages[0, :-1, 1, 5], percentages[0, :-1, 1, 6]]
    tauerr_avg = [percentages[0, :, 2, 5], percentages[0, :, 2, 6]]
    errorfill(range(1, 9), percentages[0, :-1, 1, 0], yerr=tauerr, color='red', plt_label='tau for unseen')
    errorfill(range(1, 10), percentages[0, :, 2, 0], yerr=tauerr_avg, color='orange', plt_label='tau average')
    errorfill(range(1, 9), percentages[1, :-1, 1, 0], yerr=percentages[1, :-1, 1, 2], color='blue', plt_label='onehot for unseen')
    errorfill(range(1, 10), percentages[1, :, 2, 0], yerr=percentages[1, :, 2, 2], color='green', plt_label='onehot average')
    plt.xlabel('Number of Buttons in Train Set')
    plt.ylabel('Successful Press Percentage')
    plt.title('Average Accuracy on Untrained Buttons')
    plt.legend()

    plt.figure(2)
    plt.plot(range(1, 9), percentages[0, :-1, 1, 0], label='tau')
    plt.plot(range(1, 9), percentages[1, :-1, 1, 0], label='onehot')
    plt.plot(range(1, 9), percentages[0, :-1, 1, 3], label='tau without outliers')
    plt.plot(range(1, 9), percentages[1, :-1, 1, 3], label='onehot without outliers')
    plt.xlabel('Number of Buttons in Train Set')
    plt.ylabel('Successful Press Percentage')
    plt.title('Average Accuracy on Untrained Buttons')
    plt.legend()

    plt.figure(3)
    plt.plot(range(1, 9), percentages[0, :-1, 1, 1], label='tau')
    plt.plot(range(1, 9), percentages[1, :-1, 1, 1], label='onehot')
    plt.plot(range(1, 9), percentages[0, :-1, 1, 4], label='tau without outliers')
    plt.plot(range(1, 9), percentages[1, :-1, 1, 4], label='onehot without outliers')
    plt.xlabel('Number of Buttons in Train Set')
    plt.ylabel('Successful Press Percentage')
    plt.title('Maximum Accuracy on Untrained Buttons')
    plt.legend()

    plt.figure(4)
    plt.errorbar(range(1, 10), percentages[0, :, 2, 0], yerr=percentages[0, :, 2, 2], label='tau')
    plt.errorbar(range(1, 10), percentages[1, :, 2, 0], yerr=percentages[1, :, 2, 2], label='onehot')
    plt.xlabel('Number of Buttons in Train Set')
    plt.ylabel('Successful Press Percentage')
    plt.title('Average Accuracy on All Buttons')
    plt.legend()

    plt.figure(5)
    plt.plot(range(1, 10), percentages[0, :, 2, 0], label='tau')
    plt.plot(range(1, 10), percentages[1, :, 2, 0], label='onehot')
    plt.plot(range(1, 10), percentages[0, :, 2, 3], label='tau without outliers')
    plt.plot(range(1, 10), percentages[1, :, 2, 3], label='onehot without outliers')
    plt.xlabel('Number of Buttons in Train Set')
    plt.ylabel('Successful Press Percentage')
    plt.title('Average Accuracy on All Buttons')
    plt.legend()

    plt.figure(6)
    plt.plot(range(1, 10), percentages[0, :, 2, 1], label='tau')
    plt.plot(range(1, 10), percentages[1, :, 2, 1], label='onehot')
    plt.plot(range(1, 10), percentages[0, :, 2, 4], label='tau without outliers')
    plt.plot(range(1, 10), percentages[1, :, 2, 4], label='onehot without outliers')
    plt.xlabel('Number of Buttons in Train Set')
    plt.ylabel('Successful Press Percentage')
    plt.title('Maximum Accuracy on All Buttons')
    plt.legend()

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root')
    parser.add_argument('versions', nargs='+')
    parser.add_argument('-e', '--error_range', default=.25, type=float)
    args = parser.parse_args()

    percentages = torch.zeros(len(args.versions), 9, 3, 7)
    for i_version, version in enumerate(args.versions):
        version_root = args.root + '/' + version
        for count in range(1, 10):
            count_root = version_root + '/' + str(count)
            sample_paths = glob(count_root + '/**/button_eval_percentages.pt', recursive=True)
            samples = [(torch.load(path), parse_path(path)) for path in sample_paths]
            sample_pcts = torch.stack([get_pcts(*sample) for sample in samples], dim=0)
            sorted_test = torch.stack([sample_pcts[:, i].sort()[0] for i in range(sample_pcts.size(1))], dim=1)
            error_low = int(sample_pcts.size(0) * args.error_range)
            error_high = int(sample_pcts.size(0) * (1 - args.error_range))
            percentages[i_version, count - 1, :] = torch.stack([torch.mean(sample_pcts, dim=0),
                                                                torch.max(sample_pcts, dim=0)[0],
                                                                torch.std(sample_pcts, dim=0),
                                                                torch.mean(sorted_test[error_low:error_high+1], dim=0),
                                                                torch.max(sorted_test[error_low:error_high+1], dim=0)[0],
                                                                sorted_test[error_low],
                                                                sorted_test[error_high],
                                                               ], dim=1)

    #print((percentages*100).int())
    print(percentages[1, :, 2, 0]*100)
    plot(percentages.numpy()*100)
