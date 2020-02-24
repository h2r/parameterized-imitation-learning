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

def errorfill(x, y, yerr, color=None, alpha_fill=0.3, ax=None, plt_label=None, linestyle='-'):
    ax = ax if ax is not None else plt.gca()
    # if color is None:
    #     color = ax._get_lines.color_cycle.next()
    if np.isscalar(yerr) or len(yerr) == len(y):
        ymin = y - yerr
        ymax = y + yerr
    elif len(yerr) == 2:
        ymin, ymax = yerr
    ax.plot(x, y, color=color, label=plt_label, linestyle=linestyle)
    ax.fill_between(x, ymax, ymin, color=color, alpha=alpha_fill)


def plot(percentages, versions):
    colors = ['red', 'blue', 'green', 'teal']
    linestyles = ['-', '--', '-.', ':']
    plt.style.use('ggplot')

    # MERL_percs = [11.0, 33.0, 100.0, 78.0, 100.0, 100.0, 100.0, 100.0, 100.0]
    # MERL_percs_errors = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    #
    # plt.figure(0)
    # errorfill(range(1, 10), MERL_percs, yerr=MERL_percs_errors, color='red', plt_label='tau')
    # plt.xlabel('Number of holes in training set')
    # plt.ylabel('Successful insertion percentage')
    # plt.title('Average Accuracy across all insertions')
    # plt.legend()

    plt.figure(1)
#    plt.errorbar(range(1, 9), percentages[0, :-1, 1, 0], yerr=percentages[0, :-1, 1, 2], label='tau')
#    plt.errorbar(range(1, 9), percentages[1, :-1, 1, 0], yerr=percentages[1, :-1, 1, 2], label='onehot')
    tauerr = [percentages[0, :-1, 1, 5], percentages[0, :-1, 1, 6]]
    tauerr_avg = [percentages[0, :, 2, 5], percentages[0, :, 2, 6]]
    #errorfill(range(1, 9), percentages[0, :-1, 1, 0], yerr=tauerr, color='red', plt_label='tau for unseen')
    for i, name in enumerate(versions):
        errorfill(range(1, 10), percentages[i, :, 2, 4], yerr=[percentages[i, :, 2, 5], percentages[i, :, 2, 6]], color=colors[i], linestyle=linestyles[i], plt_label=name)
    #errorfill(range(1, 10), percentages[1, :, 2, 0], yerr=percentages[1, :, 2, 2], color='green', plt_label='onehot')
    plt.xlabel('Number of goals in the training set')
    plt.ylabel('Success percentage')
    plt.title('Median Success Percentage in Simulation')
    plt.legend()
    '''
    plt.figure(2)
    plt.plot(range(1, 9), percentages[0, :-1, 1, 0], label='All Arrangements')
    #plt.plot(range(1, 9), percentages[1, :-1, 1, 0], label='onehot')
    plt.plot(range(1, 9), percentages[0, :-1, 1, 3], label='Median-centered 50%'+' of Arrangements')
    #plt.plot(range(1, 9), percentages[1, :-1, 1, 3], label='onehot without outliers')
    plt.xlabel('Number of Buttons in Train Set')
    plt.ylabel('Successful Press Percentage')
    plt.title('Average Accuracy on Untrained Buttons With Tau')
    plt.legend()

    plt.figure(3)
    plt.plot(range(1, 9), percentages[0, :-1, 1, 7], label='100th')
    #plt.bar(range(1, 9), percentages[0, :-1, 1, 6], label='75th')
    plt.plot(range(1, 9), percentages[0, :-1, 1, 5], label='50th')
    #plt.bar(range(1, 9), percentages[0, :-1, 1, 4], label='25th')
    plt.plot(range(1, 9), percentages[0, :-1, 1, 3], label='0th')
#    plt.plot(range(1, 9), percentages[0, :-1, 1, 4], label='tau without outliers')
#    plt.plot(range(1, 9), percentages[1, :-1, 1, 4], label='onehot without outliers')
    plt.xlabel('Number of Buttons in Train Set')
    plt.ylabel('Successful Press Percentage')
    plt.title('Accuracy on Untrained Buttons by Percentile')
    plt.legend()

    plt.figure(4)
    plt.errorbar(range(1, 10), percentages[0, :, 2, 0], yerr=percentages[0, :, 2, 2], label='tau')
    #plt.errorbar(range(1, 10), percentages[1, :, 2, 0], yerr=percentages[1, :, 2, 2], label='onehot')
    plt.xlabel('Number of Buttons in Train Set')
    plt.ylabel('Successful Press Percentage')
    plt.title('Average Accuracy on All Buttons')
    plt.legend()

    plt.figure(5)
    plt.plot(range(1, 10), percentages[0, :, 2, 0], label='tau')
    #plt.plot(range(1, 10), percentages[1, :, 2, 0], label='onehot')
    plt.plot(range(1, 10), percentages[0, :, 2, 3], label='tau without outliers')
#    plt.plot(range(1, 10), percentages[1, :, 2, 3], label='onehot without outliers')
    plt.xlabel('Number of Buttons in Train Set')
    plt.ylabel('Successful Press Percentage')
    plt.title('Average Accuracy on All Buttons')
    plt.legend()

    plt.figure(6)
    plt.plot(range(1, 10), percentages[0, :, 2, 1], label='tau')
    #plt.plot(range(1, 10), percentages[1, :, 2, 1], label='onehot')
    plt.plot(range(1, 10), percentages[0, :, 2, 4], label='tau without outliers')
    #plt.plot(range(1, 10), percentages[1, :, 2, 4], label='onehot without outliers')
    plt.xlabel('Number of Buttons in Train Set')
    plt.ylabel('Successful Press Percentage')
    plt.title('Maximum Accuracy on All Buttons')
    plt.legend()
    '''
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root')
    parser.add_argument('versions', nargs='+')
    parser.add_argument('-e', '--error_range', default=.25, type=float)
    args = parser.parse_args()

    percentages = torch.zeros(len(args.versions), 9, 3, 8)
    for i_version, version in enumerate(args.versions):
        version_root = args.root + '/' + version
        for count in range(1,4):
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
                                                                sorted_test[int(sample_pcts.size(0) * 0)],
                                                                sorted_test[int(sample_pcts.size(0) * .49)],
                                                                sorted_test[error_low],
                                                                sorted_test[error_high],
                                                                sorted_test[int(sample_pcts.size(0) * .99)],
                                                               ], dim=1)

    #print((percentages*100).int())
    #print(percentages[1, :, 2, 0]*100)
    plot(percentages.numpy()*100, args.versions)
