from train import train
from simulation.sim import sim, get_start
from util.parse_data import clean_kuka_data, parse_raw_data, create_lmdb
from src.model import Model
from sim_eval import get_tau, process_images

from torch import device as torch_device
from torch.cuda import is_available as use_cuda

from argparse import ArgumentParser
from os import makedirs
from shutil import rmtree
from random import sample
from itertools import combinations

# Should clean these up
import torch
import os
import sys
import csv
import time
import pygame
import argparse
from pygame.locals import *
from PIL import Image
import numpy as np
import itertools
import torch
import inspect


ALL_CASES = ['00', '01', '02',
             '10', '11', '12',
             '20', '21', '22']

descriptors = set()
def print_open_fds(print_all=False):
    global descriptors
    (frame, filename, line_number, function_name, lines, index) = inspect.getouterframes(inspect.currentframe())[1]
    fds = set(os.listdir('/proc/self/fd/'))
    new_fds = fds - descriptors
    closed_fds = descriptors - fds
    descriptors = fds

    if print_all:
        print("{}:{} ALL file descriptors: {}".format(filename, line_number, fds))

    if new_fds:
        print("{}:{} new file descriptors: {}".format(filename, line_number, new_fds))
    if closed_fds:
        print("{}:{} closed file descriptors: {}".format(filename, line_number, closed_fds))

class Config:
    pass


def arrange(num_but, max_arr):
    arrangements = list(combinations(ALL_CASES, num_but))
    return sample(arrangements, min(len(arrangements), max_arr))


def distance(a, b):
    dist = [np.abs(a[i] - b[i]) for i in range(len(a))]
    dist = [e**2 for e in dist]
    dist = sum(dist)
    return np.sqrt(dist)


def evaluate(config):
    checkpoint = torch.load(config.weights, map_location='cpu')
    model = Model(**checkpoint['kwargs'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    goals_x = [0, 1, 2]
    goals_y = [0, 1, 2]

    goal_pos = [[(200, 150), (200, 300), (200, 450)],
                [(400, 150), (400, 300), (400, 450)],
                [(600, 150), (600, 300), (600, 450)]]

    tau_opts = [[(0, 0), (0, 1), (0, 2)],
                [(1, 0), (1, 1), (1, 2)],
                [(2, 0), (2, 1), (2, 2)]]

    # These are magic numbers
    RECT_X = 60
    RECT_Y = 60
    ESCAPE_KEY = pygame.K_ESCAPE

    pygame.init()

    #Window
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("2D Simulation")
    pygame.mouse.set_visible(1)
    # Set the cursor
    curr_pos = get_start()

    #Background
    background = pygame.Surface(screen.get_size())
    background = background.convert()
    background.fill((211, 211, 211))
    screen.blit(background, (0, 0))
    pygame.display.flip()

    clock = pygame.time.Clock()
    rates = torch.zeros(3, 3)

    for r in range(3):
        for c in range(3):
            gx = c
            gy = r
            tau = get_tau(gx, gy, tau_opts)
            trials = torch.zeros(config.eval_traj)
            print('Evaluating button %d, %d' % (gx, gy))

            for i in range(config.eval_traj):
                # Set the cursor
                curr_pos = get_start()

                run = 1
                eof = None
                rgb = None
                depth = None

                while run:
                    clock.tick(config.framerate)
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            raise Exception('Game quit!!')
                        if event.type == pygame.KEYUP:
                            if event.key == ESCAPE_KEY:
                                raise Exception('You canceled it while it was evaluating!!')

                    vanilla_rgb_string = pygame.image.tostring(screen,"RGBA",False)
                    vanilla_rgb_pil = Image.frombytes("RGBA",(800,600),vanilla_rgb_string)
                    resized_rgb = vanilla_rgb_pil.resize((160,120))
                    rgb = np.array(resized_rgb)[:,:,:3]
                    rgb = process_images(rgb, True)
                    if torch.any(torch.isnan(rgb)):
                        rgb.zero_()

                    vanilla_depth = Image.fromarray(np.uint8(np.zeros((120,160))))
                    depth = process_images(vanilla_depth, False).zero_()

                    div = [200, 175] if config.normalize else [1, 1]
                    sub = [400, 325] if config.normalize else [0, 0]
                    norm_pos = [(curr_pos[0] - sub[0]) / div[0], (curr_pos[1] - sub[1]) / div[1]]
                    if eof is None:
                        eof = torch.FloatTensor([norm_pos[0], norm_pos[1], 0.0] * 5)
                    else:
                        eof = torch.cat([torch.FloatTensor([norm_pos[0], norm_pos[1], 0.0]), eof[0:12]])

                    # Calculate the trajectory
                    if config.use_tau:
                        in_tau = torch.FloatTensor(tau)
                        if config.color:
                            in_tau = 2 * in_tau / 255 - 1
                    else:
                        in_tau = torch.LongTensor([tau[0]*3 + tau[1]])

                    out, aux = model(rgb, depth, eof.view(1, -1), in_tau.view(1, -1))
                    out = out.squeeze()
                    delta_x = out[0].item()
                    delta_y = out[1].item()
                    new_pos = [curr_pos[0] + delta_x, curr_pos[1] + delta_y]

                    if (new_pos[0] < 0) or (new_pos[0] > 800) or (new_pos[1] < 0) or (new_pos[1] > 600)\
                    or (distance(new_pos, curr_pos) < config.stop_tolerance)\
                    or (run == config.max_iters): # Is it out of bounds / stopped / out of time
                        trials[i] = 1 if distance(new_pos, goal_pos[gx][gy]) < 55 else 0# Is it on the button
                        break

                    curr_pos = new_pos
                    run += 1

                    screen.fill((211,211,211))
                    for x, y in list(itertools.product(goals_x, goals_y)):
                        color = tau_opts[x,y] if config.color else (0, 0, 255)
                        pygame.draw.rect(screen, color, pygame.Rect(goal_pos[x][y][0]-RECT_X/2, goal_pos[x][y][1]-RECT_Y/2, RECT_X, RECT_Y))
                    pygame.draw.circle(screen, (0,0,0), [int(v) for v in curr_pos], 20, 0)
                    pygame.display.update()

            print('Successfully pressed the button %d out of %d times for an accuracy percentage of %d' % (torch.sum(trials), config.eval_traj, 100 * torch.sum(trials) / config.eval_traj))
            rates[r, c] = torch.sum(trials) / config.eval_traj

    torch.save(rates, config.weights[:config.weights.rfind('/')] + '/button_eval_percentages.pt')

    pygame.quit()
    return 0



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Input to data cleaner')
    parser.add_argument('-r', '--data_root', required=True, help='Path to cleaned un-compiled data')
    parser.add_argument('-d', '--data_file', required=True, help='path of temporary compiled dataset')
    parser.add_argument('-o', '--out_root', required=True, help='Path to folder containing outputs')
    parser.add_argument('-n', '--num_epochs', default=100, type=int, help='Number of epochs per train')
    parser.add_argument('-b', '--batch_size', default=64, type=int, help='Batch Size')
    parser.add_argument('-lr', '--learning_rate', default=0.0005, type=float, help='Learning Rate')
    parser.add_argument('-de', '--device', default="cuda:0", type=str, help='The cuda device')
    parser.add_argument('-m', '--max_arrangements', default=10, type=int, help='Number of arrangements per button count')
    parser.add_argument('-s', '--scale', default=1, type=float, help='Scaling factor for non-image data')
    parser.add_argument('-ib', '--ignore_bias', default=False, dest='ignore_bias', action='store_true', help='Flag to not include biases in layers')
    parser.add_argument('-l1', '--lambda_l1', default=1, type=float, help='l1 loss weight')
    parser.add_argument('-l2', '--lambda_l2', default=.01, type=float, help='l2 loss weight')
    parser.add_argument('-lc', '--lambda_c', default=.005, type=float, help='c loss weight')
    parser.add_argument('-la', '--lambda_aux', default=1, type=float, help='aux loss weight')
    parser.add_argument('-l_two', '--l2_norm', default=0.002, type=float, help='l2 norm constant')
    parser.add_argument('-opt', '--optimizer', default='adam', help='Optimizer, currently options are "adam" and "novograd"')
    parser.add_argument('-si', '--sim', default=False, dest='sim', action='store_true', help='Flag indicating data is from 2d sim')
    parser.add_argument('-zf', '--zero_eof', default=False, dest='zero_eof', action='store_true', help='Flag to only use current position in eof')
    parser.add_argument('-nt', '--num_traj', default=100, type=int, help='Number of trajectories per button.')
    parser.add_argument('-et', '--eval_traj', default=100, type=int, help='Number of trajectories per button in evaluation.')
    parser.add_argument('-f', '--framerate', default=1000, type=int, help='(Max) framerate of simulation.')
    parser.add_argument('-st', '--stop_tolerance', default=.001, type=float, help='Movement speed threshold to be considered stopped.')
    parser.add_argument('-mi', '--max_iters', default=100, type=int, help='Max eval iterations per trajectory.')
    args = parser.parse_args()

    config = Config

    # Train params
    config.root_dir         = args.data_root + '/'
    config.data_file        = args.data_file
    config.out_root         = args.out_root
    config.num_epochs       = args.num_epochs
    config.save_rate        = args.num_epochs + 2 # Never happens
    config.batch_size       = args.batch_size
    config.learning_rate    = args.learning_rate
    config.max_arrangements = args.max_arrangements
    config.scale            = args.scale
    config.use_bias         = not args.ignore_bias
    config.lambda_l1        = args.lambda_l1
    config.lambda_l2        = args.lambda_l2
    config.lambda_c         = args.lambda_c
    config.lambda_aux       = args.lambda_aux
    config.l2_norm          = args.l2_norm
    config.optimizer        = args.optimizer
    config.sim              = args.sim
    config.zero_eof         = args.zero_eof
    config.abstract_tau     = True

    if config.sim:
        config.eof_size = 15
        config.tau_size = 2
        config.aux_size = 2
        config.out_size = 7
    else:
        config.eof_size = 15
        config.tau_size = 3
        config.aux_size = 6
        config.out_size = 6


    if use_cuda():
        config.device = torch_device(args.device)
    else:
        config.device = torch_device('cpu')

    # Compilation params
    config.dest_dir     = config.data_file
    config.test_cases   = ()
    config.split_percen = .99
    config.simulation   = config.sim
    config.max_traj     = args.num_traj

    # Data gathering params
    config.color       = False
    config.normalize   = False
    config.num_traj    = args.num_traj
    config.framerate   = args.framerate
    config.save_folder = config.root_dir

    # Evaluation params
    config.eval_traj      = args.eval_traj
    config.stop_tolerance = args.stop_tolerance
    config.max_iters      = args.max_iters

    try:
        assert(os.path.exists(config.root_dir))
        for x, y in list(itertools.product([0, 1, 2], [0, 1, 2])):
            assert(os.path.exists(config.root_dir + str(x) + str(y)))
    except AssertionError:
        if config.sim:
            makedirs(config.root_dir)
            for x, y in list(itertools.product([0, 1, 2], [0, 1, 2])):
                if sim(x, y, str(x)+str(y), config):
                    raise Exception('You exited the data gathering!!')

            ("Cleaning Data...")
            clean_kuka_data(config.root_dir, ALL_CASES)

    makedirs(config.dest_dir)
    makedirs(config.out_root)

    h = config.dest_dir
    taus = ['onehot', 'tau']

    for num_buttons in range(8,9):
        for arrangement in arrange(num_buttons, config.max_arrangements):
            for use_tau in [0, 1]:
                print_open_fds(True)
                config.save_path   = config.out_root + '/' + taus[use_tau] + '/' + str(num_buttons) + '/' + str(arrangement)
                config.use_tau     = use_tau
                config.train_cases = arrangement
                config.weights     = config.save_path + '/best_checkpoint.tar'

                makedirs(config.save_path)

                ##
                '''
                config.dest_dir = h + '/' + str(arrangement)
                config.data_file = h + '/' + str(arrangement)
                makedirs(config.dest_dir)
                '''
                ##

                # Make dataset
                for mode in ['train', 'test']:
                    parse_raw_data(mode, config)
                    create_lmdb(mode, config)

                # Train on dataset
                train(config)

                # Delete dataset
                rmtree(config.dest_dir)
                makedirs(config.dest_dir)

                # eval
                if config.sim:
                    evaluate(config)
