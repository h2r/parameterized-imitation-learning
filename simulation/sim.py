#!/usr/bin/python3
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

def line(m, b):
    def inline(x):
        return m*x + b
    return inline

# note we want tau to be col, row == x, y
# This is like the get goal method
def get_tau(goal_x, goal_y, options):
    return options[goal_x][goal_y]

def get_start(win_y=600, win_x=800):
    horiz = False #np.random.uniform() < .5
    if horiz:
        min_x = 20
        max_x = win_x - min_x
        min_y = 15
        max_y = 30
    else:
        min_x = win_x - 40
        max_x = win_x - 20
        min_y = 20
        max_y = win_y - min_y
    x = round(np.random.uniform(min_x, max_x))
    y = round(np.random.uniform(min_y, max_y))
    return x, y

def get_next_move(curr_x, curr_y, goal_x, goal_y):
    linefunc = line(.1, 10)
    # Horizonal Movement
    if abs(curr_x - goal_x) <= .1:
        x = goal_x - curr_x
    else:
        x = (goal_x - curr_x)/linefunc(abs(curr_x - goal_x))
    '''
    if abs(curr_x - goal_x) <= 5:
        x = goal_x - curr_x
    elif abs(curr_x - goal_x) > 5 and abs(curr_x - goal_x) <= 50:
        x = (goal_x - curr_x)/10
    else:
        if curr_x > goal_x:
            x = (goal_x - curr_x)/100 + np.random.uniform(-3,1)
        else:
            x = (goal_x - curr_x)/100 + np.random.uniform(-1,3)
    '''

    # Vertical Movement
    if abs(curr_y - goal_y) <= .1:
        y = goal_y - curr_y
    else:
        y = (goal_y - curr_y)/linefunc(abs(curr_y - goal_y))

    return x, y

def sim(gx, gy, name, config):
    task = name
    task_path = config.save_folder + '/' + task + '/'
    os.makedirs(task_path, exist_ok=True)
    save_folder = None
    writer = None
    text_file = None

    goals_x = [0, 1, 2]
    goals_y = [0, 1, 2]

    goal_pos = [[(200, 150), (200, 300), (200, 450)],
                [(400, 150), (400, 300), (400, 450)],
                [(600, 150), (600, 300), (600, 450)]]

    tau_opts = [[(0.56, 0.22), (0.56, 0.15), (0.56, 0.08)],
                [(0.49, 0.22), (0.49, 0.15), (0.49, 0.08)],
                [(0.42, 0.22), (0.42, 0.15), (0.42, 0.08)]]

    # Note that we have the goal positions listed above. The ones that are listed are the current ones that we are using
    counter = 0

    # These are magic numbers
    RECT_X = 60
    RECT_Y = 60
    SPACEBAR_KEY = 32 # pygame logic
    S_KEY = 115

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

    save_counter = 0
    i_frame = 0
    prev_pos = None

    #tau_opts = np.random.randint(0, 255, (3,3,3)) if config.color else goal_pos
    tau = get_tau(gx, gy, tau_opts)

    for i_trajecory in range(config.num_traj):
        folder = task_path + str(i_trajecory) + '/'
        os.makedirs(folder, exist_ok=True)
        save_folder = folder
        with open(save_folder + 'vectors.txt', 'w') as text_file:
            writer = csv.writer(text_file)
            print("===Start Recording===")
            prev_pos = curr_pos
            while True:
                clock.tick(config.framerate)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return 1
                    if event.type == pygame.KEYUP:
                        if event.key == S_KEY: # sets the cursor postion near the relative start position
                            print("Cursor set to start position")
                            curr_pos = get_start()
                        if event.key == pygame.K_ESCAPE:
                            return 1

                if save_counter % 3 == 0:
                    pygame.image.save(screen, save_folder+str(i_frame)+"_rgb.png")
                    depth = Image.fromarray(np.uint8(np.zeros((600,800))))
                    depth.save(save_folder + str(i_frame) + "_depth.png")
                    vel = np.array(curr_pos)-np.array(prev_pos)
                    div = [200, 175] if config.normalize else [1, 1]
                    sub = [400, 325] if config.normalize else [0, 0]
                    save_tau = list(tau)
                    if config.color:
                        save_tau = [2*float(st)/255-1 for st in save_tau]
                    elif config.normalize:
                        save_tau = [(save_tau[0] - sub[0]) / div[0], (save_tau[1] - sub[1]) / div[1]]
                    norm_pos = [(curr_pos[0] - sub[0]) / div[0], (curr_pos[1] - sub[1]) / div[1]]
                    norm_goal = [(goal_pos[gx][gy][0] - sub[0]) / div[0], (goal_pos[gx][gy][1] - sub[1]) / div[1]]
                    writer.writerow([i_frame] + norm_pos + [0, 0, 0, 0, 0, vel[0], vel[1], 0, 0, 0, 0, 0] + save_tau + norm_goal)
                    i_frame += 1
                    prev_pos = curr_pos
                    print(i_frame, curr_pos, vel)
                save_counter += 1

                # Calculate the trajectory
                delta_x, delta_y = get_next_move(curr_pos[0], curr_pos[1], goal_pos[gx][gy][0], goal_pos[gx][gy][1])
                new_pos = [curr_pos[0] + delta_x, curr_pos[1] + delta_y]
                curr_pos = new_pos

                if (curr_pos[0] == goal_pos[gx][gy][0]) and (curr_pos[1] == goal_pos[gx][gy][1]):
                    vel = (0, 0, 0)
                    for _ in range(5):
                        pygame.image.save(screen, save_folder + str(i_frame) + "_rgb.png")
                        depth = Image.fromarray(np.uint8(np.zeros((600,800))))
                        depth.save(save_folder + str(i_frame) + "_depth.png")
                        # Record data
                        div = [200, 175] if config.normalize else [1, 1]
                        sub = [400, 325] if config.normalize else [0, 0]
                        save_tau = list(tau)
                        if config.color:
                            save_tau = [2*float(st)/255-1 for st in save_tau]
                        elif config.normalize:
                            save_tau = [(save_tau[0] - sub[0]) / div[0], (save_tau[1] - sub[1]) / div[1]]
                        norm_pos = [(curr_pos[0] - sub[0]) / div[0], (curr_pos[1] - sub[1]) / div[1]]
                        norm_goal = [(goal_pos[gx][gy][0] - sub[0]) / div[0], (goal_pos[gx][gy][1] - sub[1]) / div[1]]
                        writer.writerow([i_frame] + norm_pos + [0, 0, 0, 0, 0, vel[0], vel[1], 0, 0, 0, 0, 0] + save_tau + norm_goal)
                        i_frame += 1
                        print(prev_pos, vel)
                    print("---Stop Recording---")
                    if text_file != None:
                        text_file.close()
                        text_file = None
                    prev_pos = None
                    save_counter = 0
                    i_frame = 0
                    # Reset for Next
                    #tau_opts = np.random.randint(0, 255, (3,3,3)) if config.color else goal_pos
                    tau = get_tau(gx, gy, tau_opts)
                    curr_pos = get_start()
                    break

                screen.fill((211,211,211))
                for x, y in list(itertools.product(goals_x, goals_y)):
                    color = tau_opts[x,y] if config.color else (0, 0, 255)
                    pygame.draw.rect(screen, color, pygame.Rect(goal_pos[x][y][0]-RECT_X/2, goal_pos[x][y][1]-RECT_Y/2, RECT_X, RECT_Y))
                pygame.draw.circle(screen, (0,0,0), [int(v) for v in curr_pos], 20, 0)
                pygame.display.update()

    pygame.quit()
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Input to 2d simulation.')
    parser.add_argument('-c', '--color', dest='color', default=False, action='store_true', help='Used to activate color simulation.')
    parser.add_argument('-no', '--normalize', dest='normalize', default=False, action='store_true', help='Used to activate position normalization.')
    parser.add_argument('-n', '--num_traj', default=300, type=int, help='Number of trajectories per button.')
    parser.add_argument('-b', '--buttons', default=None, nargs='*', help='Buttons to simulate, formatted like "0,0" or "2,1". Default is all.')
    parser.add_argument('-f', '--framerate', default=300, type=int, help='Framerate of simulation.')
    parser.add_argument('-s', '--save_folder', default='datas', help='Path of the folder to save data to.')
    args = parser.parse_args()

    os.makedirs(args.save_folder)

    goals_x = [0, 1, 2]
    goals_y = [0, 1, 2]

    if args.buttons is None:
        args.buttons = list(itertools.product(goals_x, goals_y))
    else:
        args.buttons = [[int(v) for v in but.split(',')] for but in args.buttons]

    print(args.buttons)
    for x, y in args.buttons:
        if sim(x, y, str(x)+"_"+str(y), args):
            break
