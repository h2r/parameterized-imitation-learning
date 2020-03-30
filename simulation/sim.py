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
from matplotlib import pyplot as plt
from scipy.signal import convolve2d
from scipy.ndimage import zoom
from queue import PriorityQueue
from multiprocessing import Pool
import torch
from torch.nn import MaxPool2d

RECT_X = 90
RECT_Y = 30

goals_x = [0, 1, 2]
goals_y = [0, 1, 2]
# goals_x = [0]
# goals_y = [0]

goal_pos = [[(100, 50), (100, 200), (100, 350)],
            [(300, 50), (300, 200), (300, 350)],
            [(500, 50), (500, 200), (500, 350)]]

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
        min_x = win_x - 50
        max_x = win_x - 40
        min_y = 50
        max_y = win_y - min_y
    x = round(np.random.uniform(min_x, max_x))
    y = round(np.random.uniform(min_y, max_y))
    rot = np.random.randint(0, 360)
    return x, y, rot

def get_next_move(curr_x, curr_y, cur_rot, goal_x, goal_y, goal_rot):
    linefunc = line(.1, 10)
    # Horizonal Movement
    if abs(curr_x - goal_x) <= .25:
        x = goal_x - curr_x
    else:
        x = ((goal_x - curr_x)/linefunc(abs(curr_x - goal_x)) + np.random.randint(-1, 2)) * (np.random.rand() - .1)
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
    if abs(curr_y - goal_y) <= .25:
        y = goal_y - curr_y
    else:
        y = ((goal_y - curr_y)/linefunc(abs(curr_y - goal_y)) + np.random.randint(-1, 2)) * (np.random.rand() - .1)

    # rotational Movement
    true_rot = derot(goal_rot, cur_rot)
    if abs(true_rot) <= .25:
        rot = true_rot
    else:
        rot = ((true_rot)/(linefunc(abs(true_rot))) + np.random.randint(-1, 2)) * (np.random.rand() - .1) * .5


    return x, y, rot


def derot(goal_rot, cur_rot):
    if abs(goal_rot - cur_rot) > 180:
        true_rot = (goal_rot - cur_rot) / (abs(goal_rot - cur_rot) + 1e-6)
        true_rot = -1 * true_rot * (360 - abs(cur_rot - goal_rot))
    else:
        true_rot = goal_rot - cur_rot
    return true_rot


def overlap(ob1, ob2):
    return not (ob1[0]+ob1[2] < ob2[0] or ob1[0] > ob2[0]+ob2[2] or ob1[1] > ob2[1]+ob2[3] or ob1[1]+ob1[3] < ob2[1])


def get_obstacles(num, buttons_offset, player_pos):
    obstacles = []
    while len(obstacles) < num:
        width  = np.random.rand()*400
        height = np.random.rand()*300
        x = np.random.rand()*(600 - width)
        y = np.random.rand()*(600 - height)


        overlap_flag = False
        for tx, ty in list(itertools.product(goals_x, goals_y)):
            button = (goal_pos[tx][ty][0]-RECT_X/2 + buttons_offset[0], goal_pos[tx][ty][1]-RECT_Y/2 + buttons_offset[1], RECT_X, RECT_Y)
            if overlap((x, y, width, height), button):
                overlap_flag = True
                break
        if not overlap_flag:
            for obstacle in obstacles:
                if overlap((x, y, width, height), obstacle):
                    overlap_flag = True
                    break
        if not overlap_flag:
            if overlap((x, y, width, height), (player_pos[0]-20, player_pos[1]-20, 40, 40)):
                overlap_flag = True
        if not overlap_flag:
            obstacles += [(x, y, width, height)]

    return obstacles


def distance(a, b):
    return (a[0]-b[0])**2 + (a[1]-b[1])**2


def get_neighbors(point):
    neighbors = 9*[point]
    for i in range(9):
        dx = (i // 3) - 1
        dy = (i %  3) - 1
        n = neighbors[i]
        neighbors[i] = (min(max(n[0]+dx, 0), 79), min(max(n[1]+dy, 0), 59))
    return neighbors


def plan(_map, player_pos):
    if _map[player_pos[0], player_pos[1]] == 1:
        return (0,0)

    x = player_pos[0]
    y = player_pos[1]
    curmax = (_map[x, y], 0, 0)
    for dx in range(max(-1, -x-5), min(2, 155-x)):
        for dy in range(max(-1, -y-5), min(2, 115-y)):
            if _map[x+dx, y+dy] > curmax[0]:
                curmax = (_map[x+dx, y+dy], dx, dy)
            elif _map[x+dx, y+dy] == curmax[0]:
                if (dx**2+dy**2) < (curmax[1]**2+curmax[2]**2):
                    curmax = (_map[x+dx, y+dy], dx, dy)

    return curmax[1], curmax[2]


def build_map(screen, goal_pos):
    vanilla_rgb_string = pygame.image.tostring(screen,"RGBA",False)
    vanilla_rgb_pil = Image.frombytes("RGBA",(800,600),vanilla_rgb_string)
    plan_rgb = np.array(vanilla_rgb_pil)[:,:,:3]
    plan_map = plan_rgb[:, :, 0].astype(np.float32)*0
    plan_map[plan_rgb[:,:,0] == 255] = -1
    plan_map = np.transpose(plan_map)
    for i in range(28):
        plan_map = convolve2d(plan_map, np.ones((3, 3)), 'same')
    plan_map[plan_map < 0] = -1
    plan_map = zoom(plan_map, 1/5)
    plan_map[plan_map < -.1] = -1
    plan_map[plan_map > -1]  = 0
    plan_map[goal_pos[0], goal_pos[1]] = 1

    tmap = torch.from_numpy(plan_map).unsqueeze(0).unsqueeze(0)
    pooler = MaxPool2d(3, stride=1, padding=1)
    for i in range(160):
        pmap = pooler(tmap)*.99
        mmap = torch.max(torch.cat([tmap, pmap], dim=1), dim=1)[0].unsqueeze(1)
        mmap[tmap < 0] = -1
        tmap = mmap
    plan_map = tmap.squeeze().numpy()

    return plan_map


def sim(gx, gy, name, config):
    task = name
    task_path = config.save_folder + '/' + task + '/'
    os.makedirs(task_path, exist_ok=True)
    save_folder = None
    writer = None
    text_file = None

    tau_opts = [[(0.56, 0.22), (0.56, 0.15), (0.56, 0.08)],
                [(0.49, 0.22), (0.49, 0.15), (0.49, 0.08)],
                [(0.42, 0.22), (0.42, 0.15), (0.42, 0.08)]]

    div = [400, 300, 180] if config.normalize else [1, 1]
    sub = [400, 300, 180] if config.normalize else [0, 0]

    #tau_opts = goal_pos

    # Note that we have the goal positions listed above. The ones that are listed are the current ones that we are using
    counter = 0

    # These are magic numbers
    SPACEBAR_KEY = 32 # pygame logic
    S_KEY = 115

    pygame.init()

    #Window
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("2D Simulation")
    pygame.mouse.set_visible(1)
    # Set the cursor
    curr_pos = get_start()
    if args.rotation:
        curr_pos = (curr_pos[0], curr_pos[1], 90)

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

    for i_trajecory in range(config.num_traj):
        if config.color:
            tau_opts = np.random.randint(0, 255, (3,3,3))
        tau = get_tau(gx, gy, tau_opts)

        rect_rot = np.random.randint(180, 360, (9,))
        # rect_rot = np.random.randint(0, 360) * np.ones(9)
        if args.rotation:
            rect_rot = np.ones(9) * 90

        # This is the offset used to decide the button position
        x_offset = np.random.randint(0, 200)
        y_offset = np.random.randint(0, 200)
        #x_offset = 150
        #y_offset = 150


        obstacles = get_obstacles(6, (x_offset, y_offset), [int(v) for v in curr_pos])
        plan_map = None
        folder = task_path + str(i_trajecory) + '/'
        os.makedirs(folder, exist_ok=True)
        save_folder = folder
        with open(save_folder + 'vectors.txt', 'w') as text_file:
            writer = csv.writer(text_file)
            print("===Start Recording===")
            prev_pos = curr_pos
            while True:
                screen.fill((211,211,211))
                #for obstacle in obstacles:
                #    pygame.draw.rect(screen, (255, 0, 0), pygame.Rect(*obstacle))
                surf2 = pygame.Surface((RECT_Y-10, RECT_Y-10), pygame.SRCALPHA)
                surf2.fill((0, 255, 0))
                for x, y in list(itertools.product(goals_x, goals_y)):
                    color = tau_opts[x, y] if config.color else (0, 0, 255)
                    #surf = pygame.Surface()
                    surf = pygame.Surface((RECT_X, RECT_Y), pygame.SRCALPHA)
                    surf.fill(color)
                    surf.blit(surf2, (5, 5))
                    surf = pygame.transform.rotate(surf, rect_rot[3*x+y])
                    surf.convert()
                    screen.blit(surf, (goal_pos[x][y][0] + x_offset, goal_pos[x][y][1] + y_offset))
                    #pygame.draw.rect(screen, color, pygame.Rect(goal_pos[x][y][0]-RECT_X/2 + x_offset, goal_pos[x][y][1]-RECT_Y/2 + y_offset, RECT_X, RECT_Y))
                    #pygame.draw.rect(screen, (255, 0, 0), pygame.Rect(goal_pos[x][y][0]-RECT_X/4 + x_offset, goal_pos[x][y][1]-RECT_Y/4 + y_offset, RECT_X/2, RECT_Y/2))
                    #pygame.draw.rect(screen, (0, 255, 0), pygame.Rect(goal_pos[x][y][0]-RECT_X/8 + x_offset, goal_pos[x][y][1]-RECT_Y/8 + y_offset, RECT_X/4, RECT_Y/4))
                surf = pygame.Surface((RECT_X, RECT_Y), pygame.SRCALPHA)
                surf.fill((0,0,0))
                surf2.fill((255, 0, 0))
                surf.blit(surf2, (5, 5))
                surf = pygame.transform.rotate(surf, int(curr_pos[2]))
                surf.convert()
                screen.blit(surf, curr_pos[:2])
                #pygame.draw.circle(screen, (0,0,0), [int(v) for v in curr_pos[:2]], 20, 0)
                #pygame.display.update()

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
                    vel[2] = derot(curr_pos[2], prev_pos[2])
                    save_tau = list(tau)
                    if config.color:
                        save_tau = [2*float(st)/255-1 for st in save_tau]
                    norm_pos = [(curr_pos[i] - sub[i]) / div[i] for i in range(3)]
                    norm_pos = norm_pos[:2] + [np.sin(np.pi * norm_pos[2]), np.cos(np.pi * norm_pos[2])]
                    norm_goal = [(goal_pos[gx][gy][0] + x_offset - sub[0]) / div[0], (goal_pos[gx][gy][1] + y_offset - sub[1]) / div[1], (rect_rot[3*gx + gy] - sub[2]) / div[2]]
                    norm_goal = norm_goal[:2] + [np.sin(np.pi * norm_goal[2]), np.cos(np.pi * norm_goal[2])]
                    writer.writerow([i_frame] + norm_pos + [':'] + list(vel) + [':'] + save_tau + [':'] + norm_goal)
                    i_frame += 1
                    prev_pos = curr_pos
                    print(i_frame, curr_pos, vel)
                save_counter += 1

                # Calculate the trajectory
                if plan_map is None:
                    g_pos = [(goal_pos[gx][gy][0] + x_offset)//5, (goal_pos[gx][gy][1] + y_offset)//5]
                    plan_map = build_map(screen, g_pos)


                delta_x, delta_y, delta_rot = get_next_move(curr_pos[0], curr_pos[1], curr_pos[2], goal_pos[gx][gy][0] + x_offset, goal_pos[gx][gy][1] + y_offset, rect_rot[3*gx + gy])#plan(plan_map, (curr_pos[0]//5, curr_pos[1]//5))#
                if delta_x == 0 and delta_y == 0:
                    delta_x = np.clip(goal_pos[gx][gy][0] + x_offset - curr_pos[0], -3, 3)
                    delta_y = np.clip(goal_pos[gx][gy][1] + y_offset - curr_pos[1], -3, 3)
                # if args.rotation:
                #     delta_rot = 0
                new_pos = [curr_pos[0] + delta_x, curr_pos[1] + delta_y, (curr_pos[2] + delta_rot) % 360]

                if (curr_pos[0] == new_pos[0]) and (curr_pos[1] == new_pos[1]) and (curr_pos[2] == new_pos[2]):
                    curr_pos = new_pos
                    vel = (0, 0, 0)
                    if save_counter > 1:
                        for _ in range(5):
                            pygame.image.save(screen, save_folder + str(i_frame) + "_rgb.png")
                            depth = Image.fromarray(np.uint8(np.zeros((600,800))))
                            depth.save(save_folder + str(i_frame) + "_depth.png")
                            # Record data
                            save_tau = list(tau)
                            if config.color:
                                save_tau = [2*float(st)/255-1 for st in save_tau]
                            norm_pos = [(curr_pos[i] - sub[i]) / div[i] for i in range(3)]
                            norm_pos = norm_pos[:2] + [np.sin(np.pi * norm_pos[2]), np.cos(np.pi * norm_pos[2])]
                            norm_goal = [(goal_pos[gx][gy][0] + x_offset - sub[0]) / div[0], (goal_pos[gx][gy][1] + y_offset - sub[1]) / div[1], (rect_rot[3*gx + gy] - sub[2]) / div[2]]
                            norm_goal = norm_goal[:2] + [np.sin(np.pi * norm_goal[2]), np.cos(np.pi * norm_goal[2])]
                            writer.writerow([i_frame] + norm_pos + [':'] + list(vel) + [':'] + save_tau + [':'] + norm_goal)
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
                    rect_rot = np.random.randint(0,180)
                    tau = get_tau(gx, gy, tau_opts)
                    curr_pos = get_start()
                    if args.rotation:
                        rect_rot = np.ones(9) * 90
                        # curr_pos = (curr_pos[0], curr_pos[1], 90)
                    break

                curr_pos = new_pos


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
    parser.add_argument('-r', '--rotation', default=True, dest='rotation', action='store_false', help='Path of the folder to save data to.')
    args = parser.parse_args()

    os.makedirs(args.save_folder)

    goals_x = [0, 1, 2]
    goals_y = [0, 1, 2]
    # goals_x = [0]
    # goals_y = [0]

    if args.buttons is None:
        args.buttons = list(itertools.product(goals_x, goals_y))
    else:
        args.buttons = [[int(v) for v in but.split(',')] for but in args.buttons]

    print(args.buttons)
    for x, y in args.buttons:
        if sim(x, y, str(x)+"_"+str(y), args):
            break
