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
from src.model import Model
import torch

# note we want tau to be col, row == x, y
# This is like the get goal method

def get_tau_():
    button = input('Please enter a button for the robot to try to press (e.g. "00", "12"): ')
    return [int(b) for b in button]


def get_tau(goal_x, goal_y, options):
    return options[goal_x][goal_y]

def get_start(win_y=600, win_x=800):
    min_x = win_x - 40
    max_x = win_x - 20
    min_y = 20
    max_y = win_y - min_y
    x = round(np.random.uniform(min_x, max_x))
    y = round(np.random.uniform(min_y, max_y))
    return x, y

def process_images(np_array_img, is_it_rgb):
    img = 2*((np_array_img - np.amin(np_array_img))/(np.amax(np_array_img)-np.amin(np_array_img))) - 1
    img = torch.from_numpy(img).type(torch.FloatTensor).squeeze()
    if(is_it_rgb):
        img = img.view(1,img.shape[2], img.shape[0], img.shape[1])
    else:
        img = img.view(1,1,img.shape[0], img.shape[1])
    return img

def sim(model, config):
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
    SPACEBAR_KEY = 32 # pygame logic
    S_KEY = pygame.K_s
    R_KEY = pygame.K_r
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

    run = True
    eof = None
    rgb = None
    depth = None

    gx, gy = get_tau_()#np.random.randint(0, 3, (2,))
    #tau_opts = np.random.randint(0, 255, (3,3,3)) if config.color else goal_pos
    tau = (gx, gy)#get_tau(gx, gy, tau_opts)
    a = 0
    while run:
        a += 1
        clock.tick(config.framerate)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                break
            if event.type == pygame.KEYUP:
                if event.key == S_KEY:
                    curr_pos = get_start()
                if event.key == R_KEY:
                    gx, gy = get_tau_()#np.random.randint(0, 3, (2,))
                    #tau_opts = np.random.randint(0, 255, (3,3,3)) if config.color else goal_pos
                    tau = (gx, gy)#get_tau(gx, gy, tau_opts)
                if event.key == ESCAPE_KEY:
                    run = False
                    break

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
        in_tau = torch.FloatTensor(tau)
        if config.color:
            in_tau = 2 * in_tau / 255 - 1
        out, aux = model(rgb, depth, eof.view(1, -1), in_tau.view(1, -1).to(eof))
        out = out.squeeze()
        delta_x = out[0].item()
        delta_y = out[1].item()
        new_pos = [curr_pos[0] + delta_x, curr_pos[1] + delta_y]
        print(eof)
        print(tau)
        print(aux)
        #print(get_tau(gx, gy, goal_pos))
        print(out)
        print(new_pos)
        print('========')
        curr_pos = new_pos

        screen.fill((211,211,211))
        for x, y in list(itertools.product(goals_x, goals_y)):
            color = tau_opts[x,y] if config.color else (0, 0, 255)
            pygame.draw.rect(screen, color, pygame.Rect(goal_pos[x][y][0]-RECT_X/2, goal_pos[x][y][1]-RECT_Y/2, RECT_X, RECT_Y))
        pygame.draw.circle(screen, (0,0,0), [int(v) for v in curr_pos], 20, 0)
        pygame.display.update()

        if a == 200:
            break
    pygame.quit()
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Input to 2d simulation.')
    parser.add_argument('-w', '--weights', required=True, help='The path to the weights to load.')
    parser.add_argument('-c', '--color', dest='color', default=False, action='store_true', help='Used to activate color simulation.')
    parser.add_argument('-no', '--normalize', dest='normalize', default=False, action='store_true', help='Used to activate position normalization.')
    parser.add_argument('-f', '--framerate', default=300, type=int, help='Framerate of simulation.')
    args = parser.parse_args()

    checkpoint = torch.load(args.weights, map_location='cpu')
    model = Model(**checkpoint['kwargs'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    sim(model, args)
