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
import matplotlib.pyplot as plt
import itertools
from src.model import Model
from src.loss_func import fix_rot
import torch
from simulation.sim import get_start, get_tau, RECT_X, RECT_Y, goals_x, goals_y, goal_pos

# note we want tau to be col, row == x, y
# This is like the get goal method
def get_tau_():
    button = input('Please enter a button for the robot to try to press (e.g. "00", "12"): ')
    return [int(b) for b in button]


def distance(a, b):
    dist = [np.abs(a[i] - b[i]) for i in range(len(a))]
    dist = [e**2 for e in dist]
    dist = sum(dist)
    return np.sqrt(dist)


def process_images(np_array_img, is_it_rgb):
    #try:
    img = 2*((np_array_img - np.amin(np_array_img))/(np.amax(np_array_img)-np.amin(np_array_img))) - 1
    #except:
    #    img = np.zeros_like(np_array_img)
    img = torch.from_numpy(img).type(torch.FloatTensor).squeeze()
    if(is_it_rgb):
        img = img.permute(2, 0, 1)
    else:
        img = img.view(1, img.shape[0], img.shape[1])
    return img.unsqueeze(0)

def sim(model, config):

    tau_opts = [[(0, 0), (0, 1), (0, 2)],
                [(1, 0), (1, 1), (1, 2)],
                [(2, 0), (2, 1), (2, 2)]]

    #tau_opts = [[(1,0), (2,0), (3,0)],  File "/home/nishanth/miniconda3/envs/py3_pytorch_cuda10/lib/python3.7/site-packages/torch/nn/modules/module.py", line 493, in __call__

    #          [(3,1), (1,1), (2,1)],
    #          [(2,2), (3,2), (1,2)]]


    # These are magic numbers
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
    tau = get_tau(gx, gy, tau_opts)#(gx, gy)
    if args.rotation:
        rect_rot = np.ones(9) * np.random.randint(0,360)
    else:
        rect_rot = np.ones(9) * np.random.randint(0,360)
        # rect_rot = np.ones(9) * 35
        # rect_rot = np.random.randint(0,360, (9,))

    x_offset = np.random.randint(0, 200)
    y_offset = np.random.randint(0, 200)

    # This is the values used for the only_rot experiments
    # x_offset = 150
    # y_offset = 150


    _gx, _gy = get_tau(gx, gy, goal_pos)
    _gx += x_offset
    _gy += y_offset
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
                    #curr_pos[2] = np.random.randint(0, 360)
                if event.key == R_KEY:
                    gx, gy = get_tau_()#np.random.randint(0, 3, (2,))
                    #tau_opts = np.random.randint(0, 255, (3,3,3)) if config.color else goal_pos
                    tau = (gx, gy)#get_tau(gx, gy, tau_opts)
                if event.key == ESCAPE_KEY:
                    run = False
                    break




        screen.fill((211,211,211))

        surf2 = pygame.Surface((RECT_Y-10, RECT_Y-10), pygame.SRCALPHA)
        surf2.fill((0, 255, 0))

        #for obstacle in obstacles:
        #    pygame.draw.rect(screen, (255, 0, 0), pygame.Rect(*obstacle))
        for x, y in list(itertools.product(goals_x, goals_y)):
            color = tau_opts[x, y] if config.color else (0, 0, 255)
            #surf = pygame.Surface((RECT_X, RECT_Y), pygame.SRCALPHA)
            #surf.fill(color)
            #surf = pygame.transform.rotate(surf, rect_rot[3*x+y])
            #surf.convert()
            #screen.blit(surf, (goal_pos[x][y][0] + x_offset, goal_pos[x][y][1] + y_offset))

            color = tau_opts[x, y] if config.color else (0, 0, 255)
            #surf = pygame.Surface()
            surf = pygame.Surface((RECT_X, RECT_Y), pygame.SRCALPHA)
            surf.fill(color)
            surf.blit(surf2, (5, 5))
            surf = pygame.transform.rotate(surf, rect_rot[3*x+y])
            print(rect_rot[3*x+y])
            surf.convert()
            screen.blit(surf, (goal_pos[x][y][0] + x_offset, goal_pos[x][y][1] + y_offset))

            # THIS IS THE OLD CODE FOR MULTICOLOR SQUARES
            # pygame.draw.rect(screen, color, pygame.Rect(goal_pos[x][y][0]-RECT_X/2 + x_offset, goal_pos[x][y][1]-RECT_Y/2 + y_offset, RECT_X, RECT_Y))
            # pygame.draw.rect(screen, (255, 0, 0), pygame.Rect(goal_pos[x][y][0]-RECT_X/4 + x_offset, goal_pos[x][y][1]-RECT_Y/4 + y_offset, RECT_X/2, RECT_Y/2))
            # pygame.draw.rect(screen, (0, 255, 0), pygame.Rect(goal_pos[x][y][0]-RECT_X/8 + x_offset, goal_pos[x][y][1]-RECT_Y/8 + y_offset, RECT_X/4, RECT_Y/4))


        #surf = pygame.Surface((RECT_X, RECT_Y), pygame.SRCALPHA)
        #surf.fill((0,0,0))
        #surf = pygame.transform.rotate(surf, int(curr_pos[2]))
        #surf = pygame.transform.rotate(surf, 90)
        #surf.convert()
        #screen.blit(surf, curr_pos[:2])

        surf = pygame.Surface((RECT_X, RECT_Y), pygame.SRCALPHA)
        surf.fill((0,0,0))
        surf2.fill((255, 0, 0))
        surf.blit(surf2, (5, 5))
        surf = pygame.transform.rotate(surf, int(curr_pos[2]))
        surf.convert()
        screen.blit(surf, curr_pos[:2])

        # OLD CIRCULAR AGENT
        # pygame.draw.circle(screen, (0,0,0), [int(v) for v in curr_pos[:2]], 20, 0)
        pygame.display.update()




        vanilla_rgb_string = pygame.image.tostring(screen,"RGBA",False)
        vanilla_rgb_pil = Image.frombytes("RGBA",(800,600),vanilla_rgb_string)
        resized_rgb = vanilla_rgb_pil.resize((160,120))
        rgb = np.array(resized_rgb)[:,:,:3]
        rgb = process_images(rgb, True)
        if torch.any(torch.isnan(rgb)):
            rgb.zero_()

        vanilla_depth = Image.fromarray(np.uint8(np.zeros((120,160))))
        depth = process_images(vanilla_depth, False).zero_()

        div = [400, 300, 180] if config.normalize else [1, 1, 1]
        sub = [400, 300, 180] if config.normalize else [0, 0, 0]
        norm_pos = [(curr_pos[i] - sub[i]) / div[i] for i in range(3)]
        norm_pos = norm_pos[:2] + [np.sin(np.pi * norm_pos[2]), np.cos(np.pi * norm_pos[2])]
        if eof is None:
            eof = torch.FloatTensor(norm_pos * 5)
        else:
            eof = torch.cat([torch.FloatTensor(norm_pos), eof[0:16] * 0])

        # Calculate the trajectory
        in_tau = torch.FloatTensor(tau)
        print_loc = 'eval_print/' + str(a)
        out = None
        aux = None
        with torch.no_grad():
            out, aux = model(rgb, depth, eof.view(1, -1), in_tau.view(1, -1).to(eof), b_print=config.print, print_path=print_loc)
            out = out.squeeze()
        delta_x = out[0].item()
        delta_y = out[1].item()
        delta_rot = out[2].item()

        # This block is a relic from when the network output sin and cos values
        # sin_cos = out[2:4]
        # mag = torch.sqrt(sin_cos[0]**2 + sin_cos[1]**2)
        # sin_cos = sin_cos / mag
        # delta_rot = (torch.atan2(sin_cos[0], sin_cos[1]).item() / 3.14159) * 180
        new_pos = [curr_pos[0] + delta_x, curr_pos[1] + delta_y, (curr_pos[2] + delta_rot) % 360]
        # sin_cos = aux.squeeze()[2:4]
        # mag = torch.sqrt(sin_cos[0]**2 + sin_cos[1]**2)
        # sin_cos = sin_cos / mag
        # aux_rot = (torch.atan2(sin_cos[0] , sin_cos[1]).view(-1, 1) / 3.14159) - 1

        print(eof.numpy())
        print(out.numpy())
        print([-1*np.sin(new_pos[2] / 180 * np.pi), -1*np.cos(new_pos[2] / 180 * np.pi)])
        print([-1*np.sin((rect_rot[3*gx+gy] / 180) * np.pi), -1*np.cos((rect_rot[3*gx+gy] / 180) * np.pi)])
        print((goal_pos[gx][gy][0] + x_offset - 400) / 400, (goal_pos[gx][gy][1] + y_offset - 300) / 300)
        print(aux.numpy())
        # print([-1*np.sin(rect_rot[3*gx+gy] / 180 * 3.14159), -1*np.cos(rect_rot[3*gx+gy] / 180 * 3.14159)])
        # print(fix_rot(torch.FloatTensor([rect_rot[3*gx+gy] / 180 - 1]).view(-1, 1), aux_rot))
        # print(fix_rot(torch.FloatTensor([rect_rot[3*gx+gy] / 180 - 1]).view(-1, 1), torch.FloatTensor([new_pos[2]/180 - 1]).view(-1, 1)))
        # print(fix_rot(torch.FloatTensor([new_pos[2]/180 - 1]).view(-1, 1), aux_rot))
        # print(eof)
        # print(tau)
        # print(aux)
        # print((_gx, _gy))
        # print(out)
        # print(new_pos)
        # print(distance(curr_pos, new_pos))
        #if (distance(curr_pos, new_pos)) < 1.5:
        #    time.sleep(5)
        print('========')
        curr_pos = new_pos

        #if a == 200:
        #    break
    pygame.quit()
    return 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Input to 2d simulation.')
    parser.add_argument('-w', '--weights', required=True, help='The path to the weights to load.')
    parser.add_argument('-c', '--color', dest='color', default=False, action='store_true', help='Used to activate color simulation.')
    parser.add_argument('-no', '--normalize', dest='normalize', default=False, action='store_true', help='Used to activate position normalization.')
    parser.add_argument('-f', '--framerate', default=300, type=int, help='Framerate of simulation.')
    parser.add_argument('-r', '--rotation', default=True, dest='rotation', action='store_false', help='Used to eval rotation.')
    parser.add_argument('-p', '--print', default=False, dest='print', action='store_true', help='Flag to print activations.')
    args = parser.parse_args()

    checkpoint = torch.load(args.weights, map_location='cpu')
    model = Model(**checkpoint['kwargs'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    sim(model, args)
