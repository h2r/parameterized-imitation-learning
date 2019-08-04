#!/usr/bin/python3
import os
import sys
import csv
import time
import pygame
from pygame.locals import *
from PIL import Image
import numpy as np
import itertools
from src.model import Model
import torch

# note we want tau to be col, row == x, y
# This is like the get goal method
def get_tau(goal_x, goal_y, rec_x=55, rec_y=55):
    min_x = goal_x + 3 - rec_x/2
    max_x = goal_x - 3 + rec_x/2
    min_y = goal_y + 3 - rec_y/2
    max_y = goal_y - 3 + rec_y/2
    x = torch.FloatTensor(1).uniform_(min_x, max_x).int().float()
    y = torch.FloatTensor(1).uniform_(min_y, max_y).int().float()
    tau = torch.cat([x,y])
    return tau

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
    img = torch.from_numpy(img).type(torch.FloatTensor)
    if(is_it_rgb):
        img = img.view(1,img.shape[2], img.shape[0], img.shape[1])
    else:
        img = img.view(1,1,img.shape[0], img.shape[1])
    return img


def sim(gx, gy):
    """
    Goal Positions: (200, 150), (400, 150), (600, 150)
                    (200, 300), (400, 300), (600, 300)
                    (200, 450), (400, 450), (600, 450)
    """
    # Note that we have the goal positions listed above. The ones that are listed are the current ones that we are using
    goals_x = [175, 400, 625]
    goals_y = [125, 300, 475]
    center_x = gx
    center_y = gy

    weights = "best_checkpoint.tar"

    model = Model(is_aux=True, nfilm = 0)
    checkpoint = torch.load(weights_loc, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

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

    run = True
    recording = False
    save_counter = 0
    idx = 0
    eof = None
    last_gripper = None

    while run:
        # Note that this is the data collection speed
        clock.tick(30)

        position = pygame.mouse.get_pos()
        prev_pos = curr_pos
        # The mouse press is the gripper
        buttons = pygame.mouse.get_pressed()
        gripper = buttons[0]

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                break
            if event.type == pygame.KEYUP:
                # Note that since we are only recording when the mouse is moving, there is never a case when the velocity is 0
                # This may cause problems because the net cannot learn to stop when the data suggests that it never does.
                # To simulate the fact that there will be a start and an end with no movement, we will save 5 instances at the beginning
                # and at the end
                if event.key == S_KEY: # sets the cursor postion near the relative start position
                    print("Cursor set to start position")
                    pygame.mouse.set_pos(get_start())
                if event.key == pygame.K_ESCAPE:
                    run = False
                    break

            if event.type == pygame.MOUSEMOTION: # This is the simulation of the arm. Note that left click simulates the gripper status
                vanilla_rgb_string = pygame.image.tostring(screen,"RGBA",False)
                vanilla_rgb_pil = Image.frombytes("RGBA",(600,800),vanilla_rgb_string)
                resized_rgb = vanilla_rgb_pil.resize((120,160))
                rgb = numpy.array(resized_rgb)
                rgb = preprocess_images(rgb, True)

                vanilla_depth = Image.fromarray(np.uint8(np.zeros((600,800))))
                depth = preprocess_images(vanilla_depth, True)

                position = event.pos
                if(eof == None):
                    eof = torch.FloatTensor([position[0], position[1], 0.0] * 5)
                else:
                    eof = eof[3:15].cat(torch.FloatTensor([position[0], position[1], 0.0]))

            # Calculate the trajectory
            tau = get_tau(gx, gy)

            x = tau[0]
            y = tau[1]

            x[x < 240] = 0
            x[((240 < x) + (x < 500)) == 2] = 1
            x[500 < x] = 2

            y[y < 240] = 0
            y[((240 < y) + (y < 400)) == 2] = 1
            y[400 < y] = 2

            tau = torch.cat([x, y])
            
            out, _ = model(rgb, depth, eof, tau)
            out = out.squeeze()
            delta_x = out[0].item()
            delta_y = out[1].item()
            new_pos = [curr_pos[0] + delta_x, curr_pos[1] + delta_y]
            pygame.mouse.set_pos(new_pos)
            curr_pos = new_pos

        screen.fill((211,211,211))
        for x, y in list(itertools.product(goals_x, goals_y)):
            pygame.draw.rect(screen, (0,0,255), pygame.Rect(x-RECT_X/2, y-RECT_Y/2, RECT_X, RECT_Y))
        pygame.draw.circle(screen, (0,0,0), pygame.mouse.get_pos(), 20, 0)
        pygame.display.update()

    pygame.quit()

if __name__ == '__main__':
    goals_x = [175, 400, 625]
    goals_y = [125, 300, 475]
    sim(goal_x[0], goal_y[2])
