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

# note we want tau to be col, row == x, y
# This is like the get goal method
def get_tau(goal_x, goal_y, rec_x=55, rec_y=55):
    min_x = goal_x + 3 - rec_x/2
    max_x = goal_x - 3 + rec_x/2
    min_y = goal_y + 3 - rec_y/2
    max_y = goal_y - 3 + rec_y/2
    x = round(np.random.uniform(min_x, max_x))
    y = round(np.random.uniform(min_y, max_y))
    return x, y

def get_start(win_y=600, win_x=800):
    min_x = win_x - 40
    max_x = win_x - 20
    min_y = 20
    max_y = win_y - min_y
    x = round(np.random.uniform(min_x, max_x))
    y = round(np.random.uniform(min_y, max_y))
    return x, y

def get_next_move(curr_x, curr_y, goal_x, goal_y):
    # Horizonal Movement
    if abs(curr_x - goal_x) <= 5:
        x = goal_x - curr_x
    elif abs(curr_x - goal_x) > 5 and abs(curr_x - goal_x) <= 50:
        x = round((goal_x - curr_x)/10)
    else:
        x = round((goal_x - curr_x)/80) + round(np.random.uniform(-3, 0))

    # Vertical Movement
    if abs(curr_y - goal_y) <= 5:
        y = goal_y - curr_y
    elif abs(curr_y - goal_y) > 5 and abs(curr_y - goal_y) <= 50:
        y = round((goal_y - curr_y)/10)
    else:
        if curr_y > goal_y:
            y = round((goal_y - curr_y)/100) + round(np.random.uniform(-3,1))
        else:
            y = round((goal_y - curr_y)/100) + round(np.random.uniform(-1,3))

    return x, y

def sim(gx, gy, name):
    task = name
    if not os.path.exists('datas2/' + task + '/'):
        os.mkdir('datas2/' + task + '/')
    save_folder = None
    writer = None
    text_file = None

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

    run = True
    recording = False
    save_counter = 0
    idx = 0
    last_pos = None
    prev_pos = None

    while run:
        # Note that this is the data collection speed
        #clock.tick(30)

        if not recording:
            recording = True
            folder = 'datas2/'+task+'/'+str(counter)+'/'
            os.mkdir(folder)
            save_folder = folder
            text_file = open(save_folder + 'vector.txt', 'w')
            writer = csv.writer(text_file)
            print("===Start Recording===")
            position = curr_pos
            prev_pos = curr_pos

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
                    curr_pos = get_start()
                if event.key == pygame.K_ESCAPE:
                    run = False
                    break
        if recording:
            if save_counter % 3 == 0:
                pygame.image.save(screen, save_folder+str(idx)+"_rgb.png")
                depth = Image.fromarray(np.uint8(np.zeros((600,800))))
                depth.save(save_folder + str(idx) + "_depth.png")
                position = curr_pos
                vel = np.array(position)-prev_pos
                writer.writerow([idx, position[0], position[1], 0, 0, 0, 0, 0, vel[0], vel[1], 0, 0, 0, 0, 0, gx, gy])
                idx += 1
                prev_pos = np.array(position)
                last_pos = position
                print(idx, position, vel)
            save_counter += 1

        # Calculate the trajectory
        if recording:
            delta_x, delta_y = get_next_move(curr_pos[0], curr_pos[1], gx, gy)
            delta_x = delta_x / 10
            delta_y = delta_y / 10
            new_pos = [curr_pos[0] + delta_x, curr_pos[1] + delta_y]
            curr_pos = new_pos

        if (np.absolute(curr_pos[0] - gx) < 1) and (np.absolute(curr_pos[1] - gy) < 1) and recording:
            recording = False
            vel = (0, 0, 0)
            for _ in range(5):
                pygame.image.save(screen, save_folder + str(idx) + "_rgb.png")
                depth = Image.fromarray(np.uint8(np.zeros((600,800))))
                depth.save(save_folder + str(idx) + "_depth.png")
                # Record data
                writer.writerow([idx, last_pos[0], last_pos[1], 0, 0, 0, 0, 0, vel[0], vel[1], vel[2], 0, 0, 0, 0, gx, gy])
                idx += 1
                print(last_pos, vel)
            print("---Stop Recording---")
            if text_file != None:
                text_file.close()
                text_file = None
            prev_pos = None
            save_counter = 0
            idx = 0
            # Reset for Next
            counter += 1
            gx, gy = get_tau(center_x, center_y)
            curr_pos = get_start()

        # Determines number of trials per button
        if counter == 300:
            break

        screen.fill((211,211,211))
        for x, y in list(itertools.product(goals_x, goals_y)):
            pygame.draw.rect(screen, (0,0,255), pygame.Rect(x-RECT_X/2, y-RECT_Y/2, RECT_X, RECT_Y))
        pygame.draw.circle(screen, (0,0,0), [int(v) for v in curr_pos], 20, 0)
        pygame.display.update()

    pygame.quit()

if __name__ == '__main__':
    goals_x = [175, 400, 625]
    goals_y = [125, 300, 475]
    counter = 0
    for x, y in [(175, 475), (625, 475), (625, 125)]:
        sim(x,y, str(x)+"_"+str(y))
