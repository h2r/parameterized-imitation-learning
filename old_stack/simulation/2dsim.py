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
def get_tau(goal_x, goal_y, colors):
    return colors[goal_x, goal_y]

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

def sim(gx, gy, name, goals_x, goals_y):
    task = name
    if not os.path.exists('datas/' + task + '/'):
        os.mkdir('datas/' + task + '/')
    save_folder = None
    writer = None
    text_file = None


    goal_pos = [[(200, 150), (400, 150), (600, 150)],
                [(200, 300), (400, 300), (600, 300)],
                [(200, 450), (400, 450), (600, 450)]]

    # Note that we have the goal positions listed above. The ones that are listed are the current ones that we are using
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

    colors = np.random.randint(0, 255, (3,3,3))
    tau = get_tau(gx, gy, colors)

    while run:
        # Note that this is the data collection speed
        #clock.tick(30)

        if not recording:
            recording = True
            folder = 'datas/'+task+'/'+str(counter)+'/'
            os.mkdir(folder)
            save_folder = folder
            text_file = open(save_folder + 'vector.txt', 'w')
            writer = csv.writer(text_file)
            print("===Start Recording===")
            prev_pos = curr_pos
            # The mouse press is the gripper
            buttons = pygame.mouse.get_pressed()

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
                vel = np.array(curr_pos)-prev_pos
                writer.writerow([idx, curr_pos[0], curr_pos[1], 0, 0, 0, 0, 0, vel[0], vel[1], 0, 0, 0, 0, 0, tau[0], tau[1], tau[2], goal_pos[gx][gy][0], goal_pos[gx][gy][1]])
                idx += 1
                prev_pos = np.array(curr_pos)
                last_pos = curr_pos
                print(idx, curr_pos, vel)
            save_counter += 1
        # Calculate the trajectory
        if recording:
            delta_x, delta_y = get_next_move(curr_pos[0], curr_pos[1], goal_pos[gx][gy][0], goal_pos[gx][gy][1])
            new_pos = [curr_pos[0] + delta_x, curr_pos[1] + delta_y]
            curr_pos = new_pos

        if (curr_pos[0] == goal_pos[gx][gy][0]) and (curr_pos[1] == goal_pos[gx][gy][1]) and recording:
            recording = False
            vel = (0, 0, 0)
            for _ in range(5):
                pygame.image.save(screen, save_folder + str(idx) + "_rgb.png")
                depth = Image.fromarray(np.uint8(np.zeros((600,800))))
                depth.save(save_folder + str(idx) + "_depth.png")
                # Record data
                writer.writerow([idx, last_pos[0], last_pos[1], 0, 0, 0, 0, 0, vel[0], vel[1], vel[2], 0, 0, 0, 0, tau[0], tau[1], tau[2], goal_pos[gx][gy][0], goal_pos[gx][gy][1]])
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
            colors = np.random.randint(0, 255, (3,3,3))
            tau = get_tau(gx, gy, colors)
            curr_pos = get_start()

        # Determines number of trials per button
        if counter == 300:
            break

        screen.fill((211,211,211))
        for x, y in list(itertools.product(goals_x, goals_y)):
            pygame.draw.rect(screen, colors[x,y], pygame.Rect(goal_pos[x][y][0]-RECT_X/2, goal_pos[x][y][1]-RECT_Y/2, RECT_X, RECT_Y))
        pygame.draw.circle(screen, (0,0,0), [int(v) for v in curr_pos], 20, 0)
        pygame.display.update()

    pygame.quit()

if __name__ == '__main__':
    goals_x = [0, 1, 2]
    goals_y = [0, 1, 2]
    counter = 0
    for x, y in [[0, 2], [2, 2], [2, 0]]: # list(itertools.product(goals_x, goals_y)):
        if counter > 1:
            sim(x,y, str(x)+"_"+str(y), goals_x, goals_y)
        counter += 1
