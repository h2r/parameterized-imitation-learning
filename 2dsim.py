#!/usr/bin/python3
import os
import sys
import csv
import time
import pygame
from pygame.locals import *
from PIL import Image
import numpy as np

task = sys.argv[1]
if not os.path.exists('datas/' + task + '/'):
	os.mkdir('datas/' + task + '/')
save_folder = None
writer = None
text_file = None

"""
Goal Positions: (200, 150), (400, 150), (600, 150)
                (200, 300), (400, 300), (600, 300)
                (200, 450), (400, 450), (600, 450)
"""
# Note that we have the goal positions listed above. The ones that are listed are the current ones that we are using
GOAL_X = 200
GOAL_Y = 150

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
last_gripper = None
prev_pos = None

while run:
	# Note that this is the data collection speed
	clock.tick(60)
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			run = False
		if event.type == pygame.KEYUP:
			# Note that since we are only recording when the mouse is moving, there is never a case when the velocity is 0
			# This may cause problems because the net cannot learn to stop when the data suggests that it never does. 
			# To simulate the fact that there will be a start and an end with no movement, we will save 5 instances at the beginning
			# and at the end
			if event.key == SPACEBAR_KEY: # recordings the recording
				recording = not recording
				vel = (0,0,0)
				if recording:
					folder = 'datas/'+task+'/'+str(time.time())+'/'
					os.mkdir(folder)
					save_folder = folder
					text_file = open(save_folder + 'vector.txt', 'w')
					writer = csv.writer(text_file)
					print("===Start Recording===")
					position = pygame.mouse.get_pos()
					buttons = pygame.mouse.get_pressed()
					gripper = buttons[0]
					for _ in range(5):
						now = time.time()
						pygame.image.save(screen, save_folder + str(idx) + "_rgb.png")
						depth = Image.fromarray(np.uint8(np.zeros((600,800))))
						depth.save(save_folder + str(idx) + "_depth.png")
						# Record data
						writer.writerow([idx, position[0], position[1], 0, 0, 0, 0, 0, vel[0], vel[1], vel[2], 0, 0, 0, gripper, now])
						idx += 1
						print(position, vel, gripper)
				if not recording:
					for _ in range(5):
						now = time.time()
						pygame.image.save(screen, save_folder + str(idx) + "_rgb.png")
						depth = Image.fromarray(np.uint8(np.zeros((600,800))))
						depth.save(save_folder + str(idx) + "_depth.png")
						# Record data
						writer.writerow([idx, last_pos[0], last_pos[1], 0, 0, 0, 0, 0, vel[0], vel[1], vel[2], 0, 0, 0, last_gripper, now])
						idx += 1
						print(last_pos, vel, last_gripper)
					print("---Stop Recording---")
					if text_file != None:
						text_file.close()
						text_file = None
					prev_pos = None
					save_counter = 0
					idx = 0
			if event.key == S_KEY: # sets the cursor postion near the relative start position
				print("Cursor set to position (760, 370)")
				pygame.mouse.set_pos([760, 370])
		if event.type == pygame.MOUSEMOTION: # This is the simulation of the arm. Note that left click simulates the gripper status
			if recording:
				if save_counter % 5 == 0:
					now = time.time()
					pygame.image.save(screen, save_folder+str(idx)+"_rgb.png")
					depth = Image.fromarray(np.uint8(np.zeros((600,800))))
					depth.save(save_folder + str(idx) + "_depth.png")
					position = event.pos
					print(idx)
					if idx == 5:
						vel = event.rel
					else:
						vel = np.array(position)-prev_pos
					gripper = event.buttons[0]
					writer.writerow([idx, position[0], position[1], 0, 0, 0, 0, 0, vel[0], vel[1], 0, 0, 0, 0, gripper, now])
					idx += 1
					prev_pos = np.array(position)
					last_pos = position
					last_gripper = gripper
					print(now, position, vel, gripper)
				save_counter += 1
	screen.fill((211,211,211))
	pygame.draw.rect(screen, (0,0,255), pygame.Rect(GOAL_X-RECT_X/2, GOAL_Y-RECT_Y/2, RECT_X, RECT_Y))
	pygame.draw.circle(screen, (255,0,0), pygame.mouse.get_pos(), 20, 1)
	pygame.display.update()

pygame.quit()