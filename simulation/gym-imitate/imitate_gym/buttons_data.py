import gym
import imitate_gym
import numpy as np

def calc_changes(start_pos, goal_pos, num_steps=20.):  
    return (goal_pos-start_pos)/num_steps

def next_move(curr_pos, goal_pos, slope):
    diffs = np.absolute(curr_pos - goal_pos)
    if diffs[0] <= 0.01:
        slope[0] = 0.
    if diffs[1] <= 0.01:
        slope[1] = 0.
    if diffs[2] <= 0.01:
        slope[2] = 0.
    return slope

env = gym.make('buttons-v0')
env.reset()
GOAL_00 = np.array([1.2,0.66,0.45])
GOAL_01 = np.array([1.2,0.76,0.45])
GOAL_02 = np.array([1.2,0.86,0.45])
GOAL_10 = np.array([1.3,0.66,0.45])
GOAL_11 = np.array([1.3,0.76,0.45])
GOAL_12 = np.array([1.3,0.86,0.45])
GOAL_20 = np.array([1.4,0.66,0.45])
GOAL_21 = np.array([1.4,0.76,0.45])
GOAL_22 = np.array([1.4,0.86,0.45])
GOALS = np.array([GOAL_00,GOAL_01,GOAL_02,GOAL_10,GOAL_11,GOAL_12,GOAL_20,GOAL_21,GOAL_22])
curr_pos = None
slope = None
for _ in range(10000):
    env.render()
    if curr_pos is None:
        obs, reward, done, info = env.step([0.,0.,-0.005,0.,0.,0.,0.,0.])
        curr_pos = obs['achieved_goal']
        slope = calc_changes(curr_pos, GOAL)
    else:
        other = np.array([0.,0.,0.,0.,0.])
        slope = next_move(curr_pos, GOAL_22, slope)
        action = np.concatenate([slope,other])
        obs, reward, done, info = env.step(action)
        curr_pos = obs['achieved_goal']

