import os
import csv
import gym
import argparse
import imitate_gym
import numpy as np
from PIL import Image

# GOAL_00 = np.array([1.2,0.66,0.45])
# GOAL_01 = np.array([1.2,0.76,0.45])
# GOAL_02 = np.array([1.2,0.86,0.45])
# GOAL_10 = np.array([1.3,0.66,0.45])
# GOAL_11 = np.array([1.3,0.76,0.45])
# GOAL_12 = np.array([1.3,0.86,0.45])
# GOAL_20 = np.array([1.4,0.66,0.45])
# GOAL_21 = np.array([1.4,0.76,0.45])
# GOAL_22 = np.array([1.4,0.86,0.45])
# GOALS = np.array([GOAL_00,GOAL_01,GOAL_02,GOAL_10,GOAL_11,GOAL_12,GOAL_20,GOAL_21,GOAL_22])
# NAMES = {0:'goal_00',1:'goal_01',2:'goal_02',3:'goal_10',4:'goal_11',5:'goal_12',6:'goal_20',7:'goal_21',8:'goal_22'}
GOAL_00 = np.array([1.2,0.56,0.45])
GOAL_01 = np.array([1.2,0.66,0.45])
GOAL_02 = np.array([1.2,0.76,0.45])
GOAL_03 = np.array([1.2,0.86,0.45])
GOAL_04 = np.array([1.2,0.96,0.45])
GOAL_10 = np.array([1.3,0.56,0.45])
GOAL_11 = np.array([1.3,0.66,0.45])
GOAL_12 = np.array([1.3,0.76,0.45])
GOAL_13 = np.array([1.3,0.86,0.45])
GOAL_14 = np.array([1.3,0.96,0.45])
GOAL_20 = np.array([1.4,0.56,0.45])
GOAL_21 = np.array([1.4,0.66,0.45])
GOAL_22 = np.array([1.4,0.76,0.45])
GOAL_23 = np.array([1.4,0.86,0.45])
GOAL_24 = np.array([1.4,0.96,0.45])
GOALS = np.array([GOAL_00,
                  GOAL_01,
                  GOAL_02,
                  GOAL_03,
                  GOAL_04,
                  GOAL_10,
                  GOAL_11,
                  GOAL_12,
                  GOAL_13,
                  GOAL_14,
                  GOAL_20,
                  GOAL_21,
                  GOAL_22,
                  GOAL_23,
                  GOAL_24])
NAMES = {0:'goal_00',
         1:'goal_01',
         2:'goal_02',
         3:'goal_03',
         4:'goal_04',
         5:'goal_10',
         6:'goal_11',
         7:'goal_12',
         8:'goal_13',
         9:'goal_14',
         10:'goal_20',
         11:'goal_21',
         12:'goal_22',
         13:'goal_23',
         14:'goal_24'}
test=np.array([GOAL_24])

def calc_changes(start_pos, goal_pos, num_steps=50.):  
    return (goal_pos-start_pos)/num_steps

def next_move(curr_pos, goal_pos, slope):
    diffs = np.absolute(curr_pos - goal_pos)
    if diffs[0] <= 0.001:
        slope[0] = 0.
    if diffs[1] <= 0.001:
        slope[1] = 0.
    if diffs[2] <= 0.001:
        slope[2] = 0.
    return slope

def run(env, goal, trial_dir, viz=False):
    if viz == False:
        with open(trial_dir+'vectors.txt', 'w') as f:
            writer = csv.writer(f)
            curr_pos = None
            slope = None
            goal_reached = False
            counter = 0
            traj_counter = 0
            other = np.array([0.,0.,0.,0.,0.])
            while True:
                rgb = env.render('rgb_array')
                if curr_pos is None and goal_reached is False:
                    obs, _, _, _ = env.step([0.,0.,0.,0.,0.,0.,0.,0.])
                    curr_pos = obs['achieved_goal']
                    slope = calc_changes(curr_pos, goal)
                elif curr_pos is not None and goal_reached is False:
                    slope = next_move(curr_pos, goal, slope)
                    if sum(np.absolute(slope)) < 0.008 and curr_pos[2] < 0.455:
                        goal_reached = True
                    action = np.concatenate([slope,other])
                    obs, _, _, _ = env.step(action)
                    img_rgb = Image.fromarray(rgb, 'RGB')
                    img_rgb = img_rgb.resize((160,120))
                    img_rgb.save(trial_dir+'{}.png'.format(traj_counter))
                    arr = [traj_counter]
                    arr += [x for x in obs['observation']]
                    arr += [0]
                    curr_pos = obs['achieved_goal']
                    traj_counter += 1
                    writer.writerow(arr)
                elif curr_pos is not None and goal_reached is True:
                    # This is to simulate the terminal state where the user would stop near the end
                    counter += 1
                    obs, _, _, _ = env.step([0.,0.,0.,0.,0.,0.,0.,0.])
                    img_rgb = Image.fromarray(rgb, 'RGB')
                    img_rgb = img_rgb.resize((160,120))
                    img_rgb.save(trial_dir+'{}.png'.format(traj_counter))
                    arr = [traj_counter]
                    arr += [x for x in obs['observation']]
                    arr += [0]
                    traj_counter += 1
                    writer.writerow(arr)
                    if counter == 20:
                        break
    else:
        curr_pos = None
        slope = None
        goal_reached = False
        counter = 0
        other = np.array([0.,0.,0.,0.,0.])
        while True:
            env.render()
            if curr_pos is None and goal_reached is False:
                obs, _, _, _ = env.step([0.,0.,0.,0.,0.,0.,0.,0.])
                curr_pos = obs['achieved_goal']
                slope = calc_changes(curr_pos, goal)
            elif curr_pos is not None and goal_reached is False:
                slope = next_move(curr_pos, goal, slope)
                if sum(np.absolute(slope)) < 0.008 and curr_pos[2] < 0.455:
                    goal_reached = True
                action = np.concatenate([slope,other])
                obs, _, _, _ = env.step(action)
                print("==================")
                print(slope)
                print(obs['observation'][7:10])
                curr_pos = obs['achieved_goal']
                print(curr_pos)
            elif curr_pos is not None and goal_reached is True:
                # This is to simulate the terminal state where the user would stop near the end
                counter += 1
                obs, _, _, _ = env.step([0.,0.,0.,0.,0.,0.,0.,0.])
                print("==================")
                print(slope)
                print(obs['observation'][7:10])
                if counter == 20:
                    break




def main(task_name, num_trials=300, viz=False):
    if viz == False:
        if not os.path.exists('../../../datas/' + task_name + '/'):
            os.mkdir('../../../datas/' + task_name + '/')
    trial_counter = 0
    for trial in range(num_trials):
        env = gym.make('buttons-v0')
        for i, goal in enumerate(GOALS):
            save_folder = '../../../datas/'+task_name+'/'+NAMES[i]+'/'
            if viz == False:
                if not os.path.exists(save_folder):
                    os.mkdir(save_folder)
            trial_dir = save_folder+str(trial_counter)+'/'
            if viz == False:
                os.mkdir(trial_dir)
            env.reset()
            run(env, goal, trial_dir, viz=viz)
        trial_counter+=1

if __name__ == '__main__':
    main("buttons3x5", num_trials=1, viz=True)
