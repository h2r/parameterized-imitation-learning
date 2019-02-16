import os
import numpy as np
from gym import utils
from imitate_gym.envs import imitate_env

#MODEL_XML_PATH = os.path.join('fetch', 'pick_and_place.xml')
HOME = '/home/jonathanchang/parameterized-imitation-learning/simulation/gym-imitate/imitate_gym/envs'
MODEL_XML_PATH = HOME+'/assets/buttons/pick_and_place.xml'
class ButtonsEnv(imitate_env.ImitateEnv, utils.EzPickle):
    def __init__(self, reward_type='dense'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0
        }
        target = np.array([1.25, 0.53, 0.4])
        imitate_env.ImitateEnv.__init__(
            self, model_path=MODEL_XML_PATH, n_substeps=25, initial_qpos=initial_qpos, reward_type=reward_type,
            distance_threshold=0.05, gripper_extra_height=0.2, target=target)
        utils.EzPickle.__init__(self)
