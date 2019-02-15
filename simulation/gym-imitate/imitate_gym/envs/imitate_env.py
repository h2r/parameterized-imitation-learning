import numpy as np
import gym
from gym import error, spaces, utils
from gym.envs.robotics import rotations, robot_env, utils

# Just to make sure that input shape is correct
def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)

class ImitateEnv(robot_env.RobotEnv):
    """
    Inheritance Chain: Env -> GoalEnv -> RobotEnv -> ImitateEnv
    
    Methods from RobotEnv
        seed(), step(), reset(), close(), render(), _reset_sim(),
        _get_viewer(): This method will be important for retrieving rgb and depth from the sim
    """
    def __init__(
        self, model_path, n_substeps, initial_qpos, reward_type, 
        distance_threshold, target, gripper_extra_height
    ):
        """
        Args: model_path (string): path to the environments XML file
              n_substeps (int): number of substeps the simulation runs on every call to step
              initial_qpos (dict): a dictionary of joint names and values that define the initial configuration
              reward_type ('sparse' or 'dense'): the reward type, i.e. sparse or dense
        self.target = target
        srray([-0.498, 0.005, -0.431 + self.gripper_extra_height]) + self.sim.data.get_site_xpos('robot0:grip')elf.target = target
              distance_threshold (float): the threshold after which a goal is considered achieved
        """
        # initial_qpos = 8 for [pos_x, pos_y, pos_z, rot_x, rot_y, rot_z, rot_w, gripper]

        self.reward_type = reward_type
        self.distance_threshold = distance_threshold
        self.target = target
        self.gripper_extra_height = gripper_extra_height

        super(ImitateEnv, self).__init__(model_path=model_path,
                                         n_substeps=n_substeps,
                                         n_actions=8,
                                         initial_qpos=initial_qpos)
    # GoalEnv Method
    # --------------------------------------------------
    def compute_reward(self, achieved_goal, goal, info):
        d = goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    # RobotEnv Methods
    # --------------------------------------------------
    def _step_callback(self):
        """
        In the step function in the RobotEnv it does: 1) self._set_action(action)
                                                      2) self.sim.step()
                                                      3) self._step_callback()
        This method can be used to provide additional constraints to the actions that happen every step.
        Could be used to fix the gripper to be a certain orientation for example
        """
        pass

    def _set_action(self, action):
        """
        Currently, I am just returning 1 number for the gripper control because there are 2 fingers in the
        robot that is used in the OpenAI gym library. If I start using movo, I would need 3 fingers so
        perhaps increase the gripper control from 2 to 3. 
        """
        assert action.shape == (8,)
        action = action.copy() # ensures that we don't change the action outside of this scope
        pos_ctrl, rot_ctrl, gripper_ctrl = action[:3], action[3:7], action[-1]
        
        # This is where I match the gripper control to number of fingers
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        
        # Create the action that we want to pass into the simulation
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])
        
        # Now apply the action to the simulation
        utils.ctrl_set_action(self.sim, action) # note self.sim is an inherited variable
        utils.mocap_set_action(self.sim, action)

    def _get_obs(self):
        """
        This is where I make sure to grab the observations for the position and the velocities. Potential
        area to grab the rgb and depth images.
        """
        ee = 'robot0:grip'
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        ee_pos = self.sim.data.get_site_xpos(ee)
        ee_quat = rotations.mat2quat(self.sim.data.get_site_xmat(ee))
        # Position and angular velocity respectively
        ee_velp = self.sim.data.get_site_xvelp(ee) * dt # to remove time dependency
        ee_velr = self.sim.data.get_site_xvelr(ee) * dt
        
        obs = np.concatenate([ee_pos, ee_quat, ee_velp, ee_velr])

        return {
            'observation': obs.copy(),
            'achieved_goal': ee_pos.copy(),
            'desired_goal': self.goal.copy()
        }	

    def _viewer_setup(self):
        """
        This could be used to reorient the starting view frame. Will have to mess around
        """
        body_id = self.sim.model.body_name2id('robot0:gripper_link')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 132.
        self.viewer.cam.elevation = -14.

    def _sample_goal(self):
        """
        Instead of sampling I am currently just setting our defined goal as the "sampled" goal
        """
        return self.target

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

    def _env_setup(self, initial_qpos):
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim)
        self.sim.forward()
        
        # Move end effector into position
        gripper_target = np.array([-0.498, 0.005, -0.431 + self.gripper_extra_height]) + self.sim.data.get_site_xpos('robot0:grip')
        gripper_rotation = np.array([1.,0.,1.,0.])
        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        for _ in range(10):
            self.sim.step()
         





