import gym
import imitate_gym

env = gym.make('buttons-v0')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
