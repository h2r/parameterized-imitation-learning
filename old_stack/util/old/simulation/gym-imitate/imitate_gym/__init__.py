from gym.envs.registration import register

register(
    id='buttons-v0',
    entry_point='imitate_gym.envs:ButtonsEnv',
)
