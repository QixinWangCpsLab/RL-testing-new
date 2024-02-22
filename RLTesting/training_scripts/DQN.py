import gymnasium as gym
# gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)
from stable_baselines3 import DQN
import numpy
from stable_baselines3.common.env_checker import check_env
import torch
from Env import EnvWrapper



def training_sript(max_steps = 100):

    env = EnvWrapper()
    env.reset()
    check_env(env)

    model = DQN("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=10)

    vec_env = model.get_env()
    obs = vec_env.reset()
    vec_env.render(mode='human')

    action_state_list = []

    for step in range(max_steps):
        action, _state = model.predict(obs, deterministic=True)
        # with open(log_path, 'a') as log_file:
        #     log_file.write(str(obs) + ',' + str(action))
        obs, reward, done, info = vec_env.step(action)
        action_state_list.append(str(obs) + ',' + str(action))
        if done:
            break
        else:
            vec_env.render(mode='human')

    return action_state_list


# training_sript()


if __name__ == '__main__':
    training_sript()