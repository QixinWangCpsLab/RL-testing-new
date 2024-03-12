import gymnasium as gym
import numpy as np
import math
import random
from stable_baselines3 import SAC


def generate_states_actions(env, num_samples=12, min_distance=0.1):
    state_action_dict = {}
    states_sampled = 0

    while states_sampled < num_samples:
        # 随机采样一个状态
        state = env.observation_space.sample()

        # 检查新状态与已有状态的距离是否足够大
        too_close = any(abs(np.array(state[0]) - np.array(existing_state[0])) < min_distance
                        for existing_state in state_action_dict.keys())

        if not too_close:
            action = env.action_space.sample()

            # 保存状态和动作
            state_action_dict[tuple(state)] = action
            states_sampled += 1

    # 随机丢弃生成的script中的一些内容
    # keys = list(state_action_dict.keys())
    # random.shuffle(keys)  # 打乱键的顺序
    # keys_to_remove = keys[:len(keys) // 4]  # 准备取走四分之一的键

    # for key in keys_to_remove:
    #     del state_action_dict[key]  # 从字典中移除选中的键

    return state_action_dict


class EnvWrapper(gym.Env):
    def __init__(self):
        self.env = gym.make("MountainCarContinuous-v0", render_mode='rgb_array')
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.rewarded_actions = {-0.54: 1}
        self.current_state = -0.54
        self.state_action_pairs = []  # List to record all (state, action) tuples
        self.Done = False
        self.distance_bound = 0.05

    def render(self):
        return self.env.render()

    def state_similarity(self, current_state, ideal_state):
        distance = abs(current_state[0] - ideal_state[0])
        # 相似度是距离的递减函数
        # 使用指数递减
        # similarity = np.exp(-distance * 10)
        similarity = 0
        if distance < 0.5:
            similarity = 1
        else:
            similarity = 0
        return similarity

    def action_similarity(self, action, ideal_action):
        # 定义动作相似度的计算
        # 简单地使用差的绝对值
        # similarity = max(1 - abs(action - ideal_action), 0)
        similarity = 0
        if abs(action - ideal_action) < 0.3:
            similarity = 1
        elif abs(action - ideal_action) < 0.5:
            similarity = 0.7
        else:
            similarity = 0
        return similarity

    def calculate_distance(self, current_state, ideal_state):
        distance = abs(current_state[0] - ideal_state[0])
        return distance

    def step(self, action):
        self.state_action_pairs.append((self.current_state, action))
        obs, reward, terminated, truncated, info = self.env.step(action)  # calls the gym env methods
        closest_state = min(self.rewarded_actions.keys(), key=lambda s: self.calculate_distance(self.current_state, s))
        distance = self.calculate_distance(self.current_state, closest_state)

        # 如果距离小于或距离阈值0.1，使用模糊逻辑调整奖励
        if distance < self.distance_bound:
            state_sim = self.state_similarity(self.current_state, closest_state)  # 用指数函数计算相似度
            action_sim = self.action_similarity(action, self.rewarded_actions[closest_state])
            # 使用状态相似度和动作相似度来计算奖励
            fuzzy_reward = state_sim * action_sim
            # if fuzzy_reward > 0.7:
            reward = fuzzy_reward * 5
        #     else:
        #         reward = -0.5
        # else:
        #     reward = -1

        # # 其他奖励逻辑保持不变
        # elif obs[0] > 0.45:
        #     reward = 5
        # elif terminated:
        #     reward = -3
        # else:
        #     reward = -1

        self.current_state = obs

        if truncated or terminated:
            self.Done = True

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None):
        obs = self.env.reset(seed=seed)
        self.current_state = obs[0]
        return obs

    def set_rewarded_actions(self, rewarded_actions):
        self.rewarded_actions = rewarded_actions
        return

    def get_state_action_pairs(self):
        pairs = self.state_action_pairs
        self.state_action_pairs = []
        return pairs

    def get_current_state(self):
        return self.current_state

    @property
    def P(self):
        return self.env.P

    @property
    def desc(self):
        return self.env.desc


env = EnvWrapper()
env.set_rewarded_actions({(-0.4, 0.6): -0.9, (0.1, -0.6): 0.9})
print(env.reset())

# 创建一个 Soft Actor-Critic (SAC) 模型
model = SAC("MlpPolicy", env, verbose=1)

# # 训练模型
model.learn(total_timesteps=1000)


# state_action_dict = generate_states_actions(env=env)
#
# print(state_action_dict)
