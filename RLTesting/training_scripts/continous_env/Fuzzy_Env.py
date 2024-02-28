import gymnasium as gym
import numpy as np


class EnvWrapper(gym.Env):
    def __init__(self):
        self.env = gym.make('Pendulum-v1', render_mode='human')
        # self.env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False, max_episode_steps = 20)
        # self.env = gym.make("CartPole-v1", max_episode_steps=200)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.rewarded_actions = {}
        self.state_action_pairs = []  # List to record all (state, action) tuples
        self.distance_bound = 0.1

    def render(self):
        return self.env.render()

    def state_similarity(self, current_state, ideal_state):
        distance = np.sqrt((current_state[0] - ideal_state[0]) ** 2 + (current_state[1] - ideal_state[1]) ** 2)
        # 相似度是距离的递减函数
        # 使用指数递减
        similarity = np.exp(-distance)
        return similarity

    def action_similarity(self, action, ideal_action):
        # 定义动作相似度的计算
        # 简单地使用差的绝对值
        similarity = max(1 - abs(action - ideal_action), 0)
        return similarity

    def calculate_distance(self, current_state, ideal_state):
        distance = np.sqrt((current_state[0] - ideal_state[0]) ** 2 + (current_state[1] - ideal_state[1]) ** 2)
        return distance

    def step(self, action):
        self.state_action_pairs.append((self.current_state, action))
        obs, reward, terminated, truncated, info = self.env.step(action)  # calls the gym env methods
        closest_state = min(self.rewarded_actions.keys(), key=lambda s: self.calculate_distance(self.current_state, s))
        distance = self.calculate_distance(self.current_state, closest_state)

        # 如果距离小于或距离阈值(1)，使用模糊逻辑调整奖励
        if distance < self.distance_bound:
            state_sim = np.exp(-distance)  # 用指数函数计算相似度
            action_sim = self.action_similarity(action, self.rewarded_actions[closest_state])
            # 使用状态相似度和动作相似度来计算奖励
            fuzzy_reward = state_sim * action_sim
            if fuzzy_reward > 0.8:
                reward = fuzzy_reward
            else:
                reward = -1

        # 其他奖励逻辑保持不变
        elif obs == 15:
            reward = 5
        elif terminated:
            reward = -3
        else:
            reward = -1

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


if __name__ == "__main__":
    env = EnvWrapper()
    env.set_rewarded_actions(rewarded_actions={0: 2, 1: 2, 2: 1})
    env.reset()
    env.render()
    observation, info = env.reset()
    for _ in range(20):

        action = env.action_space.sample()  # agent policy that uses the observation and info
        observation_new, reward, terminated, truncated, info = env.step(action)
        print(observation, action, reward)
        observation = observation_new

        if terminated or truncated:
            observation, info = env.reset()
    env.close()
