import gymnasium as gym
import numpy as np
import random

def get_random_station_action_rewarder(env):
    state_action_dict = {}
    lake_size = int(env.observation_space.n ** 0.5)  # Assuming the lake is a square
    goal_state = env.observation_space.n - 1  # Assuming the goal state is the last one

    # 遍历所有状态
    for state in range(env.observation_space.n):
        safe_actions = []

        # 检查当前状态下的所有动作
        for action in range(env.action_space.n):
            # 获得执行动作后的潜在结果列表
            transitions = env.P[state][action]

            # 检查每个潜在结果，确保它不会导致掉入冰窟（H）或走出边界
            for transition in transitions:
                prob, next_state, reward, done = transition
                if prob == 1.0:
                    # 如果下一个状态是终点，则这个动作是安全的
                    if next_state == goal_state:
                        safe_actions.append(action)
                        break
                    row, col = divmod(next_state, lake_size)
                    # 检查是否会走出边界
                    if action == 0 and col == 0:  # 左动作，当前在最左列
                        continue
                    if action == 1 and row == (lake_size - 1):  # 下动作，当前在最下行
                        continue
                    if action == 2 and col == (lake_size - 1):  # 右动作，当前在最右列
                        continue
                    if action == 3 and row == 0:  # 上动作，当前在最上行
                        continue
                    # 检查下一个状态是否是洞（H）
                    if env.desc.reshape(-1)[next_state] != b'H':
                        safe_actions.append(action)
                        break  # 适用于确定性环境，无需检查其他transition

        # 如果有安全的动作，随机选择一个
        if safe_actions:
            action = random.choice(safe_actions)
            state_action_dict[state] = action

    # state15表示已经到达终点，不需要采取其他任何动作
    if 15 in state_action_dict:
        del state_action_dict[15]

    if 14 in state_action_dict:
        del state_action_dict[14]

    # 随机丢弃生成的script中的一些内容
    keys = list(state_action_dict.keys())
    random.shuffle(keys)  # 打乱键的顺序
    keys_to_remove = keys[:len(keys) // 4]  # 准备取走四分之一的键

    for key in keys_to_remove:
        del state_action_dict[key]  # 从字典中移除选中的键

    return state_action_dict


class EnvWrapper(gym.Env):
    def __init__(self):
        self.env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=False, max_episode_steps=200,
                            render_mode="rgb_array")
        # self.env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False, max_episode_steps = 20)
        # self.env = gym.make("CartPole-v1", max_episode_steps=200)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.rewarded_actions = {}
        self.current_state = 0
        self.state_action_pairs = []  # List to record all (state, action) tuples
        self.Done = False
        self.distance_bound = 1

    def render(self):
        return self.env.render()

    def state_similarity(self, current_state, ideal_state):
        # 定义状态相似度的计算
        # 假设状态相似度是基于状态索引的差异
        # 在FrozenLake环境中，可以假设每个状态在一个4x4的网格中，可以计算二维空间中的距离
        current_x, current_y = divmod(current_state, 4)
        ideal_x, ideal_y = divmod(ideal_state, 4)
        distance = np.sqrt((current_x - ideal_x) ** 2 + (current_y - ideal_y) ** 2)
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
        # 计算两个状态之间的距离
        current_x, current_y = divmod(current_state, 4)
        ideal_x, ideal_y = divmod(ideal_state, 4)
        distance = np.sqrt((current_x - ideal_x) ** 2 + (current_y - ideal_y) ** 2)
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
            if fuzzy_reward > 0.7:
                reward = fuzzy_reward * 5
            else:
                reward = -0.5

        # # 其他奖励逻辑保持不变
        # elif obs == 15:
        #     reward = 5
        # elif terminated:
        #     reward = -3
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
