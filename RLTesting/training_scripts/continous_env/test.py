import math

import gymnasium as gym
import numpy as np
import random
from stable_baselines3 import SAC


def generate_states_actions(env, num_samples=10, min_distance=0.1):
    state_action_dict = {}
    states_sampled = 0

    while states_sampled < num_samples:
        # 随机采样一个状态
        state = env.observation_space.sample()

        # 检查新状态与已有状态的距离是否足够大
        too_close = any(abs(np.array(state[0]) - np.array(existing_state[0])) < min_distance
                        for existing_state in state_action_dict.keys())

        if not too_close:
            # 使用策略网络或其他方法来选择动作
            # state_tensor = torch.from_numpy(state).float().unsqueeze(0)
            # action = policy_net(state_tensor).detach().numpy()[0]
            action = env.action_space.sample()

            standard_state = [state[0], math.sqrt(1-math.pow(state[0], 2)), state[2]]

            if random.random() < 0.5:
                standard_state[1] = -standard_state[1]

            # 保存状态和动作
            state_action_dict[tuple(standard_state)] = action
            states_sampled += 1

    # 随机丢弃生成的script中的一些内容
    keys = list(state_action_dict.keys())
    random.shuffle(keys)  # 打乱键的顺序
    keys_to_remove = keys[:len(keys) // 4]  # 准备取走四分之一的键

    for key in keys_to_remove:
        del state_action_dict[key]  # 从字典中移除选中的键

    return state_action_dict


# 假设你有一个训练好的策略网络
# policy_net = ...

# 创建环境
env = gym.make('Pendulum-v1', render_mode='human')


# 生成状态和动作
state_action_dict = generate_states_actions(env)

# result = 0
# for _ in state_action_dict.keys():
#     if math.pow(_[0], 2) + math.pow(_[1], 2) < 0.9:
#         result += 1
#
# print(result)

# # 创建环境
# env = gym.make('Pendulum-v1')

# 创建一个 Soft Actor-Critic (SAC) 模型
model = SAC("MlpPolicy", env, verbose=1)

# # 训练模型
# model.learn(total_timesteps=10000)
#
# # 保存模型
# model.save("sac_pendulum")

# 重新加载模型
model = SAC.load("sac_pendulum")

# 测试学习好的模型
obs = env.reset()[0]
for i in range(1000):
    action, _states = model.predict(obs)
    # print(env.step(action))
    obs, rewards, truncated, dones, info = env.step(action)
    env.render()
    if dones or truncated:
        obs = env.reset()[0]

env.close()


