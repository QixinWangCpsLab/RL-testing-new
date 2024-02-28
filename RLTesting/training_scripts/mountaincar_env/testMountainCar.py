import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy

# 创建环境
env_name = "MountainCarContinuous-v0"
env = gym.make(env_name, render_mode='human')

# 由于我们使用的是基于神经网络的模型，我们通常需要vectorized环境来加速训练
# DummyVecEnv是最简单的vectorized环境，它在单个进程中同步运行多个环境
vec_env = DummyVecEnv([lambda: env])

# 创建PPO模型
model = PPO("MlpPolicy", vec_env, verbose=1)

# 训练模型
model.learn(total_timesteps=100)

# # 评估模型
# mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
# print(f"Mean reward: {mean_reward} +/- {std_reward}")

# 测试训练好的模型
obs = env.reset()[0]
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, truncated, dones, info = env.step(action)
    env.render()
    if dones:
        obs = env.reset()[0]

env.close()