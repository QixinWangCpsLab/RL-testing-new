import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from training_scripts.Env import EnvWrapper
from stable_baselines3.common.logger import configure



def training_script(max_steps = 100):
    # 创建一个游戏环境，例如Frozen-lake
    env = EnvWrapper()
    rewarded_actions = {
        0: 2,
        1: 2,
        2: 1,
        6: 1,
        10: 1,
        14: 2
    }
    env.set_rewarded_actions(rewarded_actions)
    initial_state = env.reset()
    env.set_current_state(initial_state[0])

    # 初始化DQN模型
    model = DQN("MlpPolicy", env, verbose=1, batch_size=1)
    new_logger = configure(folder="logs", format_strings=["stdout", "log", "csv", "tensorboard"])
    model.set_logger(new_logger)

    vec_env = model.get_env()
    obs = vec_env.reset()
    vec_env.render(mode='rgb_array')

    action_state_list = []

    # 训练循环
    for step in range(max_steps):
        # 选择一个动作
        action, _states = model.predict(obs, deterministic=True)

        # 环境执行动作
        new_obs, reward, done, info = vec_env.step(action)

        action_state_list.append(str(obs) + ',' + str(action) + ',' + str(reward))

        # 存储新的转换到回放缓冲区
        model.replay_buffer.add(obs, new_obs, action, reward, done, info)

        # 检查回放缓冲区是否有足够的数据来进行学习
        if model.replay_buffer.size() > model.batch_size:
            # 执行一步学习
            model.train(gradient_steps=1)

        # 将新的观察结果设置为下一步的初始状态
        obs = new_obs

        # 检查是否结束
        if done:
            # 重置环境状态
            obs = env.reset()
            break

    # 保存模型
    model.save("dqn")

    # 关闭环境
    env.close()

    return action_state_list