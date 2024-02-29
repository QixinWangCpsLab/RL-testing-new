import time

import bug_lib as BL
import subprocess
import os
from datetime import date
import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys

sys.path.insert(0, '/')
import training_scripts
#
# import training_scripts.DQN_step_by_step as DQNS
from config_parser import parserConfig

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from training_scripts.Env import EnvWrapper
from stable_baselines3.common.logger import configure

import random

from get_and_train_models import *


def get_Frozen_lake_Env(rewarded_actions={0: 2, 1: 2, 2: 1, 6: 1, 10: 1, 14: 2}):
    # 创建一个游戏环境，例如Frozen-lake
    env = EnvWrapper()

    env.set_rewarded_actions(rewarded_actions)
    initial_state = env.reset()
    return env


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


def round_loop(config):
    BL.recover_project(config)
    BL.inject_bugs(config)

    # pip reinstall SB3 repository
    os.chdir("../..")
    os.system('pip install -e .')

    for round in range(config['rounds']):
        print("round: " + str(round) + "----")

        log_dir = os.path.join(config['root_dir'], 'RLTesting', '../logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_name = 'time_' + str(date.today()) + str(config['specified_bug_id']) + 'round_' + str(round)
        log_path = os.path.join(log_dir, log_name)
        with open(log_path, 'a') as log_file:
            log_file.write(str(config))
            log_file.write("\n-------------\n")

        # 每个round都需要重新随机生成env的rewarded_actions
        # rewarded_actions = {0: 2, 1: 2, 2: 1, 6: 1, 10: 1, 14: 2}
        env = get_Frozen_lake_Env()
        rewarded_actions = get_random_station_action_rewarder(env)
        env.set_rewarded_actions(rewarded_actions)
        with open(log_path, 'a') as log_file:
            log_file.write('rewarded_actions' + str(rewarded_actions))
            log_file.write("\n-------------\n")

        # 根据config中的'model_type'选择模型和训练函数
        if config['model_type'] == 'dqn':
            model_path = os.path.join('RLTesting', '../logs', 'dqn.zip')
            model = get_DQN_Model(env=env, model_path=model_path)
            train_func = train_DQN_model_new
        elif config['model_type'] == 'ppo':
            model_path = os.path.join('RLTesting', '../logs', 'ppo.zip')
            model = get_PPO_Model(env=env, model_path=model_path)
            train_func = train_PPO_model
        elif config['model_type'] == 'a2c':
            model_path = os.path.join('RLTesting', '../logs', 'a2c.zip')
            model = get_A2C_Model(env=env, model_path=model_path)
            train_func = train_A2C_model
        else:
            raise ValueError("Unknown model type in config: " + config['model_type'])

        for epoch in range(config['epoches']):
            # 使用选定的训练函数进行训练
            actions_in_epoch = train_func(model, model_path=model_path)
            with open(log_path, 'a') as log_file:
                log_file.write('epoch: ' + str(epoch) + '\n')
                log_file.write(str(actions_in_epoch))
                log_file.write("\n-------------\n")
            # time.sleep(0.2)

        os.remove(model_path)


def main(bug_version_list):
    config = parserConfig()

    for bug_version in bug_version_list:
        config['specified_bug_id'] = bug_version
        # print(bug_version, config['specified_bug_id'])
        if bug_version in [[7], ]:
            config['model_type'] = 'ppo'
        elif bug_version in [[8], ]:
            config['model_type'] = 'a2c'
        # 判断bug_version是否是[]
        elif not bug_version:
            print('bug free')
        else:
            config['model_type'] = 'dqn'
        round_loop(config)


# initialize bug_version_list
bug_version_list = [
    [],
    # [0],
    # [1],
    # [2],
    # [3],
    # [4],
    #
    # [6],
    # [7],
    # [8],
    # [9],
    # [10],
]

main(bug_version_list)
