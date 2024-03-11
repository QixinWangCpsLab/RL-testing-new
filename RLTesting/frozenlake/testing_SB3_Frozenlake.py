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

sys.path.insert(0, './training_scripts/')
#
# import training_scripts.DQN_step_by_step as DQNS
from config_parser import parserConfig

import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from frozenlake.Frozenlake_Env import *
from stable_baselines3.common.logger import configure

import random

from get_and_train_models import *


def round_loop(config, rounds=25, epochs=300, bug_list=[], model_type='dqn'):
    # BL.recover_project(config)
    # BL.inject_bugs(config=config, bug_id_list=bug_list)

    # # pip reinstall SB3 repository
    # os.chdir(config['root_dir'])
    # os.system('pip install -e .')

    print(model_type)
    for round in range(rounds):
        print("round: " + str(round) + "----")

        log_dir = os.path.join(config['root_dir'], 'RLTesting', 'logs', 'Frozenlake', model_type, str(bug_list))
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_name = 'time_' + str(date.today()) + str(bug_list) + 'round_' + str(round)
        log_path = os.path.join(log_dir, log_name)
        with open(log_path, 'a') as log_file:
            log_file.write(str(config))
            log_file.write("\n-------------\n")

        # 每个round都需要重新随机生成env的rewarded_actions
        # rewarded_actions = {0: 2, 1: 2, 2: 1, 6: 1, 10: 1, 14: 2}
        env = EnvWrapper()
        env.reset()
        rewarded_actions = get_random_station_action_rewarder(env)
        env.set_rewarded_actions(rewarded_actions)
        with open(log_path, 'a') as log_file:
            log_file.write('rewarded_actions' + str(rewarded_actions))
            log_file.write("\n-------------\n")

        model_path=os.path.join('logs', 'model.zip')
        model = None
        # 根据config中的'model_type'选择模型和训练函数
        if model_type == 'dqn':
            model_path = os.path.join(config['root_dir'], 'RLTesting', 'logs', 'Frozenlake', model_type, 'dqn.zip')
            model = get_DQN_Model(env=env, model_path=model_path)
            train_func = train_DQN_model_new
        elif model_type == 'ppo':
            model_path = os.path.join(config['root_dir'], 'RLTesting', 'logs', 'Frozenlake', model_type, 'ppo.zip')
            model = get_PPO_Model(env=env, model_path=model_path)
            train_func = train_PPO_model
        elif model_type == 'a2c':
            model_path = os.path.join(config['root_dir'], 'RLTesting', 'logs', 'Frozenlake', model_type, 'a2c.zip')
            model = get_A2C_Model(env=env, model_path=model_path)
            train_func = train_A2C_model
        else:
            raise ValueError("Unsupported model type")

        for epoch in range(epochs):
            # 使用选定的训练函数进行训练
            actions_in_epoch = train_func(model, model_path=model_path)
            with open(log_path, 'a') as log_file:
                log_file.write('epoch: ' + str(epoch) + '\n')
                log_file.write(str(actions_in_epoch))
                log_file.write("\n-------------\n")
            # time.sleep(0.2)

        os.remove(model_path)


def inject_bugs(bug_list):
    config=parserConfig()
    BL.recover_project(config)
    BL.inject_bugs(config=config, bug_id_list=bug_list)

    # pip reinstall SB3 repository
    os.chdir(config['root_dir'])
    os.system('pip install -e .')


# 默认使用dqn进行训练
def main(bug_version, rounds, epochs, model_type):

    # # 对于特殊的bug version指定model type
    # if bug_version in [[7], [15]]:
    #     model_type = 'ppo'
    # elif bug_version in [[8], [11], [12], [13], [14]]:
    #     model_type = 'a2c'
    # elif len(bug_version) > 0:
    #     model_type = 'dqn'

    round_loop(config=parserConfig(), rounds=rounds, epochs=epochs, bug_list=bug_version, model_type=model_type)

