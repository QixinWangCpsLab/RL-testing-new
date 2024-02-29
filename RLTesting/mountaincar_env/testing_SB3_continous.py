import sys

sys.path.insert(0, '../')
import RLTesting.bug_lib as BL
import os
from datetime import date

import Mountaincar_Env as Env

import RLTesting.get_and_train_models as G_T_modles


def main(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    for round in range(10):
        log_name = 'time_' + str(date.today()) + 'round_' + str(round)
        log_path = os.path.join(dir_name, log_name)
        with open(log_path, 'a') as log_file:
            log_file.write(str('mountaincar continous bug free'))
            log_file.write("\n-------------\n")

        env = Env.EnvWrapper()
        rewarded_actions = Env.generate_states_actions(env)
        env.set_rewarded_actions(rewarded_actions)

        with open(log_path, 'a') as log_file:
            log_file.write('rewarded_actions' + str(rewarded_actions))
            log_file.write("\n-------------\n")

        model_path = str(round) + 'model.zip'

        model = G_T_modles.get_SAC_Model(env=env, model_path=model_path)

        for epoch in range(300):
            actions_in_epoch = G_T_modles.train_SAC_model(model=model, max_steps=100, model_path=model_path)

            with open(log_path, 'a') as log_file:
                log_file.write('epoch: ' + str(epoch) + '\n')
                log_file.write(str(actions_in_epoch))
                log_file.write("\n-------------\n")

        os.remove(model_path)

main('mountaincar_bugfree')

#
# def round_loop(config):
#     BL.recover_project(config)
#     BL.inject_bugs(config)
#
#     # pip reinstall SB3 repository
#     os.chdir("../..")
#     os.system('pip install -e .')
#
#     for round in range(config['rounds']):
#         print("round: " + str(round) + "----")
#
#         log_dir = os.path.join(config['root_dir'], 'RLTesting', 'logs')
#         if not os.path.exists(log_dir):
#             os.makedirs(log_dir)
#         log_name = 'time_' + str(date.today()) + str(config['specified_bug_id']) + 'round_' + str(round)
#         log_path = os.path.join(log_dir, log_name)
#         with open(log_path, 'a') as log_file:
#             log_file.write(str(config))
#             log_file.write("\n-------------\n")
#
#         # 每个round都需要重新随机生成env的rewarded_actions
#         # rewarded_actions = {0: 2, 1: 2, 2: 1, 6: 1, 10: 1, 14: 2}
#         env = Fuzzy_Env.EnvWrapper()
#         rewarded_actions = get_random_station_action_rewarder(env)
#         env.set_rewarded_actions(rewarded_actions)
#         with open(log_path, 'a') as log_file:
#             log_file.write('rewarded_actions' + str(rewarded_actions))
#             log_file.write("\n-------------\n")
#
#         # 根据config中的'model_type'选择模型和训练函数
#         if config['model_type'] == 'dqn':
#             model_path = os.path.join('RLTesting', 'logs', 'dqn.zip')
#             model = get_DQN_Model(env=env, model_path=model_path)
#             train_func = train_DQN_model_new
#         elif config['model_type'] == 'ppo':
#             model_path = os.path.join('RLTesting', 'logs', 'ppo.zip')
#             model = get_PPO_Model(env=env, model_path=model_path)
#             train_func = train_PPO_model
#         elif config['model_type'] == 'a2c':
#             model_path = os.path.join('RLTesting', 'logs', 'a2c.zip')
#             model = get_A2C_Model(env=env, model_path=model_path)
#             train_func = train_A2C_model
#         else:
#             raise ValueError("Unknown model type in config: " + config['model_type'])
#
#         for epoch in range(config['epoches']):
#             # 使用选定的训练函数进行训练
#             actions_in_epoch = train_func(model, model_path=model_path)
#             with open(log_path, 'a') as log_file:
#                 log_file.write('epoch: ' + str(epoch) + '\n')
#                 log_file.write(str(actions_in_epoch))
#                 log_file.write("\n-------------\n")
#             # time.sleep(0.2)
#
#         os.remove(model_path)
#
#
# def main(bug_version_list):
#     config = parserConfig()
#
#     for bug_version in bug_version_list:
#         config['specified_bug_id'] = bug_version
#         # print(bug_version, config['specified_bug_id'])
#         if bug_version in [[7], ]:
#             config['model_type'] = 'ppo'
#         elif bug_version in [[8], ]:
#             config['model_type'] = 'a2c'
#         # 判断bug_version是否是[]
#         elif not bug_version:
#             print('bug free')
#         else:
#             config['model_type'] = 'dqn'
#         round_loop(config)
