import bug_lib as BL
import os
from datetime import date

import mountaincar.Mountaincar_Env as Env

import get_and_train_models as G_T_modles

from config_parser import parserConfig


def round_loop(config, rounds, epochs, bug_list, model_type, max_steps = 200):
    # BL.recover_project(config)
    # BL.inject_bugs(config=config, bug_id_list=bug_list)

    # # pip reinstall SB3 repository
    # os.chdir(config['root_dir'])
    # os.system('pip install -e .')

    for round in range(rounds):
        print("round: " + str(round) + "----")

        log_dir = os.path.join(config['root_dir'], 'RLTesting', 'logs', 'Mountaincar', model_type, str(bug_list))
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        log_name = 'time_' + str(date.today()) + str(bug_list) + 'round_' + str(round)
        log_path = os.path.join(log_dir, log_name)
        with open(log_path, 'a') as log_file:
            log_file.write(str(config))
            log_file.write("\n-------------\n")

        env = Env.EnvWrapper()
        env.reset()
        rewarded_actions = Env.generate_states_actions(env)
        env.set_rewarded_actions(rewarded_actions)
        with open(log_path, 'a') as log_file:
            log_file.write('rewarded_actions' + str(rewarded_actions))
            log_file.write("\n-------------\n")

        # 根据config中的'model_type'选择模型和训练函数
        if model_type == 'dqn':
            model_path = os.path.join(config['root_dir'], 'RLTesting', 'logs', 'Mountaincar', model_type, 'dqn.zip')
            model = G_T_modles.get_DQN_Model(env=env, model_path=model_path)
            train_func = G_T_modles.train_DQN_model_new
        elif model_type == 'sac':
            model_path = os.path.join(config['root_dir'], 'RLTesting', 'logs', 'Mountaincar', model_type,  'sac.zip')
            model = G_T_modles.get_SAC_Model(env=env, model_path=model_path)
            train_func = G_T_modles.train_SAC_model
        elif model_type == 'ppo':
            model_path = os.path.join(config['root_dir'], 'RLTesting', 'logs', 'Mountaincar', model_type,  'ppo.zip')
            model = G_T_modles.get_PPO_Model(env=env, model_path=model_path)
            train_func = G_T_modles.train_PPO_model
        elif model_type == 'a2c':
            model_path = os.path.join(config['root_dir'], 'RLTesting', 'logs', 'Mountaincar', model_type,  'a2c.zip')
            model = G_T_modles.get_A2C_Model(env=env, model_path=model_path)
            train_func = G_T_modles.train_A2C_model
        else:
            raise ValueError("Unsupported model type")

        for epoch in range(epochs):
            actions_in_epoch = train_func(model=model, max_steps=max_steps, model_path=model_path)
            with open(log_path, 'a') as log_file:
                log_file.write('epoch: ' + str(epoch) + '\n')
                log_file.write(str(actions_in_epoch))
                log_file.write("\n-------------\n")

        os.remove(model_path)


# 默认使用sac进行训练
def main(bug_version, rounds, epochs, model_type, max_steps=200):

    # 对于特殊的bug version指定model type
    if bug_version in [[7], ]:
        model_type = 'ppo'
    elif bug_version in [[5], ]:
        model_type = 'sac'
    elif bug_version in [[8], ]:
        model_type = 'a2c'

    round_loop(config=parserConfig(), rounds=rounds, epochs=epochs, bug_list=bug_version, model_type=model_type, max_steps=max_steps)

