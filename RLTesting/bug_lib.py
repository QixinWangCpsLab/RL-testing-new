import os
import random
from itertools import combinations
import shutil
from config_parser import parserConfig


bug_group = [
    {
        'relative_path': "/stable_baselines3/dqn/dqn.py",
        'lineno': -1, # no use
        'original_lines': ['tau: float = 1.0,'],
        'injected_lines': ['tau: float = 2.0,  # should be within 0 and 1, buggy'],
        'realife_bug': False,
        'description': "tau (float) – the soft update coefficient. tau: float = 2.0,  # should be within 0 and 1, buggy",
        'category': "Updating network",
    }, # 0th bug
    {
        'relative_path': "/stable_baselines3/dqn/dqn.py",
        'lineno': -1, # no use
        'original_lines': ['th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)'],
        'injected_lines': ['# th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)  # buggy'],
        'realife_bug': False,
        'description': "Disable clip grad norm. This may lead to gradient explosion.",
        'category': "Updating network",
    }, # 1st bug
    {
        'relative_path': "/stable_baselines3/dqn/dqn.py",
        'lineno': -1,  # no use
        'original_lines': ['polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)'],
        'injected_lines': ['polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, self.tau)  # 错误的方法'],
        'realife_bug': False,
        'description': "Anything about this bug",
        'category': '?',
    },  # 2nd bug
    {
        'relative_path': "/stable_baselines3/dqn/dqn.py",
        'lineno': -1,  # no use
        'original_lines': ['if self._n_calls % max(self.target_update_interval // self.n_envs, 1) == 0:'],
        'injected_lines': ['if self._n_calls % max(0 // self.n_envs, 1) == 0:'],
        'realife_bug': False,
        'description': "Anything about this bug",
        'category': "?",
    },  # 3rd bug
    {
        'relative_path': "/stable_baselines3/dqn/dqn.py",
        'lineno': -1,  # no use
        'original_lines': ['self.exploration_initial_eps,', 'self.exploration_final_eps, #'],
        'injected_lines': ['self.exploration_final_eps,', 'self.exploration_initial_eps,'],
        'realife_bug': False,
        'description': "Wrong exploration eps schedule",
        'category': "Exploring the environment",
    },  # 4th bug
    {
        'relative_path': "/stable_baselines3/sac/sac.py",
        'lineno': -1,  # no use
        'original_lines': ['next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)', 'target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values'],
        'injected_lines': ['next_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values', 'target_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)'],
        'realife_bug': True,
        'description': "#76 SAC: Wrong target q-value in SAC.",
        'category': 'Training',
    },  # 5th bug
    {
        'relative_path': "/stable_baselines3/common/on_policy_algorithm.py",
        'lineno': -1,  # no use
        'original_lines': ['self._last_episode_starts,  # type: ignore[arg-type]', 'self._last_episode_starts = dones'],
        'injected_lines': ['dones,', '#delete this line'],
        'realife_bug': True,
        'description': "#105 on policy algorithm: rollout collect current 'dones' instead of last 'dones'.",
        'category': "Training",
    },  # 6th bug
    {
        'relative_path': "/stable_baselines3/ppo/ppo.py",
        'lineno': -1,  # no use
        'original_lines': ['entropy_loss = -th.mean(-log_prob)'],
        'injected_lines': ['entropy_loss = -log_prob.mean()'],
        'realife_bug': True,
        'description': "#130 PPO: wrong entropy loss computation in PPO.",
        'category': "Loss function",
    },  # 7th bug
    {
        'relative_path': "/stable_baselines3/a2c/a2c.py",
        'lineno': -1,  # no use
        'original_lines': ['entropy_loss = -th.mean(-log_prob)'],
        'injected_lines': ['entropy_loss = -log_prob.mean()'],
        'realife_bug': True,
        'description': "#130 A2C: wrong entropy loss computation in A2C.",
        'category': "Loss function",
    },  # 8th bug
    {
        'relative_path': "/stable_baselines3/dqn/policies.py",
        'lineno': -1,  # no use
        'original_lines': ['#9th bug: 1', '#9th bug: 2', '#9th bug: 3', '#9th bug: 4', 'net_args = self._update_features_extractor(self.net_args, features_extractor=None)', 'return QNetwork(**net_args).to(self.device)'],
        'injected_lines': ['self.features_extractor = features_extractor_class(self.observation_space, **self.features_extractor_kwargs)',
                           'self.features_dim = self.features_extractor.features_dim',
                           '"features_extractor": self.features_extractor,',
                           '"features_dim": self.features_dim,',
                           '',
                           'return QNetwork(**self.net_args).to(self.device)'],
        'realife_bug': True,
        'description': "#132 DQN： main and target network accidentally shared feature extractor network.",
        'category': "model",
    },  # 9th bug
    {
        'relative_path': "/stable_baselines3/common/on_policy_algorithm.py",
        'lineno': -1,  # no use
        'original_lines': ['with th.no_grad(): #10th bug: 1', '# Compute value for the last timestep', 'values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]'],
        'injected_lines': ['', '', ''],
        'realife_bug': True,
        'description': "#183 On Policy algorithm： wrpmh advantages estimation for on policy algorithm.",
        'category': 'Training',
    },  # 10th bug


    {
        'relative_path': "/stable_baselines3/a2c/a2c.py",
        'lineno': -1,  # no use
        'original_lines': ['th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)'],
        'injected_lines': ['th.nn.utils.clip_grad_norm_(self.policy.parameters(), 1e8)'],
        'realife_bug': False,
        'description': "Gradient cropping is designed to prevent the problem of gradient explosion; if the cropping threshold is set very high, cropping does not actually occur and may lead to unstable training.",
        'category': "Updating network",
    },  # 11th bug
    {
        'relative_path': "/stable_baselines3/a2c/a2c.py",
        'lineno': -1,  # no use
        'original_lines': ['self.policy_kwargs["optimizer_kwargs"] = dict(alpha=0.99, eps=rms_prop_eps, weight_decay=0)'],
        'injected_lines': ['self.policy_kwargs["optimizer_kwargs"] = dict(alpha=0.1, eps=rms_prop_eps, weight_decay=0)'],
        'realife_bug': False,
        'description': "The alpha parameter of RMSprop controls the decay rate of the moving average. If it is set too low, the moving average will quickly forget old gradient information, causing the optimization to become very oscillatory; \
                        if it is set too high, the optimizer will rely too much on old gradient information, which may lead to too slow training.",
        'category': '?',

    },  # 12th bug
    {
        'relative_path': "/stable_baselines3/a2c/a2c.py",
        'lineno': -1,  # no use
        'original_lines': ['self.policy_kwargs["optimizer_kwargs"] = dict(alpha=0.99, eps=rms_prop_eps, weight_decay=0)'],
        'injected_lines': ['self.policy_kwargs["optimizer_kwargs"] = dict(alpha=0.9999, eps=rms_prop_eps, weight_decay=0)'],
        'realife_bug': False,
        'description': "Same as 12 th bug",
        'category': '?',
    }, # 13th bug
    {
        'relative_path': "/stable_baselines3/a2c/a2c.py",
        'lineno': -1,  # no use
        'original_lines': ['advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)'],
        'injected_lines': ['advantages = (advantages - advantages.mean())'],
        'realife_bug': False,
        'description': "This error causes the dominance function to not be normalized correctly, which can lead to less efficient training, \
            since normalization helps to speed up learning and improve the performance of the strategy",
        'category': 'Training',
    }, # 14th bug
        {
        'relative_path': "/stable_baselines3/ppo/ppo.py",
        'lineno': -1,  # no use
        'original_lines': ['self.clip_range = get_schedule_fn(self.clip_range)'],
        'injected_lines': ['self.clip_range = get_schedule_fn(self.learning_rate)'],
        'realife_bug': False,
        'description': "Wrong mistake on purpose without any meaning or explaination",
        'category': "model?",
    }, # 15th bug
    {
        'relative_path': "/stable_baselines3/dqn/dqn.py",
        'lineno': -1,  # no use
        'original_lines': ['with th.no_grad():'],
        'injected_lines': ['if True:'],
        'realife_bug': False,
        'description': "Without th.no_grad()",
        'category': 'Wrong network update',
    }, # 16th bug
    {
        'relative_path': "/stable_baselines3/dqn/dqn.py",
        'lineno': -1,  # no use
        'original_lines': ['target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values'],
        'injected_lines': ['target_q_values = replay_data.rewards + replay_data.dones * self.gamma * next_q_values'],
        'realife_bug': False,
        'description': "Wrong calculation on target q values",
        'category': "Training",
    }, # 17th bug
    {
        'relative_path': "/stable_baselines3/dqn/dqn.py",
        'lineno': -1,  # no use
        'original_lines': ['self.policy.optimizer.zero_grad()'],
        'injected_lines': [''],
        'realife_bug': False,
        'description': "No zero grad",
        'category': "Optimizer",
    }, # 18th bug
    {
        'relative_path': "/stable_baselines3/dqn/dqn.py",
        'lineno': -1,  # no use
        'original_lines': ['next_q_values = self.q_net_target(replay_data.next_observations)'],
        'injected_lines': ['next_q_values = self.q_net_target(replay_data.observations)'],
        'realife_bug': False,
        'description': "Wrong next q values calculation",
        'category': "Training",
    }, # 19th bug
    {
        'relative_path': "/stable_baselines3/dqn/dqn.py",
        'lineno': -1,  # no use
        'original_lines': ['current_q_values = self.q_net(replay_data.observations)'],
        'injected_lines': ['current_q_values = self.q_net(replay_data.next_observations)'],
        'realife_bug': False,
        'description': "Wrong current q values calculation",
        'category': "Training",
    }, # 20th bug
    {
        'relative_path': "/stable_baselines3/dqn/policies.py",
        'lineno': -1,  # no use
        'original_lines': ['net_arch = [64, 64]#bug 21'],
        'injected_lines': ['net_arch = [1, 1]#bug 21'],
        'realife_bug': False,
        'description': "Wrong Network architecture",
        'category': "Model",
    }, # 21th bug
    {
        'relative_path': "/stable_baselines3/dqn/dqn.py",
        'lineno': -1,  # no use
        'original_lines': ['self.exploration_rate = self.exploration_schedule(self._current_progress_remaining)'],
        'injected_lines': ['#self.exploration_rate = self.exploration_schedule(self._current_progress_remaining)'],
        'realife_bug': False,
        'description': "Fixed exploration rate",
        'category': "Exploring the enviroment",
    }, # 22th bug
    {
        'relative_path': "/stable_baselines3/dqn/dqn.py",
        'lineno': -1,  # no use
        'original_lines': ['target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values'],
        'injected_lines': ['target_q_values = replay_data.rewards + (1 - replay_data.dones) * next_q_values'],
        'realife_bug': False,
        'description': "Wrong target q calculation",
        'category': "Training",
    }, # 23th bug
    {
        'relative_path': "/stable_baselines3/a2c/a2c.py",
        'lineno': -1,  # no use
        'original_lines': ['th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)'],
        'injected_lines': ['#th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)'],
        'realife_bug': False,
        'description': "Disable clip grad norm. This may lead to gradient explosion.",
        'category': "Updating network",
    }, # 24th bug
    {
        'relative_path': "/stable_baselines3/common/buffers.py",
        'lineno': -1,  # no use
        'original_lines': ['next_values = self.values[step + 1]'],
        'injected_lines': ['next_values = self.values[step]'],
        'realife_bug': False,
        'description': "bug in actor critic advantage calculation. This bug will influence all algorithms using advantage, such as a2c, ppo",
        'category': "Training",
    }, # 25th bug
    {
        'relative_path': "/stable_baselines3/common/buffers.py",
        'lineno': -1,  # no use
        'original_lines': ['delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]'],
        'injected_lines': ['delta = self.rewards[step] + next_values * next_non_terminal - self.values[step]'],
        'realife_bug': False,
        'description': "bug in actor critic advantage calculation. This bug will influence all algorithms using advantage, such as a2c, ppo",
        'category': "Training",
    }, # 26th bug
    {
        'relative_path': "/stable_baselines3/common/buffers.py",
        'lineno': -1,  # no use
        'original_lines': ['for step in reversed(range(self.buffer_size)):'],
        'injected_lines': ['for step in range(self.buffer_size):'],
        'realife_bug': False,
        'description': "bug in actor critic advantage calculation. This bug will influence all algorithms using advantage, such as a2c, ppo. Not sure whether this will cause a error or not.",
        'category': "Training",
    }, # 27th bug
    {
        'relative_path': "/stable_baselines3/common/policies.py",
        'lineno': -1,  # no use
        'original_lines': ['net_arch = dict(pi=[64, 64], vf=[64, 64])'],
        'injected_lines': ['net_arch = dict(pi=[1, 1], vf=[1, 1])'],
        'realife_bug': False,
        'description': "bug in A2C policy.",
        'category': "Model",
    }, # 28th bug
    {
        'relative_path': "/stable_baselines3/a2c/a2c.py",
        'lineno': -1,  # no use
        'original_lines': ['policy_loss = -(advantages * log_prob).mean()'],
        'injected_lines': ['policy_loss = (advantages * log_prob).mean()'],
        'realife_bug': False,
        'description': "bug in A2C train. Wrong policy loss.",
        'category': 'Loss function',
    }, # 29th bug
    {
        'relative_path': "/stable_baselines3/a2c/a2c.py",
        'lineno': -1,  # no use
        'original_lines': ['entropy_loss = -th.mean(-log_prob)', 'entropy_loss = -th.mean(entropy)'],
        'injected_lines': ['entropy_loss = th.mean(-log_prob)', 'entropy_loss = th.mean(entropy)'],
        'realife_bug': False,
        'description': "bug in A2C train. Wrong policy loss. Wrong entropy loss calculation",
        'category': "Loss funtction",
    }, # 30th bug
    {
        'relative_path': "/stable_baselines3/a2c/a2c.py",
        'lineno': -1,  # no use
        'original_lines': ['loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss'],
        'injected_lines': ['loss = policy_loss + self.ent_coef * entropy_loss + value_loss'],
        'realife_bug': False,
        'description': "bug in A2C train. Wrong loss.",
        'category': "Loss function",
    }, # 31th bug
    {
        'relative_path': "/stable_baselines3/a2c/a2c.py",
        'lineno': -1,  # no use
        'original_lines': ['self.policy.optimizer.zero_grad()'],
        'injected_lines': ['#self.policy.optimizer.zero_grad()'],
        'realife_bug': False,
        'description': "bug in A2C train. Forget to run zero grad function",
        'category': "Optimizer",
    }, # 32th bug


    # 25th, 26th, 27th 28th bug also apply for ppo
    {
        'relative_path': "/stable_baselines3/ppo/ppo.py",
        'lineno': -1,  # no use
        'original_lines': ['clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]'],
        'injected_lines': ['clip_range = 0'],
        'realife_bug': False,
        'description': "Disable clip range change during training. This may cause error.",
        'category': "Training",
    }, # 33th bug
    {
        'relative_path': "/stable_baselines3/ppo/ppo.py",
        'lineno': -1,  # no use
        'original_lines': ['th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)'],
        'injected_lines': ['#th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)'],
        'realife_bug': False,
        'description': "Disable clip grad norm. This may lead to gradient explosion.",
        'category': "Updating network",
    }, # 34th bug
    {
        'relative_path': "/stable_baselines3/ppo/ppo.py",
        'lineno': -1,  # no use
        'original_lines': ['self.policy.optimizer.zero_grad()'],
        'injected_lines': ['#self.policy.optimizer.zero_grad()'],
        'realife_bug': False,
        'description': "Redundant",
        'category': "optimizer",
    }, # 35th bug
    {
        'relative_path': "/stable_baselines3/ppo/ppo.py",
        'lineno': -1,  # no use
        'original_lines': ['entropy_loss = -th.mean(-log_prob)', 'entropy_loss = -th.mean(entropy)'],
        'injected_lines': ['entropy_loss = th.mean(-log_prob)', 'entropy_loss = th.mean(entropy)'],
        'realife_bug': False,
        'description': "bug in ppo train. Wrong policy loss. Wrong entropy loss calculation",
        'category' : "Loss function",
    }, # 36th bug
    {
        'relative_path': "/stable_baselines3/ppo/ppo.py",
        'lineno': -1,  # no use
        'original_lines': ['self.policy.optimizer.zero_grad()'],
        'injected_lines': ['#self.policy.optimizer.zero_grad()'],
        'realife_bug': False,
        'description': "bug in ppo train. Forget to run zero grad function",
        'category': "optimizer",
    }, # 37th bug
    {
        'relative_path': "/stable_baselines3/ppo/ppo.py",
        'lineno': -1,  # no use
        'original_lines': ['loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss'],
        'injected_lines': ['loss = policy_loss + self.ent_coef * entropy_loss + value_loss'],
        'realife_bug': False,
        'description': "bug in ppo train. Wrong loss.",
        'category': "Loss function",
    }, # 38th bug
    {
        'relative_path': "/stable_baselines3/ppo/ppo.py",
        'lineno': -1,  # no use
        'original_lines': ['policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()'],
        'injected_lines': ['policy_loss = -advantages.mean()'],
        'realife_bug': False,
        'description': "bug in ppo train. No clip and no ratio.",
        'category': "Loss function",
    }, # 39th bug

    
]



# If there's no confliction return True, otherwise return False
def check_injection_validation(bug_id_ist):
    
    # for temp_line in temp_bug[original_lines]:
        # if temp_line not in relative_file_data:
                
    return True



def inject_bugs(config, bug_id_list):
    # if config['specific_bug_flag']:
    # bug_id_list = config['specified_bug_id']
    # else:
        # print('???????') # need to randomly generate bug versions
    
    if not check_injection_validation(bug_id_list):
        return "bug version invalid!!"
    
    for bug_id in bug_id_list:
        temp_bug = bug_group[bug_id]

        temp_bug_path = config['root_dir'] + temp_bug['relative_path']

        with open(temp_bug_path, 'r+') as relative_file:
            relative_file_data = relative_file.read()
            print(relative_file_data)

            # 替换文件内容
            for bug_line_index in range(len(temp_bug['original_lines'])):
                relative_file_data = relative_file_data.replace(
                    temp_bug['original_lines'][bug_line_index],
                    temp_bug['injected_lines'][bug_line_index]
                )

            # 移动文件指针到开头
            relative_file.seek(0)
            # 写入修改后的数据
            relative_file.write(relative_file_data)
            # 截断文件，删除旧内容后面的数据
            relative_file.truncate()
            
            
# def recover_project(config):
#     # 设置主文件夹和archive文件夹的路径
#     main_folder = config['root_dir']
#     archive_folder = os.path.join(main_folder, 'archived_code')

#     # 自动获取archive文件夹下的所有子文件夹
#     subfolders = [f for f in os.listdir(archive_folder) if os.path.isdir(os.path.join(archive_folder, f))]

#     # 遍历每个子文件夹
#     for subfolder in subfolders:
#         archive_subfolder_path = os.path.join(archive_folder, subfolder)
#         main_subfolder_path = os.path.join(main_folder, subfolder)
    
#         # 确保目标子文件夹存在
#         os.makedirs(main_subfolder_path, exist_ok=True)
    
#         # 遍历archive中的每个文件
#         for filename in os.listdir(archive_subfolder_path):
#             # 源文件路径
#             file_source = os.path.join(archive_subfolder_path, filename)
        
#             # 目标文件路径
#             file_destination = os.path.join(main_subfolder_path, filename)
        
#             # 复制文件
#             shutil.copy(file_source, file_destination)
#     return
        
def recover_project(config):
    main_folder = config['root_dir']
    archive_folder = os.path.join(main_folder, 'archived_code')

    # 确保存档文件夹存在
    if not os.path.exists(archive_folder):
        print(f"Archive folder not found: {archive_folder}")
        return

    # 获取archive文件夹下的所有子文件夹
    subfolders = [f for f in os.listdir(archive_folder) if os.path.isdir(os.path.join(archive_folder, f))]

    for subfolder in subfolders:
        archive_subfolder_path = os.path.join(archive_folder, subfolder)
        main_subfolder_path = os.path.join(main_folder, subfolder)

        # 如果目标文件夹存在，则先删除（shutil.copytree要求目标文件夹不存在）
        if os.path.exists(main_subfolder_path):
            shutil.rmtree(main_subfolder_path)

        # 复制整个目录树
        shutil.copytree(archive_subfolder_path, main_subfolder_path)
        
        
def cover_then_inject_bugs(bug_list):
    config=parserConfig()
    recover_project(config)
    inject_bugs(config=config, bug_id_list=bug_list)

    # pip reinstall SB3 repository
    os.chdir(config['root_dir'])
    os.system('pip install -e .')