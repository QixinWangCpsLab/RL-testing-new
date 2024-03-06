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
        'description': "Anything about this bug"
    }, # 0th bug
    {
        'relative_path': "/stable_baselines3/dqn/dqn.py",
        'lineno': -1, # no use
        'original_lines': ['th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)'],
        'injected_lines': ['# th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)  # buggy'],
        'realife_bug': False,
        'description': "Anything about this bug"
    }, # 1st bug
    {
        'relative_path': "/stable_baselines3/dqn/dqn.py",
        'lineno': -1,  # no use
        'original_lines': ['polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)'],
        'injected_lines': ['polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, self.tau)  # 错误的方法'],
        'realife_bug': False,
        'description': "Anything about this bug"
    },  # 2nd bug
    {
        'relative_path': "/stable_baselines3/dqn/dqn.py",
        'lineno': -1,  # no use
        'original_lines': ['if self._n_calls % max(self.target_update_interval // self.n_envs, 1) == 0:'],
        'injected_lines': ['if self._n_calls % max(0 // self.n_envs, 1) == 0:'],
        'realife_bug': False,
        'description': "Anything about this bug"
    },  # 3rd bug
    {
        'relative_path': "/stable_baselines3/dqn/dqn.py",
        'lineno': -1,  # no use
        'original_lines': ['self.exploration_initial_eps,', 'self.exploration_final_eps, #'],
        'injected_lines': ['self.exploration_final_eps,', 'self.exploration_initial_eps,'],
        'realife_bug': False,
        'description': "Anything about this bug"
    },  # 4th bug
    {
        'relative_path': "/stable_baselines3/sac/sac.py",
        'lineno': -1,  # no use
        'original_lines': ['next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)', 'target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values'],
        'injected_lines': ['next_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values', 'target_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)'],
        'realife_bug': True,
        'description': "#76 SAC: Wrong target q-value in SAC."
    },  # 5th bug
    {
        'relative_path': "/stable_baselines3/common/on_policy_algorithm.py",
        'lineno': -1,  # no use
        'original_lines': ['self._last_episode_starts,  # type: ignore[arg-type]', 'self._last_episode_starts = dones'],
        'injected_lines': ['dones', '#delete this line'],
        'realife_bug': True,
        'description': "#105 on policy algorithm: rollout collect current 'dones' instead of last 'dones'."
    },  # 6th bug
    {
        'relative_path': "/stable_baselines3/ppo/ppo.py",
        'lineno': -1,  # no use
        'original_lines': ['entropy_loss = -th.mean(-log_prob)'],
        'injected_lines': ['entropy_loss = -log_prob.mean()'],
        'realife_bug': True,
        'description': "#130 PPO: wrong entropy loss computation in PPO."
    },  # 7th bug
    {
        'relative_path': "/stable_baselines3/a2c/a2c.py",
        'lineno': -1,  # no use
        'original_lines': ['entropy_loss = -th.mean(-log_prob)'],
        'injected_lines': ['entropy_loss = -log_prob.mean()'],
        'realife_bug': True,
        'description': "#130 A2C: wrong entropy loss computation in A2C."
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
        'description': "#132 DQN： main and target network accidentally shared feature extractor network."
    },  # 9th bug
    {
        'relative_path': "/stable_baselines3/common/on_policy_algorithm.py",
        'lineno': -1,  # no use
        'original_lines': ['with th.no_grad(): #10th bug: 1', '# Compute value for the last timestep', 'values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]'],
        'injected_lines': ['', '', ''],
        'realife_bug': True,
        'description': "#183 On Policy algorithm： wrpmh advantages estimation for on policy algorithm."
    },  # 10th bug


    {
        'relative_path': "/stable_baselines3/a2c/a2c.py",
        'lineno': -1,  # no use
        'original_lines': ['th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)'],
        'injected_lines': ['th.nn.utils.clip_grad_norm_(self.policy.parameters(), 1e8)'],
        'realife_bug': False,
        'description': "Gradient cropping is designed to prevent the problem of gradient explosion; if the cropping threshold is set very high, cropping does not actually occur and may lead to unstable training."
    },  # 11th bug
    {
        'relative_path': "/stable_baselines3/a2c/a2c.py",
        'lineno': -1,  # no use
        'original_lines': ['self.policy_kwargs["optimizer_kwargs"] = dict(alpha=0.99, eps=rms_prop_eps, weight_decay=0)'],
        'injected_lines': ['self.policy_kwargs["optimizer_kwargs"] = dict(alpha=0.1, eps=rms_prop_eps, weight_decay=0)'],
        'realife_bug': False,
        'description': "The alpha parameter of RMSprop controls the decay rate of the moving average. If it is set too low, the moving average will quickly forget old gradient information, causing the optimization to become very oscillatory; \
                        if it is set too high, the optimizer will rely too much on old gradient information, which may lead to too slow training."
    },  # 12th bug
    {
        'relative_path': "/stable_baselines3/a2c/a2c.py",
        'lineno': -1,  # no use
        'original_lines': ['self.policy_kwargs["optimizer_kwargs"] = dict(alpha=0.99, eps=rms_prop_eps, weight_decay=0)'],
        'injected_lines': ['self.policy_kwargs["optimizer_kwargs"] = dict(alpha=0.9999, eps=rms_prop_eps, weight_decay=0)'],
        'realife_bug': False,
        'description': "Same as 12 th bug"
    }, # 13th bug
    {
        'relative_path': "/stable_baselines3/a2c/a2c.py",
        'lineno': -1,  # no use
        'original_lines': ['advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)'],
        'injected_lines': ['advantages = (advantages - advantages.mean())'],
        'realife_bug': False,
        'description': "This error causes the dominance function to not be normalized correctly, which can lead to less efficient training, \
            since normalization helps to speed up learning and improve the performance of the strategy"
    }, # 14th bug
        {
        'relative_path': "/stable_baselines3/ppo/ppo.py",
        'lineno': -1,  # no use
        'original_lines': ['self.clip_range = get_schedule_fn(self.clip_range)'],
        'injected_lines': ['self.clip_range = get_schedule_fn(self.learning_rate)'],
        'realife_bug': False,
        'description': "Wrong mistake on purpose without any meaning or explaination"
    }, # 15th bug
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