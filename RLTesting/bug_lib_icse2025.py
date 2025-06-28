import os
import random
from itertools import combinations
import shutil
from config_parser import parserConfig


bug_group = [
    {
        'relative_path': "/stable_baselines3/common/on_policy_algorithm.py",
        'lineno': -1,  # no use
        'original_lines': ['self._last_episode_starts,  # type: ignore[arg-type]', 'self._last_episode_starts = dones'],
        'injected_lines': ['dones,', '#delete this line'],
        'realife_bug': True,
        'description': "#105 on policy algorithm: rollout collect current 'dones' instead of last 'dones'.",
        'category': "Updating network Bugs",
    },  # 0th bug
    {
        'relative_path': "/stable_baselines3/a2c/a2c.py",
        'lineno': -1,  # no use
        'original_lines': ['th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)'],
        'injected_lines': ['#th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)'],
        'realife_bug': False,
        'description': "Training Bugs: Disabling gradient clipping during backpropagation",
        'category': "Training Bugs",
    }, # 1th bug
    {
        'relative_path': "/stable_baselines3/common/buffers.py",
        'lineno': -1,  # no use
        'original_lines': ['delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]'],
        'injected_lines': ['delta = self.rewards[step] + next_values - self.values[step]'],
        'realife_bug': False,
        'description': "Updating network Bugs: Missing multiplication of γ and next_non_terminal in advantage calculation",
        'category': "Updating network Bugs",
    }, # 2th bug
    {
        'relative_path': "/stable_baselines3/common/buffers.py",
        'lineno': -1,  # no use
        'original_lines': ['for step in reversed(range(self.buffer_size)):'],
        'injected_lines': ['for step in range(self.buffer_size):'],
        'realife_bug': False,
        'description': "Updating network Bugs: Incorrect traveral direction in TD(lambda) estimator calcualtion.",
        'category': "Updating network Bugs",
    }, # 3th bug
    {
        'relative_path': "/stable_baselines3/common/policies.py",
        'lineno': -1,  # no use
        'original_lines': ['net_arch = dict(pi=[64, 64], vf=[64, 64])'],
        'injected_lines': ['net_arch = dict(pi=[5, 5], vf=[5, 5])'],
        'realife_bug': False,
        'description': "Model Bugs: Overly simple network structure.",
        'category': "Model Bugs",
    }, # 4th bug
    {
        'relative_path': "/stable_baselines3/a2c/a2c.py",
        'lineno': -1,  # no use
        'original_lines': ['policy_loss = -(advantages * log_prob).mean()'],
        'injected_lines': ['policy_loss = (advantages * log_prob).mean()'],
        'realife_bug': False,
        'description': "Updating network Bugs: Incorrect policy loss calculation.",
        'category': 'Updating network Bugs',
    }, # 5th bug
    {
        'relative_path': "/stable_baselines3/a2c/a2c.py",
        'lineno': -1,  # no use
        'original_lines': ['loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss'],
        'injected_lines': ['loss = policy_loss + self.ent_coef * entropy_loss + value_loss'],
        'realife_bug': False,
        'description': "Updating network Bugs: Incorrect loss calculation.",
        'category': "Updating network Bugs",
    }, # 6th bug
    {
        'relative_path': "/stable_baselines3/a2c/a2c.py",
        'lineno': -1,  # no use
        'original_lines': ['self.policy.optimizer.zero_grad()', 'th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)'],
        'injected_lines': ['#self.policy.optimizer.zero_grad()', '#th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)'],
        'realife_bug': False,
        'description': "Training Bugs: Neglecting to reset gradients before backpropagation",
        'category': "Training Bugs",
    }, # 7th bug
    {
        'relative_path': "/stable_baselines3/a2c/a2c.py",
        'lineno': -1,  # no use
        'original_lines': ['loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss'],
        'injected_lines': ['loss = self.ent_coef * entropy_loss + self.vf_coef * value_loss'],
        'realife_bug': False,
        'description': "Updating network Bugs: Incorrect loss calculation (only use value loss as loss)",
        'category': "Updating network Bugs",
    }, # 8th bug
    {
        'relative_path': "/stable_baselines3/a2c/a2c.py",
        'lineno': -1,  # no use
        'original_lines': ['loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss'],
        'injected_lines': ['loss = policy_loss'],
        'realife_bug': False,
        'description': "Updating network Bugs: Incorrect loss calculation (only use policy loss as loss)",
        'category': "Updating network Bugs",
    }, # 9th bug
    {
        'relative_path': "/stable_baselines3/common/buffers.py",
        'lineno': -1,  # no use
        'original_lines': ['delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]'],
        'injected_lines': ['delta = self.rewards[step] + self.gamma * next_values * next_non_terminal'],
        'realife_bug': False,
        'description': "Updating network Bugs: Missing subtraction of the V value in advantage calculation.",
        'category': "Updating network Bugs",
    }, # 10th bug
    {
        'relative_path': "/stable_baselines3/common/buffers.py",
        'lineno': -1,  # no use
        'original_lines': ['delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]'],
        'injected_lines': ['delta = self.rewards[step] + self.gamma * next_values - self.values[step]'],
        'realife_bug': False,
        'description': "Updating network Bugs: Missing multiplication of the next_non_terminal in advantage calculation",
        'category': "Updating network Bugs",
    }, # 11th bug
    {
        'relative_path': "/stable_baselines3/common/buffers.py",
        'lineno': -1,  # no use
        'original_lines': ['delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]'],
        'injected_lines': ['delta = self.rewards[step] + next_values * next_non_terminal - self.values[step]'],
        'realife_bug': False,
        'description': "Updating network Bugs: Missing multiplication of the forget factor γ in advantage calculation",
        'category': "Updating network Bugs",
    }, # 12th bug
    {
        'relative_path': "/stable_baselines3/common/buffers.py",
        'lineno': -1,  # no use
        'original_lines': ['delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]'],
        'injected_lines': ['delta = self.gamma * next_values * next_non_terminal - self.values[step]'],
        'realife_bug': False,
        'description': "Updating network Bugs: Missing addition of reward r in advantage calculation",
        'category': "Updating network Bugs",
    }, # 13th bug
    {
        'relative_path': "/stable_baselines3/common/buffers.py",
        'lineno': -1,  # no use
        'original_lines': ['self.advantages[step] = last_gae_lam'],
        'injected_lines': ['self.advantages[step] = delta'],
        'realife_bug': False,
        'description': "Updating network Bugs: Directly using delta as advantage value.",
        'category': "Updating network Bugs",
    }, # 14th bug
    {
        'relative_path': "/stable_baselines3/common/buffers.py",
        'lineno': -1,  # no use
        'original_lines': ['self.returns = self.advantages + self.values'],
        'injected_lines': ['self.returns = self.advantages'],
        'realife_bug': False,
        'description': "Updating network Bugs: Directly using advantage value as TD(lambda) estimator",
        'category': "Updating network Bugs",
    }, # 15th bug
    {
        'relative_path': "/stable_baselines3/common/buffers.py",
        'lineno': -1,  # no use
        'original_lines': ['self.returns = self.advantages + self.values'],
        'injected_lines': ['self.returns = self.values'],
        'realife_bug': False,
        'description': "Updating network Bugs: Directly using state value estimation as TD(lambda) estimator",
        'category': "Updating network Bugs",
    }, # 16th bug
    {
        'relative_path': "/stable_baselines3/common/on_policy_algorithm.py",
        'lineno': -1,  # no use
        'original_lines': ['actions = actions.cpu().numpy()'],
        'injected_lines': ['actions = actions.cpu().numpy().astype(np.int32)'],
        'realife_bug': True,
        'description': "https://github.com/hill-a/stable-baselines/issues/109",
        'category': "Tensor and inputs bugs",
    }, # 17th bug
    {
        'relative_path': "/stable_baselines3/td3/td3.py",
        'lineno': -1,  # no use
        'original_lines': ['next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)'],
        'injected_lines': ['next_actions = (self.actor_target(replay_data.next_observations)).clamp(-1, 1)'],
        'realife_bug': False,
        'description': "Exploring the environment bugs: Missing addition of noise in action sampling",
        'category': "xploring the environment bugs",
    }, # 18th bug
    {
        'relative_path': "/stable_baselines3/ppo/ppo.py",
        'lineno': -1,  # no use
        'original_lines': ['th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)'],
        'injected_lines': ['#th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)'],
        'realife_bug': False,
        'description': "Training Bugs: Disabling gradient clipping during backpropagation",
        'category': "Training Bugs",
    }, # 19th bug
    {
        'relative_path': "/stable_baselines3/ppo/ppo.py",
        'lineno': -1,  # no use
        'original_lines': ['loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss'],
        'injected_lines': ['loss = policy_loss + self.ent_coef * entropy_loss + value_loss'],
        'realife_bug': False,
        'description': "Updating network Bugs: Incorrect loss calculation. ",
        'category': "Updating network Bugs",
    }, # 20th bug
    {
        'relative_path': "/stable_baselines3/ppo/ppo.py",
        'lineno': -1,  # no use
        'original_lines': ['self.policy.optimizer.zero_grad()', 'th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)'],
        'injected_lines': ['#self.policy.optimizer.zero_grad()', '#th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)'],
        'realife_bug': False,
        'description': "Training Bugs: Neglecting to reset gradients before backpropagation",
        'category': "Training Bugs",
    }, # 21th bug
    {
        'relative_path': "/stable_baselines3/ppo/ppo.py",
        'lineno': -1,  # no use
        'original_lines': ['loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss'],
        'injected_lines': ['loss = self.ent_coef * entropy_loss + self.vf_coef * value_loss'],
        'realife_bug': False,
        'description': "Updating network Bugs: Incorrect loss calculation (only use value loss as loss)",
        'category': "Updating network Bugs",
    }, # 22th bug
    {
        'relative_path': "/stable_baselines3/ppo/ppo.py",
        'lineno': -1,  # no use
        'original_lines': ['loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss'],
        'injected_lines': ['loss = policy_loss + self.ent_coef * entropy_loss'],
        'realife_bug': False,
        'description': "Updating network Bugs: Incorrect loss calculation (only use policy loss as loss)",
        'category': "Updating network Bugs",
    }, # 23th bug
    {
        'relative_path': "/stable_baselines3/ppo/ppo.py",
        'lineno': -1,  # no use
        'original_lines': ['advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)'],
        'injected_lines': ['advantages = (advantages - advantages.mean())'],
        'realife_bug': False,
        'description': "Updating network Bugs: Incorrect advantage normalization",
        'category': "Updating network Bugs",
    }, # 24th bug
    {
        'relative_path': "/stable_baselines3/ppo/ppo.py",
        'lineno': -1,  # no use
        'original_lines': ['advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)'],
        'injected_lines': ['advantages = (advantages) / (advantages.std() + 1e-8)'],
        'realife_bug': False,
        'description': "Updating network Bugs: Incorrect advantage normalization",
        'category': "Updating network Bugs",
    }, # 25th bug
    {
        'relative_path': "/stable_baselines3/ppo/ppo.py",
        'lineno': -1,  # no use
        'original_lines': ['ratio = th.exp(log_prob - rollout_data.old_log_prob)'],
        'injected_lines': ['ratio = th.exp(rollout_data.old_log_prob- log_prob)'],
        'realife_bug': False,
        'description': "Updating network Bugs: Incorrect similarity ratio calculation between old and new policy",
        'category': "Updating network Bugs",
    }, # 26th bug
    {
        'relative_path': "/stable_baselines3/ppo/ppo.py",
        'lineno': -1,  # no use
        'original_lines': ['policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()'],
        'injected_lines': ['policy_loss = -th.max(policy_loss_1, policy_loss_2).mean()'],
        'realife_bug': False,
        'description': "Updating network Bugs: Incorrect policy loss calculation(change min to max)",
        'category': "Updating network Bugs",
    }, # 27th bug
    {
        'relative_path': "/stable_baselines3/td3/td3.py",
        'lineno': -1,  # no use
        'original_lines': ['next_actions = (self.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)'],
        'injected_lines': ['next_actions = (self.actor_target(replay_data.next_observations) + noise)'],
        'realife_bug': False,
        'description': "Updating network Bugs: Disabling action clamp in training",
        'category': "Updating network Bugs",
    }, # 28th bug
    {
        'relative_path': "/stable_baselines3/td3/td3.py",
        'lineno': -1,  # no use
        'original_lines': ['next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)'],
        'injected_lines': ['next_q_values, _ = th.max(next_q_values, dim=1, keepdim=True)'],
        'realife_bug': False,
        'description': "Updating network Bugs: Incorrectnext q value calculation(change min to max)",
        'category': "Updating network Bugs",
    }, # 29th bug
    {
        'relative_path': "/stable_baselines3/td3/td3.py",
        'lineno': -1,  # no use
        'original_lines': ['target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values'],
        'injected_lines': ['target_q_values = replay_data.rewards'],
        'realife_bug': False,
        'description': "Updating network Bugs: Incorrect target q values calculation(only use reward as q value)",
        'category': "Updating network Bugs",
    }, # 30th bug
    {
        'relative_path': "/stable_baselines3/td3/td3.py",
        'lineno': -1,  # no use
        'original_lines': ['target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values'],
        'injected_lines': ['target_q_values = replay_data.rewards + self.gamma * next_q_values'],
        'realife_bug': False,
        'description': "Updating network Bugs: Incorrect target q values calculation(remove replay_data.dones)",
        'category': "Updating network Bugs",
    }, # 31th bug
    {
        'relative_path': "/stable_baselines3/td3/td3.py",
        'lineno': -1,  # no use
        'original_lines': ['target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values'],
        'injected_lines': ['target_q_values = replay_data.rewards + (1 - replay_data.dones) * next_q_values'],
        'realife_bug': False,
        'description': "Updating network Bugs: Incorrect target q values calculation(remove forget factor γ)",
        'category': "Updating network Bugs",
    }, # 32th bug
    {
        'relative_path': "/stable_baselines3/td3/td3.py",
        'lineno': -1,  # no use
        'original_lines': ['self.critic.optimizer.zero_grad()'],
        'injected_lines': ['#self.critic.optimizer.zero_grad()'],
        'realife_bug': False,
        'description': "Training Bugs: Neglecting to reset gradients before backpropagation",
        'category': "Training Bugs",
    }, # 33th bug
    {
        'relative_path': "/stable_baselines3/td3/td3.py",
        'lineno': -1,  # no use
        'original_lines': ['self.critic.optimizer.step()'],
        'injected_lines': ['#self.critic.optimizer.step()'],
        'realife_bug': False,
        'description': "Training Bugs: Neglecting to step gradients in backpropagation",
        'category': "Training Bugs",
    }, # 34th bug
    {
        'relative_path': "/stable_baselines3/td3/td3.py",
        'lineno': -1,  # no use
        'original_lines': ['if self._n_updates %% self.policy_delay == 0:'],
        'injected_lines': ['if self._n_updates %% 1 == 0:'],
        'realife_bug': False,
        'description': "Updating network Bugs: Disabling delay update",
        'category': "Updating network Bugs",
    }, # 35th bug
    {
        'relative_path': "/stable_baselines3/td3/td3.py",
        'lineno': -1,  # no use
        'original_lines': ['self.actor.optimizer.zero_grad()'],
        'injected_lines': ['#self.actor.optimizer.zero_grad()'],
        'realife_bug': False,
        'description': "Training Bugs: Neglecting to reset gradients before backpropagation",
        'category': "Training Bugs",
    }, # 36th bug
    {
        'relative_path': "/stable_baselines3/td3/td3.py",
        'lineno': -1,  # no use
        'original_lines': ['self.actor.optimizer.step()'],
        'injected_lines': ['#self.actor.optimizer.step()'],
        'realife_bug': False,
        'description': "Training Bugs: Neglecting to step gradients in backpropagation",
        'category': "Training Bugs",
    }, # 37th bug
    {
        'relative_path': "/stable_baselines3/td3/td3.py",
        'lineno': -1,  # no use
        'original_lines': ['polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)', 'polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)'],
        'injected_lines': ['polyak_update(self.critic.parameters(), self.critic_target.parameters(), 0)', 'polyak_update(self.actor.parameters(), self.actor_target.parameters(), 0)'],
        'realife_bug': False,
        'description': "Updating network Bugs: Disabling soft update",
        'category': "Updating network Bugs",
    }, # 38th bug
    {
        'relative_path': "/stable_baselines3/td3/td3.py",
        'lineno': -1,  # no use
        'original_lines': ['polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)', 'polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)'],
        'injected_lines': ['#polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)', '#polyak_update(self.actor.parameters(), self.actor_target.parameters(), self.tau)'],
        'realife_bug': False,
        'description': "Updating network Bugs: Missing soft update on target network",
        'category': "Updating network Bugs",
    }, # 39th bug
    {
        'relative_path': "/stable_baselines3/td3/policies.py",
        'lineno': -1,  # no use
        'original_lines': ['net_arch = [400, 300]'],
        'injected_lines': ['net_arch = [5, 5]'],
        'realife_bug': False,
        'description': "Model Bugs: Overly simple network structure",
        'category': "Model Bugs",
    }, # 40th bug
    {
        'relative_path': "/stable_baselines3/a2c/a2c.py",
        'lineno': -1,  # no use
        'original_lines': ['actions = rollout_data.actions'],
        'injected_lines': ['actions = rollout_data.actions.long()'],
        'realife_bug': False,
        'description': "Tensor and inputs bugs: In correct Action dtyoe(use int instead of float)",
        'category': "Tensor and inputs bugs",
    }, # 41th bug
    {
        'relative_path': "/stable_baselines3/ppo/ppo.py",
        'lineno': -1,  # no use
        'original_lines': ['actions = rollout_data.actions'],
        'injected_lines': ['actions = rollout_data.actions.long()'],
        'realife_bug': False,
        'description': "Tensor and inputs bugs: In correct Action dtyoe(use int instead of float)",
        'category': "Tensor and inputs bugs",
    }, # 42th bug
    {
        'relative_path': "/stable_baselines3/td3/td3.py",
        'lineno': -1,  # no use
        'original_lines': ['current_q_values = self.critic(replay_data.observations, replay_data.actions)'],
        'injected_lines': ['current_q_values = self.critic(replay_data.observations, replay_data.actions.long())'],
        'realife_bug': False,
        'description': "Tensor and inputs bugs: In correct Action dtyoe(use int instead of float)",
        'category': "Tensor and inputs bugs",
    }, # 43th bug
]


def check_injection_validation(bug_id_ist):
    return True



def inject_bugs(config, bug_id_list):
    if not check_injection_validation(bug_id_list):
        return "bug version invalid!!"
    
    for bug_id in bug_id_list:
        temp_bug = bug_group[bug_id]

        temp_bug_path = config['root_dir'] + temp_bug['relative_path']

        with open(temp_bug_path, 'r+') as relative_file:
            relative_file_data = relative_file.read()
            print(relative_file_data)

            for bug_line_index in range(len(temp_bug['original_lines'])):
                relative_file_data = relative_file_data.replace(
                    temp_bug['original_lines'][bug_line_index],
                    temp_bug['injected_lines'][bug_line_index]
                )

            relative_file.seek(0)
            relative_file.write(relative_file_data)
            relative_file.truncate()
            
        
def recover_project(config):
    main_folder = config['root_dir']
    archive_folder = os.path.join(main_folder, 'archived_code')

    if not os.path.exists(archive_folder):
        print(f"Archive folder not found: {archive_folder}")
        return

    subfolders = [f for f in os.listdir(archive_folder) if os.path.isdir(os.path.join(archive_folder, f))]

    for subfolder in subfolders:
        archive_subfolder_path = os.path.join(archive_folder, subfolder)
        main_subfolder_path = os.path.join(main_folder, subfolder)
        
        if os.path.exists(main_subfolder_path):
            shutil.rmtree(main_subfolder_path)

        shutil.copytree(archive_subfolder_path, main_subfolder_path)
        
        
def cover_then_inject_bugs(bug_list):
    config=parserConfig()
    recover_project(config)
    inject_bugs(config=config, bug_id_list=bug_list)

    # pip reinstall SB3 repository
    os.chdir(config['root_dir'])
    os.system('pip install -e .')

def cover_then_inject_bugs_without_pip(bug_list):
    config=parserConfig()
    recover_project(config)
    inject_bugs(config=config, bug_id_list=bug_list)

    # # pip reinstall SB3 repository
    # os.chdir(config['root_dir'])
    # os.system('pip install -e .')