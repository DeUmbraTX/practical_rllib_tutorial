"""
This is how you actually use the pollicy
"""
from typing import Dict
import os
import time

from ray.rllib.env import EnvContext
from ray.rllib.agents.ppo import ppo

from your_constants import YOUR_ROOT
from your_rllib_trainer import config
from your_rllib_config import cust_config


config.update(cust_config)
config['num_workers'] = 0

env_config = {'my_config_val': 'bratwurst'}  # just dummy, fill in your param
env_config = EnvContext(env_config,worker_index=1)

from your_rllib_environment import YourEnvironment
trainer = ppo.PPOTrainer(config, env=YourEnvironment)

run = 'YourTrainer_YourEnvironment_1e97d_00000_0_2022-02-20_13-13-00'
checkpoint = 'checkpoint_000500/checkpoint-500'

restore_point = os.path.join(YOUR_ROOT, run, checkpoint)
trainer.restore(restore_point)

env = YourEnvironment(env_config)

obs = env.reset()

# Note that they both use the same policy
robot_1_high_action = trainer.compute_single_action(obs['robot_1_high'], policy_id='high_level_policy')
robot_2_high_action = trainer.compute_single_action(obs['robot_2_high'], policy_id='high_level_policy')

action_dict = {'robot_1_high': robot_1_high_action,
                'robot_2_high': robot_2_high_action}

obs, rew, done, info = env.step(action_dict)

env.render()

def is_all_done(done: Dict) -> bool:
    for key, val in done.items():
        if not val:
            return False
    return True

while not is_all_done(done):
    action_dict = {}
    assert 'robot_1_low' in obs or 'robot_2_low' in obs
    if 'robot_1_low' in obs and not done['robot_1_low']:
        action_dict['robot_1_low'] = trainer.compute_single_action(obs['robot_1_low'], policy_id='low_level_policy')
    if 'robot_2_low' in obs and not done['robot_2_low']:
        action_dict['robot_2_low'] = trainer.compute_single_action(obs['robot_2_low'], policy_id='low_level_policy')
    obs, rew, done, info = env.step(action_dict)
    print("Reward: ", rew)
    env.render()
    #time.sleep(1)

time.sleep(5)