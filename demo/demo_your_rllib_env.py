"""
In a loop call the environmeht.
In reality, the environment will be called by RLlib, but doing it manually allows us to test this
part and see what is going on.
"""
from typing import Dict

import time
import random

from ray.rllib.env import EnvContext

from your_constants import NUM_CHICKENS, NUM_DIRECTIONS
from your_rllib_environment import YourEnvironment

env_config = {'is_use_visualization': True}
config = EnvContext(env_config,worker_index=1)


env = YourEnvironment(config)

env.reset()

action_dict = {'robot_1_high': random.choice(range(NUM_CHICKENS)),
                'robot_2_high': random.choice(range(NUM_CHICKENS))}

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
        action_dict['robot_1_low'] = random.choice(range(NUM_DIRECTIONS))
    if 'robot_2_low' in obs and not done['robot_2_low']:
        action_dict['robot_2_low'] = random.choice(range(NUM_DIRECTIONS))
    obs, rew, done, info = env.step(action_dict)
    print("Reward: ", rew)
    env.render()
    time.sleep(.1)

time.sleep(50)