import os

from ray.rllib import Policy
from ray.rllib.env import EnvContext
from ray.rllib.agents.ppo import ppo, PPOTorchPolicy
from ray.rllib.models.tf.complex_input_net import ComplexInputNetwork

from your_constants import YOUR_ROOT
from your_rllib_trainer import config
from your_rllib_config import cust_config


config.update(cust_config)
config['num_workers'] = 0

env_config = {'is_use_visualization': True}
env_config = EnvContext(env_config,worker_index=1)

from your_rllib_environment import YourEnvironment
trainer = ppo.PPOTrainer(config, env=YourEnvironment)

# Change these for your run
#run = 'YourTrainer_YourEnvironment_deec5_00000_0_2022-03-18_20-42-43'
run = 'custom_execution_plan_2022-02-24'
checkpoint = 'checkpoint_000500/checkpoint-500'

restore_point = os.path.join(YOUR_ROOT, run, checkpoint)
trainer.restore(restore_point)

print("************************ high level policy **************************************")
# Note the output size of 10
policy: Policy = trainer.get_policy('high_level_policy')
# https://github.com/ray-project/ray/blob/releases/1.10.0/rllib/models/torch/complex_input_net.py
model: ComplexInputNetwork = policy.model
for m in model.variables():
    print(m.shape)

print("************************ low level policy **************************************")
# Note the output size of 8
policy: Policy = trainer.get_policy('low_level_policy')
model: ComplexInputNetwork = policy.model
for m in model.variables():
    print(m.shape)
