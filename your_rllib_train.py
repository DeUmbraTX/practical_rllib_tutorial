import ray
from ray import tune

from your_rllib_trainer import YourTrainer, config
from your_rllib_config import cust_config


config.update(cust_config)
config['num_workers'] = 0   # say what this is, for running locally?

# noinspection PyUnresolvedReferences
ray.init(local_mode=True)  # in local mode you can debug it

RUN_WITH_TUNE = True
NUM_ITERATIONS = 5  # Results in Tensorboard shown with 500 iterations (about an hour)

# Tune is the system for keeping track of all of the running jobs, originally for
# hyperparameter tuning
if RUN_WITH_TUNE:

    tune.registry.register_trainable("YourTrainer", YourTrainer)
    stop = {
            "training_iteration": NUM_ITERATIONS  # Each iteration is some number of episodes
        }
    results = tune.run("YourTrainer", stop=stop, config=config, verbose=1, checkpoint_freq=10)

    # You can probably just do PPO or DQN but we wanted to show how to customize
    #results = tune.run("PPO", stop=stop, config=config, verbose=1, checkpoint_freq=10)

    # Results at YOUR_ROOT/YourTrainer

else:
    from your_rllib_environment import YourEnvironment
    trainer = YourTrainer(config, env=YourEnvironment)

    # You can just do PPO or DQN but we wanted to show how to customize
    #from ray.rllib.agents.ppo import PPOTrainer
    #trainer = PPOTrainer(config, env=YourEnvironment)

    trainer.train()

    # Results at YOUR_ROOT/YourTrainer_YourEnvironment_YYYY_MM_DD_SS-NN-XXXXXXXXX