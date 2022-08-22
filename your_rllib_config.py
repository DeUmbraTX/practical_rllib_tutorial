from typing import Dict

from ray.rllib.policy.policy import PolicySpec
from your_rllib_environment import YourEnvironment

from your_openai_spaces import high_level_obs_space, high_level_action_space, \
    low_level_obs_space,low_level_action_space


def policy_map_fn(agent_id: str, _episode=None, _worker=None, **_kwargs) -> str:
    """
    Maps agent_id to policy_id
    """
    if 'high' in agent_id:
        return 'high_level_policy'
    elif 'low' in agent_id:
        return 'low_level_policy'
    else:
        raise RuntimeError(f'Invalid agent_id: {agent_id}')


def get_multiagent_policies() -> Dict[str,PolicySpec]:
    policies: Dict[str,PolicySpec] = {}  # policy_id to policy_spec

    policies['high_level_policy'] = PolicySpec(
                policy_class=None, # use default in trainer, or could be YourHighLevelPolicy
                observation_space=high_level_obs_space,
                action_space=high_level_action_space,
                config={}
    )

    policies['low_level_policy'] = PolicySpec(
        policy_class=None,  # use default in trainer, or could be YourLowLevelPolicy
        observation_space=low_level_obs_space,
        action_space=low_level_action_space,
        config={}
    )

    return policies


policies = get_multiagent_policies()

# see https://github.com/ray-project/ray/blob/releases/1.10.0/rllib/agents/trainer.py
cust_config = {
        #"env": "logan_env",
        "simple_optimizer": True,
        "ignore_worker_failures": True,
        "batch_mode": "complete_episodes",
        "env": YourEnvironment,
        "env_config": {
            "is_use_visualization": False,
        },
        "framework": "torch",
        "multiagent": {
            "policies": policies,
            "policy_mapping_fn": policy_map_fn,
            "policies_to_train": list(policies.keys()),
            "count_steps_by": "env_steps",
            "observation_fn": None,
            "replay_mode": "independent",
            "policy_map_cache": None,
            "policy_map_capacity": 100,
        },
    }
