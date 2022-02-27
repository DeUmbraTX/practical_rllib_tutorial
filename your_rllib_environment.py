from typing import Dict

from ray.rllib import MultiAgentEnv
from ray.rllib.env import EnvContext

from your_target_system import YourTargetSystem
from your_constants import STEP_REWARD, ARRIVAL_REWARD, TIMEOUT, TIMEOUT_REWARD
from pygame_visualization.env_visualization import Visualization



# MultiAgentEvent subclass of gym.Env
class YourEnvironment(MultiAgentEnv):
    def __init__(self, config:EnvContext):
        self.is_use_visualization = config['is_use_visualization']
        if self.is_use_visualization:
            self.visualization = Visualization()
        else:
            self.visualization = None
        self.target_system = YourTargetSystem()
        self.observation_space = None  # is_atari bug


    def reset(self):
        # Start a new chicken yard
        self.target_system.initialize_yard()
        yard = self.target_system.get_yard()

        obs_robot_1_high = {'chicken_oceans': yard.chicken_ocean,
                       'chicken_positions': yard.chicken_positions,
                       'robot_ocean': yard.robot_1_ocean,
                       'robot_position': yard.robot_1_position}
        obs_robot_2_high = {'chicken_oceans': yard.chicken_ocean,
                       'chicken_positions': yard.chicken_positions,
                       'robot_ocean': yard.robot_2_ocean,
                       'robot_position': yard.robot_2_position}

        # Because only high-level robots returned, only their policies will be called
        return {'robot_1_high': obs_robot_1_high, 'robot_2_high': obs_robot_2_high}


    def step(self, action_dict):
        obs, rew, done, info = {}, {}, {}, {}

        #******************************************************************************
        # Part 1: get actions from RLlib policies and execute in your environment
        # *****************************************************************************
        # first timestep, we get actions for which chicken to befriend for both robots
        if 'robot_1_high' in action_dict:
            assert 'robot_2_high' in action_dict
            self.target_system.robot_1.set_chicken_choice(action_dict['robot_1_high'])
            self.target_system.robot_2.set_chicken_choice(action_dict['robot_2_high'])
        else:
            # other timesteps, we do low-level actions, once a robot reaches its chicken,
            # it will no longer get actions
            if 'robot_1_low' in action_dict:
                self.target_system.robot_1.move(action_dict['robot_1_low'])
            if 'robot_2_low' in action_dict:
                self.target_system.robot_2.move(action_dict['robot_2_low'])
            # but at least one should be in there
            assert 'robot_1_low' in action_dict or 'robot_2_low' in action_dict

        # ******************************************************************************
        # Part 2: get state of your environment after taking actions
        # ******************************************************************************
        yard = self.target_system.get_yard()

        # ******************************************************************************
        # Part 3: pass relevant information to RLlib
        # ******************************************************************************

        def is_robot_still_active(robot_id: str) -> bool:
            if robot_id == '1':
                return 'robot_1_high' in action_dict or 'robot_1_low' in action_dict
            elif robot_id == '2':
                return 'robot_2_high' in action_dict or 'robot_2_low' in action_dict
            else:
                raise RuntimeError(f'Invalid robot id: {robot_id}')

        # check if robots are still active; if so, pass state for next low-level action

        if is_robot_still_active('1'):
            obs['robot_1_low'] = {'robot_position': yard.robot_1_position,
                               'chicken_positions': yard.chicken_positions}
            if yard.timestep > TIMEOUT:
                rew['robot_1_high'] = TIMEOUT_REWARD
                rew['robot_1_low'] = TIMEOUT_REWARD
                done['robot_1_high'] = True
                done['robot_1_low'] = True
                # probably not really necessary
                obs['robot_1_high'] = {'chicken_oceans': yard.chicken_ocean,
                                       'chicken_positions': yard.chicken_positions,
                                       'robot_ocean': yard.robot_1_ocean,
                                       'robot_position': yard.robot_1_position}
            elif self.target_system.is_robot_at_chicken(self.target_system.robot_1):
                # if it made it the robot is all done
                rew['robot_1_high'] = self.target_system.get_chicken_reward(self.target_system.robot_1)
                rew['robot_1_low'] = ARRIVAL_REWARD
                done['robot_1_high'] = True
                done['robot_1_low'] = True
                # probably not really necessary
                obs['robot_1_high'] = {'chicken_oceans': yard.chicken_ocean,
                                       'chicken_positions': yard.chicken_positions,
                                       'robot_ocean': yard.robot_1_ocean,
                                       'robot_position': yard.robot_1_position}
            else:
                # if the robot didn't make it to the chicken, keep going onl low-level policy
                # and don't return anything for the high-level policy
                rew['robot_1_low'] = STEP_REWARD
                done['robot_1_low'] = False

        if is_robot_still_active('2'):
            obs['robot_2_low'] = {'robot_position': yard.robot_2_position,
                                  'chicken_positions': yard.chicken_positions}
            if yard.timestep > TIMEOUT:
                rew['robot_2_high'] = TIMEOUT_REWARD
                rew['robot_2_low'] = TIMEOUT_REWARD
                done['robot_2_high'] = True
                done['robot_2_low'] = True
                # probably not really necessary
                obs['robot_2_high'] = {'chicken_oceans': yard.chicken_ocean,
                                       'chicken_positions': yard.chicken_positions,
                                       'robot_ocean': yard.robot_2_ocean,
                                       'robot_position': yard.robot_2_position}
            elif self.target_system.is_robot_at_chicken(self.target_system.robot_2):
                # if it made it the robot is all done
                rew['robot_2_high'] = self.target_system.get_chicken_reward(self.target_system.robot_2)
                rew['robot_2_low'] = ARRIVAL_REWARD
                done['robot_2_high'] = True
                done['robot_2_low'] = True
                # probably not really necessary
                obs['robot_2_high'] = {'chicken_oceans': yard.chicken_ocean,
                                       'chicken_positions': yard.chicken_positions,
                                       'robot_ocean': yard.robot_2_ocean,
                                       'robot_position': yard.robot_2_position}
            else:
                # if the robot didn't make it to the chicken, keep going onl low-level policy
                # and don't return anything for the high-level policy
                rew['robot_2_low'] = STEP_REWARD
                done['robot_2_low'] = False

        def is_all_done(done: Dict) -> bool:
            for key, val in done.items():
                if not val:
                    return False
            return True
        done['__all__'] = is_all_done(done)  # say if episode is over

        # ignore info
        return obs, rew, done, info


    def close(self):
        pass

    def render(self, mode='human'):
        self.visualization.render(self.target_system)
