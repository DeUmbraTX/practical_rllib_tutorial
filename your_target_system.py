"""
This is what you want to opimize.

Optimizing this system is the whole point of what we are doing.
"""
from typing import Optional, Union
import numpy as np
from dataclasses import dataclass

from your_autonomous_agent import YourAutonomousAgent
from your_constants import NUM_OCEAN_FEATURES, NUM_CHICKENS, CLOSENESS_THRESHOLD


@dataclass
class YardState:
    chicken_ocean: np.ndarray
    chicken_positions: np.ndarray
    robot_1_position: np.ndarray
    robot_1_ocean: np.ndarray
    robot_2_position: np.ndarray
    robot_2_ocean: np.ndarray
    timestep: int


class YourTargetSystem:
    def __init__(self):
        self.chicken_ocean: Optional[np.ndarray, None] = None
        self.chicken_positions: Optional[np.ndarray, None] = None
        self.robot_1 = YourAutonomousAgent()
        self.robot_2 = YourAutonomousAgent()
        self.timestep: Optional[int, None] = None

    def initialize_yard(self) -> None:
        """
        Get a new chicken yard. For each of the NUM_CHICKEN chickens, compute their
        positions randomly and compute the results of phychological testing
        using the OCEAN model.
        """
        self.chicken_positions = np.random.rand(NUM_CHICKENS, 2)
        self.chicken_ocean = np.random.rand(NUM_CHICKENS, NUM_OCEAN_FEATURES)
        # normalize
        for i in range(NUM_CHICKENS):
            self.chicken_ocean[i,:] = self.chicken_ocean[i,:] / np.linalg.norm(self.chicken_ocean[i,:])
        self.robot_1.initialize()
        self.robot_2.initialize()
        self.timestep = 0

    def get_yard(self) -> YardState:
        self.timestep += 1
        return YardState(
            chicken_ocean=self.chicken_ocean,
            chicken_positions=self.chicken_positions,
            robot_1_position=self.robot_1.get_position(),
            robot_1_ocean=self.robot_1.get_ocean(),
            robot_2_position=self.robot_2.get_position(),
            robot_2_ocean=self.robot_2.get_ocean(),
            timestep = self.timestep
        )

    def get_chicken_reward(self, robot: YourAutonomousAgent) -> float:
        chicken_index = self.get_chicken_index(robot)
        assert not chicken_index is None
        # reward based on how compatable they are
        reward = float(np.dot(robot.get_ocean(),self.chicken_ocean[chicken_index,:]))
        return reward

    def get_chicken_index(self, robot: YourAutonomousAgent) -> Union[int,None]:
        def is_at(pos_1: np.ndarray, pos_2: np.ndarray) -> bool:
            dist = np.linalg.norm(pos_1 - pos_2)
            return dist < CLOSENESS_THRESHOLD
        robot_pos = robot.get_position()
        for i in range(NUM_CHICKENS):
            if is_at(robot_pos, self.chicken_positions[i, :]):
                return i
        return None

    def is_robot_at_chicken(self, robot: YourAutonomousAgent) -> bool:
        chicken_index = self.get_chicken_index(robot)
        if chicken_index is None:
            return False
        else:
            return True

    def is_done(self) -> bool:
        # done if both robots are at chickens
        raise NotImplementedError
