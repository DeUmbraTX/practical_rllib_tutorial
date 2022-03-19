from typing import Optional
import numpy as np

from your_constants import NUM_OCEAN_FEATURES, X, Y, MOVE_AMOUNT


# go 8 directions N, NE, E, SE, S, SW, W, NW
N = 0
NE = 1
E = 2
SE = 3
S = 4
SW = 5
W = 6
NW = 7


class YourAutonomousAgent:
    def __init__(self):
        self.position: Optional[np.ndarray, None] = None
        self.ocean: Optional[np.ndarray, None] = None
        self.chicken_choice: Optional[int, None] = None

    def initialize(self) -> None:
        """
        Robot should perform introspection using OCEAN model and generate
        a random position.
        """
        self.position = np.random.rand(2,)
        self.ocean = np.random.rand(NUM_OCEAN_FEATURES, )
        self.ocean = self.ocean / np.linalg.norm(self.ocean)

    def get_position(self) -> np.ndarray:
        return self.position

    def get_ocean(self) -> np.ndarray:
        return self.ocean

    def set_chicken_choice(self, chicken: int) -> None:
        self.chicken_choice = chicken

    def move(self, direction: int) -> None:
        # 0,0 on top-left, for y down is positive
        if direction == N:
            self.position[X] += 0.0
            self.position[Y] += -MOVE_AMOUNT
        elif direction == NE:
            self.position[X] += MOVE_AMOUNT
            self.position[Y] += -MOVE_AMOUNT
        elif direction == E:
            self.position[X] += MOVE_AMOUNT
            self.position[Y] += 0.0
        elif direction == SE:
            self.position[X] += MOVE_AMOUNT
            self.position[Y] += MOVE_AMOUNT
        elif direction == S:
            self.position[X] += 0.0
            self.position[Y] += MOVE_AMOUNT
        elif direction == SW:
            self.position[X] += -MOVE_AMOUNT
            self.position[Y] += MOVE_AMOUNT
        elif direction == W:
            self.position[X] += -MOVE_AMOUNT
            self.position[Y] += 0.0
        elif direction == NW:
            self.position[X] += -MOVE_AMOUNT
            self.position[Y] += -MOVE_AMOUNT
        else:
            raise Exception(f'Invalid movement: {direction}')

        # keep the robots on the chicken field
        if self.position[X] > 1.0:
            self.position[X] = 1.0
        elif self.position[X] < 0.0:
            self.position[X] = 0.0
        if self.position[Y] > 1.0:
            self.position[Y] = 1.0
        elif self.position[Y] < 0.0:
            self.position[Y] = 0.0
