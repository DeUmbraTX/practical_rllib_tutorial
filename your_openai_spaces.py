"""
You pick a chicken to befriend and then get the location of that chicken

A couple of articles on hierarchical reinforcement learning
https://thegradient.pub/the-promise-of-hierarchical-reinforcement-learning/
https://towardsdatascience.com/hierarchical-reinforcement-learning-a2cca9b76097

"""
import numpy as np

from gym import spaces

from your_constants import NUM_OCEAN_FEATURES, NUM_CHICKENS, NUM_DIRECTIONS


#************************************************************
# Observation spaces
#************************************************************

# 10 chickens follow the OCEAN model of personality
# I think it has no less validity here than in many applications
chicken_ocean_space = spaces.Box(
            shape=(NUM_CHICKENS,NUM_OCEAN_FEATURES),
            dtype=np.float,
            low=0.0,
            high=1.0)

chicken_position_space = spaces.Box(
            shape=(NUM_CHICKENS,2),
            dtype=np.float,
            low=0.0,
            high=1.0)

robot_position_space = spaces.Box(
            shape=(2,),
            dtype=np.float,
            low=0.0,
            high=1.0)

robot_ocean_space = spaces.Box(
            shape=(NUM_OCEAN_FEATURES,),
            dtype=np.float,
            low=0.0,
            high=1.0)

high_level_obs_space = spaces.Dict({
        'chicken_oceans': chicken_ocean_space,
        'chicken_positions': chicken_position_space,
        'robot_ocean': robot_ocean_space,
        'robot_position': robot_position_space,
})

# At the low level, you don't care about OCEAN, you've already chosen your chicken
low_level_obs_space = spaces.Dict({
        'robot_position': robot_position_space,
        'chicken_positions': chicken_position_space,
})


#************************************************************
# Action spaces
#************************************************************

# Which chicken to choose
high_level_action_space = spaces.Discrete(NUM_CHICKENS)

# go 8 directions N, NE, E, SE, S, SW, W, NW
low_level_action_space = spaces.Discrete(NUM_DIRECTIONS)
