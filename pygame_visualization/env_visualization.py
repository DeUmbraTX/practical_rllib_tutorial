import os

import pygame

from your_target_system import YourTargetSystem
from your_constants import NUM_CHICKENS


size = width, height = 1000,1000
speed = [2, 2]
background = 255, 255, 255

# 0,0 is top left with y going down

icon_dir = os.path.dirname(__file__)

class Visualization:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode(size)
        self.chickens = []
        self.chicken_rects = []
        for i in range(NUM_CHICKENS):
            chicken = pygame.image.load(os.path.join(icon_dir,'chicken_public_domain.svg'))
            chicken = pygame.transform.scale(chicken, (50, 50))
            chicken_rect = chicken.get_rect()
            self.chickens.append(chicken)
            self.chicken_rects.append(chicken_rect)
        self.robots = []
        self.robot_rects = []
        for i in range(2):
            robot = pygame.image.load(os.path.join(icon_dir, 'robot_public_domain.svg'))
            robot = pygame.transform.scale(robot, (40, 70))
            robot_rect = robot.get_rect()
            self.robots.append(robot)
            self.robot_rects.append(robot_rect)

    def render(self, target_system: YourTargetSystem):
        pygame.event.get()
        self.screen.fill(background)
        for i in range(NUM_CHICKENS):
            pos = target_system.chicken_positions[i,:]
            x = int(pos[0] * 1000)
            y = int(pos[1] * 1000)
            self.chicken_rects[i].x = x
            self.chicken_rects[i].y = y
            self.screen.blit(self.chickens[i], self.chicken_rects[i])
        for i in range(2):
            if i == 0:
                robot = target_system.robot_1
            else:
                robot = target_system.robot_2
            pos = robot.get_position()
            x = int(pos[0] * 1000)
            y = int(pos[1] * 1000)
            self.robot_rects[i].x = x
            self.robot_rects[i].y = y
            self.screen.blit(self.robots[i], self.robot_rects[i])
        pygame.display.flip()
