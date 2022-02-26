import time
import sys, pygame

import random

def get_random_position():
    x = random.random() * 1000
    y = random.random() * 1000
    return x, y


pygame.init()

size = width, height = 1000,1000
speed = [2, 2]
background = 255, 255, 255

screen = pygame.display.set_mode(size)

# 0,0 is top left with y going down


robot = pygame.image.load("robot.png")
robot = pygame.transform.scale(robot, (50, 50))
robot_rect = robot.get_rect()
robot_rect.x = 50
robot_rect.y = 200

chicken = pygame.image.load("chicken.png")
chicken = pygame.transform.scale(chicken, (50, 50))
chicken_rect = chicken.get_rect()
chicken_rect.x = 100
chicken_rect.y = 300


while 1:
    for event in pygame.event.get():
        if event.type == pygame.QUIT: sys.exit()

    chicken_rect = chicken_rect.move(speed)

    x,y = get_random_position()
    robot_rect.x = x
    robot_rect.y = y


    screen.fill(background)
    screen.blit(robot, robot_rect)
    screen.blit(chicken, chicken_rect)
    pygame.display.flip()
    time.sleep(1)
