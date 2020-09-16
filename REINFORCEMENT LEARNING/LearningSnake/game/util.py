import pygame

SNAKE_SPEED = 55
SNAKE_SIZE = FOOD_SIZE = 10
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

KEY_MAPPING = {0 :pygame.K_UP,
             1 : pygame.K_LEFT,
             2 :pygame.K_DOWN,
             3 :pygame.K_RIGHT}

OPPOSITE = {0 : 2,
            1 : 3,
            2 : 0,
            3 : 1}
