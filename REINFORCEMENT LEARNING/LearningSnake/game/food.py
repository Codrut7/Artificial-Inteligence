import pygame
import random
import util

class Food():
    def __init__(self, surface):

        self.x = None
        self.y = None
        self.surface = surface
    
    def generate_food(self, snake):
        
        surface_width, surface_height = self.surface.get_size()

        if snake is None:
            print('Genetic alghoritm shit right here ??')
        else:
            is_food_on_snake = True
            
            while is_food_on_snake:
                is_food_on_snake = False
                self.x = round(random.randrange(0, surface_width - util.SNAKE_SIZE) / 10.0) * 10.0
                self.y = round(random.randrange(0, surface_width - util.SNAKE_SIZE) / 10.0) * 10.0
                for block in snake.snake_list:
                    is_food_on_snake = True if self.x == block[0] and self.y == block[1] else is_food_on_snake
            
            
    def draw_food(self):
        pygame.draw.rect(self.surface, util.BLACK, [self.x, self.y, util.FOOD_SIZE, util.FOOD_SIZE])

