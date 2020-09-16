from collections import deque
import math
import pygame
import numpy as np

WIDTH = 500
HEIGHT = 500

direction_mapping = {pygame.K_UP : 0,
             pygame.K_LEFT : 1,
             pygame.K_DOWN : 2,
             pygame.K_RIGHT : 3}


# Replay memory settings
BUFFER_LENGTH = 50_000
MIN_TRAIN_BUFFER_LENGTH = 10_000
BATCH_SIZE = 64

# Enviroment settings
EPISODES = 20_000

class Enviroment():
    
    def __init__(self, snake, food):
        self.snake = snake
        self.food = food
        self.buffer = deque(maxlen=BUFFER_LENGTH)
        self.last_distance = self.get_apple_distance()
        self.mem_count = 0
        self.average_reward = 0

    def step(self, current_state, action):
       
        # Do the new move and calculate the reward
        new_state = None

        movement = self.process_action(action)
        self.snake.move_snake(movement)
        self.snake.direction = movement

        reward, done = self.get_reward()
        if not done:
            new_state = self.get_state()
            
        self.last_distance = self.get_apple_distance()
        self.buffer.append((current_state, action, reward, new_state, done))
        
        # Increase the buffer size
        self.mem_count += 1
        self.mem_count = min(self.mem_count, BUFFER_LENGTH)
            
        return new_state, reward, done

    def process_action(self, action):
        
        last_direction = direction_mapping[self.snake.direction]

        if action == 2:
            return self.snake.direction # hold the last direction
        elif last_direction == 0 or last_direction == 2: # if moving up or down:
            if action == 0: # 0 NN output go left
                return pygame.K_LEFT
            else: # 1 NN output go right
                return pygame.K_RIGHT
        else: # if moving left or right
            if action == 0: # 0 NN output go up
                return pygame.K_UP
            else: # 1 NN output go down
                return pygame.K_DOWN

    def get_batch(self):

        if len(self.buffer) < MIN_TRAIN_BUFFER_LENGTH:
            return

        batch = np.random.choice(self.mem_count, BATCH_SIZE, replace=False)
        return np.array(self.buffer)[batch]

    def get_state(self):
        
        # Get the snake direction
        snake_dir =  direction_mapping[self.snake.direction]
        direction = np.zeros(4)
        direction[snake_dir] = 1
        # Get the apple position relative to the snake
        apple = np.zeros(5)
        apple[0] = 1 if self.snake.y > self.food.y else 0
        apple[1] = 1 if self.snake.x > self.food.x else 0
        apple[2] = 1 if self.snake.y < self.food.y else 0
        apple[3] = 1 if self.snake.x < self.food.x else 0
        apple[4] = self.get_apple_distance() / WIDTH

        # Get the obstacles (body or wall) relative to the snake
        obstacle = np.zeros(12)
        # Wall obstacles
        obstacle[0] = self.snake.y / WIDTH # snake is upper part of the screen
        obstacle[1] = self.snake.x / WIDTH # snake is in the left part of the screen
        obstacle[2] = (HEIGHT - self.snake.y) / WIDTH # snake is in the lower part of the screen
        obstacle[3] = (WIDTH - self.snake.x) / WIDTH # snake is in the right part of the screen
        # Get the obstacles (body or wall) relative to the snake
        obstacle[4] = 1 if self.snake.y < (HEIGHT / 2) else 0 # snake is upper part of the screen
        obstacle[5] = 1 if self.snake.x < (WIDTH / 2) else 0 # snake is in the left part of the screen
        obstacle[6] = 1 if self.snake.y > (HEIGHT / 2) else 0 # snake is in the lower part of the screen
        obstacle[7] = 1 if self.snake.x > (WIDTH / 2) else 0 # snake is in the right part of the screen
        
        # Body obstacles
        if len(self.snake.snake_list) > 3:
            for block in self.snake.snake_list:
                if block[0] == self.snake.x and self.snake.y < block[1]:
                    obstacle[8] = 1 # there is a snake block above the head
                if block[1] == self.snake.y and block[0] < self.snake.x:
                    obstacle[9] = 1 # there is a snake block in the left of the head
                if block[0] == self.snake.x and self.snake.y > block[1]:
                    obstacle[10] = 1 # there is snake block below the head
                if block[1] == self.snake.y and block[0] > self.snake.x:
                    obstacle[11] = 1 # there is a snake block in the right of the head
        
        state = np.concatenate((direction, apple, obstacle), 0)

        return state  

    def get_reward(self):
        
        done = self.snake.is_dead()
        current_distance = self.get_apple_distance()
        reward = -2 if current_distance >= self.last_distance else 1
        reward = -100 if done else reward # if snake dies -100 points
        reward = 25 if self.snake.has_eaten else reward # + 10 points if the snake has eaten

        return reward, done

    def get_apple_distance(self):
        return ((self.food.x - self.snake.x)**2 + (self.food.y - self.snake.y)**2) ** (1/2)
