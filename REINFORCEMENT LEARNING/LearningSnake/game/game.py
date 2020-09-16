import pygame
import snake
import food
import util
import sys
import time
import numpy as np
sys.path.insert(0, r"D:\Projects\Reinforcement Learning\Learningsnake\model")
import enviroment
import agent
# Init the game library
pygame.init()

# Display the screen and update any changes
surface = pygame.display.set_mode((500, 500))
surface.fill(util.WHITE)
pygame.display.set_caption('snk')
pygame.display.update()

clock = pygame.time.Clock()

snk = snake.Snake(surface)
food = food.Food(surface)
food.generate_food(snk)

game_over = False
last_key = pygame.K_END
env = enviroment.Enviroment(snk, food)
ag = agent.DQLAgent(21, 3, env)

human = False
snk.direction = util.KEY_MAPPING[np.random.randint(0, 4)] if not human else None

idx = 0
update_counter = 0

state = env.get_state()

while not game_over:
    if human:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN and util.OPPOSITE[last_key] != event.key:
                last_key = event.key
                snk.direction = last_key
                env.step(last_key)

            if event.type==pygame.QUIT:
                game_over=True
    
        #print(env.get_batch())
        env.step(last_key)
        food_eaten = snk.eat_food(food, last_key)
        food.generate_food(snk) if food_eaten else None
        game_over = snk.is_dead()
        snk.draw_snk()
        food.draw_food()
    else:
        #Handle the events for window to work
        pygame.event.get()

        # snake goes brr
        action = ag.get_action(state)
        new_state, reward, done = env.step(state, action)
        env.average_reward += reward
        
        # Snake control commands
        food_eaten = snk.eat_food(food)
        food.generate_food(snk) if food_eaten else None
        snk.draw_snake()
        food.draw_food()

        if new_state is None:
            snk = snake.Snake(surface)
            snk.direction = util.KEY_MAPPING[np.random.randint(0, 4)]
            env.snake = snk
            food.generate_food(snk)
            ag.decay_epsilon()
            update_counter += 1

        ag.train(update_counter)
        
        if idx % 99 == 0:
            print(env.average_reward)
            env.average_reward = 0
        
        idx += 1
        state = new_state if new_state is not None else env.get_state()

    pygame.display.update()
    clock.tick(util.SNAKE_SPEED)
# Close the game library
pygame.quit()