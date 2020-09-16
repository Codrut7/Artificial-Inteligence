import pygame
import util
import copy

class Snake():
    """
        Snake object used as the RL agent.
    """
    def __init__(self, surface):
        
        self.surface = surface
        self.snake_list = [[200, 200, util.SNAKE_SIZE, util.SNAKE_SIZE]]
        self.direction = None
        self.x = self.snake_list[0][0]
        self.y = self.snake_list[0][1]
        self.has_eaten = False

    def move_snake(self, direction):
        """
        Move the snake in a certain direction.
        Args :
             direction : the direction in wich the snake is moving.
        """
        self.has_eaten = False
        self.direction = direction
        self.update_tail()
        moved_x, moved_y = self.get_movement(direction)
        
        self.snake_list[0][0] += moved_x
        self.snake_list[0][1] += moved_y
        self.x = self.snake_list[0][0]
        self.y = self.snake_list[0][1]

    def update_tail(self):

        if len(self.snake_list) > 1:
            for i in range(len(self.snake_list)-1, 0, -1):
                self.snake_list[i][0] = copy.deepcopy(self.snake_list[i-1][0])
                self.snake_list[i][1] = copy.deepcopy(self.snake_list[i-1][1])


    def get_movement(self, direction):
        """
            Get movement of the snake based on the pressed key.
        """
        moved_x = 0
        moved_y = 0

        if direction == pygame.K_LEFT:
            moved_x -= util.SNAKE_SIZE
        if direction == pygame.K_UP:
            moved_y -= util.SNAKE_SIZE
        if direction == pygame.K_RIGHT:
            moved_x += util.SNAKE_SIZE
        if direction == pygame.K_DOWN:
            moved_y += util.SNAKE_SIZE

        return moved_x, moved_y

    def draw_snake(self):
        """
            Draw the snake at each game frate.
            Rectangle given by snake coordinates + width and height of the snake block.
        """
        self.surface.fill(util.WHITE) # clear the last frame snake

        for block in self.snake_list:
            pygame.draw.rect(self.surface, (255, 0, 0), [block[0], block[1], util.SNAKE_SIZE, util.SNAKE_SIZE])
        

    def is_dead(self):
        """
            Check if the game is over for the model.
        """
        is_dead = False

        head_x, head_y = self.snake_list[0][0], self.snake_list[0][1]
        surface_width, surface_height = self.surface.get_size()

        if head_x < 10 or head_y < 10 or head_x > surface_width - 10 or head_y > surface_height - 10:
            is_dead = True


        if len(self.snake_list) > 3:
            for block in self.snake_list[1:]:
                if block[0] == head_x and block[1] == head_y:
                    is_dead = True

        return is_dead

    def eat_food(self, food):
        """
            Check if the head is on the food. If so make it bigger.
        """
        head_x, head_y = self.snake_list[0][0], self.snake_list[0][1]
        moved_x, moved_y = self.get_movement(self.direction)

        if food.x == head_x and food.y == head_y:
            head = [head_x + moved_x, head_y + moved_y, util.SNAKE_SIZE, util.SNAKE_SIZE]
            self.snake_list.insert(0, head)
            self.has_eaten = True
            return self.has_eaten
        else:
            return False

