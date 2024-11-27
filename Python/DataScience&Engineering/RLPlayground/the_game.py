import time

import pygame
from enum import Enum
from collections import namedtuple
import random
import numpy as np


#https://www.youtube.com/watch?v=nyjbcRQ-uQ8&list=PLZbbT5o_s2xoWNVdDudn51XM8lOuZ_Njv&index=1
#https://www.youtube.com/watch?v=qfovbG84EBg&list=PLQVvvaa0QuDezJFIOU5wDdfy4e9vdnx-7&index=6
#https://www.youtube.com/results?search_query=text+mining+with+python
#https://deeplizard.com/learn/video/_N5kpSMDf4o
#https://deeplizard.com/learn/video/XE3krf3CQls
#https://deeplizard.com/learn/video/8krd5qKVw-Q

# reset
# reward
# play(action)
# game_iteration
# collided
class direction_clusters(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Points = namedtuple("Point", "x,y")
BLOCK_SIZE = 20
SPEED = 50
# rgb colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
YELLOW = (37, 150, 190)
BLUE = (84, 159, 227)
BLACK = (0, 0, 0)

pygame.init()
font = pygame.font.Font('.\\font\\arial.ttf', 25)
font1 = pygame.font.Font(".\\font\\Jellyka_Castle's_Queen.ttf", 100)


class Snake:
    def __init__(self, width=640, height=480):
        # dimensions of game screen
        self.w = width
        self.h = height
        # init game screen
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption("Snake-Tıss")
        self.clock = pygame.time.Clock()
        self.reset()
        self.generation=0
    def reset(self):
        self.direction = direction_clusters.RIGHT
        # initially snake's head located here
        self.head = Points(self.w / 2, self.h / 2)
        # initially snake has 3 blocks:1 head,2 body
        self.snake = [self.head,
                      Points(self.head.x - BLOCK_SIZE, self.head.y),
                      Points(self.head.x - (2 * BLOCK_SIZE), self.head.y)]
        self.points = 0
        self.apple = None
        self.place_apple()
        self.frame_iter=0

    def place_apple(self):
        # in order not to place apple end of the screen,we subcribed one block size
        # if apple placed,for instance, width-block_size+10 (800 in this case), our pretty snake suffers
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.apple = Points(x, y)
        # check whether apple located in snake
        if self.apple in self.snake:
            self.place_apple()

    def update_gui(self, custom_text=None):
        self.display.fill(BLACK)
        for pt in self.snake:
            if self.snake.index(pt)!=0:
                pygame.draw.rect(self.display, BLUE, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            elif self.snake.index(pt)==0:
                pygame.draw.rect(self.display, WHITE, pygame.Rect(self.head.x, self.head.y, BLOCK_SIZE, BLOCK_SIZE))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.apple.x, self.apple.y, BLOCK_SIZE, BLOCK_SIZE))

        score_text = font.render("Score: " + str(self.points), True, WHITE)
        generation_number = font.render("Generation: " + str(self.generation), True, WHITE)
        self.display.blit(score_text, [self.w-130, 0])
        self.display.blit(generation_number, [0, 0])

        if custom_text.__contains__(":"):
            first_part,second_part=custom_text.split(":")
            first_part = font1.render(first_part, True, WHITE)
            second_part = font1.render(second_part, True, WHITE)
            self.display.blit(first_part, [self.w / 2-30, self.h / 2])
            self.display.blit(second_part, [self.w / 2-30, self.h / 2+100])
        else:
            custom_text = font1.render(custom_text, True, WHITE)
            self.display.blit(custom_text, [self.w / 2, self.h / 2])
        pygame.display.flip()

    def play(self,action):
        self.frame_iter+=1
        # input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            # if action.type == pygame.KEYDOWN:
            #     if action.key == pygame.K_RIGHT:
            #         self.direction = direction_clusters.RIGHT
            #     elif action.key == pygame.K_LEFT:
            #         self.direction = direction_clusters.LEFT
            #     elif action.key == pygame.K_UP:
            #         self.direction = direction_clusters.UP
            #     elif action.key == pygame.K_DOWN:
            #         self.direction = direction_clusters.DOWN
        # move
        self.move(action)
        self.snake.insert(0, self.head)
        reward=0

        game_over = False
        # check whether game continues
        if self.collided() or self.frame_iter>100*len(self.snake):
            game_over = True
            # return variables to main function
            reward-=10
            return reward,game_over, self.points
        # new apple
        if self.head == self.apple:
            self.place_apple()
            self.points += 10
            reward += 10
        else:
            # remove last index of array. unless we remove, array grove until game is over
            self.snake.pop()

        # update game
        self.update_gui(custom_text=(""))
        self.clock.tick(SPEED)
        # return variables to main function
        return reward,game_over, self.points

    def collided(self,pts=None):
        if pts is None:
            pts=self.head

        if pts.x > self.w - BLOCK_SIZE or pts.x < 0 or pts.y > self.h - BLOCK_SIZE or pts.y < 0:
            return True
        if pts in self.snake[1:]:
            return True
        return False

    def move(self, action):
        all_moves=[direction_clusters.RIGHT,direction_clusters.DOWN,direction_clusters.LEFT,direction_clusters.UP]
        index=all_moves.index(self.direction)
        #[straight,right,left]
        if np.array_equal(action,[1,0,0]):
            new_move=all_moves[index]

        elif np.array_equal(action,[0,1,0]):
            new_move=all_moves[(1+index)%4]

        elif np.array_equal(action,[0,0,1]):
            new_move=all_moves[(index-1)%4]
        self.direction=new_move
        x = self.head.x
        y = self.head.y
        if self.direction is direction_clusters.RIGHT:
            x += BLOCK_SIZE
        elif self.direction is direction_clusters.LEFT:
            x -= BLOCK_SIZE
        elif self.direction is direction_clusters.DOWN:
            y += BLOCK_SIZE
        elif self.direction is direction_clusters.UP:
            y -= BLOCK_SIZE
        self.head = Points(x, y)


# if __name__ == "__main__":
#     game = Snake()
#     while True:
#         game_over, points = game.play()
#         if game_over is True:
#             break
#     print("Don't lose your hope. So far you got: ", points)
#     pygame.quit()
