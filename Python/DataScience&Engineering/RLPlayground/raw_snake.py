import pygame
from enum import Enum
from collections import namedtuple
import random

class direction_clusters(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Points = namedtuple("Point", "x,y")
BLOCK_SIZE = 20
SPEED = 20
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
            pygame.draw.rect(self.display, BLUE, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(self.display, RED, pygame.Rect(self.apple.x, self.apple.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.points), True, WHITE)
        self.display.blit(text, [0, 0])

        custom_text = font1.render(custom_text, True, WHITE)
        self.display.blit(custom_text, [self.w / 2, self.h / 2])

        pygame.display.flip()

    def play(self):
        # input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT:
                    self.direction = direction_clusters.RIGHT
                elif event.key == pygame.K_LEFT:
                    self.direction = direction_clusters.LEFT
                elif event.key == pygame.K_UP:
                    self.direction = direction_clusters.UP
                elif event.key == pygame.K_DOWN:
                    self.direction = direction_clusters.DOWN
        # move
        self.move(self.direction)
        self.snake.insert(0, self.head)
        reward=0

        game_over = False
        # check whether game continues
        if self.collided():
            game_over = True
            # return variables to main function
            return game_over, self.points
        # new apple
        if self.head == self.apple:
            self.place_apple()
            self.points += 10
        else:
            # remove last index of array. unless we remove, array grove until game is over
            self.snake.pop()

        # update game
        self.update_gui(custom_text="Göt Hüdai")
        self.clock.tick(SPEED)
        # return variables to main function
        return game_over, self.points

    def collided(self):
        pts=self.head
        if pts.x > self.w - BLOCK_SIZE or pts.x < 0 or pts.y > self.h - BLOCK_SIZE or pts.y < 0:
            return True
        if pts in self.snake[1:]:
            return True
        return False

    def move(self, direction):
        x = self.head.x
        y = self.head.y
        if direction is direction_clusters.RIGHT:
            x += BLOCK_SIZE
        elif direction is direction_clusters.LEFT:
            x -= BLOCK_SIZE
        elif direction is direction_clusters.DOWN:
            y += BLOCK_SIZE
        elif direction is direction_clusters.UP:
            y -= BLOCK_SIZE
        self.head = Points(x, y)


if __name__ == "__main__":
    game = Snake()
    while True:
        game_over, points = game.play()
        if game_over is True:
            break
    print("Don't lose your hope. So far you got: ", points)
    pygame.quit()
