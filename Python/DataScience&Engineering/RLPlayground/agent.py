import time

import cv2
import pygame
import torch
import random
import numpy as np
from model import Linear_QNet, QTrainer
import the_game
from the_game import Snake, direction_clusters, Points, BLOCK_SIZE
from collections import deque
from plotter import plot

MEM = 100_000
BATCH = 1000
LR = 1e-3


class Agent:
    def __init__(self):
        self.n_games = 0
        # random
        self.epsilon = 0
        # discount rate to calculate new Q 0<gamma<1
        self.gamma = 0
        # if maxlen exeeds deque function pops the first element
        self.memory = deque(maxlen=MEM)
        # [state shape,hidden layer amount,output shape]
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, LR, self.gamma)

    def get_state(self, game):
        head = game.snake[0]
        left = Points(head.x - BLOCK_SIZE, head.y)
        right = Points(head.x + BLOCK_SIZE, head.y)
        up = Points(head.x, head.y - BLOCK_SIZE)
        down = Points(head.x, head.y + BLOCK_SIZE)

        left_dir = game.direction == direction_clusters.LEFT
        right_dir = game.direction == direction_clusters.RIGHT
        up_dir = game.direction == direction_clusters.UP
        down_dir = game.direction == direction_clusters.DOWN

        state = [
            # if going straight causes dead
            (right_dir and game.collided(right)) or
            (left_dir and game.collided(left)) or
            (up_dir and game.collided(up)) or
            (down_dir and game.collided(down)),

            # Danger right
            (up_dir and game.collided(right)) or
            (down_dir and game.collided(left)) or
            (left_dir and game.collided(up)) or
            (right_dir and game.collided(down)),

            # Danger left
            (down_dir and game.collided(right)) or
            (up_dir and game.collided(left)) or
            (right_dir and game.collided(up)) or
            (left_dir and game.collided(down)),
            # moves
            left_dir,
            right_dir,
            up_dir,
            down_dir,
            # apple coordinates
            game.apple.x < game.head.x,  # food located left
            game.apple.x > game.head.x,  # food located right
            game.apple.y < game.head.y,  # food located up
            game.apple.y > game.head.y  # food located down
        ]
        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH:
            sample = random.sample(self.memory, BATCH)
        else:
            sample = self.memory
        # in order to unpack deque variable * is used
        states, actions, rewards, next_states, dones = zip(*sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.n_games
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move
 # https://www.youtube.com/watch?v=p15xzjzR9j0

def train():
    plot_score = []
    plot_mean_score = []
    total_score = 0
    record = 0
    agent = Agent()
    game = Snake()
    screen=game.display
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('output1.avi', fourcc, 60.0, (640,480))
    while True:
        pixels = cv2.rotate(pygame.surfarray.pixels3d(screen), cv2.ROTATE_90_CLOCKWISE)
        pixels = cv2.flip(pixels, 1)
        pixels = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
        out.write(pixels)
        game.generation=agent.n_games
        old_state = agent.get_state(game)
        old_move = agent.get_action(old_state)
        reward, done, score = game.play(old_move)
        new_state = agent.get_state(game)
        agent.train_short_memory(old_state, old_move, reward, new_state, done)
        agent.remember(old_state, old_move, reward, new_state, done)

        if done:
            game.reset()
            agent.n_games += 1
            agent.train_long_memory()
            if score > record:
                score = record
                agent.model.save()
            print("Episode:{} Score:{}".format(agent.n_games, score))
            # plot_score.append(score)
            # total_score += score
            # mean_score = total_score / agent.n_games
            # plot_mean_score.append(mean_score)
            # plot(plot_score, plot_mean_score)


if __name__ == "__main__":
    train()
