import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

action_space = [0, 1, 2, 3, 4]  # right, down, left, up, space
state_size = 5  # placeholder


class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN,self).__init__()
        print(state_size)
        input_size = state_size[0] # * state_size[1]
        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return (self.fc4(x))


class Agent():
    def __init__(self, state_size, action_size,max_mem_size):
        self.state_size = state_size
        self.action_size = action_size
        self.max_mem_size =max_mem_size
        self.memory = deque(maxlen=self.max_mem_size)
        self.gamma = 0.95  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.0001
        self.model = DQN(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() < self.epsilon: # observe env arbitrarily
            return random.randrange(self.action_size)

        state = torch.FloatTensor(state).flatten().unsqueeze(0)  # ? Adds an extra dimension to the tensor. often used to match
        # the expected input shape of a neural network

        with torch.no_grad(): # return most probable action based on model output
            estimated_act_vals = self.model(state)
            return torch.argmax(estimated_act_vals, dim=1).item()

    def replay(self, batch_size):
        if len(self.memory) < batch_size: # memory doesn't have enough samples
            return

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward

            if not done:
                next_state = torch.FloatTensor(next_state).flatten().unsqueeze(0)
                target = (reward + self.gamma * torch.max(self.model(next_state)).item())
            state = torch.FloatTensor(state).flatten().unsqueeze(0)
            target_f = self.model(state)  # Compute Q-values for all actions
            target_f1 = target_f.clone()  # Clone to avoid modifying the original tensor
            target_f1[0][action] = target  # Update only the action taken

            loss = nn.MSELoss()(target_f1, target_f)  # Now we avoid recomputation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        if len(self.memory) > self.max_mem_size:
            self.memory.pop()  # Remove the oldest experience


if __name__ == "__main__":
    agent = Agent(state_size=state_size, action_size=len(action_space),max_mem_size=2000)
    episodes = 1000  # Number of episodes to train
    batch_size = 32  # Batch size for replay

    for episode in range(episodes):
        state = np.random.rand(state_size)
        done = False
        while not done:
            action=agent.act(state)
            next_state = np.random.rand(state_size)  # Replace with Tetris game logic for next state
            reward = 1  # Replace with Tetris game reward logic
            done = False  # Replace with game over condition
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if done:
                print(f"episode: {episode}/{episodes}, score: {reward}, epsilon: {agent.epsilon}")
                break

            agent.replay(batch_size)
