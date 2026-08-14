import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np

class SnakeQNetwork(nn.Module):
    def __init__(self, input_dim=11, hidden_dim=256, output_dim=3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def save_checkpoint(self, filepath='./saved_models/snake_dqn.pth'):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self.state_dict(), filepath)

class DQNTrainer:
    def __init__(self, network, lr=0.001, gamma=0.9):
        self.network = network
        self.gamma = gamma
        self.optimizer = optim.Adam(network.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def train_step(self, states, actions, rewards, next_states, dones):
        states = torch.tensor(np.array(states), dtype=torch.float)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float)
        actions = torch.tensor(np.array(actions), dtype=torch.long)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float)

        if len(states.shape) == 1:
            states = states.unsqueeze(0)
            next_states = next_states.unsqueeze(0)
            actions = actions.unsqueeze(0)
            rewards = rewards.unsqueeze(0)
            dones = (dones,)

        predictions = self.network(states)
        targets = predictions.clone()

        for i in range(len(dones)):
            q_value = rewards[i]
            if not dones[i]:
                q_value = rewards[i] + self.gamma * torch.max(self.network(next_states[i]))
            targets[i][torch.argmax(actions[i]).item()] = q_value

        self.optimizer.zero_grad()
        loss = self.loss_fn(targets, predictions)
        loss.backward()
        self.optimizer.step()