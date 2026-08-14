import torch
import numpy as np
import random
from collections import deque
from snake_env import SnakeEnvironment, Heading, GridPoint, GRID_SIZE
from q_network import SnakeQNetwork, DQNTrainer
from visualizer import plot_training_progress

MEMORY_CAPACITY = 100_000
BATCH_SIZE = 1000
LEARNING_RATE = 0.001

class SnakeAgent:
    def __init__(self):
        self.episodes = 0
        self.epsilon = 0  # Exploration factor
        self.gamma = 0.9  # Discount factor
        self.memory = deque(maxlen=MEMORY_CAPACITY)
        self.model = SnakeQNetwork(input_dim=11, hidden_dim=256, output_dim=3)
        self.trainer = DQNTrainer(self.model, lr=LEARNING_RATE, gamma=self.gamma)

    def observe_environment(self, env):
        head = env.head
        point_l = GridPoint(head.x - GRID_SIZE, head.y)
        point_r = GridPoint(head.x + GRID_SIZE, head.y)
        point_u = GridPoint(head.x, head.y - GRID_SIZE)
        point_d = GridPoint(head.x, head.y + GRID_SIZE)

        dir_l = env.heading == Heading.WEST
        dir_r = env.heading == Heading.EAST
        dir_u = env.heading == Heading.NORTH
        dir_d = env.heading == Heading.SOUTH

        state = [
            # Danger ahead
            (dir_r and env.is_collision(point_r)) or
            (dir_l and env.is_collision(point_l)) or
            (dir_u and env.is_collision(point_u)) or
            (dir_d and env.is_collision(point_d)),

            # Danger right
            (dir_u and env.is_collision(point_r)) or
            (dir_d and env.is_collision(point_l)) or
            (dir_l and env.is_collision(point_u)) or
            (dir_r and env.is_collision(point_d)),

            # Danger left
            (dir_d and env.is_collision(point_r)) or
            (dir_u and env.is_collision(point_l)) or
            (dir_r and env.is_collision(point_u)) or
            (dir_l and env.is_collision(point_d)),

            # Heading vectors
            dir_l, dir_r, dir_u, dir_d,

            # Apple relative direction
            env.apple.x < env.head.x,
            env.apple.x > env.head.x,
            env.apple.y < env.head.y,
            env.apple.y > env.head.y
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay_experience(self):
        sample_batch = random.sample(self.memory, BATCH_SIZE) if len(self.memory) > BATCH_SIZE else self.memory
        states, actions, rewards, next_states, dones = zip(*sample_batch)
        self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_step(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def select_action(self, state):
        self.epsilon = max(5, 80 - self.episodes)
        move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move_idx = random.randint(0, 2)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state_tensor)
            move_idx = torch.argmax(prediction).item()

        move[move_idx] = 1
        return move

def train():
    scores = []
    average_scores = []
    total_score = 0
    best_score = 0
    agent = SnakeAgent()
    env = SnakeEnvironment()

    while True:
        old_state = agent.observe_environment(env)
        action = agent.select_action(old_state)
        reward, done, score = env.step(action)
        new_state = agent.observe_environment(env)

        agent.train_step(old_state, action, reward, new_state, done)
        agent.remember(old_state, action, reward, new_state, done)

        if done:
            env.reset()
            agent.episodes += 1
            agent.replay_experience()

            if score > best_score:
                best_score = score
                agent.model.save_checkpoint()

            scores.append(score)
            total_score += score
            avg_score = total_score / agent.episodes
            average_scores.append(avg_score)

            print(f"Episode: {agent.episodes} | Score: {score} | Best: {best_score} | Avg: {avg_score:.2f}")
            plot_training_progress(scores, average_scores)

if __name__ == '__main__':
    train()