import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
UI_FONT = pygame.font.SysFont('arial', 20)

class Heading(Enum):
    EAST = 1
    WEST = 2
    NORTH = 3
    SOUTH = 4

GridPoint = namedtuple('GridPoint', 'x, y')

BACKGROUND_IMAGE = pygame.image.load("snakegameimage.png")
BACKGROUND_IMAGE = pygame.transform.scale(BACKGROUND_IMAGE, (640, 480))

COLOR_BG = (15, 15, 20)
COLOR_SNAKE_HEAD = (0, 230, 150)
COLOR_SNAKE_BODY = (0, 180, 120)
COLOR_APPLE = (230, 50, 50)
COLOR_TEXT = (240, 240, 240)

GRID_SIZE = 20
GAME_SPEED = 40

class SnakeEnvironment:
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Autonomous Snake - Reinforcement Learning')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.heading = Heading.EAST
        self.head = GridPoint(self.width // 2, self.height // 2)
        self.body = [
            self.head,
            GridPoint(self.head.x - GRID_SIZE, self.head.y),
            GridPoint(self.head.x - (2 * GRID_SIZE), self.head.y)
        ]
        self.score = 0
        self.apple = None
        self._spawn_apple()
        self.steps_count = 0  # Steps taken without eating food

    def _spawn_apple(self):
        x = random.randint(0, (self.width - GRID_SIZE) // GRID_SIZE) * GRID_SIZE
        y = random.randint(0, (self.height - GRID_SIZE) // GRID_SIZE) * GRID_SIZE
        self.apple = GridPoint(x, y)
        if self.apple in self.body:
            self._spawn_apple()

    def step(self, action):
        self.steps_count += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self._update_heading(action)
        self.body.insert(0, self.head)

        reward = 0
        is_done = False

        # Collision with wall/self OR stalled without eating food for too long
        if self.is_collision() or self.steps_count > 100 * len(self.body):
            is_done = True
            reward = -10
            return reward, is_done, self.score

        # Check if apple is eaten
        if self.head == self.apple:
            self.score += 1
            reward = 10
            self.steps_count = 0  # FIX: Reset step timer when food is eaten
            self._spawn_apple()
        else:
            self.body.pop()

        self._render()
        self.clock.tick(GAME_SPEED)
        return reward, is_done, self.score

    def is_collision(self, point=None):
        if point is None:
            point = self.head
        if point.x >= self.width or point.x < 0 or point.y >= self.height or point.y < 0:
            return True
        if point in self.body[1:]:
            return True
        return False

    def _render(self):
        self.screen.blit(BACKGROUND_IMAGE, (0, 0))

        for segment in self.body:
            color = COLOR_SNAKE_HEAD if segment == self.head else COLOR_SNAKE_BODY
            pygame.draw.rect(
                self.screen,
                color,
                pygame.Rect(segment.x, segment.y, GRID_SIZE, GRID_SIZE)
            )

        pygame.draw.rect(
            self.screen,
            COLOR_APPLE,
            pygame.Rect(self.apple.x, self.apple.y, GRID_SIZE, GRID_SIZE)
        )

        score_surface = UI_FONT.render(
            f'Score: {self.score}',
            True,
            COLOR_TEXT
        )
        self.screen.blit(score_surface, (10, 10))
        pygame.display.flip()


    def _update_heading(self, action):
        order = [Heading.EAST, Heading.SOUTH, Heading.WEST, Heading.NORTH]
        idx = order.index(self.heading)

        if np.array_equal(action, [1, 0, 0]):
            new_heading = order[idx]
        elif np.array_equal(action, [0, 1, 0]):
            new_heading = order[(idx + 1) % 4]
        else:
            new_heading = order[(idx - 1) % 4]

        self.heading = new_heading

        x, y = self.head.x, self.head.y
        if self.heading == Heading.EAST:
            x += GRID_SIZE
        elif self.heading == Heading.WEST:
            x -= GRID_SIZE
        elif self.heading == Heading.SOUTH:
            y += GRID_SIZE
        elif self.heading == Heading.NORTH:
            y -= GRID_SIZE

        self.head = GridPoint(x, y)