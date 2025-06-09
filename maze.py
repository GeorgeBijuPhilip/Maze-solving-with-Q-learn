import pygame
from random import choice
import numpy as np
import gym
from gym import spaces
import random
from collections import defaultdict

# Pygame setup
RES = WIDTH, HEIGHT = 600, 600
TILE = 50
cols, rows = WIDTH // TILE, HEIGHT // TILE
sc = pygame.display.set_mode(RES)
clock = pygame.time.Clock()

class Cell:
    def __init__(self, x, y):
        self.x, self.y = x, y
        self.walls = {'top': True, 'right': True, 'bottom': True, 'left': True}
        self.visited = False
    def draw_current_cell(self):
        x, y = self.x * TILE, self.y * TILE
        pygame.draw.rect(sc, pygame.Color('#f70067'), (x + 2, y + 2, TILE - 4, TILE - 4))
    def draw(self):
        x, y = self.x * TILE, self.y * TILE
        if self.visited:
            pygame.draw.rect(sc, pygame.Color('#1e1e1e'), (x, y, TILE, TILE))
        border_color = pygame.Color('white')
        if self.walls['top']:
            pygame.draw.line(sc, border_color, (x, y), (x + TILE, y), 2)
        if self.walls['right']:
            pygame.draw.line(sc, border_color, (x + TILE, y), (x + TILE, y + TILE), 2)
        if self.walls['bottom']:
            pygame.draw.line(sc, border_color, (x + TILE, y + TILE), (x, y + TILE), 2)
        if self.walls['left']:
            pygame.draw.line(sc, border_color, (x, y + TILE), (x, y), 2)
    def check_cell(self, x, y):
        find_index = lambda x, y: x + y * cols
        if x < 0 or x > cols - 1 or y < 0 or y > rows - 1:
            return None
        return grid_cells[find_index(x, y)]
    def check_neighbors(self):
        neighbors = []
        top = self.check_cell(self.x, self.y - 1)
        right = self.check_cell(self.x + 1, self.y)
        bottom = self.check_cell(self.x, self.y + 1)
        left = self.check_cell(self.x - 1, self.y)
        if top and not top.visited:
            neighbors.append(top)
        if right and not right.visited:
            neighbors.append(right)
        if bottom and not bottom.visited:
            neighbors.append(bottom)
        if left and not left.visited:
            neighbors.append(left)
        return choice(neighbors) if neighbors else None

def remove_walls(current, next_cell):
    dx = current.x - next_cell.x
    if dx == 1:
        current.walls['left'] = False
        next_cell.walls['right'] = False
    elif dx == -1:
        current.walls['right'] = False
        next_cell.walls['left'] = False
    dy = current.y - next_cell.y
    if dy == 1:
        current.walls['top'] = False
        next_cell.walls['bottom'] = False
    elif dy == -1:
        current.walls['bottom'] = False
        next_cell.walls['top'] = False

# Maze generation
grid_cells = [Cell(col, row) for row in range(rows) for col in range(cols)]
current_cell = grid_cells[0]
stack = []

def generate_maze():
    global current_cell
    while True:
        current_cell.visited = True
        next_cell = current_cell.check_neighbors()
        if next_cell:
            stack.append(current_cell)
            remove_walls(current_cell, next_cell)
            current_cell = next_cell
        elif stack:
            current_cell = stack.pop()
        else:
            break

generate_maze()

class MazeEnv(gym.Env):
    def __init__(self):
        super(MazeEnv, self).__init__()
        self.width = cols
        self.height = rows
        self.observation_space = spaces.Discrete(self.width * self.height)
        self.action_space = spaces.Discrete(4)
        self.grid_cells = grid_cells
        self.reset()
    def reset(self):
        self.current_position = (0, 0)
        self.steps = 0
        return self.get_state(self.current_position)
    def step(self, action):
        x, y = self.current_position
        current_cell = self.get_cell(x, y)
        moved = False
        if action == 0 and not current_cell.walls['top']:
            y -= 1
            moved = True
        elif action == 1 and not current_cell.walls['right']:
            x += 1
            moved = True
        elif action == 2 and not current_cell.walls['bottom']:
            y += 1
            moved = True
        elif action == 3 and not current_cell.walls['left']:
            x -= 1
            moved = True
        x = max(0, min(x, self.width - 1))
        y = max(0, min(y, self.height - 1))
        self.current_position = (x, y)
        self.steps += 1
        done = (x, y) == (self.width - 1, self.height - 1)
        if done:
            reward = 1.0
        elif moved:
            reward = -0.01
        else:
            reward = -0.1
        return self.get_state(self.current_position), reward, done, {}
    def get_state(self, position):
        x, y = position
        return y * self.width + x
    def get_cell(self, x, y):
        return self.grid_cells[y * self.width + x]
    def render(self, mode='human', path=None):
        sc.fill(pygame.Color('#a6d5e2'))
        for cell in self.grid_cells:
            cell.draw()
        if path is not None:
            for (x, y) in path:
                pygame.draw.rect(sc, pygame.Color('red'), (x * TILE + 10, y * TILE + 10, TILE - 20, TILE - 20))
        start_x, start_y = 0, 0
        pygame.draw.rect(sc, pygame.Color('white'), (start_x * TILE + 5, start_y * TILE + 5, TILE - 10, TILE - 10))
        goal_x, goal_y = self.width - 1, self.height - 1
        pygame.draw.rect(sc, pygame.Color('green'), (goal_x * TILE + 5, goal_y * TILE + 5, TILE - 10, TILE - 10))
        x, y = self.current_position
        pygame.draw.rect(sc, pygame.Color('#f70067'), (x * TILE + 5, y * TILE + 5, TILE - 10, TILE - 10))
        pygame.display.flip()
        pygame.event.pump()
        clock.tick(30)

env = MazeEnv()
q_table = defaultdict(lambda: np.zeros(env.action_space.n))
alpha = 0.1
gamma = 0.99
epsilon = 0.1
num_episodes = 1000

def epsilon_greedy_policy(state):
    if random.uniform(0, 1) < epsilon:
        return env.action_space.sample()
    else:
        return np.argmax(q_table[state])

for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = epsilon_greedy_policy(state)
        next_state, reward, done, _ = env.step(action)
        best_next_action = np.argmax(q_table[next_state])
        q_table[state][action] += alpha * (reward + gamma * q_table[next_state][best_next_action] - q_table[state][action])
        state = next_state
        total_reward += reward
        if episode % 100 == 0:
            env.render()
    print(f"Episode {episode} finished with reward: {total_reward}")

print("Training complete!")

def highlight_optimal_path():
    path = []
    state = env.reset()
    done = False
    while not done:
        path.append(env.current_position)
        action = np.argmax(q_table[state])
        next_state, _, done, _ = env.step(action)
        state = next_state
    return path

optimal_path = highlight_optimal_path()
env.render(path=optimal_path)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
