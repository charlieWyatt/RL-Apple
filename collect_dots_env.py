import gym
from gym import spaces
import numpy as np
import pygame
import sys

class CollectDotsEnv(gym.Env):
    """Custom Environment that follows gym interface, using acceleration."""
    metadata = {'render.modes': ['console', 'human']}
    
    def __init__(self, grid_size=10, max_steps=1000, max_accel=0.1, max_vel=0.5, cell_size=60):
        super(CollectDotsEnv, self).__init__()
        self.penalise_wall = True # change this depending on training cycle
        self.penalise_distance_to_apple = False # change this depending on training cycle
        self.apple_reward = 50 # change this depending on training cycle
        
        self.grid_size = grid_size
        self.cell_size = cell_size  # Size of each grid cell in pixels
        self.screen_size = self.grid_size * self.cell_size  # Calculate screen size based on grid and cell size
        self.max_steps = max_steps
        self.max_accel = max_accel
        self.max_vel = max_vel
        # Define action and observation space
        # Actions are continuous acceleration values in 2D space
        self.action_space = spaces.Box(low=-self.max_accel, high=self.max_accel, shape=(2,), dtype=np.float32)
        
        # Observation will be the position of agent and dot, normalized
        self.observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
        self.dots_collected = 0
        
        # Initialize agent velocity and position
        self.agent_vel = np.zeros(2)

        # Initialise pygame variables for rendering
        self.clock = None
        self.fps = 30
        self.screen = None
        self.screen_size = 600
        self.reset()

    def set_max_steps(self, max_steps):
        self.max_steps = max_steps

    def reset(self):
        # Reset the state of the environment to an initial state
        self.agent_pos = np.random.rand(2) * (self.grid_size - 1)
        self.agent_vel = np.zeros(2)  # Reset velocity
        self.dot_pos = np.random.rand(2) * (self.grid_size - 1)
        self.steps = 0
        self.dots_collected = 0  # Reset the dot collection counter for the new episode
        return self._get_obs()

    def step(self, action):
        # Update agent velocity and position based on acceleration action
        self.agent_vel += action
        self.agent_pos += self.agent_vel
        self.steps += 1

        if np.linalg.norm(self.agent_vel) > self.max_vel:
            self.agent_vel = self.agent_vel / np.linalg.norm(self.agent_vel) * self.max_vel
        self.agent_pos += self.agent_vel
        self.steps += 1
        self.agent_pos = np.clip(self.agent_pos, 0, self.grid_size - 1)
        
        # Check if the episode is done
        done = bool(self.steps >= self.max_steps)
        
        # Calculate reward
        reward = 0

        if self.penalise_distance_to_apple:
            reward = -np.linalg.norm(self.agent_pos - self.dot_pos) # penalised for being further away from the dot. Value is always negative
        
        if np.linalg.norm(self.agent_pos - self.dot_pos) < 1:
            reward += self.apple_reward  # Reward for collecting a dot
            self.dot_pos = np.random.rand(2) * (self.grid_size - 1) # Respawn dot
            self.dots_collected += 1
        
        # penalise heavily if agent is on the wall
        if self.penalise_wall:
            is_on_wall_x = self.agent_pos[0] == 0 or self.agent_pos[0] == self.grid_size - 1
            is_on_wall_y = self.agent_pos[1] == 0 or self.agent_pos[1] == self.grid_size - 1
            is_on_wall = is_on_wall_x or is_on_wall_y
            wall_penalty = 0
            if is_on_wall:
                wall_penalty = -5
            reward += wall_penalty
        
        return self._get_obs(), reward, done, {"dots_collected": self.dots_collected}

    def _get_obs(self):
        # Normalize positions for observation
        return np.array([self.agent_pos[0] / self.grid_size, self.agent_pos[1] / self.grid_size,
                         self.dot_pos[0] / self.grid_size, self.dot_pos[1] / self.grid_size])

    def render(self, mode='human'):
        if mode not in ['human', 'console']:
            raise NotImplementedError("Render mode not supported")
        
        if mode == 'console':
            print(f"Agent position: {self.agent_pos}, Dot position: {self.dot_pos}")
            return

        if self.screen is None or self.clock is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
            pygame.display.set_caption('Collect Dots Environment')
            self.clock = pygame.time.Clock()

        cell_size = self.screen_size / self.grid_size
        self.screen.fill((255, 255, 255))  # Fill the screen with white
        dot_rect = pygame.Rect(self.dot_pos[0] * cell_size, self.dot_pos[1] * cell_size, cell_size, cell_size)
        pygame.draw.rect(self.screen, (255, 0, 0), dot_rect)
        agent_rect = pygame.Rect(self.agent_pos[0] * cell_size, self.agent_pos[1] * cell_size, cell_size, cell_size)
        pygame.draw.rect(self.screen, (0, 0, 255), agent_rect)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        pygame.display.flip()
        self.clock.tick(self.fps)

    def close(self):
        if hasattr(self, 'screen'):
            pygame.quit()
