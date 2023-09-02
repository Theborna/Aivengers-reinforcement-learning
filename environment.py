import matplotlib.pyplot as plt
from typing import Optional, Tuple, TypeVar, Any, Dict
import gym
import gym.spaces as spaces
import numpy as np

ObsType = TypeVar("ObsType")

class GridEnv(gym.Env):
    metadata = {"num_actions": 2}
    
    def __init__(self, rewards: np.ndarray):
        # Initialize the Grid Environment
        num_actions = GridEnv.metadata.get("num_actions")
        size = rewards.shape[-1]
        self.size = size

        # Define action and observation spaces
        self.action_space = spaces.Discrete(num_actions)
        self.observation_space = spaces.Discrete(size**2)

        # Initialize the agent's state and rewards
        self.state = np.array([0, 0])
        self.rewards = rewards
        
        # Define actions and end state
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
        }
        self._end_state = np.array([size - 1, size - 1])
        
        # Initialize the path taken by the agent
        self.way = np.zeros((size, size))
        self.way[0, 0] = 1

    def step(self, action: ObsType) -> Tuple[ObsType, float, bool, Dict[str, Any]]:
        # Take a step in the environment based on the chosen action
        direction = self._action_to_direction[action]
        past_state = self.state.copy()
        self.state = np.clip(
            self.state + direction, 0, self.size - 1
        )
        self.way[self.state[0], self.state[1]] = 1
        
        # Check if the agent has reached the end state
        done = np.array_equal(self.state, self._end_state)
        
        # Get information about the current state
        info = self._get_information()
        
        # Calculate the reward for the current step
        reward = self.rewards[self.state[0], self.state[1]]
        
        # Comment explaining the -200 penalty for revisiting a state
        reward = reward + np.array_equal(self.state, past_state) * -200
        
        return self._get_observation(), reward, done, info
    

    def render(self, alpha, gamma, k):
        way = self.way[::-1]
        
        plt.matshow(way, cmap='RdBu', resample=True)
        n = way.shape[0]

        plt.hlines(y=np.arange(0, n)+0.5, xmin=np.full(n, 0)-0.5, xmax=np.full(n, n)-0.5, color="black")
        plt.vlines(x=np.arange(0, n)+0.5, ymin=np.full(n, 0)-0.5, ymax=np.full(n, n)-0.5, color="black")
        plt.axis('off')
        plt.title(f'Parameters $\\alpha = {alpha}, \\gamma = {gamma}, k={k}$', fontsize=14)  # Adjust the title font size

        for (i, j), z in np.ndenumerate(self.rewards[::-1]):
            plt.text(j, i, '{:.0f}'.format(z), ha='center', va='center', fontsize='xx-small',color='k')        
        total_points = np.sum(self.rewards * self.way)
        
        plt.text(n // 2, n + 1, f'Total Points: {total_points}', ha='center', fontsize=12)  # Add total points at the bottom

        plt.show()


        
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[ObsType, Dict[str, Any]]:
        # Reset the environment to the initial state
        self.state = np.array([0, 0])
        
        observation = self._get_observation()
        info = self._get_information()
        
        # Clear the path taken by the agent
        self.way = np.zeros((self.size, self.size))
        self.way[0, 0] = 1

        return observation, info
    
    def _get_observation(self):
        # Get the current observation (not used in this example)
        return self.state[0] + self.state[1] * self.size
    
    def _get_information(self):
        # Get information about the current state (not used in this example)
        return {"distance": np.sum(self.state - self._end_state)}
