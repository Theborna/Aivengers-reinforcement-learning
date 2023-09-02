
# Reinforcement Learning on a Gridworld Environment

This code implements a basic reinforcement learning agent using Q-learning to navigate a gridworld environment. 

The environment consists of a grid of rewards, with the agent starting in the top left corner. The goal is to navigate to the highest reward states.

## Importing Libraries

First we import the necessary libraries:

```python
from environment import GridEnv
import utils
import pandas as pd
import time
import numpy as np
```

- `GridEnv` contains the gridworld environment class
- `utils` contains some helper functions
- `pandas` is used to load in the grid data
- `time` will be used to track runtime
- `numpy` provides array operations and math functions

## Initializing the Environment

Next we set up the environment:

```python
grid = pd.read_excel("Grid.xlsx", header=None)
rewards_grid = grid.values
rewards_grid = rewards_grid[::-1] # Reverse the grid to match the environment's coordinate system

env = GridEnv(rewards_grid)
env.reset()
```

- We load the grid data from an Excel file into a Pandas DataFrame
- Extract the values into a NumPy array
- Reverse the array so it matches the (y,x) coordinate system of the environment
- Create a GridEnv instance, passing in the grid of rewards
- Reset the environment twice to start from the initial state

## Defining Parameters

Now we define some key parameters:

```python
env.reset()

state_size = env.observation_space.n
num_actions = env.action_space.n

print('State Shape:', state_size)
print('Number of actions:', num_actions)
```

- `state_size`: The number of states in the environment
- `num_actions`: The number of possible actions the agent can take

We print these to confirm the environment setup.

## Initializing Q-values

The core of Q-learning is maintaining a table of Q-values for each state-action pair. We initialize this table to zeros:

```python 
Q = np.zeros((state_size, num_actions))
```

## Defining Algorithm Parameters

We also define some key parameters for the Q-learning algorithm:

```python
GAMMA = 1       # discount factor
ALPHA = 0.5     # learning rate 
E_GREEDY = True # Wether to follow an e-greedy policy at first
```

- `GAMMA`: Discount factor for future rewards
- `ALPHA`: Learning rate for updating Q-values
- `E_GREEDY`: Whether to use an epsilon-greedy policy for action selection

## Q-Value Update Function

This function updates a Q-value using the Bellman equation and a soft update:

```python
def updated_q(q, state, action, next_state, reward, done, 
             alpha=ALPHA, gamma=GAMMA):
    
    max_qs = np.max(q[next_state,:])
    estimated_qs = reward + gamma * max_qs * (1 - done)
    
    q_value = q[state, action]
    updated_qs = utils.lerp(q_value, estimated_qs, alpha)
    
    return updated_qs
```

The Bellman equation gives us the relationship between Q-values:

$$Q(s,a) = r + \gamma \max_{a'} Q(s', a')$$

Where:

- $Q(s,a)$ is the Q-value for state $s$ and action $a$
- $r$ is the immediate reward
- $\gamma$ is the discount factor
- $\max_{a'} Q(s', a')$ is the max Q-value for the next state $s'$

We use this to get an estimated Q-value:

Then we do a soft update between the old Q-value and the estimated Q-value:

$$Q(s,a) \leftarrow (1 - \alpha) Q(s,a) + \alpha (r + \gamma \max_{a'} Q(s', a'))$$

Where $\alpha$ is the learning rate.

By using a soft update, the Q-values slowly converge towards the optimal values.

## Training Loop

Now we define the main training loop:

```python
start = time.time()

num_episodes = 10_000
max_num_timesteps = 10000

total_point_history = []
num_p_av = 100 # number of total points to use for averaging 

epsilon = 1.0 # initial ε value for ε-greedy policy

for i in range(num_episodes):

  # Reset the environment to the initial state and get the initial state
  state, _ = env.reset()
  
  total_points = 0
  
  for t in range(max_num_timesteps):
  
    # From the current state S choose an action A using an ε-greedy policy
    q_values = Q[state, :]
    action = utils.get_action(q_values, epsilon = epsilon if E_GREEDY else 0 )

    # Take action A and receive reward R and the next state S'
    next_state, reward, done, _ = env.step(action)
    
    # Update Q-value for (S,A) 
    q_values[action] = updated_q(Q, state=state, action=action, 
                                next_state=next_state, reward=reward, done=done)
                                
    Q[state, :] = q_values
    
    state = next_state.copy()  
    total_points += reward
    
    if done:
      break
      
  total_point_history.append(total_points)

  # Update epsilon
  epsilon = utils.get_new_eps(epsilon) 
  
  # Print average reward
  av_latest_points = np.mean(total_point_history[-num_p_av:])
  print(f"\rEpisode {i+1} | Total point average of the last {num_p_av} episodes: {av_latest_points:.2f}", end="")

  # Check if solved
  if av_latest_points >= 390:
    k = i + 1
    print(f"\n\nEnvironment solved in {i+1} episodes!")  
    break
```

- Set number of episodes and max timesteps per episode
- Initialize trackers
- Loop through episodes
  - Reset environment, get initial state
  - Loop through timesteps
    - Select action using ε-greedy policy
    - Take action, get reward and next state
    - Update Q-value for (state, action) 
    - Update state
  - Track total points
  - Update ε
  - Print running average
  - Check if environment solved
  
## Results

Finally we print out some results:

```python
tot_time = time.time() - start
print(f"\nTotal Runtime: {tot_time:.2f} s ({(tot_time/60):.2f} min)") 

# Plot the total point history along with the moving average
env.render(ALPHA, GAMMA, k)
utils.plot_history(total_point_history)

# utils.plot_q_values(Q, rewards_grid[::-1])
```

- Print total runtime
- Render environment visualization
- Plot reward history
- Could also plot Q-values

This shows a basic implementation of Q-learning on a simple gridworld environment. The agent is able to learn an effective policy to navigate to high reward states.