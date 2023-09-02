from environment import GridEnv
import utils
import pandas as pd
import time
import numpy as np

grid = pd.read_excel("Grid.xlsx", header=None)
rewards_grid = grid.values
rewards_grid = rewards_grid[::-1]  # Reverse the grid to match the environment's coordinate system

env = GridEnv(rewards_grid)
env.reset()

env.reset()

state_size = env.observation_space.n
num_actions = env.action_space.n

print('State Shape:', state_size)
print('Number of actions:', num_actions)

Q = np.zeros((state_size, num_actions))

GAMMA = 1                 # discount factor
ALPHA = 0.5               # learning rate  
E_GREEDY = True           # Wether to follow an e-greedy policy at first

def updated_q(q, state, action, next_state, reward, done, alpha=ALPHA, gamma=GAMMA):
    max_qs = np.max(q[next_state,:])
    estimated_qs = reward + gamma * max_qs * (1 - done)
    q_value = q[state, action]
    updated_qs = utils.lerp(q_value, estimated_qs, alpha)
    return updated_qs


start = time.time()

num_episodes = 10_000
max_num_timesteps = 10000

total_point_history = []

num_p_av = 100    # number of total points to use for averaging
epsilon = 1.0     # initial ε value for ε-greedy policy

k = num_episodes

for i in range(num_episodes):
    
    # Reset the environment to the initial state and get the initial state
    state, _  = env.reset()
    total_points = 0

    
    for t in range(max_num_timesteps):
        
        # From the current state S choose an action A using an ε-greedy policy
        q_values = Q[state, :]
        action = utils.get_action(q_values, epsilon = epsilon if E_GREEDY else 0 )
        
        # Take action A and receive reward R and the next state S'
        next_state, reward, done, _ = env.step(action)
        
        
        q_values[action] = updated_q(Q, state=state, action=action ,next_state=next_state, reward=reward, done=done)
        
        Q[state, :] = q_values
        state = next_state.copy()
        total_points += reward
        
        if done:
            break
            
    total_point_history.append(total_points)
    av_latest_points = np.mean(total_point_history[-num_p_av:])
    
    # Update the ε value
    epsilon = utils.get_new_eps(epsilon)

    print(f"\rEpisode {i+1} | Total point average of the last {num_p_av} episodes: {av_latest_points:.2f}", end="")

    if (i+1) % num_p_av == 0:
        print(f"\rEpisode {i+1} | Total point average of the last {num_p_av} episodes: {av_latest_points:.2f}")

    # We will consider that the environment is solved if we get an
    # average of 390 points in the last 100 episodes.
    if av_latest_points >= 390:
        k = i + 1
        print(f"\n\nEnvironment solved in {i+1} episodes!")
        break
    
tot_time = time.time() - start

print(f"\nTotal Runtime: {tot_time:.2f} s ({(tot_time/60):.2f} min)")

# Plot the total point history along with the moving average
env.render(ALPHA, GAMMA, k)
utils.plot_history(total_point_history)
# utils.plot_q_values(Q, rewards_grid[::-1])