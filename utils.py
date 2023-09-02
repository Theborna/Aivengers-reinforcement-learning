import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

E_DECAY = 0.999   # ε-decay rate for the ε-greedy policy.
E_MIN = 0.01      # Minimum ε value for the ε-greedy policy.

def lerp(x, y, t):
    """
    Linearly interpolate between two values x and y based on a parameter t.
    
    Parameters:
    - x: The starting value.
    - y: The ending value.
    - t: The interpolation parameter (between 0 and 1).
    
    Returns:
    - The interpolated value between x and y based on t.
    """
    return (1 - t) * x + t * y

def get_action(q_values: np.ndarray, epsilon):
    """
    Choose an action based on ε-greedy policy.

    Parameters:
    - q_values: A NumPy array representing the Q-values for each action.
    - epsilon: The exploration-exploitation parameter (0 for pure exploitation, 1 for pure exploration).

    Returns:
    - The selected action based on ε-greedy policy.
    """
    if np.random.uniform(0, 1) > epsilon:
        return np.argmax(q_values)
    else:
        return np.random.choice(np.arange(q_values.shape[-1]))

def get_new_eps(epsilon):
    """
    Update the ε value for ε-greedy policy using exponential decay.

    Parameters:
    - epsilon: The current ε value.

    Returns:
    - The updated ε value after decay.
    """
    return max(E_MIN, E_DECAY * epsilon)


def plot_history(point_history, **kwargs):
    """
    Plots the total number of points received by the agent after each episode together
    with the moving average (rolling mean). 

    Args:
        point_history (list):
            A list containing the total number of points the agent received after each
            episode.
        **kwargs: optional
            window_size (int):
                Size of the window used to calculate the moving average (rolling mean).
                This integer determines the fixed number of data points used for each
                window. The default window size is set to 10% of the total number of
                data points in point_history, i.e. if point_history has 200 data points
                the default window size will be 20.
            lower_limit (int):
                The lower limit of the x-axis in data coordinates. Default value is 0.
            upper_limit (int):
                The upper limit of the x-axis in data coordinates. Default value is
                len(point_history).
            plot_rolling_mean_only (bool):
                If True, only plots the moving average (rolling mean) without the point
                history. Default value is False.
            plot_data_only (bool):
                If True, only plots the point history without the moving average.
                Default value is False.
    """

    lower_limit = 0
    upper_limit = len(point_history)

    window_size = (upper_limit * 10) // 100

    plot_rolling_mean_only = False
    plot_data_only = False

    if kwargs:
        if "window_size" in kwargs:
            window_size = kwargs["window_size"]

        if "lower_limit" in kwargs:
            lower_limit = kwargs["lower_limit"]

        if "upper_limit" in kwargs:
            upper_limit = kwargs["upper_limit"]

        if "plot_rolling_mean_only" in kwargs:
            plot_rolling_mean_only = kwargs["plot_rolling_mean_only"]

        if "plot_data_only" in kwargs:
            plot_data_only = kwargs["plot_data_only"]

    points = point_history[lower_limit:upper_limit]

    # Generate x-axis for plotting.
    episode_num = [x for x in range(lower_limit, upper_limit)]

    # Use Pandas to calculate the rolling mean (moving average).
    rolling_mean = pd.DataFrame(points).rolling(window_size).mean()

    plt.figure(figsize=(10, 7), facecolor="white")

    if plot_data_only:
        plt.plot(episode_num, points, linewidth=1, color="cyan")
    elif plot_rolling_mean_only:
        plt.plot(episode_num, rolling_mean, linewidth=2, color="magenta")
    else:
        plt.plot(episode_num, points, linewidth=1, color="cyan")
        plt.plot(episode_num, rolling_mean, linewidth=2, color="magenta")

    text_color = "black"

    ax = plt.gca()
    ax.set_facecolor("black")
    plt.grid()
    plt.xlabel("Episode", color=text_color, fontsize=30)
    plt.ylabel("Total Points", color=text_color, fontsize=30)
    # yNumFmt = mticker.StrMethodFormatter("{x:,}")
    # ax.yaxis.set_major_formatter(yNumFmt)
    ax.tick_params(axis="x", colors=text_color)
    ax.tick_params(axis="y", colors=text_color)
    plt.show()
    
def plot_q_values(Q, rewards):
    num_states, num_actions = Q.shape
    grid_size = int(np.sqrt(num_states))  # Assuming a square grid
    
    # Create a grid to represent the vectors
    x, y = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
    
    # Create vectors for each state based on Q-values
    vectors = np.zeros((grid_size, grid_size, 2))
    for state in range(num_states):
        row = state // grid_size
        col = state % grid_size
        q_right = Q[state, 1]  # Q-value for action 1 (right)
        q_down = Q[state, 0]   # Q-value for action 0 (down)
        
        # Normalize the vector
        vector_length = np.sqrt(q_right**2 + q_down**2)
        if vector_length > 0:
            vectors[row, col, 0] = q_right / vector_length
            vectors[row, col, 1] = q_down / vector_length
    
    # Create a figure
    plt.figure(figsize=(8, 6))
    
    # Plot the rewards as a heatmap
    rewards_heatmap = plt.imshow(rewards, cmap='coolwarm', interpolation='nearest', aspect='auto', extent=[0, grid_size-1, 0, grid_size-1])
    plt.xlabel('X')
    plt.title('Rewards Heatmap')
    plt.colorbar(rewards_heatmap, label='Values')
    
    # Overlay the Q-value vectors on top of the heatmap
    plt.quiver(x, y, vectors[:, :, 0], vectors[:, :, 1], angles='xy', scale_units='xy', scale=1, color='black')
    
    plt.tight_layout()
    plt.show()

