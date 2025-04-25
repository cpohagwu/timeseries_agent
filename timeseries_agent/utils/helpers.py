import numpy as np
import torch
import pandas as pd
from typing import Union, Tuple, Optional

def get_state_tensor(series: np.ndarray,
                       idx: int,
                       lookback: int,
                       normalize: bool = True) -> torch.Tensor:
    """
    Extracts the state (lookback window) ending at index `idx-1`.

    Args:
        series (np.ndarray): The full time series data (num_steps, num_features).
        idx (int): The current time step index in the original series.
                   The lookback window will be series[idx-lookback : idx].
        lookback (int): Number of past time steps in the state.
        normalize (bool): Whether to normalize the state window.

    Returns:
        torch.Tensor: The state tensor of shape (lookback, num_features).
    """
    if idx < lookback:
        # Not enough history for a full lookback window
        raise IndexError(f"Index {idx} is too small for lookback {lookback}")

    start_idx = idx - lookback
    end_idx = idx
    x_window = series[start_idx:end_idx, :].copy() # Ensure it's a copy

    # Apply optional normalization to the input window
    if normalize:
        mean = np.mean(x_window, axis=0, keepdims=True)
        std = np.std(x_window, axis=0, keepdims=True)
        x_window = (x_window - mean) / (std + 1e-8) # Epsilon for stability

    return torch.tensor(x_window, dtype=torch.float32)


def calculate_reward(current_state_last_val: float,
                      next_state_last_val: float,
                      sampled_action: int) -> int:
    """
    Calculates the reward based on the action taken and the actual outcome.
    Action 0: Predict Up
    Action 1: Predict Down
    Action 2: Predict Same

    Args:
        current_state_last_val (float): The value of the target variable at the
                                       last step of the current state window.
        next_state_last_val (float): The value of the target variable at the
                                     last step of the next state window (the actual outcome).
        sampled_action (int): The action taken by the agent (0, 1, or 2).

    Returns:
        int: Reward (+1 for correct prediction, -1 for incorrect).
    """
    actual_change = 2 # Default to 'Same'
    if next_state_last_val > current_state_last_val:
        actual_change = 0 # Actual outcome is 'Up'
    elif next_state_last_val < current_state_last_val:
        actual_change = 1 # Actual outcome is 'Down'
    else:
        actual_change = 2 # Actual outcome is 'Same'

    # Reward is 1 if predicted action matches actual change, -1 otherwise
    return 1 if sampled_action == actual_change else -1


def sample_action(probabilities: torch.Tensor, epsilon: float) -> Tuple[int, torch.Tensor]:
    """
    Samples an action using epsilon-greedy strategy from action probabilities.

    Args:
        probabilities (torch.Tensor): Tensor of action probabilities (output of softmax).
                                      Shape: (output_size,)
        epsilon (float): The probability of taking a random action (exploration).

    Returns:
        Tuple[int, torch.Tensor]: A tuple containing:
            - action (int): The sampled action index (0, 1, or 2).
            - log_prob (torch.Tensor): The log probability of the sampled action.
    """
    output_size = probabilities.shape[0]
    if torch.rand(1).item() < epsilon:
        # Explore: Sample randomly
        action = torch.randint(output_size, (1,)).item()
    else:
        # Exploit: Sample from the network's probability distribution
        action = torch.argmax(probabilities).item()

    # Calculate log probability of the chosen action
    log_prob = torch.log(probabilities[action] + 1e-9) # Add epsilon for stability

    return action, log_prob