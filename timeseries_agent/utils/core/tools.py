import numpy as np
import torch
import pandas as pd
from typing import Union, Tuple, Optional, Dict, Any, List
from .rewards import calculate_directional_reward, calculate_proportional_reward, calculate_threshold_reward

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

def discount_rewards(rewards: List[float], gamma: float) -> torch.Tensor:
    """
    Calculates discounted rewards for a given sequence of rewards.

    Args:
        rewards (List[float]): A list of immediate rewards.
        gamma (float): The discount factor (between 0 and 1).
            Note: When gamma is close to 1 (e.g., 0.99), future rewards are highly valued; when it's closer to 0, immediate rewards dominate. 

    Returns:
        torch.Tensor: A tensor of discounted rewards, of the same length as the input rewards.
    """
    discounted = []
    running_add = 0
    rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
    running_add = torch.tensor(0.0, dtype=torch.float32)
    for r in reversed(rewards_tensor):
        running_add = r + gamma * running_add
        discounted.insert(0, running_add.item())
    return torch.tensor(discounted, dtype=torch.float32)

def calculate_reward(current_state_last_val: float,
                      next_state_last_val: float,
                      sampled_action: int,
                      reward_config: Optional[Dict[str, Any]] = None,
                      class_weights: Optional[Dict[int, float]] = None) -> float:
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
        reward_config (Optional[Dict[str, Any]]): Configuration for reward calculation.
            Supports different methods:
            - 'directional': Basic directional accuracy (default if no config)
            - 'proportional': Scaled by price change magnitude
            - 'threshold': Based on significant price movements
            Each method accepts additional parameters as defined in rewards.py

    Returns:
        float: Reward value based on prediction accuracy and chosen reward method.
    """
    if reward_config is None:
        # Default to directional reward if no config provided
        base_reward = calculate_directional_reward(current_state_last_val, next_state_last_val, sampled_action)
    else:
        method = reward_config.get('method', 'directional')
    
        # Calculate base reward based on method
        if method == 'directional':
            reward_map = reward_config.get('reward_map')
            base_reward = calculate_directional_reward(
                current_state_last_val, 
                next_state_last_val, 
                sampled_action,
                reward_map
            )
        
        elif method == 'proportional':
            scale = reward_config.get('scale', 1.0)
            min_change_pct = reward_config.get('min_change_pct', 0.0001)
            base_reward = calculate_proportional_reward(
                current_state_last_val,
                next_state_last_val,
                sampled_action,
                scale,
                min_change_pct
            )
        
        elif method == 'threshold':
            threshold = reward_config.get('threshold', 0.01)
            reward_map = reward_config.get('reward_map')
            base_reward = calculate_threshold_reward(
                current_state_last_val,
                next_state_last_val,
                sampled_action,
                threshold,
                reward_map
            )
        else:
            raise ValueError(f"Unknown reward method: {method}")
        
    # Apply class weights if provided
    if class_weights and sampled_action in class_weights:
        base_reward *= class_weights[sampled_action]
        
    return base_reward
