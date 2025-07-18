"""Reward utilities for time series decision making."""

import pandas as pd
import numpy as np
from typing import Callable, Dict, Any, Optional, Union

def calculate_directional_reward(
    current_state_last_val: float,
    next_state_last_val: float,
    sampled_action: int,
    reward_map: Optional[Dict[str, float]] = None
) -> float:
    """Enhanced version of the original calculate_reward with customizable values.
    
    Args:
        current_state_last_val (float): Current value
        next_state_last_val (float): Next value
        sampled_action (int): Action taken (0=Up, 1=Down, 2=Same)
        reward_map (Dict[str, float]): Custom reward values
            Default: {'correct': 1.0, 'incorrect': -1.0}
    
    Returns:
        float: Reward value
    """
    rewards = reward_map or {'correct': 1.0, 'incorrect': -1.0}
    
    if next_state_last_val > current_state_last_val:
        actual = 0  # Up
    elif next_state_last_val < current_state_last_val:
        actual = 1  # Down
    else:
        actual = 2  # Same

    return rewards['correct'] if sampled_action == actual else rewards['incorrect']

def calculate_proportional_reward(
    current_state_last_val: float,
    next_state_last_val: float,
    sampled_action: int,
    scale: float = 1.0,
    min_change_pct: float = 0.0001
) -> float:
    """Calculate reward proportional to the magnitude of price change.
    
    Args:
        current_state_last_val (float): Current value
        next_state_last_val (float): Next value
        sampled_action (int): Action taken (0=Up, 1=Down, 2=Same)
        scale (float): Scaling factor for reward magnitude
        min_change_pct (float): Minimum % change required for non-zero reward
        
    Returns:
        float: Scaled reward value
    """
    pct_change = (next_state_last_val - current_state_last_val) / (current_state_last_val) if current_state_last_val != 0 else 0.0
    # No reward for very small changes
    if abs(pct_change) < min_change_pct:
        return 0.0
    
    # Determine if prediction was correct
    if pct_change > 0:  # Price went up
        correct = (sampled_action == 0)
    elif pct_change < 0:  # Price went down
        correct = (sampled_action == 1)
    else:  # No change or negligible change
        correct = (sampled_action == 2)

    # Scale reward by change magnitude
    base_reward = abs(pct_change) * scale
    return base_reward if correct else -base_reward

def calculate_threshold_reward(
    current_state_last_val: float,
    next_state_last_val: float,
    sampled_action: int,
    threshold: float = 0.01,
    reward_map: Optional[Dict[str, float]] = None
) -> float:
    """Calculate reward based on significant price movements.
    
    Args:
        current_state_last_val (float): Current value
        next_state_last_val (float): Next value
        sampled_action (int): Action taken (0=Up, 1=Down, 2=Same)
        threshold (float): Required price change threshold
        reward_map (Dict[str, float]): Custom reward values
            Default: {'correct': 1.0, 'incorrect': -1.0, 'neutral': 0.0}
    
    Returns:
        float: Reward value
    """
    rewards = reward_map or {
        'correct': 1.0,
        'incorrect': -1.0,
        'neutral': 0.0
    }

    threshold_change = (next_state_last_val - current_state_last_val)
    # Determine significance of move
    if abs(threshold_change) < threshold:
        return rewards['neutral']
    
    # For significant moves, reward based on direction
    if threshold_change > threshold:
        actual = 0  # Significant up move
    elif threshold_change < -threshold:
        actual = 1  # Significant down move
    else:
        actual = 2  # No significant move

    return rewards['correct'] if sampled_action == actual else rewards['incorrect']
