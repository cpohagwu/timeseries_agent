"""Timeseries Agent: A reinforcement learning library for time series data.

This package provides tools for applying reinforcement learning to time series data,
with a focus on financial, economic and infinite-runner-style game (like the Chrome dinosaur game) applications.
"""

# Expose high-level API
from .api import (
    create_env,
    create_agent,
    train_agent,
    train_from_csv,
    load_agent
)

# Expose core classes for advanced usage
from .data import SequentialTimeSeriesDataset
from .agents import PPOAgent, ReinforceAgent, ReinforceStepAgent
from .utils import get_state_tensor, calculate_reward

__all__ = [
    # High-level API
    'create_env',
    'create_agent',
    'train_agent',
    'train_from_csv',
    'load_agent',
    
    # Core classes
    'SequentialTimeSeriesDataset',
    'get_state_tensor',
    'calculate_reward',
    'PPOAgent',
    'ReinforceAgent',
    'ReinforceStepAgent'
]
