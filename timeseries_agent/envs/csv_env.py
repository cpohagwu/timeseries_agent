"""CSV environment wrapper for sequential decision making."""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Union, Dict, Any
import torch
from collections import Counter

from ..utils.core.tools import get_state_tensor, calculate_reward
from ..data import SequentialTimeSeriesDataset

class CsvEnv:
    """Environment wrapper for CSV-based time series data.
    
    Provides a standardized interface for sequential decision-making with CSV data,
    integrating with existing SequentialTimeSeriesDataset for training.

    Args:
        data (pd.DataFrame): Input DataFrame
        feature_cols (List[str]): Columns to use as state features
        target_col (str): Column used for calculating rewards
        lookback (int): Number of past timesteps to include in state
        normalize_state (bool): Whether to normalize state features
        reward_config (Optional[Dict[str, Any]]): Reward configuration
            Default: {'method': 'directional'}
            Supports 'directional', 'proportional', 'threshold' methods.
    
    """
    def __init__(
        self,
        data: pd.DataFrame,
        feature_cols: List[str],
        target_col: str,
        lookback: int = 10,
        normalize_state: bool = True,
        reward_config: Optional[Dict[str, Any]] = None,
        test_size: float = 0.2
    ):
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input 'data' must be a pandas DataFrame")
        if not feature_cols:
            raise ValueError("feature_cols cannot be empty")
        if target_col not in data.columns:
            raise ValueError(f"target_col '{target_col}' not found in DataFrame")
        
        # Split data into train/test
        train_size = int(len(data) * (1 - test_size))
        self.train_df = data.iloc[:train_size].copy()
        self.test_df = data.iloc[train_size:].copy()
        
        # Use train data as default
        self.is_train = True
        self.df = self.train_df
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.lookback = lookback
        self.normalize_state = normalize_state
        self.reward_config = reward_config
        
        # Convert feature columns to numpy for get_state_tensor
        self.features = self.df[feature_cols].values.astype(np.float32)
        self.targets = self.df[target_col].values.astype(np.float32)
        
        # Calculate class weights based on the true distribution
        self.class_weights = self._calculate_class_weights()
        
        # Track current position
        self.pointer = lookback
        self._check_data_length()
        
    def _calculate_class_weights(self) -> Dict[int, float]:
        """Calculate class weights based on the true distribution of movements."""
        # Calculate true movements
        movements = []
        for i in range(len(self.targets) - 1):
            curr_val = self.targets[i]
            next_val = self.targets[i + 1]
            movement = 0 if next_val > curr_val else 1 if next_val < curr_val else 2
            movements.append(movement)


        # Count occurrences
        total = len(movements)
        class_counts = Counter(movements)

        # Calculate inverse frequency weights
        weights = {}
        for cls in class_counts:
            if class_counts[cls] > 0:
                weights[cls] = total / (len(class_counts) * class_counts[cls])
            else:
                weights[cls] = 1.0  # Default weight for unseen classes
                
        return weights

    def _check_data_length(self):
        """Validate data length against lookback window."""
        if len(self.df) <= self.lookback:
            raise ValueError(
                f"DataFrame length {len(self.df)} too short for lookback {self.lookback}"
            )

    def reset(self) -> torch.Tensor:
        """Reset environment to initial state.
        
        Returns:
            torch.Tensor: Initial state tensor of shape (lookback, num_features)
        """
        self.pointer = self.lookback
        return get_state_tensor(
            self.features, 
            self.pointer,
            self.lookback,
            self.normalize_state
        )

    def step(self, action: int) -> Tuple[torch.Tensor, float, bool, Dict]:
        """Take an action in the environment.
        
        Args:
            action (int): Action to take (0=Up, 1=Down, 2=Same)
            
        Returns:
            Tuple containing:
            - torch.Tensor: Next state tensor
            - float: Reward for the action
            - bool: Whether episode is done
            - dict: Additional info
        """
        if self.pointer >= len(self.df) - 1:  # Need space for both pointer and pointer+1
            raise ValueError(f"Environment has reached the end of data at index {self.pointer}")

        # Calculate reward 
        # Extract values from the *target column* for reward calculation
        # These are the values at the *end* of the respective lookback windows
        # The current_val is at the end of the window used to predict `action`.
        # The next_val is at the time_idx, which is the actual outcome.
        current_val = self.targets[self.pointer]
        next_val = self.targets[self.pointer + 1]
        reward = calculate_reward(
            current_val, 
            next_val, 
            action, 
            self.reward_config, 
            self.class_weights
        )

        # Get next state
        self.pointer += 1
        next_state = get_state_tensor(
            self.features,
            self.pointer,
            self.lookback,
            self.normalize_state
        )

        done = self.pointer >= len(self.df) - 1
        true_action = 0 if next_val > current_val else 1 if next_val < current_val else 2
        info = {"timestep": self.pointer,
                "true_action": true_action} # True action based on actual next value

        return next_state, reward, done, info

    def get_state(self) -> torch.Tensor:
        """Get current state without advancing environment.
        
        Returns:
            torch.Tensor: Current state tensor
        """
        return get_state_tensor(
            self.features,
            self.pointer,
            self.lookback,
            self.normalize_state
        )

    def set_train_mode(self, train: bool = True):
        """Switch between train and test datasets.
        
        Args:
            train (bool): If True, use training data, else use test data
        """
        self.is_train = train
        self.df = self.train_df if train else self.test_df
        # Update features and targets
        self.features = self.df[self.feature_cols].values.astype(np.float32)
        self.targets = self.df[self.target_col].values.astype(np.float32)
        # Reset environment
        self.pointer = self.lookback
        # Recalculate class weights for current dataset
        self.class_weights = self._calculate_class_weights()
        
    def to_dataset(self, train: bool = True) -> SequentialTimeSeriesDataset:
        """Convert environment to SequentialTimeSeriesDataset for training.
        
        Args:
            train (bool): If True, use training data, else use test data
            
        Returns:
            SequentialTimeSeriesDataset: Dataset for training/validation
        """
        data = self.train_df if train else self.test_df
        return SequentialTimeSeriesDataset(
            data=data,
            lookback=self.lookback
        )

    @classmethod
    def from_csv(
        cls,
        filepath: str,
        feature_cols: List[str],
        target_col: str,
        **kwargs
    ) -> "CsvEnv":
        """Create environment directly from CSV file.
        
        Args:
            filepath (str): Path to CSV file
            feature_cols (List[str]): Feature column names
            target_col (str): Target column name
            **kwargs: Additional arguments passed to constructor
            
        Returns:
            CsvEnv: Initialized environment
        """
        df = pd.read_csv(filepath)
        return cls(
            data=df,
            feature_cols=feature_cols,
            target_col=target_col,
            **kwargs
        )
