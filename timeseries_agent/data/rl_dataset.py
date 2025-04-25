import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from typing import Union, Tuple

class RLTimeSeriesDataset(Dataset):
    """
    Dataset for Reinforcement Learning on time series.
    Provides access to indices to allow sequential processing in the RL loop.

    Args:
        data (pd.DataFrame): Input DataFrame containing time series data.
        lookback (int): Number of past time steps to use as state (X).
                        Note: Actual state extraction happens outside the dataset.
    """
    def __init__(self,
                 data: pd.DataFrame,
                 lookback: int):
        super().__init__()

        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input 'data' must be a pandas DataFrame.")
        if lookback <= 0:
            raise ValueError("Lookback must be a positive integer.")

        self.series_length = len(data)
        self.lookback = lookback

        # The dataset length allows fetching indices up to the point
        # where a *next* state can also be determined.
        # We need index `i` and `i+1` to calculate reward.
        # The state for index `i` uses data up to `i-1`.
        # The state for index `i+1` uses data up to `i`.
        # The latest possible `i` is `series_length - 1`.
        # So, we need indices from `lookback` up to `series_length - 1`.
        self.num_steps = self.series_length - lookback
        if self.num_steps <= 0:
             raise ValueError("Not enough data points for the given lookback.")


    def __len__(self) -> int:
        """Returns the total number of valid starting steps."""
        return self.num_steps

    def __getitem__(self, idx: int) -> int:
        """
        Returns the time series index corresponding to the dataset index.
        This index represents the *current* time step for which a state
        and action will be determined in the RL loop.

        Args:
            idx (int): The dataset index (from 0 to len(self)-1).

        Returns:
            int: The corresponding time series index (from `lookback` to `series_length - 1`).
        """
        if not 0 <= idx < self.num_steps:
            raise IndexError("Dataset index out of bounds.")
        # Map dataset index (0 to num_steps-1) to time series index (lookback to series_length-1)
        time_series_idx = idx + self.lookback
        return time_series_idx