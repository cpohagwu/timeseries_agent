import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional
from dataclasses import dataclass

@dataclass
class TimeSeriesData:
    """Container for train and test datasets with visualization methods."""
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    data_type: str

    def plot_series(self, save_path: Optional[str] = None) -> None:
        """Plot both train and test series."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
        
        # Plot training data
        ax1.plot(self.train_df['value'], label='Value', linewidth=2)
        ax1.plot(self.train_df['feature2'], label='Feature2', alpha=0.7)
        ax1.set_title(f'Training Data ({self.data_type})')
        ax1.set_xlabel('Time Step')
        ax1.set_ylabel('Value')
        ax1.legend()
        ax1.grid(True)
        
        # Plot test data
        ax2.plot(self.test_df['value'], label='Value', linewidth=2)
        ax2.plot(self.test_df['feature2'], label='Feature2', alpha=0.7)
        ax2.set_title(f'Test Data ({self.data_type})')
        ax2.set_xlabel('Time Step')
        ax2.set_ylabel('Value')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

    def plot_directions(self, save_path: Optional[str] = None) -> None:
        """Plot direction distributions for both datasets."""
        def get_directions(df):
            return np.sign(df['value'].diff()).map({1.0: 'Up', -1.0: 'Down', 0.0: 'Same'})
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot training directions
        train_dirs = get_directions(self.train_df).value_counts()
        train_dirs.plot(kind='bar', ax=ax1, color=['green', 'red', 'blue'])
        ax1.set_title(f'Training Direction Distribution ({self.data_type})')
        ax1.set_ylabel('Count')
        
        # Plot test directions
        test_dirs = get_directions(self.test_df).value_counts()
        test_dirs.plot(kind='bar', ax=ax2, color=['green', 'red', 'blue'])
        ax2.set_title(f'Test Direction Distribution ({self.data_type})')
        ax2.set_ylabel('Count')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()

def create_random_walk(n_steps: int = 1000, 
                      step_size: float = 1.0,
                      train_test_split: float = 0.8,
                      noise_factor: float = 0.1) -> TimeSeriesData:
    """
    Generate a random walk time series with a second noisy feature.
    
    Args:
        n_steps (int): Total number of time steps
        volatility (float): Scale of random steps
        train_test_split (float): Proportion of data for training
        noise_factor (float): Amount of noise in feature2
    
    Returns:
        TimeSeriesData: Container with train and test DataFrames
    """
    # Generate random walk
    steps = np.random.choice([-step_size, 0, step_size], size=n_steps, p=[1/3, 1/3, 1/3])

    value_col = np.cumsum(steps)
    
    # Create second feature with noise
    feature2_col = np.roll(value_col, 1) + np.random.randn(n_steps) * noise_factor
    feature2_col[0] = value_col[0]
    
    # Create full DataFrame
    full_df = pd.DataFrame({
        'value': value_col,
        'feature2': feature2_col
    })
    
    # Split into train and test
    split_idx = int(n_steps * train_test_split)
    train_df = full_df[:split_idx].copy()
    test_df = full_df[split_idx:].copy()
    
    return TimeSeriesData(train_df, test_df, 'Random Walk')

def create_simple_timeseries(num_repeats: int = 3,
                           noise_factor: float = 0.1,
                           train_test_split: float = 0.8) -> TimeSeriesData:
    """
    Creates a sample time series with multiple patterns.
    
    Args:
        num_repeats (int): Number of times to repeat the patterns
        noise_factor (float): Amount of noise in feature2
        train_test_split (float): Proportion of data for training
    
    Returns:
        TimeSeriesData: Container with train and test DataFrames
    """
    # Create patterns
    data1 = np.array([1, 2, 2] * 5)
    data2 = np.array([1, 0, 0] * 5)
    data3 = np.array([0, -2, -2, 0] * 5)
    data4 = np.array([np.sin(x) for x in np.linspace(0, 8 * np.pi, 20)])
    
    # Concatenate with repeats
    value_col = np.concatenate([data1, data2, data3, data4] * num_repeats)
    
    # Create second feature
    feature2_col = np.roll(value_col, 1) + np.random.randn(len(value_col)) * noise_factor
    feature2_col[0] = value_col[0]
    
    # Create full DataFrame
    full_df = pd.DataFrame({
        'value': value_col,
        'feature2': feature2_col
    })
    
    # Split into train and test
    split_idx = int(len(full_df) * train_test_split)
    train_df = full_df[:split_idx].copy()
    test_df = full_df[split_idx:].copy()
    
    return TimeSeriesData(train_df, test_df, 'Sample Patterns')
