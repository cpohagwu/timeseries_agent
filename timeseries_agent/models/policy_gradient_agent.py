import torch
import torch.nn as nn
import torch.optim as optim
import lightning as L
from typing import List, Any, Union
import torch.nn.functional as F
import numpy as np
import pandas as pd

# Import helper functions
from ..utils.helpers import get_state_tensor, calculate_reward, sample_action


class PolicyGradientAgent(L.LightningModule):
    """
    A Reinforcement Learning agent using Policy Gradient implemented with PyTorch Lightning.
    Trains by interacting sequentially with time series data.

    Args:
        full_data (pd.DataFrame): The entire time series DataFrame.
        target_column (Union[str, int]): Name or index of the target column for reward calculation.
        input_features (int): Number of features in the input time series.
        lookback (int): Number of past time steps used as state.
        hidden_layers (List[int]): Neurons in each hidden layer.
        output_size (int): Number of possible actions (e.g., 3 for Up, Down, Same).
        learning_rate (float): Learning rate for the optimizer.
        normalize_state (bool): Whether to normalize the state window.
        epsilon_start (float): Initial epsilon for epsilon-greedy exploration.
        epsilon_end (float): Final epsilon value.
        epsilon_decay_epochs (int): Number of epochs over which epsilon decays.
        activation_fn (nn.Module): Activation function. Defaults to nn.Tanh().
    """
    def __init__(self,
                 full_data: pd.DataFrame,
                 target_column: Union[str, int],
                 input_features: int,
                 lookback: int,
                 hidden_layers: List[int],
                 output_size: int = 3,
                 learning_rate: float = 1e-3,
                 normalize_state: bool = True,
                 epsilon_start: float = 0.9,
                 epsilon_end: float = 0.05,
                 epsilon_decay_epochs_rate: float = 0.5,
                 num_training_epochs: int = 1000,
                 activation_fn: nn.Module = nn.Tanh(),
                 eval_noise_factor: float = 0.0,
                 ):
        super().__init__()

        if not hidden_layers:
            raise ValueError("hidden_layers list cannot be empty.")

        # Store essential parameters needed for the RL loop
        self.full_series_np = full_data.values.astype(np.float32)
        self.lookback = lookback
        self.normalize_state = normalize_state
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_epochs = int(num_training_epochs * epsilon_decay_epochs_rate)
        self.output_size = output_size
        self.eval_noise_factor = eval_noise_factor

         # Determine the index of the target column for reward calculation
        if isinstance(target_column, str):
            if target_column not in full_data.columns:
                raise ValueError(f"Target column '{target_column}' not found in DataFrame.")
            self.target_col_idx = full_data.columns.get_loc(target_column)
        elif isinstance(target_column, int):
            if target_column < 0 or target_column >= full_data.shape[1]:
                raise IndexError("Target column index is out of bounds.")
            self.target_col_idx = target_column
        else:
            raise TypeError("target_column must be str or int.")


        # Save hyperparameters for logging
        self.save_hyperparameters(ignore=['full_data', 'activation_fn'])
        # Save checkpoint for model loading

        self.learning_rate = learning_rate # Ensure lr is saved for configure_optimizers
        self.activation_fn = activation_fn

        # --- Policy Network ---
        input_size = input_features * lookback
        layers = []
        layers.append(nn.Linear(input_size, hidden_layers[0]))
        layers.append(self.activation_fn)

        for i in range(len(hidden_layers) - 1):
            layers.append(nn.Linear(hidden_layers[i], hidden_layers[i+1]))
            layers.append(self.activation_fn)

        layers.append(nn.Linear(hidden_layers[-1], output_size))
        # Output raw logits, Softmax will be applied before sampling

        self.network = nn.Sequential(*layers)

        # Internal state for RL loop progress
        self.automatic_optimization = False # We need manual optimization for PG


    def forward(self, state_tensor: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the policy network.

        Args:
            state_tensor (torch.Tensor): Input state tensor of shape
                                         (batch_size, lookback, input_features)
                                         or (lookback, input_features) if batch_size=1.

        Returns:
            torch.Tensor: Action logits of shape (batch_size, output_size).
        """
        # Handle potential batch dimension
        if state_tensor.ndim == 3:
             batch_size, lookback, features = state_tensor.size()
             state_flat = state_tensor.view(batch_size, -1) # Flatten: (batch, lookback * features)
        elif state_tensor.ndim == 2:
             lookback, features = state_tensor.size()
             state_flat = state_tensor.view(1, -1) # Add batch dim: (1, lookback * features)
        else:
            raise ValueError(f"Unexpected state_tensor ndim: {state_tensor.ndim}")

        logits = self.network(state_flat)
        return logits

    def get_epsilon(self) -> float:
        """Calculates the current epsilon value based on the training epoch."""
        # Exponential decay
        decay_rate = (self.epsilon_end / self.epsilon_start)**(1/self.epsilon_decay_epochs)
        epsilon = self.epsilon_start * (decay_rate ** self.current_epoch)
        epsilon = max(self.epsilon_end, epsilon)
        return epsilon


    # --- Reinforcement Learning Loop ---
    # We override training_step to implement the sequential RL interaction
    def training_step(self, batch: Any, batch_idx: int):
        """
        Performs the Policy Gradient update loop for one epoch sequentially.
        The 'batch' from DataLoader likely contains indices, but we iterate
        through the time series step-by-step here.
        """
        # Get the optimizer
        opt = self.optimizers()

        # Calculate current epsilon
        current_epsilon = self.get_epsilon()

        total_reward_epoch = 0
        total_loss_epoch = torch.tensor(0.0)
        steps_in_epoch = 0

        # Manually iterate through the valid time steps for the epoch
        # The RLTimeSeriesDataset provides the correct indices via the dataloader
        total_loss_epoch = torch.tensor(0.0).to(self.device)
        for time_idx in batch: # Assuming batch contains time indices
             time_idx = time_idx.item() # Get index from tensor

             # Ensure we don't go past the end of the series when getting next state
             if time_idx >= len(self.full_series_np) - 1:
                 continue

             steps_in_epoch += 1

             # 1. Get Current State
             state = get_state_tensor(self.full_series_np, time_idx, self.lookback, self.normalize_state)
             state = state.to(self.device) # Move state to correct device

             # 2. Get Action Probabilities from Policy Network
             logits = self(state) # Shape: (1, output_size)
             probabilities = F.softmax(logits, dim=1).squeeze(0) # Shape: (output_size,)

             # 3. Sample Action (using epsilon-greedy)
             action, log_prob = sample_action(probabilities, current_epsilon)

             # 4. Get Next State and Calculate Reward
             next_state = get_state_tensor(self.full_series_np, time_idx + 1, self.lookback, self.normalize_state)

             # Extract values from the *target column* for reward calculation
             # These are the values at the *end* of the respective lookback windows
             current_val = self.full_series_np[time_idx - 1, self.target_col_idx]
             next_val = self.full_series_np[time_idx, self.target_col_idx] # Actual value at current time_idx

             reward = calculate_reward(current_val, next_val, action)
             total_reward_epoch += reward

             # 5. Calculate Policy Gradient Loss
             # We want to maximize expected reward, so minimize negative log_prob * reward
             loss = -log_prob * reward
             loss = loss.to(self.device)  # Move loss to agent's device
             total_loss_epoch += loss # Accumulate the loss tensor

             # 6. Backpropagate and Optimize
             opt.zero_grad() # Zero gradients before backpropagation
             self.manual_backward(loss) # Perform backward pass
             torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0) # Gradient clipping
             opt.step() # Update weights

        # --- Logging ---
        avg_reward = total_reward_epoch / steps_in_epoch if steps_in_epoch > 0 else 0
        avg_loss = total_loss_epoch / steps_in_epoch if steps_in_epoch > 0 else torch.tensor(0.0) # Ensure avg_loss is a tensor

        self.log('train_reward', avg_reward, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_loss', avg_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('epsilon', current_epsilon, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # training_step in manual opt mode should return None or the loss dict
        return {'loss': avg_loss, 'reward': avg_reward}

    # Override configure_optimizers
    def configure_optimizers(self) -> optim.Optimizer:
        """Configures the optimizer."""
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def validation_step(self, batch: Any, batch_idx: int):
        """Evaluates the agent's greedy policy on a validation dataset."""
        self.eval()  # Set agent to evaluation mode
        total_test_reward = 0
        pass_count = 0
        num_test_steps = 0

        noisy_data_np = self.full_series_np.copy()

        if self.eval_noise_factor > 0:
            # Generate noise based on the standard deviation of the dataset
            noise = np.random.randn(*self.full_series_np.shape) * self.eval_noise_factor * self.full_series_np.std(axis=0)

            # Log the noise factor for tracking
            self.log('eval_noise_factor', self.eval_noise_factor, on_step=False, on_epoch=True, prog_bar=True, logger=True)

            # Add noise to the entire dataset for evaluation
            noisy_data_np += noise

        with torch.no_grad():  # Disable gradients during evaluation
            for i in range(self.lookback, len(noisy_data_np) - 1):
                num_test_steps += 1
                state = get_state_tensor(noisy_data_np, i, self.lookback, self.normalize_state)
                state = state.to(self.device)
                logits = self(state)
                probabilities = F.softmax(logits, dim=1)
                action = torch.argmax(probabilities).item()  # Greedy action

                # Calculate actual reward
                current_val = noisy_data_np[i - 1, self.target_col_idx]
                next_val = noisy_data_np[i, self.target_col_idx]
                reward = calculate_reward(current_val, next_val, action)
                total_test_reward += reward
                if reward == 1:
                    pass_count += 1

        avg_test_reward = total_test_reward / num_test_steps if num_test_steps > 0 else 0
        pass_percentage = pass_count / num_test_steps * 100 if num_test_steps > 0 else 0

        self.log('val_avg_reward', avg_test_reward, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_pass_percentage', pass_percentage, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return {'val_avg_reward': avg_test_reward, 'val_pass_percentage': pass_percentage}
