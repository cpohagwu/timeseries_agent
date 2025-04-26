"""
Example script demonstrating how to use the ModelTuner to train multiple models
with different hyperparameter configurations.
"""

import pandas as pd
import numpy as np
import torch
import sys
import os
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

import lightning as L
from torch.utils.data import DataLoader
from timeseries_agent import RLTimeSeriesDataset, PolicyGradientAgent
from timeseries_agent.tuning.tuner import ModelTuner

# --- 1. Create Sample Data (same as in example_train.py) ---
data1 = np.array([1, 2, 2] * 5)
data2 = np.array([1, 0, 0] * 5)
data3 = np.array([0, -2, -2, 0] * 5)
data4 = np.array([np.sin(x) for x in np.linspace(0, 8 * np.pi, 20)])
value_col = np.concatenate([data1, data2, data3, data4] * 3)

# Create a second feature
feature2_col = np.roll(value_col, 1) + np.random.randn(len(value_col)) * 0.1
feature2_col[0] = value_col[0]

# Create DataFrame
data_df = pd.DataFrame({'value': value_col, 'feature2': feature2_col})

# --- 2. Define hyperparameter ranges to test ---
param_ranges = {
    'learning_rate': [0.001, 0.0005],
    'lookback': [5, 7],
    'hidden_layers': [
        [100, 100, 10],
        [200, 100, 50],
    ],
    'epsilon_start': [1.0, 0.9],
    'epsilon_end': [0.01, 0.05],
    'epsilon_decay_epochs': [5, 10],
}

# Base parameters that will be used for all models
base_params = {
    'normalize_state': True,  # This parameter will be the same for all models
    'eval_noise_factor': 0.1,  # Noise factor for evaluation
}

# --- 3. Create and run the tuner ---
tuner = ModelTuner(
    data_df=data_df,
    base_log_dir="logs",  # Will use the same logs dir as example_train.py
    target_column="value",
)

# Train models with different hyperparameter combinations
results = tuner.train(
    param_ranges=param_ranges,
    num_epochs=10,  # Adjust based on your needs
    base_params=base_params,
)

# --- 4. Display Results ---
print("\nTuning Results (sorted by validation reward):")
print(results)

# Print best performing model details
best_model = results.iloc[0]
print("\nBest Model Configuration:")
for param, value in best_model.items():
    print(f"{param}: {value}")

print(f"\nBest model checkpoint saved at: {best_model['model_dir']}")

# --- 5. Train Final Model with Best Parameters ---
print("\nTraining final model with best parameters...")

# Extract best hyperparameters
best_params = {
    'learning_rate': best_model['learning_rate'],
    'lookback': best_model['lookback'],
    'hidden_layers': best_model['hidden_layers'],
    'epsilon_start': best_model['epsilon_start'],
    'epsilon_end': best_model['epsilon_end'],
    'epsilon_decay_epochs': best_model['epsilon_decay_epochs'],
}

# Combine with base parameters
final_params = {**base_params, **best_params}

# Create dataset for final model
final_dataset = RLTimeSeriesDataset(
    data=data_df,
    lookback=best_params['lookback'],
)
dataloader = DataLoader(final_dataset, batch_size=len(final_dataset), shuffle=False, num_workers=0)

# Create final model with best parameters
agent = PolicyGradientAgent(
    full_data=data_df,
    target_column="value",
    input_features=data_df.shape[1],
    output_size=3,
    **final_params  # This includes both base_params and best_params
)

# Create trainer for final model
trainer = L.Trainer(
    max_epochs=1000,  # Increased epochs for final training
    accelerator='auto',
    devices='auto',
    log_every_n_steps=1,
    enable_checkpointing=True,
    logger=L.pytorch.loggers.CSVLogger("logs", name="tuning_final_model"),
    deterministic=True,
)

print("\nTraining final model with best parameters...")
trainer.fit(agent, dataloader)
print("\nTraining finished.")

print("\nEvaluating final model...")
trainer.validate(model=agent, dataloaders=dataloader)
print("Evaluation finished.")

# Save final model
model_dir = trainer.logger.log_dir
trainer.save_checkpoint(os.path.join(model_dir, "final_model.ckpt"))
print(f"Final model saved to: {os.path.join(model_dir, 'final_model.ckpt')}")

# Plot logs
logs = pd.read_csv(os.path.join(model_dir, "metrics.csv"))
logs.set_index('epoch', inplace=True)
logs[['train_reward', 'train_loss']].plot(subplots=True, figsize=(10, 6))
plt.savefig(os.path.join(model_dir, "final_model_logs.png"))
print(f"Training logs plotted and saved to: {os.path.join(model_dir, 'final_model_logs.png')}")
plt.show()