## TimeSeries RL Agent Example Notebook
# Welcome to the RLTimeSeriesAgent example notebook! This notebook demonstrates how to train a reinforcement learning agent using the RLTimeSeriesAgent package. The agent is designed to work with time series data and can be used for various tasks such as forecasting, anomaly detection, and more.
# The package is built on top of PyTorch and PyTorch Lightning, making it easy to integrate with existing PyTorch workflows. The agent uses a policy gradient approach for training, allowing it to learn from the environment and improve its performance over time.
# In this example, we will create a sample time series dataset, set up the RL environment parameters, define the agent model, and train the agent using PyTorch Lightning. We will also evaluate the trained model and plot the training logs.

## Install timestreams-agent package
# !pip install -i https://test.pypi.org/simple/ timeseries-agent --quiet
# !pip install lightning torch --quiet
# !pip install yfinance --quiet

# Load Dataset and Train Model

## Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import lightning as L
from lightning.pytorch import seed_everything
seed_everything(42, workers=True) # sets seeds for numpy, torch and python.random.

# Import from the package. Note: Comment out the sys.path.append line if running in a Jupyter notebook or similar environment.
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from timeseries_agent import RLTimeSeriesDataset, PolicyGradientAgent


## --- 1. Create Sample Data ---
# This is a sample dataset for demonstration purposes. In practice, you would load your own time series data.
# The dataset consists of a time series with 4 different patterns, each repeated 5 times.
data1 = np.array([1, 2, 2] * 5)
data2 = np.array([1, 0, 0] * 5)
data3 = np.array([0, -2, -2, 0] * 5)
data4 = np.array([np.sin(x) for x in np.linspace(0, 8 * np.pi, 20)])
# Concatenate the data to create a single time series
value_col = np.concatenate([data1, data2, data3, data4] * 3)

# Create a second feature by rolling the first feature and adding noise
feature2_col = np.roll(value_col, 1) + np.random.randn(len(value_col)) * 0.1
feature2_col[0] = value_col[0] # Set the first value to avoid NaN after rolling

# Create a DataFrame with the time series data
data_df = pd.DataFrame({'value': value_col, 'feature2': feature2_col})
print("Sample DataFrame Head:")
print(data_df.head())
print(f"\nDataFrame Shape: {data_df.shape}")


## --- 2. Setup RL Environment Parameters ---
LOOKBACK = 7                        # Number of past time steps to consider for the agent
TARGET_COLUMN = 'value'             # Column for reward calculation and action
NORMALIZE_STATE = True              # Normalize state if True
NUM_FEATURES = data_df.shape[1]     # Number of features in the dataset
OUTPUT_SIZE = 3                     # Number of actions (Up, Down, Same)


## --- 3. Dataset and DataLoader ---
# The DataLoader will provide indices for the agent to process sequentially
# Batch size = 1 ensures step-by-step processing if needed, but the agent
# loop handles sequence internally now. Larger batch size for the loader
# just means the agent processes more indices per `training_step` call.
# We use batch_size = length of dataset to process the whole epoch in one go.

try:
    full_dataset = RLTimeSeriesDataset(
        data=data_df,
        lookback=LOOKBACK,
    )
    print(f"\nRL Dataset created. Num valid steps: {len(full_dataset)}")

    # Example: Get the time series index for the first dataset item
    first_ts_idx = full_dataset[0]
    print(f"First dataset item corresponds to time series index: {first_ts_idx}") # Should be == LOOKBACK

    # DataLoader: Load all indices for the epoch at once
    # Shuffle=False is crucial for sequential processing if needed,
    # although our agent processes sequentially internally anyway.
    dataloader = DataLoader(full_dataset, batch_size=len(full_dataset), shuffle=False, num_workers=0)
    print(f"DataLoader created. Batches: {len(dataloader)}")

except ValueError as e:
    print(f"\nError creating dataset: {e}")
    exit()


## --- 4. Model (Agent) Definition ---
# The agent is a PyTorch Lightning module that handles the RL training loop.
# It uses a policy gradient approach to learn from the environment.
hidden_layers_config = [100, 100, 10]           # Flexible hidden layers
learning_rate = 0.001                           # Initial learning rate
epsilon_start = 1.0                             # Initial exploration rate
epsilon_end = 0.01                              # Final exploration rate
epsilon_decay_epochs = 1000                     # Number of epochs for epsilon decay

agent = PolicyGradientAgent(
    full_data=data_df,                          # Agent needs access to full pandas DataFrame for reward calculation
    target_column=TARGET_COLUMN, 
    input_features=NUM_FEATURES,
    lookback=LOOKBACK,      
    hidden_layers=hidden_layers_config, 
    output_size=OUTPUT_SIZE,
    learning_rate=learning_rate,
    normalize_state=NORMALIZE_STATE, 
    epsilon_start=epsilon_start, 
    epsilon_end=epsilon_end, 
    epsilon_decay_epochs=epsilon_decay_epochs,
    # activation_fn=nn.ReLU(),            # Optional: specify activation function
    # eval_noise_factor=0.1,              # Optional: noise for evaluation
)

print("\nAgent Policy Network Summary:")
print(agent.network) # Print model architecture


## --- 5. Training with PyTorch Lightning ---
# Move the agent's model to the appropriate device (GPU if available)
accelerator = ('gpu', agent.to('cuda')) if torch.cuda.is_available() else ('cpu', agent.to('cpu'))
num_epochs = 1000 # Number of training epochs

trainer = L.Trainer(
    max_epochs=num_epochs,
    accelerator='auto',         # Use 'gpu' if available, otherwise 'cpu'
    devices='auto',             # Use all available devices (e.g., GPUs)
    log_every_n_steps=1,        # Logging happens per epoch in our setup
    enable_checkpointing=False, # Disable default checkpointing if not needed
    logger=L.pytorch.loggers.CSVLogger("logs", name="rl_agent"), # Log to CSV
    deterministic=True,         # Ensure deterministic behavior for reproducibility
)

print(f"\nStarting RL training on {accelerator[0].upper()}...")
# Train the agent
# The dataloader provides indices; the agent runs the RL loop in training_step
trainer.fit(model=agent, dataloaders=dataloader)
print("\nTraining finished.")


## --- 6. Evaluate Trained Model ---
# We will evaluate the trained model using the same dataloader.
# The evaluation will use the same indices as the training, but the agent will not update its weights.
# If eval_noise_factor is set, the agent will add noise to the actions during evaluation.
# This is useful for testing the agent's robustness to noise.
trainer.validate(model=agent, dataloaders=dataloader)
print("\nEvaluation finished.")


## --- 7. Save Trained Model ---
model_dir = trainer.logger.log_dir 
trainer.save_checkpoint(os.path.join(model_dir, "policy_agent.ckpt")) # Save the model checkpoint
print(f"Trained model saved to: {os.path.join(model_dir, 'policy_agent.ckpt')}")


## --- 8. (Optional) Plotting Logs ---
# You can plot the logs saved by the CSVLogger ('logs/rl_agent/version_X/metrics.csv')
logs = pd.read_csv(os.path.join(model_dir, "metrics.csv"))
logs.set_index('epoch', inplace=True)
logs[['train_reward', 'train_loss']].plot(subplots=True, figsize=(10, 6))
plt.show()
plt.savefig(os.path.join(model_dir, "training_logs.png")) # Save the plot
print("Training logs plotted and saved to: ", os.path.join(model_dir, "training_logs.png"))
