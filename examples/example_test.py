# After training a reinforcement learning agent using the timeseries_agent package, you can test the agent's performance on new data.
# This example demonstrates how to set up a test environment, load the trained agent, and evaluate its performance on a sample dataset. The test loop simulates live data updates and calculates rewards based on the agent's actions.

# Load Trained Model and Run Test

## Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import torch.nn.functional as F

# Import from the package. Note: Comment out the sys.path.append line if running in a Jupyter notebook or similar environment.
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from timeseries_agent import PolicyGradientAgent
from timeseries_agent.utils.helpers import get_state_tensor, calculate_reward

## --- 1. Sample Test Data (at least 2x LOOKBACK) ---
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
test_df = pd.DataFrame({'value': value_col, 'feature2': feature2_col})


## --- 2. Instantiate PolicyGradientAgent directly ---
model_dir = "logs/rl_agent/version_0"                       # Path to the trained agent version
model_path = os.path.join(model_dir, "policy_agent.ckpt")   # Path to the model checkpoint
hparams_path = os.path.join(model_dir, "hparams.yaml")      # Path to the hyperparameters file

# Load the trained agent from checkpoint
# Note: The PolicyGradientAgent must be initialized with the full dataset for the first time.
# This could be a dummy dataset or the same dataset used for training which must contain a column with the same name as TARGET_COLUMN inn the hparams.yaml file.
agent = PolicyGradientAgent.load_from_checkpoint(model_path, hparams_file=hparams_path, full_data=test_df)
agent.eval() # Set to evaluation mode
print(f"Loaded trained model from: {model_path}")

LOOKBACK = agent.lookback                       # Use lookback from trained agent
TARGET_COLUMN = agent.hparams.target_column     # Use target column from trained agent
NORMALIZE_STATE = agent.normalize_state         # Use normalization from trained agent

# Initialize with LOOKBACK data points
current_data = test_df.iloc[:LOOKBACK].copy()
print("\nInitial Test DataFrame (first LOOKBACK period):")
print(current_data)


## --- 3. Test Loop (Simulating Live Data Update) ---
print("\nStarting test loop with sudo live data updates...")
total_test_reward = 0
test_pass_count = 0
num_test_steps = 0
predicted_actions = []
true_actions = []

# Loop through the test data starting from LOOKBACK index
# This simulates a live data update where the agent receives new data points sequentially.
for i in range(LOOKBACK, len(test_df)):
    num_test_steps += 1 # Increment test step count
    # Get state from current data
    state = get_state_tensor(current_data.values, LOOKBACK, LOOKBACK, NORMALIZE_STATE) # Last index of current_data
    state = state.to(agent.device) # Move to the same device as the agent

    # Agent predicts action
    with torch.no_grad():                           # No gradient tracking needed for evaluation
        logits = agent(state)                       # Forward pass through the agent
        probabilities = F.softmax(logits, dim=1)    # Convert logits to probabilities
        action = torch.argmax(probabilities).item() # Greedy action selection
        # action = torch.multinomial(probabilities, 1).item() # Sample action based on probabilities (optional)

    # Get actual reward (using next data point from test_df)
    current_val = test_df.iloc[LOOKBACK - 1 + (i - LOOKBACK), test_df.columns.get_loc(TARGET_COLUMN)]   # Value at end of lookback window in current_data
    next_val = test_df.iloc[LOOKBACK + (i - LOOKBACK), test_df.columns.get_loc(TARGET_COLUMN)]          # Next actual value in test_df
    reward = calculate_reward(current_val, next_val, action)
    total_test_reward += reward
    if reward == 1:
        test_pass_count += 1 # Count correct predictions (reward = 1)

    predicted_actions.append(action) # Store predicted action for plotting later
    # Determine true action based on next_val and current_val
    # 0: Up, 1: Down, 2: Same
    if next_val > current_val:
        true_action = 0  
    elif next_val < current_val:
        true_action = 1 
    else:
        true_action = 2  
    true_actions.append(true_action) # Store true action for plotting later
    
    print(f"\n--- Step {i - LOOKBACK + 1} ---")
    print(f"Time Index: {i}, Action: {action}, Reward: {reward}")

    # Update current_data with new data point (simulating live update)
    current_data = pd.concat([current_data, test_df.iloc[[i]]], ignore_index=True)
    current_data = current_data.iloc[1:].reset_index(drop=True) # Shift window

    print("\nUpdated Test DataFrame (last LOOKBACK period):")
    print(current_data)
    time.sleep(0) # Simulate delay for live data update (optional)

# Calculate average test reward and pass percentage
avg_test_reward = total_test_reward / num_test_steps if num_test_steps > 0 else 0
test_pass_percentage = test_pass_count / num_test_steps * 100 if num_test_steps > 0 else 0

print("\n--- Testing Finished ---")
print(f"Total Test Steps: {num_test_steps}")
print(f"Average Test Reward: {avg_test_reward:.4f}")
print(f"Test Pass Percentage (Accuracy): {test_pass_percentage:.2f}%")


## --- 4. (Optional) Plotting predicted vs true actions ---
plt.figure(figsize=(10, 6))
plt.plot(true_actions, label='True Actions', marker='o', linestyle='-', color='blue')
plt.plot(predicted_actions, label='Predicted Actions', marker='x', linestyle='--', color='red')
plt.xlabel('Time Step')
plt.ylabel('Action (0, 1, or 2)')
plt.title('Predicted Actions vs True Actions')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(model_dir, "predicted_vs_true_actions.png")) # Save the plot
print("Plot saved to:", os.path.join(model_dir, "predicted_vs_true_actions.png"))
plt.show()


## --- 5. (Optional) Plotting the predictions on TARGET_COLUMN ---
# Define a dict for mapping actions to labels
ACTION_LABELS = {0: 'Up', 1: 'Down', 2: 'Same'}
data = test_df.iloc[LOOKBACK-1:-1].copy() # Exclude the last row as it doesn't have a next value
data['predicted_action'] = np.array(predicted_actions)

# Create a new column for each action label
data[ACTION_LABELS[0]] = [1 if x == 0 else 0 for x in data['predicted_action']]
data[ACTION_LABELS[1]] = [1 if x == 1 else 0 for x in data['predicted_action']]
data[ACTION_LABELS[2]] = [1 if x == 2 else 0 for x in data['predicted_action']]

def plot_signal_line_chart(df: pd.DataFrame, title: str = "Predicted Actions on TARGET_COLUMN") -> None:
    """Plot the TARGET_COLUMN values and the predicted actions on a line chart.
    Args:
        df (pd.DataFrame): DataFrame containing the time series data and predicted actions.
        title (str): Title of the plot.
    """
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(16, 4))

    # Plot the TARGET_COLUMN values as a line chart
    ax.plot(df.index, df[TARGET_COLUMN], label=TARGET_COLUMN, color='blue')

    # Scatter plot for the actions
    up_actions = df[df[ACTION_LABELS[0]] == 1]
    down_actions = df[df[ACTION_LABELS[1]] == 1]
    same_actions = df[df[ACTION_LABELS[2]] == 1]
    ax.scatter(up_actions.index, up_actions[TARGET_COLUMN], marker='^', color='green', label=ACTION_LABELS[0])
    ax.scatter(down_actions.index, down_actions[TARGET_COLUMN], marker='v', color='red', label=ACTION_LABELS[1])
    ax.scatter(same_actions.index, same_actions[TARGET_COLUMN], marker='o', color='orange', label=ACTION_LABELS[2])

    # Set labels and legend
    ax.set_xlabel('Time Step')
    ax.set_ylabel(TARGET_COLUMN)
    ax.set_title(title)
    ax.legend()

    # Display the plot
    plt.grid(True)
    plt.savefig(os.path.join(model_dir, "predicted_actions_on_target_column.png")) # Save the plot
    print("Plot saved to:", os.path.join(model_dir, "predicted_actions_on_target_column.png"))
    plt.show()

plot_signal_line_chart(data)
