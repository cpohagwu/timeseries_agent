"""Visualization utilities for TimeSeries Agent."""

import os
import numpy as np
import pandas as pd
import lightning as L
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.animation as animation
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_recall_fscore_support

AGENT_CLASSES = {
    "PPOAgent": "ppo",
    "ReinforceAgent": "reinforce",
    "ReinforceStepAgent": "reinforce_step"
}

ACTION_LABELS = {0: 'Up', 1: 'Down', 2: 'Same'}

def plot_training_metrics(agent: L.LightningModule):
    """
    Plots training metrics for reward/accuracy and loss/epsilon.

    Args:
        agent (L.LightningModule): The Lightning module containing training metrics.

    """
    logs = pd.read_csv(os.path.join(agent.trainer.logger.log_dir, "metrics.csv"))
    logs.set_index('epoch', inplace=True)

    agent_class = agent.__class__.__name__
    if agent_class in AGENT_CLASSES:
        agent_type = AGENT_CLASSES[agent_class]
    
    log_dir = agent.trainer.logger.log_dir

    # Plot for train_reward and train_accuracy
    fig, ax1 = plt.subplots(figsize=(10, 3))
    ax1.plot(logs.index, logs['train_reward'], color='tab:blue', label='Reward')
    ax1.set_ylabel('Reward', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax2 = ax1.twinx()
    ax2.plot(logs.index, logs['train_accuracy'], color='tab:orange', label='Accuracy')
    ax2.set_ylabel('Accuracy', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    ax1.set_xlabel('Epoch') # Set x-axis label
    plt.title("Training Metrics")
    fig.tight_layout()
    if log_dir:
        plt.savefig(os.path.join(log_dir, "training_metrics_reward.png"))
    plt.show()

    # Plot for train_loss and train_epsilon/train_entropy
    ax2_values = logs['train_epsilon'] if agent_type in ['reinforce_step', 'reinforce'] else logs['train_entropy']
    fig, ax1 = plt.subplots(figsize=(10, 3))
    ax1.plot(logs.index, logs['train_loss'], color='tab:blue', label='Loss')
    ax1.set_ylabel('Loss', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax2 = ax1.twinx()
    ax2.plot(logs.index, ax2_values, color='tab:orange', label='Epsilon' if agent_type in ['reinforce_step', 'reinforce'] else 'Entropy')
    ax2.set_ylabel('Epsilon' if agent_type in ['reinforce_step', 'reinforce'] else 'Entropy', color='tab:orange')
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    ax1.set_xlabel('Epoch')  # Set x-axis label
    plt.title("Training Metrics")
    fig.tight_layout()
    if log_dir:
        plt.savefig(os.path.join(log_dir, "training_metrics_loss.png"))
    plt.show()

def plot_prediction_density(y_true: list, y_pred: list, save_path: str = None):
    """
    Creates a density plot of predicted vs true values.

    Args:
        y_true (list): List of true values.
        y_pred (list): List of predicted values.
        save_path (str, optional): Path to save the plot. Defaults to None.
    """
    # Set style
    plt.style.use('classic')
    plt.rcParams['figure.autolayout'] = True  # Automatically adjust layout to fit elements
   
    # Create figure with larger size
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor('white')

    sns.kdeplot(y_pred, fill=True, color='skyblue', label='Predicted', alpha=0.6)
    sns.kdeplot(y_true, fill=True, color='orange', label='True', alpha=0.4)
    plt.title('Density Plot of Predicted vs True Distribution')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_confusion_matrix_n_metrics(y_true: list, y_pred: list, save_path: str = None) -> None:
    """
    Creates and plots an enhanced confusion matrix with additional metrics.
    
    Args:
        y_true: List of true actions
        predicted_actions: List of predicted actions
        save_path: Optional path to save the plot
    """
    # Set style
    plt.style.use('classic')
    plt.rcParams['figure.autolayout'] = True  # Automatically adjust layout to fit elements
   
    # Create figure with larger size
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('white')
    
    # Compute confusion matrix and metrics
    cm = confusion_matrix(y_true, y_pred, labels=list(ACTION_LABELS.keys()))
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, 
                                                             average='weighted')
    
    # Create display object
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[ACTION_LABELS[i] for i in sorted(ACTION_LABELS.keys())]
    )
    
    # Plot with enhanced styling
    disp.plot(
        ax=ax,
        cmap='Blues',
        values_format='d',
        colorbar=True,
        xticks_rotation=45
    )
    
    # Add title and labels with enhanced styling
    plt.title("Confusion Matrix", 
              fontsize=14, pad=20)
    plt.xlabel("Predicted", fontsize=12, labelpad=10)
    plt.ylabel("True", fontsize=12, labelpad=10)
    
    # Add metrics text box
    metrics_text = (f"Accuracy: {accuracy:.2%}\n"
                   f"Precision: {precision:.2%}\n"
                   f"Recall: {recall:.2%}\n"
                   f"F1 Score: {f1:.2%}")
    
    plt.text(1.45, 0.5, metrics_text,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
             transform=ax.transAxes, fontsize=11,
             verticalalignment='center')
    
    # Adjust layout
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()

def _process_df_for_animation(df: pd.DataFrame, y_true: list, y_pred: list, lookback: int) -> pd.DataFrame:
    df = df.iloc[lookback:-1].copy()
    df.index.name = 'Time Step'

    # One-hot encode predicted actions for visualization
    for action, label in ACTION_LABELS.items():
        df[label] = [1 if x == action else 0 for x in y_pred]

    # Add predicted and true actions to the DataFrame
    df['y_pred'] = y_pred
    df['y_true'] = y_true
    return df

def plot_animated_signal_line_chart(df: pd.DataFrame, target_column: str, 
                                    y_true: list, y_pred: list, lookback: int, save_path:str = None) -> None:
    """
    Creates an animated plot showing the evolution of predictions over time.

    Args:
        df: DataFrame containing the time series data
        target_column: Name of the target column to plot
        y_true (list): List of true values.
        y_pred (list): List of predicted values.
        lookback: Lookback window
        save_path: Optional path to save the animation
    """

    # Prepare data
    df = _process_df_for_animation(df=df, y_true=y_true, y_pred=y_pred, lookback=lookback)

    plt.style.use('classic')
    # Move plot to the right side to make space for the legend
    plt.rcParams['figure.autolayout'] = True  # Automatically adjust layout to fit elements

    # Create figure and axis with increased size and margins
    fig, ax = plt.subplots(figsize=(16, 8))
    fig.patch.set_facecolor('white')

    # Initialize line and scatter plots
    line, = ax.plot([], [], label=target_column, color='#2E86C1', linewidth=2)
    # The scatter labels should now indicate they represent predictions
    scatter_up = ax.scatter([], [], marker='^', color='#27AE60', s=100,
                             label=f'{ACTION_LABELS[0]} (Predicted)', alpha=0.7)
    scatter_down = ax.scatter([], [], marker='v', color='#E74C3C', s=100,
                              label=f'{ACTION_LABELS[1]} (Predicted)', alpha=0.7)
    scatter_same = ax.scatter([], [], marker='o', color='#F39C12', s=100,
                              label=f'{ACTION_LABELS[2]} (Predicted)', alpha=0.7)

    # Set axis limits
    ax.set_xlim(df.index.min() - 1, df.index.max() + 1)
    ax.set_ylim(df[target_column].min() * 1.1, df[target_column].max() * 1.1)

    # Add labels and title
    ax.set_xlabel('Time Step', fontsize=14)
    ax.set_ylabel(target_column, fontsize=14)
    ax.set_title('Signal and Predicted Actions (Animation)', fontsize=16)
    ax.legend(loc='center left', bbox_to_anchor=(1.01, 0.5))
    ax.grid(True, linestyle='--', alpha=0.7)

    # Store the current prediction arrow text
    prediction_arrows = []

    def update(frame):
        # Remove all previous prediction arrows
        for arrow in prediction_arrows:
            arrow.remove()
        prediction_arrows.clear()

        # Add prediction arrow for the *next* point (i.e., at the current frame index)
        # This arrow will appear before the line graph reaches this point
        if frame < len(df):
            next_action_index = df.index[frame]

            # The prediction is for the current 'frame' index
            next_predicted_action = df['y_pred'].iloc[frame]
            current_val_for_arrow = df[target_column].iloc[frame]

            arrow_colors = {0: '#27AE60', 1: '#E74C3C', 2: '#F39C12'}
            arrow_symbols = {0: '↑', 1: '↓', 2: '→'}

            arrow = ax.text(next_action_index, current_val_for_arrow, arrow_symbols[next_predicted_action],
                            color=arrow_colors[next_predicted_action], fontsize=20,
                            ha='center', va='bottom', weight='bold')
            prediction_arrows.append(arrow)


        # Update line data up to the current frame
        line.set_data(df.index[:frame], df[target_column][:frame])

        # Update scatter data for *predicted actions* up to the current frame
        if frame > 0:
            # Get data up to (but not including) the current frame for predicted actions
            current_predicted_data = df.iloc[:frame]

            # Filter for predicted actions using the one-hot encoded columns
            up_actions = current_predicted_data[current_predicted_data['Up'] == 1]
            down_actions = current_predicted_data[current_predicted_data['Down'] == 1]
            same_actions = current_predicted_data[current_predicted_data['Same'] == 1]

            scatter_up.set_offsets(np.c_[up_actions.index, up_actions[target_column]])
            scatter_down.set_offsets(np.c_[down_actions.index, down_actions[target_column]])
            scatter_same.set_offsets(np.c_[same_actions.index, same_actions[target_column]])

        return (line, scatter_up, scatter_down, scatter_same) + tuple(prediction_arrows)

    # Create animation
    ani = animation.FuncAnimation(
        fig, update, frames=len(df) + 1, # +1 to show the last prediction arrow
        interval=200, blit=True, repeat=False
    )

    # Save animation if path provided
    if save_path:
        ani.save(save_path, writer='pillow', fps=5)

    # Set figure size to accommodate legend
    plt.subplots_adjust(right=0.85)
    plt.show()
    return ani

def plot_evolution_of_fitness_scores_across_generations(results):
    """
    Plots the distribution of fitness scores for each generation.
    
    Args:
        results (pd.DataFrame): DataFrame containing the tuning results.
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=results, x='generation', y='fitness')
    plt.title('Fitness Score Distribution by Generation')
    plt.xlabel('Generation')
    plt.ylabel('Fitness Score')
    plt.show()

def plot_average_diversity_scores_across_generations(results):
    """
    Plots the average diversity score for each generation.
    
    Args:
        results (pd.DataFrame): DataFrame containing the tuning results.
    """
    avg_diversity_by_gen = results.groupby('generation')['diversity_score'].mean()
    plt.figure(figsize=(10, 6))
    plt.plot(avg_diversity_by_gen.index, avg_diversity_by_gen.values, marker='o')
    plt.title('Average Population Diversity by Generation')
    plt.xlabel('Generation')
    plt.ylabel('Average Diversity Score')
    plt.grid(True)
    plt.show()
