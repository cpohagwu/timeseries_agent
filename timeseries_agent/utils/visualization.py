"""Visualization utilities for TimeSeries Agent."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_recall_fscore_support

def plot_signal_line_chart(df: pd.DataFrame, target_column: str, action_labels: dict, 
                          save_path: str = None) -> None:
    """
    Plot time series data with action markers and action distribution comparison.
    
    Args:
        df: DataFrame containing the time series data and predictions
        target_column: Name of the target column to plot
        action_labels: Dictionary mapping action indices to their labels
        save_path: Optional path to save the plot
    """
    plt.style.use('classic')
    plt.rcParams['figure.autolayout'] = True  # Automatically adjust layout to fit elements
    
    # Create figure and axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), dpi=100, height_ratios=[3, 1])
    fig.patch.set_facecolor('white')
    
    # Plot main signal on top subplot
    line = ax1.plot(df.index, df[target_column], label=target_column, 
                   color='#2E86C1', linewidth=2, alpha=0.8)
    
    # Plot actions with enhanced markers
    up_actions = df[df[action_labels[0]] == 1]
    down_actions = df[df[action_labels[1]] == 1]
    same_actions = df[df[action_labels[2]] == 1]
    
    # Add markers with enhanced styling
    ax1.scatter(up_actions.index, up_actions[target_column], 
               marker='^', color='#27AE60', s=100, label=action_labels[0],
               alpha=0.7, edgecolor='white', linewidth=1)
    ax1.scatter(down_actions.index, down_actions[target_column], 
               marker='v', color='#E74C3C', s=100, label=action_labels[1],
               alpha=0.7, edgecolor='white', linewidth=1)
    ax1.scatter(same_actions.index, same_actions[target_column], 
               marker='o', color='#F39C12', s=100, label=action_labels[2],
               alpha=0.7, edgecolor='white', linewidth=1)
    
    # Add action distribution comparison on bottom subplot
    x = np.arange(len(action_labels))
    width = 0.35  # Width of bars
    
    # Count distributions
    true_dist = pd.Series(df['true_action']).value_counts().sort_index()
    pred_dist = df['predicted_action'].value_counts().sort_index()
    
    # Create grouped bars
    rects1 = ax2.bar(x - width/2, true_dist, width, label='True', 
                     color='#3498DB', alpha=0.7)
    rects2 = ax2.bar(x + width/2, pred_dist, width, label='Predicted',
                     color='#E74C3C', alpha=0.7)
    
    # Customize bottom subplot
    ax2.set_ylabel('Count', fontsize=14)
    ax2.set_title('Action Distribution Comparison', fontsize=16, pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels([action_labels[i] for i in range(len(action_labels))])
    ax2.legend()
    
    # Add value labels on bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax2.annotate(f'{int(height)}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    # Enhance grid
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Customize axes
    ax1.set_xlabel('Time Step', fontsize=14)
    ax1.set_ylabel(target_column, fontsize=14)
    
    # Add subtitles
    ax1.set_title('Signal and Predicted Actions', fontsize=16, pad=10)
    
    # Enhance legend
    ax1.legend(bbox_to_anchor=(1.01, 1), loc='upper left', 
              fontsize=12, frameon=True, facecolor='white', framealpha=1)
    
    # Add accuracy text
    accuracy = sum(df['predicted_action'] == df['true_action']) / len(df['true_action']) * 100
    ax1.text(0.02, 0.92, f'Accuracy: {accuracy:.1f}%', 
             transform=ax1.transAxes, fontsize=14,
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    
    # Adjust layout
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    
    plt.show()

def plot_confusion_matrix(true_actions: list, predicted_actions: list, 
                         action_labels: dict, save_path: str = None) -> None:
    """
    Creates and plots an enhanced confusion matrix with additional metrics.
    
    Args:
        true_actions: List of true actions
        predicted_actions: List of predicted actions
        action_labels: Dictionary mapping action indices to their labels
        save_path: Optional path to save the plot
    """
    # Set style
    plt.style.use('classic')
    plt.rcParams['figure.autolayout'] = True  # Automatically adjust layout to fit elements
   
    # Create figure with larger size
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('white')
    
    # Compute confusion matrix and metrics
    cm = confusion_matrix(true_actions, predicted_actions, labels=list(action_labels.keys()))
    accuracy = accuracy_score(true_actions, predicted_actions)
    precision, recall, f1, _ = precision_recall_fscore_support(true_actions, predicted_actions, 
                                                             average='weighted')
    
    # Create display object
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[action_labels[i] for i in sorted(action_labels.keys())]
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

def create_animated_prediction_plot(df: pd.DataFrame, target_column: str,
                                  action_labels: dict, save_path: str = None) -> None:
    """
    Creates an animated plot showing the evolution of predictions over time.

    Args:
        df: DataFrame containing the time series data and predictions
        target_column: Name of the target column to plot
        action_labels: Dictionary mapping action indices to their labels
        save_path: Optional path to save the animation
    """

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
                             label=f'{action_labels[0]} (Predicted)', alpha=0.7)
    scatter_down = ax.scatter([], [], marker='v', color='#E74C3C', s=100,
                              label=f'{action_labels[1]} (Predicted)', alpha=0.7)
    scatter_same = ax.scatter([], [], marker='o', color='#F39C12', s=100,
                              label=f'{action_labels[2]} (Predicted)', alpha=0.7)

    # Set axis limits
    ax.set_xlim(df.index.min() - 1, df.index.max() + 1)
    ax.set_ylim(df[target_column].min() * 1.1, df[target_column].max() * 1.1)

    # Add labels and title
    ax.set_xlabel('Time Step', fontsize=14)
    ax.set_ylabel(target_column, fontsize=14)
    ax.set_title('Signal and Predicted Actions (Live)', fontsize=16)
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
            next_predicted_action = df['predicted_action'].iloc[frame]
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
