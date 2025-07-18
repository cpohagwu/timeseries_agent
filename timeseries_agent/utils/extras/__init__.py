"""Extra utilities for visualization and synthetic data generation."""

from .visualization import (plot_training_metrics, plot_prediction_density, 
                            plot_confusion_matrix_n_metrics, plot_animated_signal_line_chart,
                            plot_evolution_of_fitness_scores_across_generations,
                            plot_average_diversity_scores_across_generations)

__all__ = [
    'plot_training_metrics',
    'plot_prediction_density',
    'plot_confusion_matrix_n_metrics',
    'plot_animated_signal_line_chart',
    'plot_evolution_of_fitness_scores_across_generations',
    'plot_average_diversity_scores_across_generations'
]
