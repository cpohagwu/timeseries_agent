# Release Notes

## Version 0.1.0 (alpha release)

### Major Changes
- Default use of genetic algorithm during hyperparameter optimization.
- New Policy Gradient algorithms introduced: PPO and REINFORCE.
- The previous Policy Gradient algorithm is now accessible as REINFORCE_STEP.
- Added an API module to further simplify training and loading trained agents.
- Optimized visualization module to quickly show relevant train and val metrics.
- Allows for user-defined custom reward functions.

### New Features
- **PPO Agent**: A new agent that implements the Proximal Policy Optimization algorithm.
- **REINFORCE Agent**: A new agent that implements the REINFORCE algorithm.
- **Genetic Tuner**: The default hyperparameter tuner is now the Genetic Algorithm tuner.
- **Custom Reward Functions**: Users can now define and use their own custom reward functions.
- **Simplified API**: The new `api` module simplifies the training and loading of agents.

### Improvements
- The `REINFORCE_STEP` agent now includes `entropy_beta` as a regularization term.
- The visualization module has been optimized for performance.

### Breaking Changes
- The default `PolicyGradientAgent` has been removed. Users should now use `PPO`, `REINFORCE`, or `REINFORCE_STEP` agents.

### Migration Guide
To migrate from previous versions, you will need to update your agent creation code to use one of the new agent types. For example, if you were using `PolicyGradientAgent`, you should now use `REINFORCE_STEP` for similar behavior.

```python
# Previous code
# from timeseries_agent import PolicyGradientAgent
# agent = PolicyGradientAgent(...)

# New code
from timeseries_agent.api import train_from_csv
agent = train_from_csv(..., agent_kwargs={'agent_type': 'reinforce_step'})
```

### Documentation Updates
- Updated `README.md` with the new features and examples.
- Added new notebooks to the `examples` directory.

### Bug Fixes
- None in this release.

## Version 0.0.29

### Major Changes
- Added support for continuing training from the best model checkpoint after hyperparameter tuning
- Introduced shared checkpoint management between grid search and genetic algorithm tuners
- Improved weight initialization consistency through checkpoint loading
- Removed `input_features` parameter from `PolicyGradientAgent()` initialization
- Removed `num_training_epochs` parameter from `tuner.train()` method. Now part of the `base_params` as `num_epochs`

### New Features
- Added `num_epochs_best_model` parameter to continue training from the best model found during tuning
- Added new tutorial notebook for genetic algorithm-based hyperparameter tuning
- Implemented population diversity tracking in genetic tuning

### Improvements
- Consolidated model checkpoint handling in base ModelTuner class
- Enhanced parameter validation and error handling

### Breaking Changes
None

### Migration Guide
To use the new checkpoint continuation feature, add num_epochs_best_model to base_params when calling train():

```python
tuner.train(
    params_grid=params_grid,
    base_params={
        'num_epochs': 10,              # epochs for each trial during tuning
        'num_epochs_best_model': 100,  # additional epochs for best model
        ...
    }
)
```

### Documentation Updates
- Added timeseries_agent_genetic_tuner_tutorial.ipynb showing how to use the genetic algorithm tuner
- Updated existing tuning tutorial with checkpoint continuation examples
- Added API documentation for new parameters and methods

### Bug Fixes
- Fixed parameter naming consistency across tuners
