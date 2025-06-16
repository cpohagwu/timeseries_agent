# Release Notes

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