<div align="center">
  <img src="doc/_static/logo.png" width="300">
</div>

TimeSeries Agent is a powerful reinforcement learning library designed for time series analysis and prediction. Built on top of PyTorch and PyTorch Lightning, it provides a flexible framework for training RL agents to work with time series data.

<div align="center">

[![PyPI version](https://badge.fury.io/py/timeseries-agent.svg)](https://pypi.org/project/timeseries-agent/)
[![Basic Tutorial](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cpohagwu/timeseries_agent/blob/main/examples/timeseries_agent_tutorial.ipynb)
[![Tuner Tutorial](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cpohagwu/timeseries_agent/blob/main/examples/timeseries_agent_tuner_tutorial.ipynb)

</div>
<div align="center">
  <img src="doc/_static/predictions_animation.gif"/>
</div>
<br>

## Key Features

* Policy gradient-based reinforcement learning for time series prediction
* Easy integration with existing PyTorch workflows
* Support for custom time series datasets (multivariate)
* Built-in state normalization and reward calculation
* Flexible neural network architecture configuration (and activation function)
* Real-time prediction capabilities

## Installation
You can install TimeSeries Agent using pip:

```bash
pip install timeseries-agent
```

## Getting Started

We provide two interactive Colab tutorials to help you get started:
1. [Basic Tutorial](https://colab.research.google.com/github/cpohagwu/timeseries_agent/blob/main/examples/timeseries_agent_tutorial.ipynb) - Walks through a complete example of training and testing a time series agent.
2. [Tuner Tutorial](https://colab.research.google.com/github/cpohagwu/timeseries_agent/blob/main/examples/timeseries_agent_tuner_tutorial.ipynb) - Shows how to use the ModelTuner to find optimal hyperparameters for your agent.

## Using Your Own Data

To use TimeSeries Agent with your own data, you need to:

1. Prepare your time series data as a pandas DataFrame.

```python
import pandas as pd
from timeseries_agent import SequentialTimeSeriesDataset, PolicyGradientAgent

# Load your time series data
data_df = pd.DataFrame({
    'value': your_target_values,
    'feature1': your_feature1_values,
    'feature2': your_feature2_values,
    # ... add more features as needed
})

# Configure the RL environment
LOOKBACK = 7                       # Number of past time steps to consider
TARGET_COLUMN = 'value'            # Column for reward calculation
NUM_FEATURES = data_df.shape[1]    # Number of features in dataset

# Create and train the agent
agent = PolicyGradientAgent(
    full_data=data_df,
    target_column=TARGET_COLUMN,
    input_features=NUM_FEATURES,
    lookback=LOOKBACK,
    hidden_layers=[100, 100, 10],  # Customize network architecture
    normalize_state=True           # Enable state normalization
)
```

Key considerations when preparing your data:
- Ensure your DataFrame has no missing values
- The target column should contain the values you want to predict
- Additional features can help improve prediction accuracy
- The lookback period determines how much historical data the agent considers
- Ensure a balanced distribution of the actions you want to predict

</div>
<div align="center">
  <img src="doc/_static/predictions_analysis.png"/>
</div>
<br>

## Development

TimeSeries Agent is actively being developed on [GitHub](https://github.com/cpohagwu/timeseries_agent). Please note that the API is subject to change as we continue to improve and enhance the library.

### Contributing

We welcome contributions to TimeSeries Agent! Whether it's improving documentation, adding new features, fixing bugs, or suggesting improvements, your help is appreciated. Feel free to submit pull requests or open discussions on GitHub.

### Issues and Bug Reports

If you encounter any issues or bugs, please report them on our [GitHub Issues page](https://github.com/cpohagwu/timeseries_agent/issues). Your feedback helps us improve the library and fix problems faster.

## License

TimeSeries Agent is released under the Apache License 2.0 License. See [LICENSE](LICENSE) file for details.
