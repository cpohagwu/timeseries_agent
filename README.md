<div align="center">
  <img src="https://drive.google.com/uc?export=view&id=1PzYzSl8Xe-B9eBNI3Na6-3EOxHAQzSZY" width="300">
</div>

TimeSeries Agent is a powerful reinforcement learning library designed for time series analysis and prediction. Built on top of PyTorch and PyTorch Lightning, it provides a flexible framework for training RL agents to work with time series data.

<div align="center">

[![PyPI version](https://badge.fury.io/py/timeseries-agent.svg)](https://test.pypi.org/project/timeseries-agent/)
[![Tutorial](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1WU9nhn5U3XGTmRmLrm2BRQ8xXoPplTNn?usp=sharing)

</div>

<div align="center">
  <img src="https://drive.google.com/uc?export=view&id=1U8imXhwcvvJGxRCpQjoO6Ypaf-F-EFFz"/>
</div>
<br>

## Key Features

* Policy gradient-based reinforcement learning for time series prediction
* Easy integration with existing PyTorch workflows
* Support for custom time series datasets
* Built-in state normalization and reward calculation
* Flexible neural network architecture configuration
* Real-time prediction capabilities

## Installation

```bash
pip install -i https://test.pypi.org/simple/ timeseries-agent
pip install lightning torch
```

## Getting Started

The easiest way to get started is by following our [Colab Tutorial](https://colab.research.google.com/drive/1WU9nhn5U3XGTmRmLrm2BRQ8xXoPplTNn?usp=sharing), which walks through a complete example of training and testing a time series agent.

## Using Your Own Data

To use TimeSeries Agent with your own data, you need to:

1. Prepare your time series data as a pandas DataFrame with at least one target column.

```python
import pandas as pd
from timeseries_agent import RLTimeSeriesDataset, PolicyGradientAgent

# Load your time series data
data_df = pd.DataFrame({
    'value': your_target_values,
    'feature1': your_feature1_values,
    'feature2': your_feature2_values,
    # ... add more features as needed
})

# Configure the RL environment
LOOKBACK = 7  # Number of past time steps to consider
TARGET_COLUMN = 'value'  # Column for reward calculation
NUM_FEATURES = data_df.shape[1]  # Number of features in dataset

# Create and train the agent
agent = PolicyGradientAgent(
    full_data=data_df,
    target_column=TARGET_COLUMN,
    input_features=NUM_FEATURES,
    lookback=LOOKBACK,
    hidden_layers=[100, 100, 10],  # Customize network architecture
    normalize_state=True  # Enable state normalization
)
```

Key considerations when preparing your data:
- Ensure your DataFrame has no missing values
- The target column should contain the values you want to predict
- Additional features can help improve prediction accuracy
- The lookback period determines how much historical data the agent considers

## License

TimeSeries Agent is released under the MIT License. See [LICENSE](LICENSE) file for details.
