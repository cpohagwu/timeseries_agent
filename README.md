<div align="center">
  <img src="https://raw.githubusercontent.com/cpohagwu/timeseries_agent/main/doc/_static/logo.png" width="300">
</div>

TimeSeries Agent is a powerful reinforcement learning library designed for time series analysis and prediction. Built on top of PyTorch and PyTorch Lightning, it provides a flexible framework for training RL agents to work with time series data. It uses modern policy gradient algorithms like PPO and REINFORCE, and offers hyperparameter tuning using genetic algorithms.
<br>
<small><i>This package is designed to predict the deviation direction of a time series that maximizes the reward (total and/or immediate).</i></small>

<div align="center">

[![PyPI version](https://badge.fury.io/py/timeseries-agent.svg)](https://pypi.org/project/timeseries-agent/)
[![PyPI Downloads](https://static.pepy.tech/badge/timeseries-agent)](https://pepy.tech/projects/timeseries-agent)
[![Pattern BM Example](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cpohagwu/timeseries_agent/blob/main/examples/01_pattern_BM_example.ipynb)
[![Genetic Tuner Tutorial](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cpohagwu/timeseries_agent/blob/main/examples/02_genetic_tuner_tutorial.ipynb)
[![Stock RL Example](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/cpohagwu/timeseries_agent/blob/main/examples/03_stock_rl_example.ipynb)

</div>
<div align="center">
  <img src="https://raw.githubusercontent.com/cpohagwu/timeseries_agent/main/doc/_static/predictions_animation.gif"/>
</div>
<br>

## Key Features

* Multiple policy gradient algorithms (PPO, REINFORCE, REINFORCE_STEP)
* Efficient hyperparameter tuning with genetic algorithms
* Simplified API for training and loading agents
* Support for custom reward functions
* Optimized visualization for training and validation metrics
* Easy integration with existing PyTorch workflows
* Support for custom time series datasets (multivariate)
* Built-in state normalization and reward calculation
* Flexible neural network architecture configuration
* Real-time prediction capabilities

## Installation

```bash
pip install timeseries-agent
```

## Getting Started

We provide three interactive Colab tutorials to help you get started:
1. [Pattern BM Example](https://colab.research.google.com/github/cpohagwu/timeseries_agent/blob/main/examples/01_pattern_BM_example.ipynb) - A benchmark synthetic dataset example.
2. [Genetic Tuner Tutorial](https://colab.research.google.com/github/cpohagwu/timeseries_agent/blob/main/examples/02_genetic_tuner_tutorial.ipynb) - Shows how to use the Genetic Tuner to find optimal hyperparameters for your agent.
3. [Stock RL Example](https://colab.research.google.com/github/cpohagwu/timeseries_agent/blob/main/examples/03_stock_rl_example.ipynb) - An example of using the agent for stock price prediction.

## Using Your Own Data

To use TimeSeries Agent with your own data, you need to prepare your time series data as a pandas DataFrame and use the `train_from_csv` function.

```python
from timeseries_agent.api import train_from_csv

agent = train_from_csv(
    csv_path="path/to/your/data.csv",
    feature_cols=['your', 'feature', 'columns'],
    target_col='your_target_column',
    env_kwargs={'lookback': 10},
    agent_kwargs={'agent_type': 'ppo'},
    trainer_kwargs={'max_epochs': 50}
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
  <img src="https://raw.githubusercontent.com/cpohagwu/timeseries_agent/main/doc/_static/predictions_analysis.png"/>
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
