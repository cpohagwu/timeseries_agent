import os
import itertools
from typing import Dict, List, Any, Union
import lightning as L
import pandas as pd
from copy import deepcopy
import torch
from torch.utils.data import DataLoader
from ..models.policy_gradient_agent import PolicyGradientAgent
from ..data.rl_dataset import RLTimeSeriesDataset

class ModelTuner:
    """
    A tuner class for training multiple PolicyGradientAgent models with different hyperparameters.
    
    Args:
        data_df (pd.DataFrame): The time series DataFrame to use for training
        base_log_dir (str): Base directory for storing logs of different model versions
        target_column (Union[str, int]): Name or index of the target column for reward calculation
        num_features (int): Number of features in the input time series
        output_size (int): Number of possible actions
    """
    def __init__(
        self,
        data_df: pd.DataFrame,
        base_log_dir: str = "logs/tuning",
        target_column: Union[str, int] = "value",
        num_features: int = None,
        output_size: int = 3,
    ):
        self.data_df = data_df
        # Ensure the log directory is relative to the current working directory
        self.base_log_dir = os.path.abspath(base_log_dir)
        self.target_column = target_column
        self.num_features = num_features or data_df.shape[1]
        self.output_size = output_size
        
        # Create base log directory if it doesn't exist
        os.makedirs(self.base_log_dir, exist_ok=True)

    def generate_parameter_grid(self, param_ranges: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """
        Generates a list of all possible parameter combinations from the given ranges.
        
        Args:
            param_ranges: Dictionary mapping parameter names to lists of possible values
            
        Returns:
            List of dictionaries, each containing a unique combination of parameters
        """
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())
        
        combinations = list(itertools.product(*param_values))
        return [dict(zip(param_names, combo)) for combo in combinations]

    def _get_next_version(self) -> int:
        """Find the next available version number for tuning results."""
        version = 0
        while os.path.exists(os.path.join(self.base_log_dir, f"tuning_results_v{version}.csv")):
            version += 1
        return version

    def train(
        self,
        param_ranges: Dict[str, List[Any]],
        num_epochs: int = 1000,
        base_params: Dict[str, Any] = None,
    ) -> pd.DataFrame:
        """
        Trains multiple models with different hyperparameter combinations.
        
        Args:
            param_ranges: Dictionary mapping parameter names to lists of possible values
            num_epochs: Number of epochs to train each model
            base_params: Optional base parameters that will be used for all models
            
        Returns:
            DataFrame containing the results for each hyperparameter combination
        """
        if base_params is None:
            base_params = {}

        param_combinations = self.generate_parameter_grid(param_ranges)
        results = []

        for model_idx, params in enumerate(param_combinations):
            print(f"\nTraining model {model_idx + 1}/{len(param_combinations)}")
            print("Parameters:", params)

            # Create dataset with current lookback
            lookback = params.get('lookback', base_params.get('lookback', 7))
            dataset = RLTimeSeriesDataset(
                data=self.data_df,
                lookback=lookback,
            )
            dataloader = DataLoader(
                dataset,
                batch_size=len(dataset),
                shuffle=False,
                num_workers=0
            )

            # Combine base_params with current params
            model_params = deepcopy(base_params or {})
            model_params.update(params)

            # Create and train model
            model = PolicyGradientAgent(
                full_data=self.data_df,
                target_column=self.target_column,
                input_features=self.num_features,
                output_size=self.output_size,
                **model_params
            )

            # Create trainer with unique log directory for this model
            logger = L.pytorch.loggers.CSVLogger(
                self.base_log_dir,
                name=f"tuning/model_{model_idx}",
                version=None  # Auto-increment version
            )

            trainer = L.Trainer(
                max_epochs=num_epochs,
                accelerator='auto',
                devices='auto',
                logger=logger,
                enable_checkpointing=True,
                deterministic=True,
            )

            # Train and validate
            trainer.fit(model=model, train_dataloaders=dataloader)
            val_results = trainer.validate(model=model, dataloaders=dataloader)

            # Get the final metrics
            metrics = {
                'val_avg_reward': val_results[0]['val_avg_reward'],
                'val_pass_percentage': val_results[0]['val_pass_percentage']
            }

            # Save model checkpoint
            ckpt_path = os.path.join(logger.log_dir, "model.ckpt")
            trainer.save_checkpoint(ckpt_path)

            # Combine parameters and metrics for results
            result_entry = {**params, **metrics, 'model_dir': logger.log_dir}
            results.append(result_entry)

        # Convert results to DataFrame and sort by validation reward
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('val_avg_reward', ascending=False)
        
        # Save results with version number
        version = self._get_next_version()
        results_path = os.path.join(self.base_log_dir, f"tuning_results_v{version}.csv")
        results_df.to_csv(results_path, index=False)
        print(f"\nTuning results saved to: {results_path}")
        
        return results_df
