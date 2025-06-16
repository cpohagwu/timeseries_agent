import os
import itertools
from typing import Dict, List, Any, Union
import lightning as L
import pandas as pd
from copy import deepcopy
import torch
from torch.utils.data import DataLoader
from ..models.policy_gradient_agent import PolicyGradientAgent
from ..data.rl_sequence_dataset import SequentialTimeSeriesDataset

class ModelTuner:
    """
    A tuner class for training multiple PolicyGradientAgent models with different hyperparameters.
    
    Args:
        data_df (pd.DataFrame): The time series DataFrame to use for training
        base_log_dir (str): Base directory for storing logs of different model versions
        target_column (Union[str, int]): Name or index of the target column for reward calculation
        output_size (int): Number of possible actions
    """
    def __init__(
        self,
        data_df: pd.DataFrame,
        base_log_dir: str = "logs/tuning",
        target_column: Union[str, int] = "value",
        output_size: int = 3,
    ):
        self.data_df = data_df
        self.num_features = data_df.shape[1] # Number of features (columns) in the DataFrame
        # Ensure the log directory is relative to the current working directory
        self.base_log_dir = os.path.abspath(base_log_dir)
        self.results_prefix = "tuning_results"
        self.target_column = target_column
        self.output_size = output_size
        
        # Store best model info
        self.best_model_checkpoint = None
        self.best_model_params = None
        
        # Create base log directory if it doesn't exist
        os.makedirs(self.base_log_dir, exist_ok=True)

    def generate_parameter_grid(self, params_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """
        Generates a list of all possible parameter combinations from the given ranges.
        
        Args:
            params_grid: Dictionary mapping parameter names to lists of possible values
            
        Returns:
            List of dictionaries, each containing a unique combination of parameters
        """
        param_names = list(params_grid.keys())
        param_values = list(params_grid.values())
        
        combinations = list(itertools.product(*param_values))
        return [dict(zip(param_names, combo)) for combo in combinations]

    def get_next_version(self) -> int:
        """Find the next available version number for tuning results."""
        version = 0
        while os.path.exists(os.path.join(self.base_log_dir, f"{self.results_prefix}_v{version}.csv")):
            version += 1
        return version

    def evaluate_model(
        self,
        params: Dict[str, Any],
        base_params: Dict[str, Any] = None,
        model_name: str = "tuning/model",
    ) -> Dict[str, Any]:
        """
        Evaluate a single model with given parameters.
        
        This is the core function for training and evaluating a model with specific parameters.
        It handles dataset creation, model initialization, training, and result collection.
        
        Args:
            params: Model parameters to evaluate
            base_params: Optional base parameters that will be used for all models
            model_name: Name for logging directory structure
            
        Returns:
            Dictionary containing evaluation results including metrics and model directory
        """
        if base_params is None:
            base_params = {}
            
        # Remove num_epochs_best_model from base_params if present
        base_params = {k: v for k, v in base_params.items() if k != 'num_epochs_best_model'}
            
        # Create dataset with current lookback
        lookback = params.get('lookback', base_params.get('lookback', 7))
        dataset = SequentialTimeSeriesDataset(
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
            output_size=self.output_size,
            **model_params
        )

        # Create trainer with unique log directory for this model
        logger = L.pytorch.loggers.CSVLogger(
            self.base_log_dir,
            name=model_name,
            version=None  # Auto-increment version
        )

        trainer = L.Trainer(
            max_epochs=base_params.get('num_epochs', 1000),
            accelerator=base_params.get('accelerator', 'auto'),
            devices=base_params.get('devices', 'auto'),
            logger=logger,
            enable_checkpointing=base_params.get('enable_checkpointing', True),
            deterministic=base_params.get('deterministic', True),
        )

        # Train and validate
        trainer.fit(model=model, train_dataloaders=dataloader)
        val_results = trainer.validate(model=model, dataloaders=dataloader)

        # Save model checkpoint
        ckpt_path = os.path.join(logger.log_dir, "model.ckpt")
        trainer.save_checkpoint(ckpt_path)

        # Combine parameters and metrics for results
        return {
            **params,
            'val_avg_reward': val_results[0]['val_avg_reward'],
            'val_pass_percentage': val_results[0]['val_pass_percentage'],
            'model_dir': logger.log_dir
        }
        
    def continue_training_best_model(
        self,
        num_epochs: int,
        params: Dict[str, Any],
        base_params: Dict[str, Any] = None,
        model_name: str = "best_model_continued",
    ) -> None:
        """
        Continue training the best model from its checkpoint for additional epochs.
        
        Args:
            num_epochs: Number of additional epochs to train
            params: Best model parameters
            base_params: Optional base parameters for training configuration
            model_name: Name for the continued training logs
        """
        if base_params is None:
            base_params = {}
            
        # Create dataset
        lookback = params.get('lookback', 7)
        dataset = SequentialTimeSeriesDataset(
            data=self.data_df,
            lookback=lookback,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=len(dataset),
            shuffle=False,
            num_workers=0
        )
        
        # Load best model from checkpoint
        model = PolicyGradientAgent.load_from_checkpoint(
            checkpoint_path=self.best_model_checkpoint,
            full_data=self.data_df,
            target_column=self.target_column,
            **params
        )
        
        # Create trainer for additional training
        logger = L.pytorch.loggers.CSVLogger(
            self.base_log_dir,
            name=model_name,
            version=None
        )
        
        trainer = L.Trainer(
            max_epochs=num_epochs,
            accelerator=base_params.get('accelerator', 'auto'),
            devices=base_params.get('devices', 'auto'),
            logger=logger,
            enable_checkpointing=base_params.get('enable_checkpointing', True),
            deterministic=base_params.get('deterministic', True),
        )
        
        # Train and validate
        trainer.fit(model=model, train_dataloaders=dataloader)
        trainer.validate(model=model, dataloaders=dataloader)
        
        # Save final model checkpoint
        final_ckpt_path = os.path.join(logger.log_dir, "final_model.ckpt")
        trainer.save_checkpoint(final_ckpt_path)
        print(f"\nFinal model saved to: {final_ckpt_path}")

    def train(
        self,
        params_grid: Dict[str, List[Any]],
        base_params: Dict[str, Any] = None,
    ) -> pd.DataFrame:
        """
        Train multiple models with different hyperparameter combinations using grid search.
        
        This is a high-level function that uses evaluate_model() to test each combination
        of parameters in the grid.
        
        Args:
            params_grid: Dictionary mapping parameter names to lists of possible values
            base_params: Optional base parameters that will be used for all models
            num_epochs: Number of epochs to train each model
            
        Returns:
            DataFrame containing the results for each hyperparameter combination
        """
        param_combinations = self.generate_parameter_grid(params_grid)
        results = []

        for model_idx, params in enumerate(param_combinations):
            print(f"\nTraining model {model_idx + 1}/{len(param_combinations)}")
            print("Parameters:", params)

            # Evaluate model with current parameters
            result = self.evaluate_model(
                params=params,
                base_params=base_params,
                model_name=f"tuning/model_{model_idx}"
            )
            results.append(result)

        # Convert results to DataFrame and sort by validation reward
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('val_avg_reward', ascending=False)
        
        # Store best model information
        best_result = results_df.iloc[0]
        self.best_model_checkpoint = os.path.join(best_result['model_dir'], "model.ckpt")
        self.best_model_params = {k: best_result[k] for k in params_grid.keys()}
        
        # Save results with version number
        version = self.get_next_version()
        results_path = os.path.join(self.base_log_dir, f"{self.results_prefix}_v{version}.csv")
        results_df.to_csv(results_path, index=False)
        print(f"\nTuning results saved to: {results_path}")
        
        # If num_epochs_best_model is provided, continue training the best model
        num_epochs_best_model = base_params.get('num_epochs_best_model', None)
        if num_epochs_best_model is not None and num_epochs_best_model > 0:
            print(f"\nContinuing training of best model for {num_epochs_best_model} epochs...")
            best_model_params = deepcopy(base_params or {})
            best_model_params.update(self.best_model_params)
            self.continue_training_best_model(
                num_epochs=num_epochs_best_model,
                params=best_model_params,
                base_params=base_params,
                model_name="tuning_best_model_continued"
            )
        
        return results_df
