import os
import itertools
from typing import Dict, List, Any, Union
import pandas as pd
from copy import deepcopy
from ..api import train_from_csv
from torch.utils.data import DataLoader

class ModelTuner:
    """
    A tuner class for training multiple models with different hyperparameters.
    
    Args:
        base_log_dir (str): Base directory for storing logs of different model versions
    """
    def __init__(
        self,
        base_log_dir: str = "logs/tuning",
    ):
        self.base_log_dir = os.path.abspath(base_log_dir)
        self.results_prefix = "tuning_results"
        
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
        
        Args:
            params: Model parameters to evaluate
            base_params: Optional base parameters that will be used for all models
            model_name: Name for logging directory structure
            
        Returns:
            Dictionary containing evaluation results including metrics and model directory
        """
        if base_params is None:
            base_params = {}
            
        # Combine base_params with current params
        all_params = deepcopy(base_params)
        all_params.update(params)

        # Separate params into their respective kwarg dicts
        env_kwargs = all_params.get('env_kwargs', {})
        agent_kwargs = all_params.get('agent_kwargs', {})
        trainer_kwargs = all_params.get('trainer_kwargs', {})
        
        # Update kwargs with params from the grid
        for key, value in params.items():
            if key in ['lookback', 'normalize_state', 'test_size']:
                env_kwargs[key] = value
            elif key in ['hidden_layers', 'output_size', 'agent_type', 'learning_rate', 'epsilon_start', 'epsilon_end', 'epsilon_decay_epochs_rate']:
                agent_kwargs[key] = value
            elif key in ['max_epochs', 'enable_checkpointing']:
                trainer_kwargs[key] = value

        trainer_kwargs['experiment_name'] = model_name

        # Train the model
        agent = train_from_csv(
            csv_path=all_params['csv_path'],
            feature_cols=all_params['feature_cols'],
            target_col=all_params['target_col'],
            env_kwargs=env_kwargs,
            agent_kwargs=agent_kwargs,
            trainer_kwargs=trainer_kwargs
        )

        # Get validation results
        val_dataset = agent.env.to_dataset(train=False)
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=len(val_dataset),
            shuffle=False,
            num_workers=0
        )
        val_results = agent.trainer.validate(agent, dataloaders=val_dataloader, verbose=False)

        # Combine parameters and metrics for results
        return {
            **params,
            'val_reward': val_results[0]['val_reward'],
            'val_accuracy': val_results[0]['val_accuracy'],
            'model_dir': agent.trainer.logger.log_dir
        }

    def train(
        self,
        params_grid: Dict[str, List[Any]],
        base_params: Dict[str, Any] = None,
    ) -> pd.DataFrame:
        """
        Train multiple models with different hyperparameter combinations using grid search.
        
        Args:
            params_grid: Dictionary mapping parameter names to lists of possible values
            base_params: Optional base parameters that will be used for all models
            
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
        results_df = results_df.sort_values('val_reward', ascending=False)
        
        # Store best model information
        best_result = results_df.iloc[0]
        self.best_model_checkpoint = os.path.join(best_result['model_dir'], "checkpoints", "last.ckpt")
        self.best_model_params = {k: best_result[k] for k in params_grid.keys()}
        
        # Save results with version number
        version = self.get_next_version()
        results_path = os.path.join(self.base_log_dir, f"{self.results_prefix}_v{version}.csv")
        results_df.to_csv(results_path, index=False)
        print(f"\nTuning results saved to: {results_path}")
        
        return results_df
