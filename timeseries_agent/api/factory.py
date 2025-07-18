"""Factory functions for creating and training agents."""

import os
from typing import List, Dict, Any, Optional
import pandas as pd
import lightning as L
from torch.utils.data import DataLoader
import torch.nn as nn

from ..envs.csv_env import CsvEnv
from ..agents.ppo_agent import PPOAgent
from ..agents.reinforce_agent import ReinforceAgent
from ..agents.reinforce_step_agent import ReinforceStepAgent

# Mapping of agent names to their classes
AGENT_CLASSES = {
    "ppo": PPOAgent,
    "reinforce": ReinforceAgent,
    "reinforce_step": ReinforceStepAgent,
}

# Import Lightning's ModelCheckpoint
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

def create_env(
    data: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    reward_config: Optional[Dict[str, Any]] = None,
    lookback: int = 10,
    normalize_state: bool = True,
    test_size: float = 0.2
) -> CsvEnv:
    """Create a CsvEnv with validation and error handling.
    
    Args:
        data (pd.DataFrame): Input data
        feature_cols (List[str]): Feature columns
        target_col (str): Target column
        reward_config (Optional[Dict[str, Any]]): Reward configuration
            Default: {'method': 'directional'}
            Supports 'directional', 'proportional', 'threshold' methods.
        lookback (int): Lookback window size
        normalize_state (bool): Whether to normalize state
        test_size (float): Proportion of data to use for testing (default 0.2)

    Returns:
        CsvEnv: Initialized environment
    """
    try:
        return CsvEnv(
            data=data,
            feature_cols=feature_cols,
            target_col=target_col,
            lookback=lookback,
            normalize_state=normalize_state,
            test_size=test_size,
            reward_config=reward_config
        )
    except Exception as e:
        raise RuntimeError(f"Failed to create environment: {str(e)}")

def create_agent(
    env: CsvEnv,
    agent_type: str = "ppo",
    **agent_kwargs: Dict[str, Any]
) -> L.LightningModule:
    """Create an agent of the specified type with sane defaults.
    
    Args:
        env (CsvEnv): Environment the agent will train in.
        agent_type (str): The type of agent to create (e.g., "ppo", "reinforce", "reinforce_step").
        **agent_kwargs: Additional agent parameters.
            activation_fn (str): Activation function for the network.
    
    Returns:
        L.LightningModule: Initialized agent.

    Raises:
        ValueError: If an unsupported `agent_type` is provided.
        RuntimeError: If agent creation fails.
    """
    agent_class = AGENT_CLASSES.get(agent_type.lower())
    if agent_class is None:
        raise ValueError(f"Unsupported agent type: {agent_type}. Available types: {list(AGENT_CLASSES.keys())}")

    try:
        agent = agent_class(
            env=env,
            **agent_kwargs
        )
        return agent
    except Exception as e:
        raise RuntimeError(f"Failed to create agent of type {agent_type}: {str(e)}")

def train_agent(
    agent: L.LightningModule,
    env: CsvEnv,
    max_epochs: int = 100,
    enable_checkpointing: bool = True,
    log_dir: str = "logs",
    experiment_name: str = "default",
    **trainer_kwargs: Dict[str, Any]
) -> L.LightningModule:
    """Train an agent in the given environment.
    
    Args:
        agent (L.LightningModule): Agent to train.
        env (CsvEnv): Training environment.
        max_epochs (int): Maximum number of training epochs.
        enable_checkpointing (bool): Save model checkpoints.
        log_dir (str): Directory for logs.
        experiment_name (str): Name for logging.
        **trainer_kwargs: Additional trainer arguments.
    
    Returns:
        L.LightningModule: Trained agent.
    """
    # Create train and validation datasets
    train_dataset = env.to_dataset(train=True)
    val_dataset = env.to_dataset(train=False)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=len(train_dataset),
        shuffle=False,
        num_workers=0
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=len(val_dataset),
        shuffle=False,
        num_workers=0
    )
    
    # Configure logger
    logger = L.pytorch.loggers.CSVLogger(
        log_dir,
        name=experiment_name
    )
    
    # Configure callbacks
    callbacks = []
    
    if enable_checkpointing:
        # Save checkpoints in the same versioned directory as the CSV logger
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(logger.log_dir, "checkpoints"),
            filename="{epoch}-{train_reward:.2f}-{train_loss:.2f}-{train_accuracy:.2f}-{train_precision:.2f}-{train_recall:.2f}-{train_f1:.2f}",
            monitor="train_reward",
            mode="max",
            save_top_k=3,
            save_last=True,
            every_n_epochs=1  # Save every epoch
        )
        callbacks.append(checkpoint_callback)
        early_stop_callback = EarlyStopping(
            monitor="train_reward", 
            patience=max_epochs*0.2, # 20% of max_epochs 
            verbose=False, 
            mode="max")
        callbacks.append(early_stop_callback)
    

    
    # Configure trainer
    default_trainer_kwargs = {
        "max_epochs": max_epochs,
        "accelerator": "auto",
        "devices": "auto",
        "enable_checkpointing": enable_checkpointing,
        "callbacks": callbacks if callbacks else None,
        "logger": logger
    }
    trainer_kwargs = {**default_trainer_kwargs, **trainer_kwargs}
    trainer = L.Trainer(**trainer_kwargs)
    
    # Train with validation
    trainer.fit(
        agent,
        train_dataloaders=train_dataloader,
        ckpt_path=trainer_kwargs.get('ckpt_path', None)  # Allow resuming from checkpoint
    )
    trainer.validate(
        agent,
        dataloaders=val_dataloader)
    
    return agent

def train_from_csv(
    csv_path: str,
    feature_cols: List[str],
    target_col: str,
    reward_config: Optional[Dict[str, Any]] = None,
    env_kwargs: Dict[str, Any] = {},
    agent_kwargs: Dict[str, Any] = {},
    trainer_kwargs: Dict[str, Any] = {}
) -> L.LightningModule:
    """Create and train an agent from CSV data.
    
    Args:
        csv_path (str): Path to CSV file.
        feature_cols (List[str]): Feature columns.
        target_col (str): Target column.
        reward_config (Optional[Dict[str, Any]]): Reward configuration.
            Default: {'method': 'directional'}
            Supports 'directional', 'proportional', 'threshold' methods.
        env_kwargs (Dict): Environment creation arguments.
        agent_kwargs (Dict): Agent creation arguments.
            Must include 'agent_type' (e.g., "ppo", "reinforce", "reinforce_step").
        trainer_kwargs (Dict): Training arguments.
    
    Returns:
        L.LightningModule: Trained agent.
    
    Example:
        >>> # Train a PPO agent
        >>> ppo_agent = train_from_csv(
        ...     'data.csv',
        ...     feature_cols=['price', 'volume'],
        ...     target_col='price',
        ...     agent_kwargs={'agent_type': 'ppo', 'hidden_layers': [128, 64]}
        ... )
        >>> # Train a Reinforce agent
        >>> reinforce_agent = train_from_csv(
        ...     'data.csv',
        ...     feature_cols=['price', 'volume'],
        ...     target_col='price',
        ...     agent_kwargs={'agent_type': 'reinforce', 'hidden_layers': [128, 64], 'epsilon_end_epochs': 100}
        ... )
    """
    # Load and preprocess data
    df = pd.read_csv(csv_path)
    # Drop rows with NaN values
    df = df.dropna(subset=feature_cols + [target_col])
    
    # Create environment
    # Merge reward_config into env_kwargs if provided
    if reward_config:
        env_kwargs['reward_config'] = reward_config
        print(f"Using reward config: {reward_config}")
        
    env = create_env(
        data=df,
        feature_cols=feature_cols,
        target_col=target_col,
        **env_kwargs
    )
    
    # Extract agent_type from agent_kwargs, default to 'ppo' if not specified
    agent_type = agent_kwargs.pop('agent_type', 'ppo')
    
    # Create and train agent
    agent = create_agent(env, agent_type=agent_type, **agent_kwargs)
    trained_agent = train_agent(agent, env, **trainer_kwargs)
    trained_agent.env = env
    return trained_agent

def load_agent(
    checkpoint_path: str,
    csv_path: str,
    feature_cols: List[str],
    target_col: str,
    agent_type: str = "ppo",
    **env_kwargs: Dict[str, Any]
) -> L.LightningModule:
    """Load a trained agent for inference.
    
    Args:
        checkpoint_path (str): Path to model checkpoint.
        csv_path (str): Path to current data.
        feature_cols (List[str]): Feature columns.
        target_col (str): Target column.
        agent_type (str): The type of agent to load (e.g., "ppo", "reinforce", "reinforce_step").
        **env_kwargs: Environment arguments.
    
    Returns:
        L.LightningModule: Agent ready for inference.

    Raises:
        ValueError: If an unsupported `agent_type` is provided.
        RuntimeError: If agent loading fails.
    """
    # Create environment for inference
    df = pd.read_csv(csv_path)
    # Drop rows with NaN values
    df = df.dropna(subset=feature_cols + [target_col])
    
    env = create_env(
        data=df,
        feature_cols=feature_cols,
        target_col=target_col,
        **env_kwargs
    )
    
    # Get the agent class
    agent_class = AGENT_CLASSES.get(agent_type.lower())
    if agent_class is None:
        raise ValueError(f"Unsupported agent type: {agent_type}. Available types: {list(AGENT_CLASSES.keys())}")

    try:
        agent = agent_class.load_from_checkpoint(
            checkpoint_path,
            env=env
        )
        agent.eval()
        return agent
    except Exception as e:
        raise RuntimeError(f"Failed to load agent of type {agent_type} from checkpoint: {str(e)}")
