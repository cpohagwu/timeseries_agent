import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
import lightning as L
from typing import List, Any, Union
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from ..envs.csv_env import CsvEnv
from ..utils.core.tools import get_state_tensor, sample_action, discount_rewards

class PPOAgent(L.LightningModule):
    """Proximal Policy Optimization (PPO) Agent for reinforcement learning tasks.

    This agent implements the PPO algorithm, which is an on-policy algorithm
    that optimizes a stochastic policy in an actor-critic setup. It uses a
    clipping mechanism to limit the policy update, ensuring stable training.

    Unique Properties:
        - Actor-Critic Architecture: Separates policy (actor) and value (critic) networks.
        - Clipped Surrogate Objective: Uses a clipped probability ratio to prevent
          large policy updates, improving training stability.
        - Entropy Regularization: Encourages exploration by adding an entropy bonus
          to the policy loss.
        - Generalized Advantage Estimation (GAE): Calculates advantages for more
          stable and efficient learning (though not explicitly GAE, it uses
          discounted returns and advantage normalization).
        - Supports various activation functions and Xavier initialization for Tanh.
    """
    def __init__(self,
                 env: CsvEnv,
                 hidden_layers: List[int] = [64, 32],
                 output_size: int = 3,
                 learning_rate: float = 1e-3,
                 epsilon_clip: float = 0.2,
                 entropy_beta: float = 0.01,
                 gamma: float = 0.99,
                 value_loss_coef: float = 0.5,
                 activation_fn: nn.Module = nn.Tanh()):
        """Initializes the PPOAgent.

        Args:
            env: The environment to interact with, a CsvEnv instance in this case.
            hidden_layers: A list of integers specifying the number of neurons
                in each hidden layer of the policy and value networks.
            output_size: The number of possible actions (output dimensions) for the policy network.
            learning_rate: The learning rate for the Adam optimizer.
            epsilon_clip: The clipping parameter for the PPO objective.
            entropy_beta: The coefficient for the entropy regularization term.
            gamma: The discount factor for future rewards.
            value_loss_coef: The coefficient for the value function loss.
            activation_fn: The activation function to use in the hidden layers.
        """
        super().__init__()

        self.env = env
        self.full_data = env.df.copy()
        self.num_features = len(env.feature_cols)
        self.lookback = env.lookback
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.epsilon_clip = epsilon_clip
        self.entropy_beta = entropy_beta
        self.gamma = gamma
        self.value_loss_coef = value_loss_coef
        self.activation_fn = activation_fn

        self.save_hyperparameters(ignore=['full_data', 'activation_fn', 'env'])

        # Shared input size
        input_size = self.num_features * self.lookback

        # --- Policy Network ---
        self.policy_net = self._build_mlp(input_size, hidden_layers, output_size)

        # --- Value Network ---
        self.value_net = self._build_mlp(input_size, hidden_layers, 1)

        self.automatic_optimization = False

    def _build_mlp(self, input_size: int, hidden_layers: List[int], output_size: int) -> nn.Sequential:
        """Builds a multi-layer perceptron (MLP) network.

        Args:
            input_size: The input dimension of the MLP.
            hidden_layers: A list of integers specifying the number of neurons
                in each hidden layer.
            output_size: The output dimension of the MLP.

        Returns:
            A `nn.Sequential` model representing the MLP.
        """
        layers = []
        last_size = input_size
        
        # Check if using Tanh activation for Xavier initialization
        use_xavier = isinstance(self.activation_fn, nn.Tanh)
        
        for size in hidden_layers:
            linear_layer = nn.Linear(last_size, size)
            if use_xavier:
                init.xavier_uniform_(linear_layer.weight)
                init.zeros_(linear_layer.bias)
            layers.append(linear_layer)
            layers.append(self.activation_fn)
            last_size = size
            
        # Output layer
        output_layer = nn.Linear(last_size, output_size)
        if use_xavier:
            init.xavier_uniform_(output_layer.weight)
            init.zeros_(output_layer.bias)
        layers.append(output_layer)
        
        return nn.Sequential(*layers)

    def forward(self, state_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Performs a forward pass through the policy and value networks.

        Args:
            state_tensor: The input state tensor. Can be 2D (single state) or 3D (batched states).

        Returns:
            A tuple containing:
                - policy_logits: The raw logits from the policy network.
                - value: The predicted value from the value network.

        Raises:
            ValueError: If the `state_tensor` has an unexpected number of dimensions.
        """
        if state_tensor.ndim == 3:
            batch_size, lookback, features = state_tensor.size()
            state_flat = state_tensor.view(batch_size, -1)
        elif state_tensor.ndim == 2:
            state_flat = state_tensor.view(1, -1)
        else:
            raise ValueError(f"Unexpected state_tensor ndim: {state_tensor.ndim}")
        return self.policy_net(state_flat), self.value_net(state_flat).squeeze(-1)

    def training_step(self, batch: Any, batch_idx: int) -> dict:
        """Performs the PPO agent update loop for one epoch/episode sequentially.

        Args:
            batch: A batch of data (timesteps) from the DataLoader.
            batch_idx: The index of the current batch.

        Returns:
            A dictionary containing logged metrics.
        """
        opt = self.optimizers()
        self.env.set_train_mode(train=True)
        self.env.reset()

        # Store experience
        states, actions, rewards, log_probs_old, values = [], [], [], [], []
        predicted_actions, true_actions = [], []

        for timestep_idx in batch:
            timestep_idx = timestep_idx.item()
            if timestep_idx >= len(self.env.df) - 1:
                continue

            self.env.pointer = timestep_idx
            state = self.env.get_state().to(self.device)
            logits, value = self(state)
            probs = F.softmax(logits, dim=-1).squeeze(0)
            probs = torch.clamp(probs, min=1e-6)

            action, log_prob = sample_action(probs, epsilon=0.0)
            _, reward, _, info = self.env.step(action)
            
            # Store experience
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs_old.append(log_prob)
            values.append(value.squeeze(0))
            predicted_actions.append(action)
            true_actions.append(info['true_action'])

        if not rewards:
            return {}

        # Process episode
        states = torch.stack(states)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        log_probs_old = torch.stack(log_probs_old).detach()
        values = torch.stack(values)

        returns = discount_rewards(rewards, self.gamma)
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-9)

        # Compute new log probs
        logits, new_values = self(states)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        # Probability ratio
        ratio = torch.exp(log_probs - log_probs_old)

        # Clipped loss
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        value_loss = F.mse_loss(new_values, returns.detach())

        # Total loss
        total_loss = policy_loss + (self.value_loss_coef * value_loss) - (self.entropy_beta * entropy)

        # Backward
        opt.zero_grad()
        self.manual_backward(total_loss)
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        opt.step()

        # Metrics
        # avg_reward = rewards.mean().item()
        acc = accuracy_score(true_actions, predicted_actions)
        precision, recall, f1, _ = precision_recall_fscore_support(true_actions, predicted_actions, average='weighted')

        self.log_dict({
            'train_reward': rewards.mean().item(),
            'train_steps': len(rewards),
            'train_loss': total_loss.detach(),
            'train_entropy': entropy,
            'train_accuracy': acc,
            'train_precision': precision,
            'train_recall': recall,
            'train_f1': f1,
        }, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def validation_step(self, batch: Any, batch_idx: int) -> dict:
        """Performs a single validation epoch/episode for the PPO agent.

        Args:
            batch: A batch of data (timesteps) from the DataLoader.
            batch_idx: The index of the current batch.

        Returns:
            A dictionary containing logged metrics.
        """
        self.env.set_train_mode(train=False)
        self.env.reset()

        rewards, predicted_actions, true_actions = [], [], []

        with torch.no_grad():
            for timestep_idx in batch:
                timestep_idx = timestep_idx.item()
                if timestep_idx >= len(self.env.df) - 1:
                    continue

                self.env.pointer = timestep_idx
                state = self.env.get_state().to(self.device)
                logits, _ = self(state)
                probs = F.softmax(logits, dim=-1)
                action = torch.argmax(probs).item()

                _, reward, _, info = self.env.step(action)
                rewards.append(reward)
                predicted_actions.append(action)
                true_actions.append(info['true_action'])

        if not rewards:
            return {}

        # avg_reward = sum(rewards) / len(rewards)
        acc = accuracy_score(true_actions, predicted_actions)
        precision, recall, f1, _ = precision_recall_fscore_support(true_actions, predicted_actions, average='weighted')

        self.log_dict({
            'val_reward': sum(rewards) / len(rewards),
            'val_steps': len(rewards),
            'val_accuracy': acc,
            'val_precision': precision,
            'val_recall': recall,
            'val_f1': f1,
        }, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def act(self, features: np.ndarray, return_probs: bool = False) -> Union[int, tuple]:
        """Determines the action to take based on the current features for real-time inference.

        Args:
            features: A NumPy array representing the current state features.
                Can be 1D (num_features,) or 2D (lookback, num_features).
            return_probs: If True, returns the action probabilities along with the action.

        Returns:
            The chosen action index (int) or a tuple (action_index, probabilities_array)
            if `return_probs` is True.

        Raises:
            ValueError: If the `features` array has an unexpected number of dimensions.
        """
        self.eval()  # Ensure evaluation mode
        with torch.no_grad():
            # Handle feature dimensions
            if features.ndim == 1:
                features = features.reshape(1, -1)
            elif features.ndim > 2:
                raise ValueError("Features must be 1D or 2D array")

            # Check if normalization is needed
            if self.env.normalize_state:
                mean = np.mean(features, axis=0, keepdims=True)
                std = np.std(features, axis=0, keepdims=True)
                features = (features - mean) / (std + 1e-8)  # Epsilon for stability
                features = features.reshape(1, -1)

            # Convert to tensor
            state = torch.tensor(features, dtype=torch.float32).to(self.device)

            # Get action probabilities from policy network (ignore value network for inference)
            logits, _ = self(state)
            probabilities = F.softmax(logits, dim=1)
            
            # Get greedy action
            action = torch.argmax(probabilities).item()
            
            if return_probs:
                return action, probabilities.squeeze().cpu().numpy()
            return action

    def configure_optimizers(self):
        """Configures the optimizer for the agent.

        Returns:
            An instance of `torch.optim.Adam` configured with the agent's parameters
            and learning rate.
        """
        return optim.Adam(self.parameters(), lr=self.learning_rate)
