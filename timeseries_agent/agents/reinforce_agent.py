import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F
import lightning as L
import numpy as np
from typing import List, Any, Union
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from ..envs.csv_env import CsvEnv
from ..utils.core.tools import get_state_tensor, sample_action, discount_rewards

class ReinforceAgent(L.LightningModule):
    """Reinforce Agent for reinforcement learning tasks.

    This agent implements the REINFORCE (Monte Carlo Policy Gradient) algorithm.
    It learns a policy by sampling episodes, calculating discounted returns,
    and updating the policy parameters using the policy gradient theorem.

    Unique Properties:
        - Monte Carlo Policy Gradient: Updates the policy based on full episode
          returns, making it suitable for episodic tasks.
        - Epsilon-Greedy Exploration: Uses an annealing epsilon-greedy strategy
          to balance exploration and exploitation during training.
        - Entropy Regularization: Adds an entropy bonus to the loss to encourage
          more diverse actions and prevent premature convergence to suboptimal policies.
        - Supports various activation functions and Xavier initialization for Tanh.
    """
    def __init__(self,
                 env: CsvEnv,
                 hidden_layers: List[int] = [64, 32],
                 output_size: int = 3,
                 learning_rate: float = 1e-3,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 1e-2,
                 epsilon_end_epochs: int = 50,
                 entropy_beta: float = 0.01,
                 activation_fn: nn.Module = nn.Tanh()):
        """Initializes the ReinforceAgent.

        Args:
            env: The environment to interact with, typically a CsvEnv instance.
            hidden_layers: A list of integers specifying the number of neurons
                in each hidden layer of the policy network.
            output_size: The number of possible actions (output dimensions) for the policy network.
            learning_rate: The learning rate for the Adam optimizer.
            gamma: The discount factor for future rewards.
            epsilon_start: The initial value of epsilon for epsilon-greedy exploration.
            epsilon_end: The final value of epsilon for epsilon-greedy exploration.
            epsilon_end_epochs: The number of epochs over which epsilon decays from start to end.
            entropy_beta: The coefficient for the entropy regularization term.
            activation_fn: The activation function to use in the hidden layers.
        """
        super().__init__()

        self.env = env
        self.full_data = env.df.copy()
        self.num_features = len(env.feature_cols)
        self.lookback = env.lookback
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_end_epochs = epsilon_end_epochs
        self.entropy_beta = entropy_beta
        self.activation_fn = activation_fn
        self.gamma = gamma

        self.save_hyperparameters(ignore=['full_data', 'activation_fn', 'env'])

        input_size = self.num_features * self.lookback
        self.policy_net = self._build_mlp(input_size, hidden_layers, output_size)
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
        use_xavier = isinstance(self.activation_fn, nn.Tanh)
        
        for size in hidden_layers:
            linear_layer = nn.Linear(last_size, size)
            if use_xavier:
                init.xavier_uniform_(linear_layer.weight)
                init.zeros_(linear_layer.bias)
            layers.append(linear_layer)
            layers.append(self.activation_fn)
            last_size = size
            
        output_layer = nn.Linear(last_size, output_size)
        if use_xavier:
            init.xavier_uniform_(output_layer.weight)
            init.zeros_(output_layer.bias)
        layers.append(output_layer)
        
        return nn.Sequential(*layers)

    def forward(self, state_tensor: torch.Tensor) -> torch.Tensor:
        """Performs a forward pass through the policy network.

        Args:
            state_tensor: The input state tensor. Can be 2D (single state) or 3D (batched states).

        Returns:
            The raw logits from the policy network.

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
        return self.policy_net(state_flat)

    def get_epsilon(self) -> float:
        """Calculates the current epsilon value for epsilon-greedy exploration.

        Epsilon decays exponentially from `epsilon_start` to `epsilon_end` over
        `epsilon_end_epochs`.

        Returns:
            The current epsilon value.
        """
        if self.epsilon_end == 0.0:
            self.epsilon_end += 1e-6
        decay_rate = (self.epsilon_end / self.epsilon_start)**(1/self.epsilon_end_epochs)
        epsilon = self.epsilon_start * (decay_rate ** self.current_epoch)
        return max(self.epsilon_end, epsilon)

    def training_step(self, batch: Any, batch_idx: int) -> dict:
        """Performs the Reinforce agent update loop for one epoch/episode sequentially.

        This method collects an episode's experience, calculates discounted returns,
        and updates the policy network using the REINFORCE algorithm.

        Args:
            batch: A batch of data (timesteps) from the DataLoader.
            batch_idx: The index of the current batch.

        Returns:
            A dictionary containing logged metrics.
        """
        opt = self.optimizers()
        current_epsilon = self.get_epsilon()
        
        self.env.set_train_mode(train=True)
        self.env.reset().to(self.device)

        states, log_probs, entropies, rewards = [], [], [], []
        predicted_actions, true_actions = [], []

        for timestep_idx in batch:
            timestep_idx = timestep_idx.item()
            if timestep_idx >= len(self.env.df) - 1:
                continue

            self.env.pointer = timestep_idx
            state = self.env.get_state().to(self.device)
            logits = self(state)
            probs = F.softmax(logits, dim=1).squeeze(0)
            probs = torch.clamp(probs, min=1e-6)
            
            action, log_prob = sample_action(probs, current_epsilon)
            _, reward, _, info = self.env.step(action)
            
            entropy = -torch.sum(probs * torch.log(probs + 1e-9))
            
            states.append(state)
            log_probs.append(log_prob)
            entropies.append(entropy)
            rewards.append(reward)
            predicted_actions.append(action)
            true_actions.append(info['true_action'])

        if not rewards:
            return {}
        
        states = torch.stack(states)
        log_probs = torch.stack(log_probs).detach()
        entropies = torch.stack(entropies)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)

        returns = discount_rewards(rewards, self.gamma)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)

        policy_loss = -torch.sum(log_probs * returns)
        entropy_loss = -self.entropy_beta * torch.sum(entropies)
        total_loss = policy_loss + entropy_loss

        opt.zero_grad()
        self.manual_backward(total_loss)
        torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        opt.step()

        acc = accuracy_score(true_actions, predicted_actions)
        precision, recall, f1, _ = precision_recall_fscore_support(true_actions, predicted_actions, average='weighted')

        self.log_dict({
            'train_reward': rewards.mean().item(),
            'train_steps': len(rewards),
            'train_loss': total_loss.detach(),
            'train_epsilon': current_epsilon,
            'train_accuracy': acc,
            'train_precision': precision,
            'train_recall': recall,
            'train_f1': f1,
        }, on_step=False, on_epoch=True, prog_bar=True, logger=True)

    def validation_step(self, batch: Any, batch_idx: int):
        """Performs a single validation epoch/episode for the Reinforce agent.

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
                logits = self(state)
                probs = F.softmax(logits, dim=1)
                action = torch.argmax(probs).item()

                _, reward, _, info = self.env.step(action)
                rewards.append(reward)
                predicted_actions.append(action)
                true_actions.append(info['true_action'])

        if not rewards:
            return {}

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
        self.eval()
        with torch.no_grad():
            if features.ndim == 1:
                features = features.reshape(1, -1)
            elif features.ndim > 2:
                raise ValueError("Features must be 1D or 2D array")

            if self.env.normalize_state:
                mean = np.mean(features, axis=0, keepdims=True)
                std = np.std(features, axis=0, keepdims=True)
                features = (features - mean) / (std + 1e-8)
                features = features.reshape(1, -1)

            state = torch.tensor(features, dtype=torch.float32).to(self.device)
            logits = self(state)
            probabilities = F.softmax(logits, dim=1)
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
