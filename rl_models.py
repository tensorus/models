import random
from dataclasses import dataclass # dataclass is imported but not used by the RL models here
from typing import Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .base import TensorusModel


class ReplayBuffer:
    '''
    A simple FIFO experience replay buffer for storing transitions.

    This buffer is commonly used in off-policy reinforcement learning algorithms
    like DQN to store experiences (state, action, reward, next_state, done)
    and sample them randomly for training to break correlations between
    consecutive experiences.
    '''

    def __init__(self, capacity: int = 10000):
        '''
        Initialize the ReplayBuffer.

        Args:
            capacity (int): The maximum number of transitions that can be stored
                          in the buffer. When the buffer is full, older transitions
                          are overwritten. Defaults to 10000.
        '''
        self.capacity = capacity
        self.buffer: List[Tuple[Any, int, float, Any, bool]] = []
        self.position = 0

    def push(self, state: Any, action: int, reward: float, next_state: Any, done: bool) -> None:
        '''
        Add a new transition to the buffer.

        If the buffer is full, the oldest transition is replaced.

        Args:
            state (Any): The current state observed from the environment.
            action (int): The action taken in the current state.
            reward (float): The reward received after taking the action.
            next_state (Any): The next state observed after taking the action.
            done (bool): A boolean indicating whether the episode terminated after
                         this transition.
        '''
        if len(self.buffer) < self.capacity:
            self.buffer.append(None) # type: ignore
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[List[Any], List[int], List[float], List[Any], List[bool]]:
        '''
        Randomly sample a batch of transitions from the buffer.

        Args:
            batch_size (int): The number of transitions to sample.

        Returns:
            Tuple[List[Any], List[int], List[float], List[Any], List[bool]]:
                A tuple containing five lists:
                - states: A list of sampled states.
                - actions: A list of sampled actions.
                - rewards: A list of sampled rewards.
                - next_states: A list of sampled next states.
                - dones: A list of sampled done flags.

        Raises:
            ValueError: If batch_size is larger than the current number of items
                        in the buffer.
        '''
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(list, zip(*batch))
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        '''
        Return the current number of transitions stored in the buffer.

        Returns:
            int: The current size of the buffer.
        '''
        return len(self.buffer)


class QLearningModel(TensorusModel):
    """Tabular Q-learning for discrete state/action spaces."""

    def __init__(self, n_states: int, n_actions: int, alpha: float = 0.1, gamma: float = 0.99, epsilon: float = 0.1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((n_states, n_actions), dtype=np.float32)

    def fit(self, env: Any, episodes: int = 10) -> None:
        for _ in range(episodes):
            state = int(env.reset())
            done = False
            while not done:
                if random.random() < self.epsilon:
                    action = random.randrange(self.n_actions)
                else:
                    action = int(np.argmax(self.q_table[state]))
                next_state, reward, done, _ = env.step(action)
                next_state = max(0, min(self.n_states - 1, int(next_state)))
                best_next = np.max(self.q_table[next_state])
                td_target = reward + self.gamma * best_next * (1 - done)
                td_error = td_target - self.q_table[state, action]
                self.q_table[state, action] += self.alpha * td_error
                state = next_state

    def predict(self, state: Any) -> int:
        state = int(state)
        return int(np.argmax(self.q_table[state]))

    def save(self, path: str) -> None:
        np.save(path, self.q_table)

    def load(self, path: str) -> None:
        self.q_table = np.load(path)


class QNetwork(nn.Module):
    '''
    A simple Multi-Layer Perceptron (MLP) used as a Q-value approximator.

    This network takes a state representation as input and outputs estimated
    Q-values for each possible action in that state. It consists of three
    fully connected layers with ReLU activations for the hidden layers.
    '''
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 64):
        '''
        Initialize the QNetwork.

        Args:
            state_dim (int): The dimensionality of the input state space.
            action_dim (int): The number of discrete actions, determining the
                              output dimensionality of the network.
            hidden (int, optional): The number of units in each hidden layer.
                                    Defaults to 64.
        '''
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        Perform the forward pass of the network.

        Args:
            x (torch.Tensor): The input tensor representing the state(s).
                              Expected shape: (batch_size, state_dim) or (state_dim,).

        Returns:
            torch.Tensor: The output tensor representing the Q-values for each action.
                          Shape: (batch_size, action_dim) or (action_dim,).
        '''
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def _to_tensor(x: Any) -> torch.Tensor:
    '''
    Convert input data to a PyTorch float tensor.

    If the input is already a PyTorch Tensor, it\'s converted to float type.
    If the input is a NumPy array, it\'s converted to a PyTorch tensor and then
    to float type.
    If the resulting tensor is a scalar (0-dimensional), it\'s unsqueezed to
    become 1-dimensional (e.g., a tensor with a single element). This is often
    useful for consistency in batch processing or network inputs.
    The function assumes the input `x` is on the CPU if it\'s a NumPy array or
    Python native type. If `x` is already a Tensor, its original device is preserved.

    Args:
        x (Any): The input data to convert. Expected to be a PyTorch Tensor,
                 NumPy array, or a type compatible with `torch.tensor()`
                 (e.g., list of numbers, a single number).

    Returns:
        torch.Tensor: The converted data as a PyTorch float tensor, potentially
                      unsqueezed if it was a scalar. The tensor will be on the
                      same device as the input if `x` was a Tensor, otherwise CPU.

    Raises:
        TypeError: If the input `x` is of a type that cannot be converted to a
                   PyTorch tensor by `torch.tensor()`.
    '''
    if isinstance(x, torch.Tensor):
        t = x.float()
    else:
        t = torch.tensor(x, dtype=torch.float32)
    if t.dim() == 0:
        t = t.unsqueeze(0)
    return t


class DQNModel(TensorusModel):
    '''
    Deep Q-Network (DQN) implementation for discrete action spaces.

    DQN uses a neural network (typically an MLP or CNN) to approximate the
    Q-value function (action-value function). It employs key techniques such as
    experience replay to store and sample transitions, and a target network
    to stabilize training by providing fixed targets for Q-value updates.
    This implementation uses a QNetwork (MLP) for both the policy and target networks.

    Attributes:
        state_dim (int): Dimensionality of the state space.
        action_dim (int): Number of discrete actions available in the environment.
        gamma (float): Discount factor for future rewards in the Bellman equation.
        epsilon (float): Exploration rate for the epsilon-greedy action selection strategy.
                         A fixed epsilon is used in this implementation.
        batch_size (int): Number of experiences to sample from the replay buffer
                          for each optimization step.
        target_update (int): Frequency (in terms_of environment steps) for updating
                             the target network weights with the policy network weights.
        policy_net (QNetwork): The main Q-network that is trained. It estimates
                               Q-values and is used to select actions.
        target_net (QNetwork): A separate Q-network with the same architecture as
                               the policy_net. Its weights are periodically copied
                               from the policy_net and are used for calculating
                               target Q-values during the optimization step,
                               providing stability to the training process.
        optimizer (torch.optim.Optimizer): The optimizer (Adam) used for training
                                           the policy_net.
        buffer (ReplayBuffer): The experience replay buffer used to store and
                               sample transitions.
        steps (int): A counter for the total number of environment steps taken,
                     primarily used to schedule target network updates.
    '''

    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 64, lr: float = 1e-3,
                 gamma: float = 0.99, epsilon: float = 0.1, batch_size: int = 32, buffer_size: int = 10000,
                 target_update: int = 10):
        '''
        Initialize the DQNModel agent.

        Args:
            state_dim (int): Dimensionality of the input state space.
            action_dim (int): Number of discrete actions the agent can take.
            hidden_size (int, optional): Number of units in the hidden layers of
                                         the QNetwork. Defaults to 64.
            lr (float, optional): Learning rate for the Adam optimizer.
                                  Defaults to 1e-3.
            gamma (float, optional): Discount factor for future rewards.
                                     Value should be between 0 and 1. Defaults to 0.99.
            epsilon (float, optional): Exploration rate for the epsilon-greedy
                                       policy. Probability of choosing a random
                                       action. Defaults to 0.1.
                                       (Note: This implementation uses a fixed epsilon.
                                       For advanced usage, consider implementing epsilon decay).
            batch_size (int, optional): Batch size for sampling experiences from
                                        the replay buffer during optimization.
                                        Defaults to 32.
            buffer_size (int, optional): Maximum capacity of the replay buffer.
                                         Defaults to 10000.
            target_update (int, optional): Frequency (in terms of environment steps)
                                           at which the target network weights are
                                           updated with the policy network weights.
                                           Defaults to 10.
        '''
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.target_update = target_update

        self.policy_net = QNetwork(state_dim, action_dim, hidden_size)
        self.target_net = QNetwork(state_dim, action_dim, hidden_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)
        self.steps = 0

    def _optimize(self) -> None:
        '''
        Perform a single optimization step on the policy network.

        This method is called typically after each environment step (or a few steps).
        It samples a batch of experiences from the replay buffer, calculates the
        loss using the Bellman equation (with the target network providing stable
        targets), and updates the policy network weights via backpropagation.
        '''
        if len(self.buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states_t = torch.stack([_to_tensor(s) for s in states])
        actions_t = torch.tensor(actions, dtype=torch.long).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        next_states_t = torch.stack([_to_tensor(s) for s in next_states])
        dones_t = torch.tensor(dones, dtype=torch.float32)

        q_values = self.policy_net(states_t).gather(1, actions_t).squeeze(-1)

        with torch.no_grad():
            next_q_values_target = self.target_net(next_states_t).max(1)[0]
            target_q_values = rewards_t + self.gamma * next_q_values_target * (1 - dones_t)

        loss = F.mse_loss(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def fit(self, env: Any, episodes: int = 10) -> None:
        '''
        Train the DQN agent by interacting with the specified environment.

        The agent interacts with the environment for a given number of episodes.
        In each episode, it takes actions based on an epsilon-greedy policy,
        stores experiences in the replay buffer, and periodically optimizes
        its Q-network. The target network is updated at regular intervals.

        Args:
            env (Any): The environment to interact with. It must implement:
                       - reset() -> state: Resets the environment and returns an
                         initial state. The state should be compatible with `state_dim`.
                       - step(action: int) -> Tuple[state, reward: float, done: bool, info: dict]:
                         Executes an action in the environment and returns the
                         next state, the reward received, a boolean indicating if
                         the episode has terminated, and an auxiliary info dictionary.
            episodes (int, optional): The total number of episodes to train for.
                                      Defaults to 10. For meaningful learning, this
                                      value typically needs to be much larger.
        '''
        for ep in range(episodes):
            state = env.reset()
            done = False
            while not done:
                if random.random() < self.epsilon:
                    action = random.randrange(self.action_dim)
                else:
                    self.policy_net.eval()
                    with torch.no_grad():
                        state_t = _to_tensor(state)
                        q_vals = self.policy_net(state_t)
                        action = int(torch.argmax(q_vals).item())
                    self.policy_net.train()

                next_state, reward, done, _ = env.step(action)
                self.buffer.push(state, action, reward, next_state, done)
                state = next_state
                self._optimize()
                if self.steps % self.target_update == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                self.steps += 1

    def predict(self, state: Any) -> int:
        '''
        Select an action for the given state using the learned Q-network (greedy policy).

        This method uses the policy network to estimate Q-values for all actions
        in the given state and returns the action with the highest Q-value.
        No exploration (epsilon-greedy) is performed here; it\'s deterministic.

        Args:
            state (Any): The current state of the environment. The type should be
                         compatible with the `state_dim` of the QNetwork and
                         processable by the `_to_tensor` utility function.

        Returns:
            int: The action selected by the greedy policy.
        '''
        self.policy_net.eval()
        with torch.no_grad():
            state_t = _to_tensor(state)
            q_values = self.policy_net(state_t)
            action = int(torch.argmax(q_values).item())
        return action

    def save(self, path: str) -> None:
        '''
        Save the DQN model\'s state to a file.

        This method saves the state dictionaries of both the policy network
        and the target network.

        Args:
            path (str): Path to the file where the model will be saved.
        '''
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict()
        }, path)

    def load(self, path: str) -> None:
        '''
        Load the DQN model\'s state from a file.

        This method loads the state dictionaries for the policy network and
        the target network. Assumes the model instance has been initialized
        with the same architecture.

        Args:
            path (str): Path to the file from which to load the model.
        '''
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.policy_net.eval()
        self.target_net.eval()


def _actor_critic_networks(state_dim: int, action_dim: int, hidden: int = 64):
    # This function remains as is
    actor = nn.Sequential(
        nn.Linear(state_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, action_dim)
    )
    critic = nn.Sequential(
        nn.Linear(state_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 1)
    )
    return actor, critic


class A2CModel(TensorusModel):
    # This class remains as is
    """Simplified Advantage Actor-Critic."""

    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 64, lr: float = 1e-3, gamma: float = 0.99):
        self.gamma = gamma
        self.actor, self.critic = _actor_critic_networks(state_dim, action_dim, hidden_size)
        self.optimizer = optim.Adam(list(self.actor.parameters()) + list(self.critic.parameters()), lr=lr)

    def fit(self, env: Any, episodes: int = 10) -> None:
        for _ in range(episodes):
            state = env.reset()
            done = False
            while not done:
                state_t = _to_tensor(state)
                probs = torch.softmax(self.actor(state_t), dim=-1)
                action = torch.multinomial(probs, 1).item()
                next_state, reward, done, _ = env.step(action)
                next_state_t = _to_tensor(next_state)
                value = self.critic(state_t)
                next_value = self.critic(next_state_t).detach()
                advantage = reward + self.gamma * (1 - done) * next_value - value
                actor_loss = -torch.log(probs[action]) * advantage.detach()
                critic_loss = advantage.pow(2)
                loss = actor_loss + critic_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                state = next_state

    def predict(self, state: Any) -> int:
        with torch.no_grad():
            probs = torch.softmax(self.actor(_to_tensor(state)), dim=-1)
            return int(torch.argmax(probs).item())

    def save(self, path: str) -> None:
        torch.save({"actor": self.actor.state_dict(), "critic": self.critic.state_dict()}, path)

    def load(self, path: str) -> None:
        data = torch.load(path, map_location="cpu")
        self.actor.load_state_dict(data["actor"])
        self.critic.load_state_dict(data["critic"])


class PPOModel(A2CModel):
    # This class remains as is
    """Proximal Policy Optimization - minimal variant."""

    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 64, lr: float = 1e-3,
                 gamma: float = 0.99, clip_epsilon: float = 0.2):
        super().__init__(state_dim, action_dim, hidden_size, lr, gamma)
        self.clip_epsilon = clip_epsilon

    def fit(self, env: Any, episodes: int = 10) -> None:
        for _ in range(episodes):
            state = env.reset()
            done = False
            while not done:
                state_t = _to_tensor(state)
                probs = torch.softmax(self.actor(state_t), dim=-1)
                action = torch.multinomial(probs, 1).item()
                log_prob_old = torch.log(probs[action])
                next_state, reward, done, _ = env.step(action)
                next_state_t = _to_tensor(next_state)
                value = self.critic(state_t)
                next_value = self.critic(next_state_t).detach()
                advantage = reward + self.gamma * (1 - done) * next_value - value
                probs_new = torch.softmax(self.actor(state_t), dim=-1)
                log_prob_new = torch.log(probs_new[action])
                ratio = torch.exp(log_prob_new - log_prob_old.detach())
                actor_loss = -torch.min(ratio * advantage.detach(),
                                       torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantage.detach())
                critic_loss = advantage.pow(2)
                loss = actor_loss + critic_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                state = next_state


class TRPOModel(A2CModel):
    # This class remains as is
    """Placeholder Trust Region Policy Optimization."""

    def fit(self, env: Any, episodes: int = 1) -> None:
        # For brevity we reuse A2C updates. Real TRPO requires complex trust region steps.
        super().fit(env, episodes)


class SACModel(TensorusModel):
    # This class remains as is
    """Simple Soft Actor Critic for discrete actions."""

    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 64, lr: float = 1e-3,
                 gamma: float = 0.99, alpha: float = 0.2, buffer_size: int = 10000, batch_size: int = 32):
        self.gamma = gamma
        self.alpha = alpha
        self.batch_size = batch_size
        self.policy_net, self.q_net = _actor_critic_networks(state_dim, action_dim, hidden_size)
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        self.optimizer = optim.Adam(list(self.policy_net.parameters()) +
                                    list(self.q_net.parameters()) +
                                    list(self.value_net.parameters()), lr=lr)
        self.replay = ReplayBuffer(buffer_size)

    def _sample_action(self, state: torch.Tensor) -> Tuple[int, torch.Tensor]:
        logits = self.policy_net(state)
        prob = torch.softmax(logits, dim=-1)
        action = torch.multinomial(prob, 1).item()
        log_prob = torch.log(prob[action])
        return action, log_prob

    def _update(self):
        if len(self.replay) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.replay.sample(self.batch_size)
        states = torch.stack([_to_tensor(s) for s in states])
        actions = torch.tensor(actions).long()
        rewards = torch.tensor(rewards).float()
        next_states = torch.stack([_to_tensor(s) for s in next_states])
        dones = torch.tensor(dones).float()

        q_vals = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        with torch.no_grad():
            next_logits = self.policy_net(next_states)
            next_prob = torch.softmax(next_logits, dim=-1)
            next_log_prob = torch.log(next_prob + 1e-8)
            next_q = self.q_net(next_states)
            next_v = (next_prob * (next_q - self.alpha * next_log_prob)).sum(dim=1)
            target_q = rewards + self.gamma * next_v * (1 - dones)
        q_loss = F.mse_loss(q_vals, target_q)

        logits = self.policy_net(states)
        prob = torch.softmax(logits, dim=-1)
        log_prob = torch.log(prob + 1e-8)
        q_new = self.q_net(states)
        policy_loss = (prob * (self.alpha * log_prob - q_new)).sum(dim=1).mean()

        self.optimizer.zero_grad()
        (q_loss + policy_loss).backward()
        self.optimizer.step()

    def fit(self, env: Any, episodes: int = 10) -> None:
        for _ in range(episodes):
            state = env.reset()
            done = False
            while not done:
                state_t = _to_tensor(state)
                action, log_prob = self._sample_action(state_t)
                next_state, reward, done, _ = env.step(action)
                self.replay.push(state_t, action, reward, _to_tensor(next_state), done)
                self._update()
                state = next_state

    def predict(self, state: Any) -> int:
        with torch.no_grad():
            logits = self.policy_net(_to_tensor(state))
            return int(torch.argmax(logits).item())

    def save(self, path: str) -> None:
        torch.save({"policy": self.policy_net.state_dict(),
                    "q_net": self.q_net.state_dict(),
                    "value_net": self.value_net.state_dict()}, path)

    def load(self, path: str) -> None:
        data = torch.load(path, map_location="cpu")
        self.policy_net.load_state_dict(data["policy"])
        self.q_net.load_state_dict(data["q_net"])
        self.value_net.load_state_dict(data["value_net"])
