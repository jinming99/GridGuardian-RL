"""
Multi-Agent RL utilities for PettingZoo environments and MARL algorithms.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Any, Callable, Tuple
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict

# Import base utilities
from .rl import RunningMeanStd, RolloutBuffer


def build_central_state(
    obs_dict: Dict[str, np.ndarray], 
    agent_order: Optional[List[str]] = None
) -> np.ndarray:
    """
    Build a centralized (joint) observation by concatenating per-agent flat observations.
    (Moved from utils_rl.py)

    Parameters
    ----------
    obs_dict : Dict[str, np.ndarray]
        Mapping agent_id -> flat observation (np.ndarray)
    agent_order : Optional[List[str]]
        Order in which to concatenate agents. If None, uses sorted keys for determinism.

    Returns
    -------
    np.ndarray
        1D flat joint observation vector.
    """
    if agent_order is None:
        agent_order = sorted(list(obs_dict.keys()))
    obs_list = []
    for aid in agent_order:
        ob = obs_dict.get(aid, None)
        if ob is None:
            continue
        arr = np.asarray(ob, dtype=np.float32).reshape(-1)
        obs_list.append(arr)
    if len(obs_list) == 0:
        return np.array([], dtype=np.float32)
    return np.concatenate(obs_list, axis=0)


class ValueNormalizer:
    """
    Simple return/value normalizer backed by RunningMeanStd.
    (Moved from utils_rl.py)

    - Call update(returns_np) each update to refresh running statistics.
    - Use normalize_np/normalize_torch to scale targets (and optionally predictions)
      with the same mean/std.
    """

    def __init__(self):
        self._rms = RunningMeanStd(shape=())

    @property
    def mean(self) -> float:
        return float(self._rms.mean)

    @property
    def std(self) -> float:
        return float(np.sqrt(self._rms.var + 1e-8))

    def update(self, returns: np.ndarray) -> None:
        x = np.asarray(returns, dtype=np.float64).reshape(-1)
        if x.size > 0:
            self._rms.update(x)

    def normalize_np(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float32) - self.mean) / self.std

    def normalize_torch(self, t):
        """Normalize a torch tensor with current running mean/std."""
        import torch
        m = torch.as_tensor(self.mean, dtype=t.dtype, device=t.device)
        s = torch.as_tensor(self.std, dtype=t.dtype, device=t.device)
        return (t - m) / s


class MultiAgentRolloutBuffer:
    """
    Rollout buffer for multi-agent environments.
    Stores per-agent trajectories and handles variable agent sets.
    """
    
    def __init__(
        self,
        agent_ids: List[str],
        obs_shapes: Dict[str, tuple],
        act_shapes: Dict[str, tuple],
        capacity: int,
        device: str = 'cpu'
    ):
        self.agent_ids = agent_ids
        self.capacity = capacity
        self.device = device
        self.position = 0
        
        # Per-agent buffers
        self.buffers = {}
        for aid in agent_ids:
            self.buffers[aid] = {
                'obs': torch.zeros((capacity,) + obs_shapes[aid], device=device),
                'actions': torch.zeros((capacity,) + act_shapes[aid], device=device),
                'rewards': torch.zeros(capacity, device=device),
                'dones': torch.zeros(capacity, device=device),
                'values': torch.zeros(capacity, device=device),
                'logprobs': torch.zeros(capacity, device=device),
            }
    
    def add(
        self,
        obs_dict: Dict[str, torch.Tensor],
        action_dict: Dict[str, torch.Tensor],
        reward_dict: Dict[str, float],
        done_dict: Dict[str, bool],
        value_dict: Dict[str, torch.Tensor],
        logprob_dict: Dict[str, torch.Tensor]
    ) -> bool:
        """Add transition for all agents."""
        if self.position >= self.capacity:
            return False
        
        for aid in self.agent_ids:
            if aid in obs_dict:
                self.buffers[aid]['obs'][self.position] = obs_dict[aid]
                self.buffers[aid]['actions'][self.position] = action_dict.get(aid, 0)
                self.buffers[aid]['rewards'][self.position] = reward_dict.get(aid, 0)
                self.buffers[aid]['dones'][self.position] = float(done_dict.get(aid, False))
                self.buffers[aid]['values'][self.position] = value_dict.get(aid, 0)
                self.buffers[aid]['logprobs'][self.position] = logprob_dict.get(aid, 0)
        
        self.position += 1
        return True
    
    def get(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Get all data and reset."""
        data = {}
        for aid, buffer in self.buffers.items():
            data[aid] = {
                k: v[:self.position] for k, v in buffer.items()
            }
        self.position = 0
        return data
    
    def compute_returns(self, gamma: float = 0.99) -> Dict[str, torch.Tensor]:
        """Compute discounted returns per agent."""
        returns = {}
        for aid, buffer in self.buffers.items():
            rewards = buffer['rewards'][:self.position]
            dones = buffer['dones'][:self.position]
            
            ret = torch.zeros_like(rewards)
            running_return = 0
            for t in reversed(range(len(rewards))):
                if dones[t]:
                    running_return = 0
                running_return = rewards[t] + gamma * running_return
                ret[t] = running_return
            
            returns[aid] = ret
        return returns


class CentralizedCritic(nn.Module):
    """
    Centralized critic for MARL that observes all agents' states.
    """
    
    def __init__(
        self,
        agent_obs_dims: Dict[str, int],
        hidden_dim: int = 256,
        use_layer_norm: bool = False
    ):
        super().__init__()
        
        # Total observation dimension
        self.agent_ids = sorted(agent_obs_dims.keys())
        total_obs_dim = sum(agent_obs_dims.values())
        
        # Value network
        layers = [
            nn.Linear(total_obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ]
        self.value_net = nn.Sequential(*layers)
        
    def forward(self, obs_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass with observations from all agents.
        
        Parameters
        ----------
        obs_dict : Dict[str, torch.Tensor]
            Observations for each agent
            
        Returns
        -------
        torch.Tensor
            Centralized value estimate
        """
        # Concatenate observations in consistent order
        obs_list = []
        for aid in self.agent_ids:
            if aid in obs_dict:
                obs_list.append(obs_dict[aid])
        
        if len(obs_list) == 0:
            return torch.tensor(0.0)
        
        central_obs = torch.cat(obs_list, dim=-1)
        return self.value_net(central_obs)


class DecentralizedActorCritic(nn.Module):
    """
    Decentralized actor-critic where each agent has its own network.
    """
    
    def __init__(
        self,
        agent_configs: Dict[str, Dict[str, Any]],
        share_parameters: bool = False
    ):
        """
        Parameters
        ----------
        agent_configs : Dict[str, Dict[str, Any]]
            Configuration for each agent: {'agent_id': {'obs_dim': ..., 'act_dim': ...}}
        share_parameters : bool
            Whether agents share network parameters
        """
        super().__init__()
        
        self.agent_ids = sorted(agent_configs.keys())
        self.share_parameters = share_parameters
        
        if share_parameters:
            # All agents use the same network
            first_config = next(iter(agent_configs.values()))
            self.shared_network = self._build_agent_network(
                first_config['obs_dim'],
                first_config['act_dim']
            )
        else:
            # Each agent has its own network
            self.agent_networks = nn.ModuleDict({
                aid: self._build_agent_network(cfg['obs_dim'], cfg['act_dim'])
                for aid, cfg in agent_configs.items()
            })
    
    def _build_agent_network(self, obs_dim: int, act_dim: int) -> nn.Module:
        """Build a single agent's actor-critic network."""
        return nn.ModuleDict({
            'actor': nn.Sequential(
                nn.Linear(obs_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, act_dim),
                nn.Tanh()
            ),
            'critic': nn.Sequential(
                nn.Linear(obs_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )
        })
    
    def forward(
        self,
        obs_dict: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        Forward pass for all agents.
        
        Returns
        -------
        actions, values : Tuple[Dict, Dict]
            Actions and values for each agent
        """
        actions = {}
        values = {}
        
        for aid in self.agent_ids:
            if aid not in obs_dict:
                continue
            
            if self.share_parameters:
                network = self.shared_network
            else:
                network = self.agent_networks[aid]
            
            actions[aid] = network['actor'](obs_dict[aid])
            values[aid] = network['critic'](obs_dict[aid])
        
        return actions, values


class CommNetwork(nn.Module):
    """
    Communication network for agent coordination.
    Implements various communication protocols (broadcast, targeted, graph-based).
    """
    
    def __init__(
        self,
        agent_ids: List[str],
        message_dim: int = 32,
        comm_type: str = 'broadcast',  # 'broadcast', 'targeted', 'graph'
        hidden_dim: int = 64
    ):
        super().__init__()
        
        self.agent_ids = agent_ids
        self.n_agents = len(agent_ids)
        self.message_dim = message_dim
        self.comm_type = comm_type
        
        # Message generation
        self.message_encoder = nn.Linear(hidden_dim, message_dim)
        
        if comm_type == 'broadcast':
            # All agents receive aggregated message
            self.message_aggregator = nn.Linear(message_dim, message_dim)
            
        elif comm_type == 'targeted':
            # Agents can send targeted messages
            self.attention = nn.MultiheadAttention(
                embed_dim=message_dim,
                num_heads=4,
                batch_first=True
            )
            
        elif comm_type == 'graph':
            # Graph neural network style communication
            self.edge_network = nn.Sequential(
                nn.Linear(message_dim * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, message_dim)
            )
    
    def forward(
        self,
        hidden_states: Dict[str, torch.Tensor],
        adjacency: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Perform communication between agents.
        
        Parameters
        ----------
        hidden_states : Dict[str, torch.Tensor]
            Hidden states from each agent
        adjacency : Optional[torch.Tensor]
            Adjacency matrix for graph-based communication
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Messages for each agent
        """
        # Generate messages
        messages = {
            aid: self.message_encoder(h) 
            for aid, h in hidden_states.items()
        }
        
        if self.comm_type == 'broadcast':
            # Average all messages
            all_messages = torch.stack(list(messages.values()))
            aggregated = self.message_aggregator(all_messages.mean(dim=0))
            return {aid: aggregated for aid in self.agent_ids}
            
        elif self.comm_type == 'targeted':
            # Use attention to select relevant messages
            message_stack = torch.stack(list(messages.values())).unsqueeze(0)
            attended, _ = self.attention(
                message_stack, message_stack, message_stack
            )
            return {
                aid: attended[0, i] 
                for i, aid in enumerate(self.agent_ids)
            }
            
        elif self.comm_type == 'graph':
            # Graph-based message passing
            if adjacency is None:
                # Fully connected by default
                adjacency = torch.ones(self.n_agents, self.n_agents)
            
            updated_messages = {}
            for i, aid in enumerate(self.agent_ids):
                neighbor_msgs = []
                for j, other_aid in enumerate(self.agent_ids):
                    if adjacency[i, j] > 0:
                        edge_input = torch.cat([
                            messages[aid], messages[other_aid]
                        ])
                        neighbor_msgs.append(self.edge_network(edge_input))
                
                if neighbor_msgs:
                    updated_messages[aid] = torch.stack(neighbor_msgs).mean(dim=0)
                else:
                    updated_messages[aid] = messages[aid]
            
            return updated_messages
        
        else:
            return messages


def create_marl_policy_wrapper(
    agent_networks: Dict[str, Any],
    deterministic: bool = True,
    device: str = 'cpu'
) -> Callable[[Dict[str, np.ndarray]], Dict[str, np.ndarray]]:
    """
    Create a policy function for multi-agent evaluation.
    
    Parameters
    ----------
    agent_networks : Dict[str, Any]
        Mapping from agent_id to network/policy
    deterministic : bool
        Whether to use deterministic actions
    device : str
        Device for computation
        
    Returns
    -------
    Callable
        Policy function mapping obs_dict -> action_dict
    """
    def policy_fn(obs_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        action_dict = {}
        
        for aid, obs in obs_dict.items():
            if aid not in agent_networks:
                # Default action for unknown agents
                action_dict[aid] = np.zeros(1)
                continue
            
            # Convert to tensor
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            
            # Get action from network
            network = agent_networks[aid]
            with torch.no_grad():
                if hasattr(network, 'get_action'):
                    action = network.get_action(obs_tensor, deterministic=deterministic)
                elif hasattr(network, 'act'):
                    action = network.act(obs_tensor, deterministic=deterministic)
                elif hasattr(network, 'forward'):
                    action = network(obs_tensor)
                    if deterministic and hasattr(action, 'mean'):
                        action = action.mean
                else:
                    # Assume network is a callable
                    action = network(obs_tensor)
            
            # Convert to numpy
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy()
            
            action_dict[aid] = action.squeeze()
        
        return action_dict
    
    return policy_fn


class IndependentPPOTrainer:
    """
    Independent PPO training for multi-agent settings.
    Each agent trains its own policy independently.
    """
    
    def __init__(
        self,
        env_factory: Callable,
        agent_configs: Dict[str, Dict[str, Any]],
        shared_params: bool = False,
        **ppo_kwargs
    ):
        """
        Parameters
        ----------
        env_factory : Callable
            Factory to create environment
        agent_configs : Dict[str, Dict[str, Any]]
            Configuration for each agent
        shared_params : bool
            Whether agents share parameters
        **ppo_kwargs
            PPO hyperparameters
        """
        self.env = env_factory()
        self.agent_ids = list(agent_configs.keys())
        
        # Create actor-critics
        self.ac = DecentralizedActorCritic(agent_configs, shared_params)
        
        # Create buffers
        self.buffer = MultiAgentRolloutBuffer(
            agent_ids=self.agent_ids,
            obs_shapes={aid: tuple(cfg['obs_shape']) for aid, cfg in agent_configs.items()},
            act_shapes={aid: tuple(cfg['act_shape']) for aid, cfg in agent_configs.items()},
            capacity=ppo_kwargs.get('buffer_size', 2048),
            device=ppo_kwargs.get('device', 'cpu')
        )
        
        # PPO parameters
        self.gamma = ppo_kwargs.get('gamma', 0.99)
        self.gae_lambda = ppo_kwargs.get('gae_lambda', 0.95)
        self.clip_epsilon = ppo_kwargs.get('clip_epsilon', 0.2)
        self.n_epochs = ppo_kwargs.get('n_epochs', 10)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.ac.parameters(),
            lr=ppo_kwargs.get('lr', 3e-4)
        )
    
    def train_step(self):
        """Single training iteration."""
        # Collect rollouts
        self.collect_rollouts()
        
        # Compute returns
        returns = self.buffer.compute_returns(self.gamma)
        
        # Update each agent
        data = self.buffer.get()
        
        for epoch in range(self.n_epochs):
            for aid in self.agent_ids:
                agent_data = data[aid]
                agent_returns = returns[aid]
                
                # Compute advantages
                advantages = agent_returns - agent_data['values']
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # PPO update (simplified)
                # ... standard PPO loss computation
        
        return {}  # Return training statistics
    
    def collect_rollouts(self):
        """Collect experience from environment."""
        # Implementation depends on specific environment interface
        pass
