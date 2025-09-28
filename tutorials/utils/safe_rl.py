"""
Safe RL utilities for Constrained MDPs - infrastructure and boilerplate.
Extracts repetitive components while keeping algorithmic logic in notebooks.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .notebook import log_info
from .rl import RolloutBuffer, RunningMeanStd, explained_variance, compute_log_prob_with_squashing


class SafeActorCritic(nn.Module):
    """
    CMDP-aware actor-critic with dual value functions (reward and cost).
{{ ... }}
    Matches Tutorial 04's architecture signatures for compatibility.
    """
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256, normalize_obs: bool = True):
        super().__init__()
        self.normalize_obs = normalize_obs
        if normalize_obs:
            try:
                self.obs_rms = RunningMeanStd(shape=(obs_dim,))
            except Exception:
                self.normalize_obs = False
                self.obs_rms = None  # type: ignore

        # Shared feature extractor
        self.features = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        )
        self.actor_mean = nn.Linear(hidden_dim, act_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.cost_head = nn.Linear(hidden_dim, 1)

        self.actor_logstd = nn.Parameter(torch.zeros(act_dim))

    def _maybe_normalize_obs(self, obs: torch.Tensor, update_rms: bool = True) -> torch.Tensor:
        if not self.normalize_obs or getattr(self, 'obs_rms', None) is None:
            return obs
        obs_np = obs.detach().cpu().numpy()
        if update_rms:
            self.obs_rms.update(obs_np)
        normed = self.obs_rms.normalize(obs_np)
        return torch.as_tensor(normed, dtype=obs.dtype, device=obs.device)

    def forward(self, obs: torch.Tensor, update_rms: bool = True):
        obs = self._maybe_normalize_obs(obs, update_rms=update_rms)
        h = self.features(obs)
        # Produce raw (unsquashed) mean; squashing handled when sampling/actions are returned
        mean_raw = self.actor_mean(h)
        std = torch.exp(self.actor_logstd)
        dist = torch.distributions.Normal(mean_raw, std)
        value = self.value_head(h)
        cost_value = self.cost_head(h)
        return dist, value, cost_value

    @torch.no_grad()
    def get_action(self, obs: torch.Tensor, deterministic: bool = False):
        # Accept numpy arrays and add batch dimension if needed
        device = self.actor_logstd.device
        if not torch.is_tensor(obs):
            obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
        else:
            obs = obs.to(device)
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)

        dist, value, cost_value = self.forward(obs, update_rms=False)
        # Sample in raw space, then tanh-squash and map to [0,1]
        if deterministic:
            z = dist.mean
        else:
            z = dist.sample()
        action_tanh = torch.tanh(z)
        action01 = (action_tanh + 1.0) * 0.5
        # Compute log-prob with tanh Jacobian correction (constants from linear scaling cancel in PPO ratios)
        log_prob = compute_log_prob_with_squashing(dist, action_tanh)
        return action01, log_prob, value, cost_value

    def evaluate_actions(self, obs: torch.Tensor, actions: torch.Tensor):
        """
        Evaluate given actions under the current policy.

        Returns
        -------
        log_probs : torch.Tensor
            Log probabilities of the actions under the policy.
        entropy : torch.Tensor
            Entropy of the action distribution (per-sample).
        values : torch.Tensor
            Reward value estimates.
        cost_values : torch.Tensor
            Cost value estimates.
        """
        dist, values, cost_values = self.forward(obs, update_rms=False)
        # Actions are stored in buffer in [0,1]; convert to tanh-space [-1,1] for corrected log-prob
        action_tanh = actions * 2.0 - 1.0
        log_probs = compute_log_prob_with_squashing(dist, action_tanh)
        # For Normal distributions, entropy is per-dimension; sum to get total entropy per sample
        entropy = dist.entropy().sum(-1)
        return log_probs, entropy, values, cost_values


class SafeRolloutBuffer(RolloutBuffer):
    """Extended rollout buffer that tracks costs for safe RL."""
    
    def __init__(self, obs_shape, act_shape, capacity: int, device='cpu'):
        super().__init__(obs_shape, act_shape, capacity, device)
        
        # Additional storage for safety
        # Reward-side targets (not present in base RolloutBuffer)
        self.returns = torch.zeros(capacity, device=device)
        self.advantages = torch.zeros(capacity, device=device)
        # Cost-side targets
        self.costs = torch.zeros(capacity, device=device)
        self.cost_values = torch.zeros(capacity, device=device)
        self.cost_returns = torch.zeros(capacity, device=device)
        self.cost_advantages = torch.zeros(capacity, device=device)
        
    def reset(self):
        """Reset buffer write position (does not zero out storage tensors)."""
        self.position = 0
        
    def add_safe(self, obs, action, reward, cost, value, cost_value, log_prob, done):
        """Add transition with cost information."""
        if not self.add(obs, action, log_prob, reward, done, value):
            return False
        
        # Add cost-specific data
        pos = self.position - 1  # Position was already incremented
        self.costs[pos] = cost
        self.cost_values[pos] = cost_value.squeeze() if hasattr(cost_value, 'squeeze') else cost_value
        return True
    
    def compute_returns_and_advantages(self, gamma: float = 0.99, gae_lambda: float = 0.95):
        """Compute returns and GAE for both rewards and costs."""
        n = self.position
        
        # Compute for rewards
        self.returns[:n] = self._compute_returns(
            self.rewards[:n], self.values[:n], self.dones[:n], gamma
        )
        self.advantages[:n] = self._compute_gae(
            self.rewards[:n], self.values[:n], self.dones[:n], gamma, gae_lambda
        )
        
        # Compute for costs
        self.cost_returns[:n] = self._compute_returns(
            self.costs[:n], self.cost_values[:n], self.dones[:n], gamma
        )
        self.cost_advantages[:n] = self._compute_gae(
            self.costs[:n], self.cost_values[:n], self.dones[:n], gamma, gae_lambda
        )
    
    def _compute_returns(self, rewards, values, dones, gamma):
        """Compute discounted returns."""
        returns = torch.zeros_like(rewards)
        running_return = 0
        
        for t in reversed(range(len(rewards))):
            if dones[t]:
                running_return = 0
            running_return = rewards[t] + gamma * running_return
            returns[t] = running_return
            
        return returns
    
    def _compute_gae(self, rewards, values, dones, gamma, gae_lambda):
        """Compute Generalized Advantage Estimation."""
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            if dones[t]:
                next_value = 0
                last_advantage = 0
                
            delta = rewards[t] + gamma * next_value - values[t]
            last_advantage = delta + gamma * gae_lambda * last_advantage
            advantages[t] = last_advantage
            
        return advantages
    
    def get_safe(self) -> Dict[str, torch.Tensor]:
        """Get all data including safety information."""
        # Create data dictionary first (safe from exceptions)
        n = int(self.position)
        
        if n == 0:
            # Return empty tensors with correct shapes
            return {
                'obs': self.obs[:0],
                'actions': self.actions[:0],
                'logprobs': self.logprobs[:0],
                'rewards': self.rewards[:0],
                'dones': self.dones[:0],
                'values': self.values[:0],
                'returns': self.returns[:0],
                'advantages': self.advantages[:0],
                'costs': self.costs[:0],
                'cost_values': self.cost_values[:0],
                'cost_returns': self.cost_returns[:0],
                'cost_advantages': self.cost_advantages[:0],
            }
        
        # Build data dictionary
        try:
            data: Dict[str, torch.Tensor] = {
                'obs': self.obs[:n].clone(),
                'actions': self.actions[:n].clone(),
                'logprobs': self.logprobs[:n].clone(),
                'rewards': self.rewards[:n].clone(),
                'dones': self.dones[:n].clone(),
                'values': self.values[:n].clone(),
                'returns': self.returns[:n].clone(),
                'advantages': self.advantages[:n].clone(),
                'costs': self.costs[:n].clone(),
                'cost_values': self.cost_values[:n].clone(),
                'cost_returns': self.cost_returns[:n].clone(),
                'cost_advantages': self.cost_advantages[:n].clone(),
            }
            # Only reset if successful
            self.position = 0
            return data
        except Exception as e:
            # Don't reset on error
            raise RuntimeError(f"Failed to get buffer data: {e}")


class LagrangianMultiplier:
    """Adaptive Lagrangian multiplier with multiple update strategies."""
    
    def __init__(
        self,
        cost_limit: float,
        variant: str = 'standard',  # 'standard', 'pid', 'adaptive'
        init_value: float = 0.0,
        lr: float = 0.035,
        upper_bound: Optional[float] = None,
        # PID parameters
        kp: float = 0.1,
        ki: float = 0.01,
        kd: float = 0.01,
        d_delay: int = 10
    ):
        self.cost_limit = cost_limit
        self.variant = variant
        self.lr = lr
        self.upper_bound = upper_bound
        
        if variant == 'standard':
            self._lambda = nn.Parameter(
                torch.tensor(max(init_value, 0.0), dtype=torch.float32),
                requires_grad=True
            )
            self.optimizer = optim.Adam([self._lambda], lr=lr)
            
        elif variant == 'pid':
            self.kp = kp
            self.ki = ki
            self.kd = kd
            self.integral = 0.0
            self.prev_error = 0.0
            self.cost_history = deque(maxlen=d_delay)
            self.cost_history.append(0.0)
            self._lambda_value = 0.0
            
        elif variant == 'adaptive':
            self._lambda_value = init_value
            self.cost_history = deque(maxlen=20)
            self.lr_scale = 1.0
            
        else:
            raise ValueError(f"Unknown Lagrangian variant: {variant}")
    
    @property
    def value(self) -> float:
        """Current λ value (always non-negative)."""
        if self.variant == 'standard':
            return float(F.relu(self._lambda).detach().item())
        else:
            return float(self._lambda_value)
    
    def update(self, mean_cost: float) -> Dict[str, float]:
        """Update λ based on constraint violation."""
        if self.variant == 'standard':
            return self._update_standard(mean_cost)
        elif self.variant == 'pid':
            return self._update_pid(mean_cost)
        elif self.variant == 'adaptive':
            return self._update_adaptive(mean_cost)
    
    def _update_standard(self, mean_cost: float) -> Dict[str, float]:
        """Standard gradient-based update."""
        self.optimizer.zero_grad()
        loss = -self._lambda * (mean_cost - self.cost_limit)
        loss.backward()
        self.optimizer.step()
        
        # Enforce bounds
        with torch.no_grad():
            if self.upper_bound is not None:
                self._lambda.clamp_(0, self.upper_bound)
            else:
                self._lambda.clamp_(0)
        
        return {
            'lambda': self.value,
            'lambda_loss': loss.item(),
            'constraint_violation': mean_cost - self.cost_limit
        }
    
    def _update_pid(self, mean_cost: float) -> Dict[str, float]:
        """PID controller update."""
        error = mean_cost - self.cost_limit
        
        # P term
        p_term = self.kp * error
        
        # I term
        self.integral += error * self.ki
        self.integral = max(0, self.integral)
        
        # D term
        if len(self.cost_history) > 1:
            d_term = self.kd * (mean_cost - self.cost_history[0])
        else:
            d_term = 0
        
        # Combine PID
        pid_output = p_term + self.integral + d_term
        self._lambda_value = np.clip(pid_output, 0, self.upper_bound or 100.0)
        
        # Update history
        self.cost_history.append(mean_cost)
        self.prev_error = error
        
        return {
            'lambda': self._lambda_value,
            'p_term': p_term,
            'i_term': self.integral,
            'd_term': d_term,
            'constraint_violation': error
        }
    
    def _update_adaptive(self, mean_cost: float) -> Dict[str, float]:
        """Adaptive learning rate based on violation history."""
        violation = mean_cost - self.cost_limit
        self.cost_history.append(mean_cost)
        
        # Adapt learning rate based on violation consistency
        if len(self.cost_history) >= 5:
            recent_violations = [c - self.cost_limit for c in list(self.cost_history)[-5:]]
            if all(v > 0 for v in recent_violations):
                self.lr_scale = min(2.0, self.lr_scale * 1.1)  # Increase aggressiveness
            elif all(v < 0 for v in recent_violations):
                self.lr_scale = max(0.5, self.lr_scale * 0.9)  # Decrease aggressiveness
        
        # Update lambda
        effective_lr = self.lr * self.lr_scale
        self._lambda_value += effective_lr * violation
        self._lambda_value = np.clip(self._lambda_value, 0, self.upper_bound or 100.0)
        
        return {
            'lambda': self._lambda_value,
            'lr_scale': self.lr_scale,
            'effective_lr': effective_lr,
            'constraint_violation': violation
        }


class BaseSafeRLTrainer(ABC):
    """Base trainer class with common infrastructure for safe RL algorithms."""
    
    def __init__(
        self,
        env_factory: Callable,
        actor_critic: nn.Module,
        cost_limit: float,
        experiment_tag: str = "safe_rl",
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_epochs: int = 10,
        batch_size: int = 64,
        buffer_size: int = 2048,
        device: str = 'cpu',
        track_metrics: Optional[list[str]] = None
    ):
        self.env_factory = env_factory
        self.env = env_factory()
        self.actor_critic = actor_critic.to(device)
        self.cost_limit = float(cost_limit)
        self.experiment_tag = experiment_tag
        self.device = device
        
        # Hyperparameters
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        
        # Optimizer
        self.optimizer = optim.Adam(actor_critic.parameters(), lr=lr)
        
        # Buffer
        obs_dim = self.env.observation_space.shape[0]
        act_dim = self.env.action_space.shape[0]
        self.buffer = SafeRolloutBuffer(
            (obs_dim,), (act_dim,), buffer_size, device
        )
        
        # Tracking
        if track_metrics is None:
            track_metrics = ['mean_cost', 'constraint_violation']
        
        # Lazy import to avoid circular dependency with tutorials.utils.ev
        from .ev import create_training_logger  # type: ignore
        self.logger = create_training_logger(
            experiment_tag,
            metrics=['mean_cost', 'constraint_violation'] + track_metrics
        )
        
        # Episode tracking
        self.episode_rewards = deque(maxlen=100)
        self.episode_costs = deque(maxlen=100)
        self.episode_lengths = deque(maxlen=100)
        self.total_steps = 0
        
    def collect_rollout(self, n_steps: int):
        """Collect experience using the environment - standard across all algorithms."""
        obs, _ = self.env.reset()
        episode_reward = 0
        episode_cost = 0
        episode_length = 0
        # Track last cumulative excess charge to derive per-step cost when needed
        last_cum_cost = 0.0
        
        for _ in range(n_steps):
            # Get action
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            
            with torch.no_grad():
                action, log_prob, value, cost_value = self.get_action(obs_tensor)
            
            # Step environment
            action_np = action.cpu().numpy().squeeze()
            next_obs, reward, terminated, truncated, info = self.env.step(action_np)
            done = terminated or truncated
            
            # Extract cost (handle different info structures)
            if 'cost' in info:
                cost = info['cost']
            elif 'reward_breakdown' in info and isinstance(info['reward_breakdown'], dict):
                # Convention: environment may expose cumulative 'excess_charge' in reward_breakdown.
                # We difference it to obtain the per-step cost to train Safe RL correctly.
                cum = float(info['reward_breakdown'].get('excess_charge', 0.0))
                cost = max(0.0, cum - last_cum_cost)
                last_cum_cost = cum
            else:
                cost = 0.0
            
            # Store transition
            self.buffer.add_safe(
                obs_tensor.squeeze(),
                action.squeeze(),
                reward,
                cost,
                value,
                cost_value,
                log_prob,
                done
            )
            
            # Update episode stats
            episode_reward += reward
            episode_cost += cost
            episode_length += 1
            self.total_steps += 1
            
            # Handle episode end
            if done:
                self.episode_rewards.append(episode_reward)
                self.episode_costs.append(episode_cost)
                self.episode_lengths.append(episode_length)
                
                obs, _ = self.env.reset()
                episode_reward = 0
                episode_cost = 0
                episode_length = 0
                last_cum_cost = 0.0
            else:
                obs = next_obs
                
        # Compute returns and advantages
        self.buffer.compute_returns_and_advantages(self.gamma, self.gae_lambda)
    
    def get_action(self, obs_tensor):
        """Get action from actor-critic - can be overridden if needed."""
        if hasattr(self.actor_critic, 'get_action'):
            return self.actor_critic.get_action(obs_tensor)
        elif hasattr(self.actor_critic, 'act'):
            return self.actor_critic.act(obs_tensor)
        else:
            # Default implementation
            dist, value, cost_value = self.actor_critic(obs_tensor)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(-1)
            return action, log_prob, value, cost_value
    
    @abstractmethod
    def update(self) -> Dict[str, float]:
        """Update policy - algorithm-specific, must be implemented in subclass."""
        pass
    
    def _ppo_update_step(self, batch_data: Dict[str, torch.Tensor], advantages: torch.Tensor) -> Dict[str, float]:
        """Shared PPO update logic for all safe RL trainers.
        
        Parameters
        ----------
        batch_data : Dict[str, torch.Tensor]
            A minibatch with keys ['obs','actions','logprobs','returns','cost_returns']
        advantages : torch.Tensor
            The precomputed advantages to use for policy gradient (e.g., combined or mode-specific)
        
        Returns
        -------
        Dict[str, float]
            Dictionary of losses and entropy for logging
        """
        obs_batch = batch_data['obs']
        actions_batch = batch_data['actions']
        old_log_probs = batch_data['logprobs']
        returns_batch = batch_data['returns']
        cost_returns_batch = batch_data['cost_returns']

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Evaluate actions
        log_probs, entropy, values, cost_values = self.actor_critic.evaluate_actions(
            obs_batch, actions_batch
        )

        # PPO loss
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        pg_loss = -torch.min(surr1, surr2).mean()

        # Value losses
        value_loss = F.mse_loss(values.squeeze(), returns_batch)
        cost_value_loss = F.mse_loss(cost_values.squeeze(), cost_returns_batch)

        # Total loss
        loss = (
            pg_loss +
            self.value_coef * (value_loss + cost_value_loss) -
            self.entropy_coef * entropy.mean()
        )

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return {
            'pg_loss': float(pg_loss.item()),
            'value_loss': float(value_loss.item()),
            'cost_value_loss': float(cost_value_loss.item()),
            'entropy': float(entropy.mean().item()),
        }

    def log_iteration(self, stats: Dict[str, float]):
        """Standardized logging across all algorithms."""
        for key, value in stats.items():
            if key in self.logger:
                self.logger[key].append(value)
    
    def checkpoint_if_needed(self, iteration: int, checkpoint_interval: int = 10):
        """Save model checkpoint if needed."""
        if checkpoint_interval > 0 and iteration > 0 and (iteration % checkpoint_interval == 0):
            from .ev import save_model_checkpoint  # type: ignore
            save_model_checkpoint(self.actor_critic, self.experiment_tag, iteration, kind='safe_rl')
    
    def train(self, total_steps: int, rollout_length: int = 2048, 
              log_interval: int = 5, checkpoint_interval: int = 10) -> pd.DataFrame:
        """Common training loop - can be overridden if needed."""
        n_iterations = total_steps // rollout_length
        
        for iteration in range(n_iterations):
            # Collect rollout
            self.collect_rollout(rollout_length)
            
            # Update policy (algorithm-specific)
            stats = self.update()
            
            # Log stats
            self.log_iteration(stats)
            
            # Periodic logging
            if iteration % log_interval == 0:
                self.print_progress(iteration, n_iterations, stats)
            
            # Checkpointing
            self.checkpoint_if_needed(iteration, checkpoint_interval)
        
        # Save final results: ensure all logger lists have equal length
        max_len = max((len(v) for v in self.logger.values()), default=0)
        if max_len > 0:
            for k, v in list(self.logger.items()):
                if len(v) < max_len:
                    # pad with NaNs to align lengths
                    pad_count = max_len - len(v)
                    self.logger[k] = v + [float('nan')] * pad_count
        results_df = pd.DataFrame(self.logger)
        from .ev import save_timeseries, save_dual_log  # type: ignore
        save_timeseries(self.experiment_tag, results_df, kind='safe_rl')
        
        # Save dual dynamics if applicable
        if 'lambda' in results_df.columns:
            dual_df = results_df[['timestep', 'lambda', 'mean_cost', 'constraint_violation']].copy()
            dual_df['budget'] = self.cost_limit
            save_dual_log(self.experiment_tag, dual_df, kind='safe_rl')
        
        return results_df
    
    def print_progress(self, iteration: int, total_iterations: int, stats: Dict[str, float]):
        """Notebook-friendly training progress."""
        msg = (
            f"Iteration {iteration}/{total_iterations}\n"
            f"  • Mean Reward: {stats.get('reward', 0):.2f}\n"
            f"  • Mean Cost: {stats.get('mean_cost', 0):.2f} (limit: {self.cost_limit})\n"
            f"  • Lambda: {stats.get('lambda', float('nan')):.4f}\n"
            f"  • Constraint Violation: {stats.get('constraint_violation', 0):.2f}"
        )
        log_info(msg)
    
    def get_diagnostics(self) -> Dict[str, float]:
        """Return standardized diagnostics for the current policy."""
        data = self.buffer.get_safe()
        
        # Value function quality
        ev_reward = explained_variance(
            data['values'].cpu().numpy(),
            data['returns'].cpu().numpy()
        ) if (len(data.get('values', [])) > 0 and len(data.get('returns', [])) > 0) else 0.0
        
        ev_cost = explained_variance(
            data['cost_values'].cpu().numpy(),
            data['cost_returns'].cpu().numpy()
        ) if (len(data.get('cost_values', [])) > 0 and len(data.get('cost_returns', [])) > 0) else 0.0
        
        # Constraint satisfaction
        mean_cost = np.mean(self.episode_costs) if self.episode_costs else 0.0
        constraint_satisfaction_rate = np.mean([c <= self.cost_limit for c in self.episode_costs]) if self.episode_costs else 0.0
        
        return {
            'explained_variance_reward': ev_reward,
            'explained_variance_cost': ev_cost,
            'mean_episode_cost': mean_cost,
            'constraint_satisfaction_rate': constraint_satisfaction_rate,
            'episodes_collected': len(self.episode_rewards),
            'total_steps': self.total_steps
        }


def create_safe_policy_wrapper(
    actor_critic: nn.Module,
    deterministic: bool = True,
    device: str = 'cpu',
    normalize_obs: bool = False,
    obs_rms: Optional[RunningMeanStd] = None
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Create a standard policy wrapper for evaluation.
    Handles observation normalization and torch conversion.
    """
    def policy_fn(obs):
        # Handle observation normalization
        if normalize_obs and obs_rms is not None:
            obs = obs_rms.normalize(obs)
        
        # Convert to tensor
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        
        with torch.no_grad():
            if hasattr(actor_critic, 'get_action'):
                action, _, _, _ = actor_critic.get_action(obs_tensor, deterministic=deterministic)
            elif hasattr(actor_critic, 'act'):
                action, _, _, _ = actor_critic.act(obs_tensor, deterministic=deterministic)
            else:
                # Default: assume forward returns (dist, value, cost_value)
                dist, _, _ = actor_critic(obs_tensor, update_rms=False)
                if deterministic:
                    action = dist.mean
                else:
                    action = dist.sample()
        
        return action.cpu().numpy().squeeze()
    
    return policy_fn


class SafetyTrackingWrapper:
    """Lightweight safety-cost wrapper that adds per-step cost to info.

    - Computes per-step `info['cost']` by differencing the cumulative
      `info['reward_breakdown']['excess_charge']` emitted by the base env.
    - Tracks `info['episode_cost']` and optionally `info['is_safe']` if cost_limit is provided.
    - Minimal dependency: avoids gym; just forwards step/reset.
    """
    def __init__(self, env: Any, cost_limit: Optional[float] = None):
        self.env = env
        self.cost_limit = cost_limit
        self._last_cum_cost = 0.0
        self.episode_cost = 0.0

    def reset(self, **kwargs):
        self._last_cum_cost = 0.0
        self.episode_cost = 0.0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        rb = info.get('reward_breakdown', {}) if isinstance(info, dict) else {}
        cum = float(rb.get('excess_charge', 0.0))
        step_cost = max(0.0, cum - self._last_cum_cost)
        self._last_cum_cost = cum
        self.episode_cost += float(step_cost)
        # Attach cost signals
        if isinstance(info, dict):
            info['cost'] = float(step_cost)
            info['episode_cost'] = float(self.episode_cost)
            if self.cost_limit is not None:
                info['cost_limit'] = float(self.cost_limit)
                info['is_safe'] = bool(self.episode_cost <= self.cost_limit)
        return obs, reward, terminated, truncated, info

    def __getattr__(self, name):
        return getattr(self.env, name)


def create_safe_ev_env(
    site: str = 'caltech',
    date_range: Tuple[str, str] = ('2019-05-01', '2019-05-31'),
    seed: int = 42,
    flatten: bool = True,
    noise: Optional[float] = None,
    noise_action: Optional[float] = None,
    dense_mode: bool = False,
    density_multiplier: float = 3.0,
    violation_weight: Optional[float] = None,
    cost_limit: Optional[float] = None,
    data_generator: Any = None,
    **env_kwargs,
):
    """Factory for CMDP-ready EV env with per-step safety cost tracking.

    - Reward remains profit - carbon.
    - Per-step constraint cost is returned in `info['cost']`.
    - Use `violation_weight` to upweight violation penalties in the underlying env.
    """
    # Local import to avoid circular initialization during package import
    from .common import create_ev_env  # type: ignore

    base = create_ev_env(
        site=site,
        date_range=date_range,
        seed=seed,
        safe_rl=False,
        flatten=flatten,
        noise=noise,
        noise_action=noise_action,
        dense_mode=dense_mode,
        density_multiplier=density_multiplier,
        violation_weight=violation_weight,
        data_generator=data_generator,
        **env_kwargs,
    )
    return SafetyTrackingWrapper(base, cost_limit=cost_limit)


def pretrain_cost_value_function(
    actor_critic: nn.Module,
    env_factory: Callable[[], Any],
    warmup_steps: int = 5000,
    batch_size: int = 64,
    gamma: float = 0.99,
    device: Optional[str] = None,
) -> Tuple[list[float], float]:
    """
    Pre-train the cost value function V_c by supervised regression on Monte Carlo
    cost returns generated from a random policy.

    Parameters
    ----------
    actor_critic : nn.Module
        Safe actor-critic model that returns (dist, value, cost_value) from forward().
        If it exposes a 'cost_head' attribute, only that head is optimized.
    env_factory : Callable[[], Env]
        Zero-arg factory that returns a fresh environment with SafetyTrackingWrapper,
        so that info['cost'] contains per-step safety costs.
    warmup_steps : int
        Target number of transitions to collect using a random policy for fitting V_c.
    batch_size : int
        Minibatch size for regression.
    gamma : float
        Discount factor for computing Monte Carlo cost returns.
    device : Optional[str]
        Torch device; falls back to the model's parameter device if None.

    Returns
    -------
    (losses_per_epoch, explained_variance)
    """
    dev = device or next(actor_critic.parameters()).device

    # Optimize only the cost head if available; otherwise the whole network
    cost_head = getattr(actor_critic, 'cost_head', None)
    params = cost_head.parameters() if cost_head is not None else actor_critic.parameters()
    optimizer = optim.Adam(params, lr=1e-3)

    # Collect random trajectories to build a supervised dataset of (obs, cost_return)
    dataset: list[tuple[np.ndarray, float]] = []
    steps_collected = 0
    env = env_factory()
    try:
        while steps_collected < int(warmup_steps):
            obs, _ = env.reset()
            episode_obs: list[np.ndarray] = []
            episode_costs: list[float] = []
            done = False
            while not done:
                episode_obs.append(np.asarray(obs, dtype=np.float32))
                action = env.action_space.sample()
                obs, _r, term, trunc, info = env.step(action)
                episode_costs.append(float(info.get('cost', 0.0)))
                done = bool(term) or bool(trunc)
                steps_collected += 1
                if steps_collected >= warmup_steps:
                    break

            # Compute discounted returns for costs
            G = 0.0
            returns: list[float] = []
            for c in reversed(episode_costs):
                G = float(c) + float(gamma) * G
                returns.append(G)
            returns.reverse()

            for o, ret in zip(episode_obs[:len(returns)], returns):
                dataset.append((o, float(ret)))
    finally:
        try:
            env.close()
        except Exception:
            pass

    losses: list[float] = []
    if len(dataset) == 0:
        return losses, float('nan')

    # Training loop
    for _epoch in range(10):
        np.random.shuffle(dataset)
        epoch_losses: list[float] = []
        for i in range(0, len(dataset), int(batch_size)):
            batch = dataset[i:i + int(batch_size)]
            obs_batch = torch.as_tensor(
                np.stack([b[0] for b in batch]), dtype=torch.float32, device=dev
            )
            ret_batch = torch.as_tensor([b[1] for b in batch], dtype=torch.float32, device=dev)
            # Forward through model to get cost values
            _dist, _val, cost_values = actor_critic(obs_batch)
            loss = F.mse_loss(cost_values.squeeze(-1), ret_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))
        losses.append(float(np.mean(epoch_losses)) if epoch_losses else float('nan'))

    # Evaluate explained variance on a small held-out slice
    test = dataset[: min(200, len(dataset))]
    if len(test) > 0:
        with torch.no_grad():
            obs_batch = torch.as_tensor(
                np.stack([b[0] for b in test]), dtype=torch.float32, device=dev
            )
            _d, _v, preds = actor_critic(obs_batch)
            preds_np = preds.squeeze(-1).detach().cpu().numpy()
        ev = float(explained_variance(preds_np, np.array([b[1] for b in test], dtype=float)))
    else:
        ev = float('nan')

    return losses, ev
