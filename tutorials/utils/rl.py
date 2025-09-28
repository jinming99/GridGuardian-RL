"""
Minimal RL utilities for Tutorial 03 - infrastructure and boilerplate only.
Core RL concepts (PPO, GAE, actor-critic) remain in the notebook for education.
"""

from __future__ import annotations

import random
from typing import Optional, Callable, Any, Dict, List

import gymnasium as gym
import numpy as np
from gymnasium import spaces
try:
    from gymnasium.wrappers import RecordEpisodeStatistics  # type: ignore
except Exception:
    RecordEpisodeStatistics = None  # type: ignore
from collections import deque

# Only import torch if available
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


# ======== Action scaling utilities (generic: work with numpy or torch tensors) ========
def get_action_scale_and_bias(action_space):
    """
    Compute scaling parameters for continuous action spaces.
    Transforms from [-1, 1] (tanh output) to action space bounds.
    """
    if hasattr(action_space, 'low') and hasattr(action_space, 'high'):
        low = action_space.low
        high = action_space.high
    else:
        # Default for EV charging
        low = np.zeros(action_space.shape[0])
        high = np.ones(action_space.shape[0])

    scale = (high - low) / 2.0
    bias = (high + low) / 2.0
    return scale, bias


def scale_action(action_tanh, scale, bias):
    """Scale action from tanh output [-1,1] to environment bounds."""
    return action_tanh * scale + bias


def unscale_action(scaled_action, scale, bias):
    """Inverse scaling for log probability computation."""
    return (scaled_action - bias) / scale


class TrainingMonitor:
    """Lightweight training monitor for consistent logging."""

    def __init__(self, window_size: int = 100):
        self.episode_returns = deque(maxlen=window_size)
        self.episode_lengths = deque(maxlen=window_size)

    def log_episode(self, info: dict):
        """Extract and store episode stats from info dict."""
        for item in info.get("final_info", []):
            if item and "episode" in item:
                self.episode_returns.append(item["episode"]["r"])
                self.episode_lengths.append(item["episode"]["l"])

    def get_stats(self) -> dict:
        """Get current statistics."""
        stats: Dict[str, float] = {}
        if self.episode_returns:
            stats["mean_episode_return"] = float(np.mean(self.episode_returns))
            stats["std_episode_return"] = float(np.std(self.episode_returns))
        if self.episode_lengths:
            stats["mean_episode_length"] = float(np.mean(self.episode_lengths))
        return stats


class DictFlatteningWrapper(gym.Wrapper):
    """Flattens Dict observations into 1D Box - pure plumbing, not RL-specific."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self._dict_space = env.observation_space
        # Flatten to a single Box space
        self.observation_space = spaces.flatten_space(self._dict_space)
        self.action_space = env.action_space

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs_f = spaces.flatten(self._dict_space, obs).astype(np.float32)
        return obs_f, info

    def step(self, action):
        obs, r, terminated, truncated, info = self.env.step(action)
        obs_f = spaces.flatten(self._dict_space, obs).astype(np.float32)
        return obs_f, float(r), bool(terminated), bool(truncated), info


class RunningMeanStd:
    """
    Tracks running mean and std for observation normalization.
    Standard Welford's online algorithm - not RL-specific.
    """

    def __init__(self, shape=(), epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count) -> None:
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        self.mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta ** 2 * self.count * batch_count / total_count

        self.var = M2 / total_count
        self.count = total_count

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / np.sqrt(self.var + 1e-8)


def build_central_state(
    obs_dict: Dict[str, np.ndarray], agent_order: Optional[List[str]] = None
) -> np.ndarray:
    """
    Moved to tutorials.utils_marl.build_central_state.
    Import from tutorials.utils_marl instead of tutorials.utils_rl.
    """
    raise ImportError(
        "build_central_state has moved to tutorials.utils_marl. "
        "Please import from tutorials.utils_marl instead."
    )


class ValueNormalizer:
    """
    Moved to tutorials.utils_marl.ValueNormalizer.
    Import from tutorials.utils_marl instead of tutorials.utils_rl.
    """

    def __init__(self, *args, **kwargs):
        raise ImportError(
            "ValueNormalizer has moved to tutorials.utils_marl. "
            "Please import from tutorials.utils_marl instead."
        )


if TORCH_AVAILABLE:
    def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0):
        """Standard orthogonal initialization for neural network layers."""
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """
        Compute fraction of variance explained by predictions.
        Standard metric for value function quality - not RL-specific math.
        """
        var_y = np.var(y_true)
        return float(np.nan) if var_y == 0 else float(1 - np.var(y_true - y_pred) / var_y)

    def compute_log_prob_with_squashing(dist, action_tanh):
        """
        Compute log probability with Jacobian correction for tanh squashing.
        Expects:
          - dist: torch Normal distribution over raw (pre-tanh) actions
          - action_tanh: actions in [-1, 1] after tanh
        Returns: log_prob per sample, shape [B]
        """
        eps = 1e-6
        # Recover raw action before tanh in a numerically stable way
        action_tanh = torch.clamp(action_tanh, -1 + eps, 1 - eps)
        action_raw = torch.atanh(action_tanh)
        # Base log-prob
        log_prob = dist.log_prob(action_raw)
        # Jacobian correction for tanh: sum over action dims
        log_prob -= torch.log(1 - action_tanh.pow(2) + eps)
        return log_prob.sum(-1)

    class PPOActor(nn.Module):
        """
        Actor network for PPO with continuous actions in [0, 1].
        Two hidden layers + learnable log_std parameter.
        """
        def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256):
            super().__init__()
            self.net = nn.Sequential(
                layer_init(nn.Linear(obs_dim, hidden_dim)), nn.Tanh(),
                layer_init(nn.Linear(hidden_dim, hidden_dim)), nn.Tanh(),
                layer_init(nn.Linear(hidden_dim, act_dim), std=0.01),
            )
            self.log_std = nn.Parameter(torch.zeros(act_dim))

        def forward(self, obs: torch.Tensor):
            mean = self.net(obs)
            log_std = self.log_std.expand_as(mean)
            return mean, log_std

        @torch.no_grad()
        def get_action(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
            """Deterministic tanh(mean) mapped to [0,1] by default."""
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            mean, log_std = self.forward(obs_t)
            std = log_std.exp()
            dist = torch.distributions.Normal(mean, std)
            if deterministic:
                a_raw = torch.tanh(mean)
            else:
                a_raw = torch.tanh(dist.sample())
            a = (a_raw + 1.0) * 0.5
            return a.squeeze(0).cpu().numpy()

    class RolloutBuffer:
        """
        Simple buffer for storing rollout data - pure data structure.
        The actual advantage computation stays in the notebook as GAE is educational.
        """

        def __init__(self, obs_shape, act_shape, capacity: int, device: str = 'cpu'):
            self.obs = torch.zeros((capacity,) + tuple(obs_shape), device=device)
            self.actions = torch.zeros((capacity,) + tuple(act_shape), device=device)
            self.logprobs = torch.zeros(capacity, device=device)
            self.rewards = torch.zeros(capacity, device=device)
            self.dones = torch.zeros(capacity, device=device)
            self.values = torch.zeros(capacity, device=device)
            self.position = 0
            self.capacity = capacity
            self.device = device

        def add(self, obs, action, logprob, reward, done, value) -> bool:
            if self.position >= self.capacity:
                return False  # Buffer full
            self.obs[self.position] = obs
            self.actions[self.position] = action
            self.logprobs[self.position] = logprob
            self.rewards[self.position] = reward
            self.dones[self.position] = done
            self.values[self.position] = value
            self.position += 1
            return True

        def get(self) -> dict[str, torch.Tensor]:
            """Returns all stored data and resets position."""
            data = {
                'obs': self.obs[: self.position],
                'actions': self.actions[: self.position],
                'logprobs': self.logprobs[: self.position],
                'rewards': self.rewards[: self.position],
                'dones': self.dones[: self.position],
                'values': self.values[: self.position],
            }
            self.position = 0
            return data


def make_vec_envs(
    env_id: str | Callable[[], gym.Env],
    num_envs: int = 1,
    seed: int = 0,
    device: str = 'cpu',  # kept for compatibility; not used by envs
    wrapper_fn: Optional[Callable[[gym.Env], gym.Env]] = None,
    *,
    async_mode: bool = True,
    record_stats: bool = True,
) -> gym.vector.VectorEnv:
    """
    Create vectorized environments for parallel rollouts.
    This is boilerplate for environment setup, not RL logic.

    Args:
        env_id: Gym environment ID or callable that creates env
        num_envs: Number of parallel environments
        seed: Base random seed
        device: Device for observations ('cpu' or 'cuda')
        wrapper_fn: Optional wrapper to apply to each env
    """

    def make_env(rank: int):
        def _thunk():
            if callable(env_id):
                env = env_id()
            else:
                env = gym.make(env_id)

            if wrapper_fn is not None:
                env = wrapper_fn(env)

            # Optionally record episode statistics for cleaner logging from vector envs
            if record_stats and RecordEpisodeStatistics is not None:
                try:
                    env = RecordEpisodeStatistics(env, deque_size=100)
                except Exception:
                    pass

            # Standard seeding
            try:
                env.action_space.seed(seed + rank)
                env.observation_space.seed(seed + rank)
                env.reset(seed=seed + rank)
            except Exception:
                pass

            return env

        return _thunk

    if num_envs == 1:
        return gym.vector.SyncVectorEnv([make_env(0)])
    else:
        if async_mode:
            return gym.vector.AsyncVectorEnv([make_env(i) for i in range(num_envs)])
        else:
            return gym.vector.SyncVectorEnv([make_env(i) for i in range(num_envs)])


def get_obs_shape(observation_space: spaces.Space) -> tuple[int, ...]:
    """Get observation shape from different space types - utility function."""
    if isinstance(observation_space, spaces.Box):
        return tuple(observation_space.shape)
    elif isinstance(observation_space, spaces.Discrete):
        return (1,)
    elif isinstance(observation_space, spaces.Dict):
        # Use flatten_space to determine final shape
        return tuple(spaces.flatten_space(observation_space).shape)
    else:
        raise NotImplementedError(f"Observation space {observation_space} not supported")


def get_action_dim(action_space: spaces.Space) -> int:
    """Get action dimension from different space types - utility function."""
    if isinstance(action_space, spaces.Box):
        return int(np.prod(action_space.shape))
    elif isinstance(action_space, spaces.Discrete):
        return int(action_space.n)
    else:
        raise NotImplementedError(f"Action space {action_space} not supported")


def set_random_seed(seed: int) -> None:
    """Set random seed for reproducibility - standard practice."""
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass


def normalize_obs(obs: np.ndarray, obs_rms: Optional[RunningMeanStd]) -> np.ndarray:
    """Normalize observations using running statistics - standard preprocessing."""
    if obs_rms is not None:
        return obs_rms.normalize(obs)
    return obs


def compute_returns_and_advantages(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    gae_lambda: Optional[float] = None,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Simple discounted returns computation.
    GAE should be shown in notebook as it's educational.
    This is just for when you want basic returns without GAE.
    """
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    dones = np.asarray(dones, dtype=np.float32)

    T = len(rewards)
    returns = np.zeros_like(rewards)
    if T == 0:
        return returns, None

    # Bootstrap with last value if not done at the end
    returns[-1] = rewards[-1] + gamma * values[-1] * (1.0 - dones[-1])
    for t in reversed(range(T - 1)):
        returns[t] = rewards[t] + gamma * returns[t + 1] * (1.0 - dones[t])

    if gae_lambda is not None:
        # User should implement GAE in notebook for learning
        raise NotImplementedError(
            "GAE computation should be implemented in the notebook for educational purposes. "
            "Use compute_returns_and_advantages with gae_lambda=None for simple returns."
        )

    return returns, None


class LinearSchedule:
    """Linear schedule for learning rate or other hyperparameters - standard utility."""

    def __init__(self, initial_value: float, final_value: float, total_steps: int):
        self.initial_value = float(initial_value)
        self.final_value = float(final_value)
        self.total_steps = int(total_steps)
        self.current_step = 0

    def get_value(self, step: Optional[int] = None) -> float:
        if step is None:
            step = self.current_step
        if step >= self.total_steps:
            return self.final_value
        progress = step / max(1, self.total_steps)
        return self.initial_value + progress * (self.final_value - self.initial_value)

    def step(self) -> float:
        value = self.get_value()
        self.current_step += 1
        return value


# ======== Deterministic policy builders for evaluation ========
if TORCH_AVAILABLE:
    def _build_action_scalers(make_env_fn, device):
        """Create a temporary env to compute action scale/bias tensors on device."""
        env_temp = make_env_fn()
        try:
            scale_np, bias_np = get_action_scale_and_bias(env_temp.action_space)
        finally:
            try:
                env_temp.close()
            except Exception:
                pass
        scale = torch.as_tensor(scale_np, dtype=torch.float32, device=device)
        bias = torch.as_tensor(bias_np, dtype=torch.float32, device=device)
        return scale, bias

    def build_ppo_policy(actor: nn.Module, make_env_fn, device: "torch.device | str" = "cpu"):
        """
        Build a deterministic PPO policy function.
        Uses tanh(mean) and rescales to env bounds.
        """
        action_scale, action_bias = _build_action_scalers(make_env_fn, device)

        def policy_fn(obs):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                mean, _ = actor(obs_t)
                action_tanh = torch.tanh(mean)
                action = scale_action(action_tanh, action_scale, action_bias)
            return action.squeeze(0).cpu().numpy()

        return policy_fn

    def build_sac_policy(actor: nn.Module, make_env_fn, device: "torch.device | str" = "cpu"):
        """
        Build a deterministic SAC policy function (use mean action).
        Uses tanh(mean) and rescales to env bounds.
        """
        action_scale, action_bias = _build_action_scalers(make_env_fn, device)

        def policy_fn(obs):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                mean, _ = actor.forward(obs_t)
                action_tanh = torch.tanh(mean)
                action = scale_action(action_tanh, action_scale, action_bias)
            return action.squeeze(0).cpu().numpy()

        return policy_fn

    def build_lstm_policy(lstm_actor: nn.Module, make_env_fn, device: "torch.device | str" = "cpu"):
        """
        Build a stateful LSTM policy function.
        Follows the tutorial convention: sigmoid(mean) to map to [0,1].
        Provides a reset() method to clear hidden state.
        """
        hidden_state = None

        def policy_fn(obs):
            nonlocal hidden_state
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                # Expecting signature: mean, _, new_hidden
                mean, _, hidden_state = lstm_actor(obs_t, hidden_state)
                action = torch.sigmoid(mean)
            return action.squeeze(0).cpu().numpy()

        def reset():
            nonlocal hidden_state
            hidden_state = None

        policy_fn.reset = reset
        return policy_fn

    def build_grpo_policy(actor: nn.Module, make_env_fn, device: "torch.device | str" = "cpu"):
        """
        Build a deterministic GRPO policy function.
        Uses tanh(mean) and rescales to env bounds.
        """
        action_scale, action_bias = _build_action_scalers(make_env_fn, device)

        def policy_fn(obs):
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                mean = actor(obs_t)
                if isinstance(mean, (tuple, list)):
                    mean = mean[0]
                action_tanh = torch.tanh(mean)
                action = scale_action(action_tanh, action_scale, action_bias)
            return action.squeeze(0).cpu().numpy()

        return policy_fn
