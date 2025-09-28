"""
Lightweight wrappers for external baselines that integrate with our adapters.
- Stable-Baselines3 (single-agent PPO/SAC)
- RLlib + PettingZoo (multi-agent PPO/APPO)
- OmniSafe (safe RL) — thin guidance, relies on env exposing info['safety_cost']

These helpers are optional and import heavy dependencies lazily inside functions.
"""
from __future__ import annotations

from typing import Callable, Any, Dict, List, Optional
import numpy as np

# -----------------------------
# Stable-Baselines3 (SB3)
# -----------------------------

def sb3_make_vec_env(env_factory: Callable[[], Any], n_envs: int = 1):
    """
    Create a DummyVecEnv around `env_factory` for SB3.
    The factory should return a Gymnasium Env with Box obs/action (e.g., DictFlatteningWrapper applied).
    """
    from stable_baselines3.common.vec_env import DummyVecEnv
    return DummyVecEnv([env_factory for _ in range(n_envs)])


def sb3_make_model(algo: str, policy: str, vec_env, **kwargs):
    """
    Build an SB3 model given algo name ('PPO' or 'SAC'), a policy string (e.g., 'MlpPolicy'), and a vec_env.
    Example:
        env = sb3_make_vec_env(lambda: DictFlatteningWrapper(make_env()))
        model = sb3_make_model('PPO', 'MlpPolicy', env, verbose=1)
    """
    if algo.upper() == 'PPO':
        from stable_baselines3 import PPO
        return PPO(policy, vec_env, **kwargs)
    elif algo.upper() == 'SAC':
        from stable_baselines3 import SAC
        return SAC(policy, vec_env, **kwargs)
    else:
        raise ValueError(f"Unsupported SB3 algo: {algo}")


def sb3_policy_fn(model) -> Callable[[np.ndarray], np.ndarray]:
    """
    Wrap an SB3 model as policy_fn(obs)->action for our evaluation helpers.
    Handles single observation input (non-vectorized) by adding a batch dim.
    """
    def _fn(obs: np.ndarray) -> np.ndarray:
        a, _ = model.predict(obs, deterministic=True)
        return np.asarray(a, dtype=np.float32)
    return _fn

# -----------------------------
# RLlib + PettingZoo (Multi-Agent)
# -----------------------------

def rllib_register_parallel_env(name: str, pz_env_factory: Callable[[], Any]) -> None:
    """
    Register a PettingZoo ParallelEnv with RLlib under `name`.
    Usage:
        rllib_register_parallel_env('ev_ma', lambda: ParallelEVEnv(...))
    """
    from ray.tune.registry import register_env
    from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

    def _make(_config=None):
        return ParallelPettingZooEnv(pz_env_factory())

    register_env(name, _make)


def rllib_collect_one_episode(algo, env, policy_mapping_fn: Callable[[str], str]) -> dict:
    """
    Roll out one episode using an RLlib Algorithm `algo` on a PettingZoo ParallelEnv wrapped by RLlib.
    Returns a dict with actions per agent and simple timeseries.
    """
    import collections

    actions_hist = collections.defaultdict(list)
    rews_hist = []

    obs, info = env.reset()
    done = {agent: False for agent in env.agents}
    terminated = {agent: False for agent in env.agents}
    truncated = {agent: False for agent in env.agents}

    while not all(terminated.values()) and not all(truncated.values()):
        act = {}
        for agent, ob in obs.items():
            pid = policy_mapping_fn(agent)
            a = algo.compute_single_action(observation=ob, policy_id=pid)
            act[agent] = a
        obs, rew, terminated, truncated, info = env.step(act)
        for agent, a in act.items():
            actions_hist[agent].append(float(np.asarray(a).squeeze()))
        rews_hist.append(np.mean(list(rew.values())))

    return {
        'actions': {k: np.array(v, dtype=np.float32) for k, v in actions_hist.items()},
        'reward': np.array(rews_hist, dtype=np.float32),
    }

# -----------------------------
# OmniSafe (Safe RL)
# -----------------------------

def check_env_has_safety_cost(env) -> bool:
    """
    Quick check that env emits info['safety_cost'] in step() when wrapped.
    """
    try:
        obs, info = env.reset()
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        return 'safety_cost' in info
    except Exception:
        return False


def make_omnisafe_ready_env(env_factory: Callable[[], Any]) -> Any:
    """
    OmniSafe expects environments with explicit safety cost signals.
    This helper just runs env_factory and asserts that info['safety_cost'] is present.
    Pair with your SafeCostWrapper in Tutorial 05.
    """
    env = env_factory()
    assert check_env_has_safety_cost(env), "Env must expose info['safety_cost']; wrap with SafeCostWrapper."
    return env

# -----------------------------
# SafePO (in-repo) import helpers
# -----------------------------

def safepo_add_path(base_dir: str = 'Safe-Policy-Optimization-main') -> None:
    """
    Ensure the Safe-Policy-Optimization repo is on sys.path so `import safepo` works.
    Call this before importing safepo.* modules.
    """
    import sys, os
    if base_dir not in sys.path:
        sys.path.insert(0, base_dir)


def safepo_import_single_agent(algo: str = 'ppo_lag', base_dir: str = 'Safe-Policy-Optimization-main'):
    """
    Import SafePO single-agent algorithm module, e.g., 'ppo_lag', 'rcpo', 'cpo'.
    Returns the imported module (e.g., safepo.single_agent.ppo_lag).
    """
    safepo_add_path(base_dir)
    import importlib
    return importlib.import_module(f'safepo.single_agent.{algo}')


def safepo_import_multi_agent(algo: str = 'mappo', base_dir: str = 'Safe-Policy-Optimization-main'):
    """
    Import SafePO multi-agent algorithm module, e.g., 'mappo', 'mappolag', 'macpo'.
    Returns the imported module (e.g., safepo.multi_agent.mappo).
    """
    safepo_add_path(base_dir)
    import importlib
    return importlib.import_module(f'safepo.multi_agent.{algo}')


# -----------------------------
# SafePO-style share_obs builder
# -----------------------------

def build_share_obs_dict(
    obs_dict: Dict[str, np.ndarray], agent_order: Optional[List[str]] = None
) -> Dict[str, np.ndarray]:
    """
    Construct a SafePO-style `share_obs` dict from a per-agent flat obs dict.

    For a centralized critic, SafePO expects `share_obs[agent]` to contain the same
    joint observation vector (concatenation of all agents' flat obs) for each agent.

    Parameters
    ----------
    obs_dict : Dict[str, np.ndarray]
        Mapping agent_id -> flat observation for that agent.
    agent_order : Optional[List[str]]
        Order of agents for concatenation to ensure determinism. If None, uses sorted keys.

    Returns
    -------
    Dict[str, np.ndarray]
        Mapping agent_id -> the same centralized (joint) observation vector.
    """
    # Import locally to avoid cyclic/hard dependencies at module import time
    from .marl import build_central_state

    central = build_central_state(obs_dict, agent_order=agent_order)
    return {aid: central.copy() for aid in obs_dict.keys()}
