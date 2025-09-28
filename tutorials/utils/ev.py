"""
Shared utilities for EV Charging tutorials (baseline, RL, Safe RL, MARL).
Keep this module minimal and reusable across notebooks.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Dict, Tuple, Iterable, Optional, Union, Any, List
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from .common import wrap_policy, wrap_ma_policy
from .notebook import log_info, log_warning, log_error, log_success, show, show_metrics

# Optional plotting dependency
try:
    import seaborn as sns  # type: ignore
except Exception:  # pragma: no cover - optional
    sns = None  # type: ignore

# -----------------------------
# Paths and simple persistence
# -----------------------------

# Determine cache base directory:
# - Use env var TUTORIALS_CACHE_DIR if set
# - Otherwise default to a 'cache' folder next to this file
CACHE_BASE = os.environ.get("TUTORIALS_CACHE_DIR") or str((Path(__file__).resolve().parent / "cache"))
CACHE_DIRS = {
    "baselines": os.path.join(CACHE_BASE, "baselines"),
    "rl": os.path.join(CACHE_BASE, "rl"),
    "safe_rl": os.path.join(CACHE_BASE, "safe_rl"),
    "marl": os.path.join(CACHE_BASE, "marl"),
}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_cache_dir(kind: str) -> str:
    d = CACHE_DIRS.get(kind, os.path.join(CACHE_BASE, kind))
    ensure_dir(d)
    return d


def save_df_gz(df: pd.DataFrame, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=False, compression="gzip")


def load_df_gz(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

# -----------------------------
# Standard cache file helpers
# -----------------------------

def cache_path(kind: str, tag: str, suffix: str) -> str:
    d = get_cache_dir(kind)
    return os.path.join(d, f"{tag}_{suffix}.csv.gz")

def save_timeseries(tag: str, ts_df: pd.DataFrame, kind: str = "rl") -> str:
    """
    Save per-episode or per-update timeseries under a standard path.
    Example filename: <CACHE_BASE>/rl/<tag>_timeseries.csv.gz
    By default, CACHE_BASE is a 'cache' folder next to this file; override via env var TUTORIALS_CACHE_DIR.
    """
    path = cache_path(kind, tag, "timeseries")
    save_df_gz(ts_df, path)
    return path

def load_timeseries(tag: str, kind: str = "rl") -> pd.DataFrame:
    return load_df_gz(cache_path(kind, tag, "timeseries"))

# -----------------------------
# Trajectory standardization helpers
# -----------------------------

def standardize_trajectory_costs(trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure all cost-like time series in a trajectory are per-step (not cumulative).

    If a series appears monotonically non-decreasing, treat it as cumulative and convert
    to per-step via first differences, preserving the original as *_cumulative.

    Keys processed: ['excess_charge', 'carbon_cost', 'profit']
    """
    if trajectory is None:
        return trajectory
    # Convert any cumulative series to per-step and keep original as *_cumulative
    for cost_key in ['excess_charge', 'carbon_cost', 'profit']:
        if cost_key in trajectory:
            arr = np.asarray(trajectory[cost_key], dtype=float).reshape(-1)
            if arr.size > 1:
                diffs = np.diff(arr)
                # Consider floating tolerance; treat as cumulative when non-decreasing
                if np.all(diffs >= -1e-9):
                    trajectory[f"{cost_key}_cumulative"] = arr.copy()
                    trajectory[cost_key] = np.diff(np.concatenate([[0.0], arr]))
    return trajectory

def seek_to_active(
    env: Any,
    key: str = 'demands',
    max_seek: int = 200,
    ensure_change: bool = True,
    lookahead: int = 20,
    change_tol: float = 1e-6,
    reset_seed: Optional[int] = None,
) -> Tuple[Any, Dict[str, Any], int]:
    """Advance an environment to a timestep where the observation shows activity.

    By default, it targets positive EV `demands` and, if requested, ensures the
    measured activity changes within a short lookahead window so downstream plots
    are not flat.

    Parameters
    ----------
    env : gymnasium.Env-like
        Environment with reset() -> (obs, info) and step(action) -> (obs, r, term, trunc, info).
    key : str
        Observation dict key to measure activity from (default: 'demands').
    max_seek : int
        Maximum number of steps to advance while seeking.
    ensure_change : bool
        If True, also ensure the measured activity changes within `lookahead` steps.
    lookahead : int
        Steps to look ahead for a change when ensure_change is True.
    change_tol : float
        Threshold for considering a change significant.

    Returns
    -------
    obs, info, steps : Tuple[Any, Dict[str, Any], int]
        The observation and info at the found timestep, and total steps advanced.
    """
    def _measure(o: Any) -> float:
        if isinstance(o, dict) and key in o and o[key] is not None:
            d = np.asarray(o[key], dtype=float).reshape(-1)
            return float(np.sum(d[d > 0]))
        return 0.0

    def _zero_action_for(o: Any) -> Any:
        if isinstance(o, dict) and key in o and o[key] is not None:
            return np.zeros_like(o[key], dtype=float)
        try:
            return np.asarray(env.action_space.sample()) * 0
        except Exception:
            return 0.0

    obs, info = env.reset(seed=reset_seed)
    steps = 0
    val = _measure(obs)

    # Phase 1: move to any timestep with positive measure
    while val <= 0.0 and steps < int(max_seek):
        obs, _, term, trunc, info = env.step(_zero_action_for(obs))
        steps += 1
        val = _measure(obs)
        if term or trunc:
            obs, info = env.reset()
            val = _measure(obs)

    if ensure_change and steps < int(max_seek):
        prev = val
        k = 0
        changed = False
        while (not changed) and (steps < int(max_seek)) and (k < int(lookahead)):
            obs_next, _, term, trunc, info_next = env.step(_zero_action_for(obs))
            steps += 1
            k += 1
            cur = _measure(obs_next)
            if abs(cur - prev) > float(change_tol) or term or trunc:
                changed = True
            obs, info = (obs_next, info_next)
            prev = cur
            if term or trunc:
                obs, info = env.reset()
                prev = _measure(obs)

    return obs, info, steps


def traj_to_timeseries_df(trajectory: Dict[str, Any], algorithm_name: str) -> pd.DataFrame:
    """
    Convert trajectory dict to standardized timeseries DataFrame.

    Expected keys include: 'reward', 'profit', 'carbon_cost', 'excess_charge', 'active_evs'.
    Missing keys are filled with zeros of appropriate length.
    """
    T = int(len(np.asarray(trajectory.get('reward', []))))
    zeros = np.zeros(T, dtype=float)
    evs = trajectory.get('active_evs', np.zeros(T, dtype=int))
    return pd.DataFrame({
        'episode': np.zeros(T, dtype=int),
        'timestep': np.arange(T, dtype=int),
        'reward': np.asarray(trajectory.get('reward', zeros), dtype=float),
        'profit': np.asarray(trajectory.get('profit', zeros), dtype=float),
        'carbon_cost': np.asarray(trajectory.get('carbon_cost', zeros), dtype=float),
        'excess_charge': np.asarray(trajectory.get('excess_charge', zeros), dtype=float),
        'active_evs': np.asarray(evs, dtype=int),
        'algorithm': algorithm_name,
    })

def save_dual_log(tag: str, dual_df: pd.DataFrame, kind: str = "safe_rl") -> str:
    """
    Save dual dynamics logs (e.g., lambda, mean_ep_cost, budget) for Safe RL.
    Example filename: tutorials/cache/safe_rl/<tag>_dual.csv.gz
    """
    path = cache_path(kind, tag, "dual")
    save_df_gz(dual_df, path)
    return path

def load_dual_log(tag: str, kind: str = "safe_rl") -> pd.DataFrame:
    return load_df_gz(cache_path(kind, tag, "dual"))

# -----------------------------
# Evaluation and sweeps (single-agent)
# -----------------------------


def evaluate_policy(
    policy_fn: Callable[[Any], np.ndarray],
    make_env_fn: Callable[..., object],
    episodes: int = 2,
    noise: Optional[float] = None,
    noise_action: Optional[float] = None,
    horizon: int = 288,
    track_metrics: Optional[list[str]] = None,
    verbose: bool = False,
    reset_seed: Optional[int] = None,
) -> Tuple[float, float, float, Dict[str, Any]]:
    """
    Evaluate a single-agent policy with optional comprehensive metrics tracking.

    Returns (mean_return, std_return, mean_safety_cost, metrics_dict).

    Notes
    -----
    Cost signal conventions:
    - Preferred: if the environment (or a safety wrapper) provides per-step safety cost
      `info['cost']`, that value is used directly.
    - Fallback: when `info['cost']` is absent, we derive a per-step cost by differencing the
      cumulative series `info['reward_breakdown']['excess_charge']` (monotone within episode).
    - Aggregation: `mean_safety_cost` returned by this function is the mean of episode totals
      (sum of per-step costs over each episode), aligning training and evaluation semantics.

    Other:
    - `policy_fn` should accept the raw observation structure produced by the env (e.g., a Dict or np.ndarray).
    """
    # Support both signature styles: make_env_fn(noise=..., noise_action=...) or zero-arg factories
    try:
        env = make_env_fn(noise=noise, noise_action=noise_action)
    except TypeError:
        # Fall back to calling without kwargs if the factory doesn't accept them
        env = make_env_fn()

    if track_metrics is None:
        track_metrics = []

    returns, costs = [], []
    metrics: Dict[str, list] = defaultdict(list)

    for ep in range(episodes):
        if verbose:
            log_info(f"Evaluating episode {ep+1}/{episodes}")

        # If a fixed reset_seed is provided, use it (ensures scenario determinism).
        # Otherwise default to varying per-episode seed.
        obs, info = env.reset(seed=(reset_seed if reset_seed is not None else ep))
        done = False
        ep_ret, ep_cost = 0.0, 0.0
        t = 0
        # Track last cumulative excess charge to compute per-step cost when SafetyWrapper not used
        last_excess_cum = 0.0

        # Episode-level tracking
        ep_actions: list[np.ndarray] = []
        ep_demands_initial: list[np.ndarray] = []
        ep_demands_final: list[np.ndarray] = []
        # Track component series to robustly aggregate (handle cumulative vs per-step)
        comp_profit_series: list[float] = []
        comp_carbon_series: list[float] = []
        trajectory = []

        while not done and t < horizon:
            # Store initial state for satisfaction/demand tracking
            if ('satisfaction' in track_metrics) or ('demands' in track_metrics):
                if isinstance(obs, dict) and 'demands' in obs:
                    initial_demands = obs['demands'].copy()
                else:
                    initial_demands = np.zeros(1)
                ep_demands_initial.append(initial_demands)

            # Get action
            action = policy_fn(obs)
            ep_actions.append(action)

            # Step
            obs_next, r, term, trunc, info = env.step(action)

            # Track trajectory if requested
            if 'trajectory' in track_metrics:
                trajectory.append({
                    't': t,
                    'obs': obs.copy() if isinstance(obs, dict) else np.array(obs).copy(),
                    'action': np.array(action).copy(),
                    'reward': float(r),
                    'info': dict(info),
                })

            # Update core metrics
            ep_ret += float(r)
            rb = info.get('reward_breakdown', {})
            # Prefer SafetyWrapper's per-step cost when present
            step_cost = info.get('cost', None)
            if step_cost is None:
                # Fallback: difference the cumulative series from reward_breakdown
                ex_cum = float(rb.get('excess_charge', 0.0))
                step_cost = max(0.0, ex_cum - last_excess_cum)
                last_excess_cum = ex_cum
            ep_cost += float(step_cost)

            # Track components if requested (store series and aggregate post-episode)
            if 'components' in track_metrics:
                comp_profit_series.append(float(rb.get('profit', 0.0)))
                comp_carbon_series.append(float(rb.get('carbon_cost', 0.0)))

            # Track demand satisfaction
            if ('satisfaction' in track_metrics) or ('demands' in track_metrics):
                if isinstance(obs_next, dict) and 'demands' in obs_next:
                    final_demands = obs_next['demands'].copy()
                else:
                    final_demands = np.zeros(1)
                ep_demands_final.append(final_demands)

            obs = obs_next
            done = bool(term) or bool(trunc)
            t += 1

        # Store episode results
        returns.append(ep_ret)
        costs.append(ep_cost)

        # Calculate episode-level demand totals if needed
        if (('satisfaction' in track_metrics) or ('demands' in track_metrics)) and ep_demands_initial:
            total_initial = float(sum(np.sum(d[d > 0]) for d in ep_demands_initial))
            total_final = float(sum(np.sum(d[d > 0]) for d in ep_demands_final))
        else:
            total_initial = 0.0
            total_final = 0.0

        # Episode-level metrics aggregation
        if 'satisfaction' in track_metrics and ep_demands_initial:
            delivered = max(0.0, total_initial - total_final)
            satisfaction = delivered / max(total_initial, 1.0)
            metrics['satisfaction'].append(satisfaction)
            metrics['total_delivered'].append(delivered)
            metrics['total_requested'].append(total_initial)

        # True satisfaction from simulator: delivered kWh / total requested kWh (episode-level)
        # Robust to wrappers (e.g., SafetyWrapper) and flattened observations.
        try:
            # Unwrap common Gym wrappers to reach base env
            base_env = env
            # Follow `.env` chain if present (older wrappers)
            for _ in range(5):
                if hasattr(base_env, 'env'):
                    base_env = getattr(base_env, 'env')
                else:
                    break
            # Prefer Gym's unwrapped if available
            base_env = getattr(base_env, 'unwrapped', base_env)

            sim = getattr(base_env, '_simulator', None)
            evs = getattr(base_env, '_evs', [])
            a_pers_to_kwh = float(getattr(base_env, 'A_PERS_TO_KWH', 0.0))
            if sim is not None and hasattr(sim, 'charging_rates') and a_pers_to_kwh > 0:
                R = np.asarray(sim.charging_rates)
                # Use the portion of the episode actually simulated (t steps)
                T_used = min(int(t), R.shape[1]) if R.ndim == 2 else int(t)
                delivered_amp_periods = float(np.sum(R[:, :T_used])) if R.ndim == 2 else 0.0
                delivered_kwh = delivered_amp_periods * a_pers_to_kwh
                requested_total_kwh = float(np.sum([float(getattr(ev, 'requested_energy', 0.0)) for ev in evs]))
                true_sat = (delivered_kwh / max(requested_total_kwh, 1e-9)) if requested_total_kwh > 0 else 0.0
                metrics['true_satisfaction'].append(true_sat)
                # Optional transparency for downstream analyses
                metrics['delivered_kwh'].append(delivered_kwh)
                metrics['requested_kwh'].append(requested_total_kwh)
        except Exception:
            # Do not fail evaluation if diagnostic computation is unavailable
            pass

        if 'components' in track_metrics:
            # Aggregate component series into episode totals.
            # If the series is monotonically non-decreasing, treat it as cumulative and use the final value.
            # Otherwise, assume per-step values and sum them.
            ep_profit_total = 0.0
            ep_carbon_total = 0.0

            if len(comp_profit_series) > 0:
                arr_p = np.asarray(comp_profit_series, dtype=float)
                if arr_p.size > 1 and np.all(np.diff(arr_p) >= -1e-9):
                    ep_profit_total = float(arr_p[-1])
                else:
                    ep_profit_total = float(np.sum(arr_p))

            if len(comp_carbon_series) > 0:
                arr_c = np.asarray(comp_carbon_series, dtype=float)
                if arr_c.size > 1 and np.all(np.diff(arr_c) >= -1e-9):
                    ep_carbon_total = float(arr_c[-1])
                else:
                    ep_carbon_total = float(np.sum(arr_c))

            metrics['profit'].append(ep_profit_total)
            metrics['carbon_cost'].append(ep_carbon_total)
            metrics['profit_per_step'].append(ep_profit_total / max(t, 1))
            metrics['carbon_per_step'].append(ep_carbon_total / max(t, 1))

        if 'actions' in track_metrics and ep_actions:
            actions_array = np.array(ep_actions)
            metrics['action_mean'].append(float(np.mean(actions_array)))
            metrics['action_std'].append(float(np.std(actions_array)))
            metrics['action_saturation_high'].append(float(np.mean(actions_array >= 0.999)))
            metrics['action_saturation_low'].append(float(np.mean(actions_array <= 0.001)))
            metrics['actions_per_episode'].append(actions_array)

        if 'demands' in track_metrics and ep_demands_initial:
            metrics['demands_initial'].append(ep_demands_initial)
            metrics['demands_final'].append(ep_demands_final)
            metrics['demand_reduction_rate'].append(1.0 - (total_final / max(total_initial, 1.0)))

        if 'trajectory' in track_metrics:
            metrics['trajectories'].append(trajectory)

    # Aggregate metrics across episodes
    aggregated_metrics: Dict[str, Any] = {}
    for key, values in metrics.items():
        if key in ['trajectories', 'actions_per_episode', 'demands_initial', 'demands_final']:
            aggregated_metrics[key] = values
        elif len(values) > 0:
            aggregated_metrics[f"{key}_mean"] = float(np.mean(values))
            aggregated_metrics[f"{key}_std"] = float(np.std(values))
            aggregated_metrics[f"{key}_all"] = values

    aggregated_metrics['eval_episodes'] = episodes
    aggregated_metrics['eval_horizon'] = horizon
    aggregated_metrics['noise_obs'] = noise
    aggregated_metrics['noise_action'] = noise_action
    # Episode-level aggregates useful for downstream safety analysis
    aggregated_metrics['episode_returns'] = list(map(float, returns))
    aggregated_metrics['episode_costs'] = list(map(float, costs))

    return float(np.mean(returns)), float(np.std(returns)), float(np.mean(costs)), aggregated_metrics


def rollout_to_trajectory(
    env,
    policy_fn: Callable[[Any], np.ndarray],
    max_steps: int = 288,
    standardize: bool = False,
    deep_copy_obs: bool = False,
) -> Dict[str, Any]:
    """Standard trajectory collection using evaluate_policy.

    Returns a dict with keys: reward, action, profit, carbon_cost, excess_charge, obs, moer, demands, est_departures.
    Missing fields are omitted gracefully.

    Parameters
    ----------
    env : gymnasium.Env-like
        Environment instance to roll out in.
    policy_fn : Callable
        Policy function mapping obs -> action.
    max_steps : int
        Maximum steps to roll out.
    standardize : bool
        If True, post-process cost-like series to per-step via standardize_trajectory_costs().
    deep_copy_obs : bool
        If True, deep-copy dict observations into the trajectory for maximum aliasing safety.
    """
    mean_ret, std_ret, mean_cost, metrics = evaluate_policy(
        policy_fn,
        lambda **kwargs: env,
        episodes=1,
        horizon=max_steps,
        track_metrics=['trajectory', 'actions', 'demands', 'components'],
        verbose=False,
    )

    traj_list = metrics.get('trajectories', [])
    if not traj_list:
        return {}
    traj_steps = traj_list[0]

    trajectory: Dict[str, Any] = {
        'reward': np.array([s.get('reward', 0.0) for s in traj_steps], dtype=float),
        'action': np.array([np.asarray(s.get('action', 0.0)) for s in traj_steps]),
        'profit': np.array([s.get('info', {}).get('reward_breakdown', {}).get('profit', 0.0) for s in traj_steps], dtype=float),
        'carbon_cost': np.array([s.get('info', {}).get('reward_breakdown', {}).get('carbon_cost', 0.0) for s in traj_steps], dtype=float),
        'excess_charge': np.array([s.get('info', {}).get('reward_breakdown', {}).get('excess_charge', 0.0) for s in traj_steps], dtype=float),
        'obs': [(__import__('copy').deepcopy(s.get('obs')) if (deep_copy_obs and isinstance(s.get('obs'), dict)) else s.get('obs')) for s in traj_steps],
    }

    # Optional context
    try:
        trajectory['moer'] = np.array([s.get('info', {}).get('moer', np.nan) for s in traj_steps], dtype=float)
    except Exception:
        pass
    try:
        trajectory['demands'] = np.array([s.get('obs', {}).get('demands') if isinstance(s.get('obs'), dict) else np.array([]) for s in traj_steps], dtype=object)
        trajectory['est_departures'] = np.array([s.get('obs', {}).get('est_departures') if isinstance(s.get('obs'), dict) else np.array([]) for s in traj_steps], dtype=object)
    except Exception:
        pass

    if standardize:
        trajectory = standardize_trajectory_costs(trajectory)
    return trajectory


def collect_trajectory_direct(
    env: Any,
    policy_or_algo: Any,
    max_steps: int = 288,
    seed: int = 0,
    deep_copy_obs: bool = True,
    standardize: bool = False,
) -> Dict[str, Any]:
    """Collect a single trajectory by stepping the env directly with a policy/algo.

    This mirrors the robust snapshotting behavior used in Tutorial 02, with:
    - Deep copies of dict observations to avoid aliasing
    - Per-step safety cost computed from info['cost'] when present, otherwise from
      differences of info['reward_breakdown']['excess_charge'] (cumulative)

    Parameters
    ----------
    env : gymnasium.Env-like
        Environment to roll out in.
    policy_or_algo : Callable or object
        Either a callable policy(obs)->action, or an object with one of
        {get_action, predict, act}. Resolved via wrap_policy().
    max_steps : int
        Maximum number of steps.
    seed : int
        Reset seed for the episode.
    deep_copy_obs : bool
        If True, deep-copy dict observations into the trajectory.
    standardize : bool
        If True, run standardize_trajectory_costs() before returning.
    """
    # Build a policy function with shared wrapper
    policy_fn = wrap_policy(policy_or_algo)

    obs, _ = env.reset(seed=seed)
    traj: Dict[str, Any] = {
        'obs': [],
        'action': [],
        'reward': [],
        'moer': [],
        'demands': [],
        'excess_charge': [],  # per-step safety cost
        'profit': [],
        'carbon_cost': [],
    }

    last_excess_cum = 0.0

    for _t in range(int(max_steps)):
        # Snapshot observation (deep copy for dicts if requested)
        if isinstance(obs, dict):
            if deep_copy_obs:
                import copy as _copy
                snap = _copy.deepcopy(obs)
            else:
                snap = obs.copy()
        else:
            snap = np.array(obs).copy()
        traj['obs'].append(snap)

        # Context variables (best-effort)
        try:
            if isinstance(obs, dict) and 'prev_moer' in obs:
                traj['moer'].append(float(np.asarray(obs['prev_moer'])[0]))
            else:
                traj['moer'].append(np.nan)
        except Exception:
            traj['moer'].append(np.nan)
        try:
            if isinstance(obs, dict) and 'demands' in obs:
                traj['demands'].append(np.asarray(obs['demands'], dtype=float).copy())
            else:
                traj['demands'].append(np.zeros(1, dtype=float))
        except Exception:
            traj['demands'].append(np.zeros(1, dtype=float))

        # Action and environment step
        action = np.asarray(policy_fn(obs), dtype=float)
        traj['action'].append(action.copy())
        obs, r, term, trunc, info = env.step(action)
        traj['reward'].append(float(r))

        # Robust per-step cost
        rb = info.get('reward_breakdown', {})
        step_cost = info.get('cost', None)
        if step_cost is None:
            ex_cum = float(rb.get('excess_charge', 0.0))
            step_cost = max(0.0, ex_cum - last_excess_cum)
            last_excess_cum = ex_cum
        traj['excess_charge'].append(float(step_cost))

        # Components
        traj['profit'].append(float(rb.get('profit', 0.0)))
        traj['carbon_cost'].append(float(rb.get('carbon_cost', 0.0)))

        if bool(term) or bool(trunc):
            break

    if standardize:
        traj = standardize_trajectory_costs(traj)
    return traj


def sweep_noise(
    policy_fn: Callable[[np.ndarray], np.ndarray],
    make_env_fn: Callable[..., object],
    Ns: Iterable[float] = (0.0, 0.05, 0.1, 0.2),
    N_as: Iterable[float] = (0.0, 0.1),
    episodes: int = 2,
    show_progress: bool = True,
) -> pd.DataFrame:
    rows = []
    Ns_list = list(Ns)
    N_as_list = list(N_as)
    total = len(Ns_list) * len(N_as_list)
    step = 0
    for n in Ns_list:
        for na in N_as_list:
            if show_progress:
                try:
                    show("progress: Sweeping noise settings", step=step, total=total, key="sweep_noise")
                except Exception:
                    pass
            m, s, c, _ = evaluate_policy(
                policy_fn, make_env_fn, episodes=episodes, noise=n, noise_action=na
            )
            rows.append({"noise": n, "noise_action": na, "return_mean": m, "return_std": s, "safety_cost": c})
            step += 1
    if show_progress:
        try:
            # Final update to 100%
            show("progress: Sweeping noise settings", step=total, total=total, key="sweep_noise")
            show("result: Noise sweep completed")
        except Exception:
            pass
    return pd.DataFrame(rows)

# -----------------------------
# Training utilities (lightweight, reusable)
# -----------------------------


def mpc_horizon_sweep(
    make_env_fn: Callable[..., Any],
    horizons: Iterable[int] = (1, 6, 12, 24),
    wrappers: Optional[list] = None,
    seed: int = 42,
    max_steps: int = 288,
    episodes: int = 1,
    **env_kwargs,
) -> Dict[str, Any]:
    """Run a standardized MPC horizon benchmark.

    Parameters
    ----------
    make_env_fn : Callable[..., Env]
        Zero-arg or kwargs-accepting factory to create a fresh environment per run.
    horizons : Iterable[int]
        MPC lookahead values to evaluate.
    wrappers : Optional[list]
        Sequence of wrappers to apply using tutorials.utils.apply_wrappers.
        Each element can be a callable (env->env) or a tuple (WrapperClass, kwargs_dict).
    seed : int
        Reset seed for environment episodes.
    max_steps : int
        Max steps per episode.
    episodes : int
        Number of episodes to average per horizon.
    env_kwargs : dict
        Forwarded to make_env_fn.

    Returns
    -------
    Dict[str, Any]
        {
          'rewards': { 'MPC-<H>': mean_return_across_episodes },
          'spread': max(rewards) - min(rewards),
          'best': name_of_best_horizon,
        }
    """
    # Lazy imports to avoid circularities at module import time
    from .common import apply_wrappers as _apply_wrappers  # type: ignore
    from .baselines import MPC  # type: ignore

    results: Dict[str, float] = {}

    for H in horizons:
        # Fresh env per horizon/episode to avoid cross-contamination
        ep_returns: list[float] = []
        for ep in range(int(episodes)):
            # Ensure env receives enough forecast steps for given horizon
            kwargs = dict(env_kwargs)
            try:
                max_h = int(H)
                cur = int(kwargs.get('moer_forecast_steps', 36))
                if cur < max_h:
                    kwargs['moer_forecast_steps'] = max_h
            except Exception:
                pass

            try:
                env = make_env_fn(**kwargs)
            except TypeError:
                env = make_env_fn()

            try:
                # Apply optional wrappers
                env = _apply_wrappers(env, wrappers)

                # Validate compatibility
                try:
                    validate_mpc_config(env, [int(H)])
                except Exception:
                    pass

                # Build MPC controller for this env and horizon
                mpc = MPC(env, lookahead=int(H))
                # Roll out and sum rewards
                obs, _ = env.reset(seed=seed + ep)
                total = 0.0
                t = 0
                done = False
                while (not done) and (t < int(max_steps)):
                    action = mpc.get_action(obs)
                    obs, r, term, trunc, _info = env.step(action)
                    total += float(r)
                    done = bool(term) or bool(trunc)
                    t += 1
                ep_returns.append(total)
            finally:
                try:
                    env.close()
                except Exception:
                    pass
        results[f"MPC-{int(H)}"] = float(np.mean(ep_returns)) if ep_returns else 0.0

    spread = float(max(results.values()) - min(results.values())) if results else 0.0
    best = max(results.items(), key=lambda kv: kv[1])[0] if results else ''
    return {
        'rewards': results,
        'spread': spread,
        'best': best,
    }


def validate_mpc_config(env: Any, horizons: Iterable[int]) -> bool:
    """Ensure environment supports required horizon lengths for MPC.

    Checks that env.moer_forecast_steps >= max(horizons) when present.
    """
    try:
        max_h = int(max(horizons))
    except Exception:
        return True
    if hasattr(env, 'moer_forecast_steps'):
        assert int(getattr(env, 'moer_forecast_steps')) >= max_h, (
            f"Forecast steps ({getattr(env, 'moer_forecast_steps')}) < max horizon ({max_h})"
        )
    return True


def create_training_logger(tag: str, metrics: list[str]) -> Dict[str, list]:
    """
    Create a standardized dict-of-lists logger for RL training loops.
    Keep minimal to encourage reuse across tutorials.
    """
    logger: Dict[str, list] = {
        'episode': [],
        'timestep': [],
        'reward': [],
        'episode_length': [],
    }
    for m in metrics:
        logger[m] = []
    return logger


def save_model_checkpoint(model: Any, tag: str, epoch: int, kind: str = "rl") -> str:
    """
    Save model checkpoint with standardized naming.
    Torch is imported lazily to avoid hard dependency at module import time.
    """
    path = os.path.join(get_cache_dir(kind), f"{tag}_model_epoch_{epoch}.pt")
    try:
        import torch  # type: ignore
        state = model.state_dict() if hasattr(model, 'state_dict') else model
        torch.save(state, path)
    except Exception as e:
        raise RuntimeError(f"Failed to save checkpoint to {path}: {e}")
    return path


def load_model_checkpoint(model_class: Any, tag: str, epoch: int, kind: str = "rl", **kwargs):
    """
    Load a model from checkpoint by instantiating model_class(**kwargs) and loading state_dict.
    Torch is imported lazily.
    """
    path = os.path.join(get_cache_dir(kind), f"{tag}_model_epoch_{epoch}.pt")
    try:
        import torch  # type: ignore
        model = model_class(**kwargs)
        state = torch.load(path, map_location='cpu')
        model.load_state_dict(state)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint from {path}: {e}")


def evaluate_policy_batch(
    policy_fn: Callable[[np.ndarray], np.ndarray],
    make_env_fn: Callable[..., object],
    episodes: int = 10,
    num_envs: int = 4,
    horizon: int = 288,
    **env_kwargs,
) -> Dict[str, np.ndarray]:
    """
    Evaluate a policy across multiple parallel environments using Gymnasium's SyncVectorEnv.

    Notes
    -----
    - This keeps the interface simple: it returns arrays for returns and lengths per episode.
    - For richer metrics (e.g., satisfaction, components), consider calling evaluate_policy() in a loop
      with track_metrics enabled, as batch tracking would add complexity here.
    """
    from gymnasium.vector import SyncVectorEnv  # type: ignore

    def _make_one():
        return make_env_fn(**env_kwargs)

    def _make_vec(n: int):
        return SyncVectorEnv([_make_one for _ in range(n)])

    ep_returns: list[float] = []
    ep_lengths: list[int] = []

    remaining = int(episodes)
    while remaining > 0:
        batch = min(num_envs, remaining)
        ven = _make_vec(batch)
        obs, infos = ven.reset()
        done_mask = np.zeros(batch, dtype=bool)
        rets = np.zeros(batch, dtype=np.float32)
        lens = np.zeros(batch, dtype=np.int32)
        # Derive a zero action matching the action space
        try:
            zero_act = np.asarray(ven.single_action_space.sample()) * 0
        except Exception:
            zero_act = 0.0

        t = 0
        while not np.all(done_mask) and t < horizon:
            # Compute actions per env (policy_fn is per-observation)
            actions = []
            for i in range(batch):
                if done_mask[i]:
                    # Placeholder action for finished envs in the vector; ignored by trackers
                    actions.append(np.array(zero_act))
                else:
                    # Handle Dict observations from vector envs
                    if isinstance(obs, dict):
                        ob_i = {k: (v[i] if isinstance(v, np.ndarray) else v) for k, v in obs.items()}
                    else:
                        ob_i = obs[i]
                    actions.append(np.asarray(policy_fn(ob_i)))
            actions = np.array(actions)

            obs, rewards, terms, truncs, infos = ven.step(actions)

            # Update stats
            rets += rewards.astype(np.float32)
            just_done = (terms | truncs) & (~done_mask)
            lens[~done_mask] += 1
            done_mask |= (terms | truncs)
            t += 1

        ep_returns.extend(rets.tolist())
        ep_lengths.extend(lens.tolist())
        ven.close()
        remaining -= batch

    return {
        'returns': np.array(ep_returns, dtype=np.float32),
        'lengths': np.array(ep_lengths, dtype=np.int32),
    }


def create_experiment_config(
    algorithm: str,
    env_config: Dict[str, Any],
    hyperparams: Dict[str, Any],
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Create a standardized experiment configuration record for logging and reproducibility.
    """
    return {
        'algorithm': algorithm,
        'env_config': dict(env_config),
        'hyperparams': dict(hyperparams),
        'seed': int(seed),
        'timestamp': pd.Timestamp.now().isoformat(),
    }


def run_training_with_monitoring(
    train_fn: Callable[..., Iterable[Dict[str, Any]]],
    config: Dict[str, Any],
    tag: str,
    checkpoint_freq: int = 10000,
    eval_freq: int = 5000,
) -> Tuple[Any, pd.DataFrame]:
    """
    Generic wrapper for training loops that yield progress dicts.

    Expectations for train_fn:
    - Returns an iterator/generator of records (dicts) containing at least keys among
      {'timestep','episode','reward','episode_length'}; extra keys are added to the log.
    - Optionally includes 'model' in records, which will be checkpointed at checkpoint_freq.

    Optional evaluation:
    - If config contains 'policy_fn' and 'make_env_fn', we run evaluate_policy(policy_fn, make_env_fn)
      every eval_freq timesteps and append 'eval_return' and 'eval_cost' to the log at that point.
    """
    logger = create_training_logger(tag, metrics=[])
    dynamic_keys: set[str] = set(logger.keys())
    final_model: Any = None

    policy_fn = config.get('policy_fn')
    make_env_fn = config.get('make_env_fn')

    for record in train_fn(config):
        if not isinstance(record, dict):
            continue

        # Merge any new keys into logger
        for k, v in record.items():
            if k == 'model':
                final_model = v
                continue
            if k not in dynamic_keys:
                logger[k] = []
                dynamic_keys.add(k)
            logger[k].append(v)

        # Backfill required keys in case missing in this record
        for req in ['episode', 'timestep', 'reward', 'episode_length']:
            if req not in record:
                if req not in logger:
                    logger[req] = []
                logger[req].append(np.nan)

        # Checkpointing
        ts = int(record.get('timestep', -1))
        if final_model is not None and checkpoint_freq > 0 and ts >= 0 and (ts % checkpoint_freq == 0):
            try:
                save_model_checkpoint(final_model, tag, ts)
            except Exception:
                pass

        # Periodic evaluation
        if (
            policy_fn is not None and make_env_fn is not None and eval_freq > 0 and ts >= 0 and (ts % eval_freq == 0)
        ):
            try:
                m, s, c, _ = evaluate_policy(policy_fn, make_env_fn, episodes=1)
                for k in ['eval_return', 'eval_return_std', 'eval_cost']:
                    if k not in logger:
                        logger[k] = []
                logger['eval_return'].append(m)
                logger['eval_return_std'].append(s)
                logger['eval_cost'].append(c)
            except Exception:
                # Skip evaluation errors to keep training running
                pass

    # Normalize list lengths (ensure all keys have same length)
    max_len = max((len(v) for v in logger.values()), default=0)
    for k, v in list(logger.items()):
        if len(v) < max_len:
            logger[k] = v + [np.nan] * (max_len - len(v))

    results_df = pd.DataFrame(logger)
    return final_model, results_df


class ExperimentTracker:
    """
    Track multiple experiments for comparison with minimal ceremony.
    Stores in-memory references to configs and results; does not enforce on-disk I/O.
    """

    def __init__(self, base_dir: str | None = None):
        self.base_dir = base_dir or get_cache_dir('rl')
        self.experiments: Dict[str, Dict[str, Any]] = {}

    def add_experiment(self, tag: str, config: Dict[str, Any], results: pd.DataFrame) -> None:
        self.experiments[tag] = {
            'config': config,
            'results': results,
            'path': os.path.join(self.base_dir, f"{tag}_experiment"),
        }

    def compare_experiments(self, tags: list[str], metric: str = 'reward') -> None:
        """
        Simple comparison plot for a metric across selected experiments.
        Expects the metric to be a column in each results DataFrame.
        """
        plt.figure(figsize=(8, 5))
        for tag in tags:
            exp = self.experiments.get(tag)
            if not exp:
                continue
            df = exp['results']
            y = df[metric].to_numpy() if metric in df.columns else np.array([])
            x = df['episode'].to_numpy() if 'episode' in df.columns else np.arange(len(y))
            if len(y) == 0:
                continue
            plt.plot(x, y, label=tag)
        plt.title(f"Experiment comparison — {metric}")
        plt.xlabel('episode' if any('episode' in exp['results'].columns for exp in self.experiments.values()) else 'index')
        plt.ylabel(metric)
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()


def run_scenario_sweep(
    algorithms: Dict[str, Any],
    env_factory: Callable,
    scenarios: Dict[str, Dict[str, Any]],
    metrics_to_track: list[str] = ['satisfaction', 'components'],
    episodes_per_scenario: int = 1,
) -> pd.DataFrame:
    """
    Run multiple algorithms across multiple scenarios and return a tidy DataFrame.

    Parameters
    ----------
    algorithms : Dict[str, Any]
        Mapping from algorithm name to either a callable policy_fn(obs)->action or an object
        exposing one of {get_action, predict, act}.
    env_factory : Callable
        Factory to create an environment. It should accept the scenario's env_kwargs.
        If it doesn't accept noise/noise_action, they are ignored gracefully.
    scenarios : Dict[str, Dict[str, Any]]
        Mapping from scenario name to a config dict. Supports optional key 'env_kwargs'.
    metrics_to_track : list[str]
        Which metrics to track via evaluate_policy (e.g., ['satisfaction','components']).
    episodes_per_scenario : int
        Number of episodes to evaluate per algorithm per scenario.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns including ['scenario','algorithm','mean_return','std_return','mean_cost', ...].
        Additional metric means from evaluate_policy with suffix '_mean' are included when present.
    """

    # Use shared policy wrapper from utils_common

    def _make_env_fn_from(env_kwargs: Dict[str, Any]) -> Callable[..., object]:
        def _fn(noise: Optional[float] = None, noise_action: Optional[float] = None):
            # Try passing noise/noise_action if env_factory supports them, otherwise ignore.
            try:
                return env_factory(**{**env_kwargs, 'noise': noise, 'noise_action': noise_action})
            except TypeError:
                return env_factory(**env_kwargs)
        return _fn

    results = []
    for scenario_name, scenario_config in scenarios.items():
        env_kwargs = scenario_config.get('env_kwargs', {})
        make_env_fn = _make_env_fn_from(env_kwargs)

        for algo_name, algo_impl in algorithms.items():
            policy_fn = wrap_policy(algo_impl)

            mean_ret, std_ret, mean_cost, metrics = evaluate_policy(
                policy_fn, make_env_fn,
                episodes=episodes_per_scenario,
                track_metrics=metrics_to_track,
            )

            row = {
                'scenario': scenario_name,
                'algorithm': algo_name,
                'mean_return': mean_ret,
                'std_return': std_ret,
                'mean_cost': mean_cost,
            }
            # Merge in any *_mean metrics collected
            for k, v in metrics.items():
                if isinstance(k, str) and k.endswith('_mean'):
                    row[k] = v
            results.append(row)

    return pd.DataFrame(results)

# -----------------------------
# Multi-Agent evaluation and sweeps
# -----------------------------

def evaluate_multi_agent_policy(
    policy_fn: Callable[[Dict[str, np.ndarray]], Dict[str, np.ndarray]],
    make_env_fn: Callable[..., Any],
    episodes: int = 2,
    horizon: int = 288,
    seeds: Optional[List[int]] = None,
    track_metrics: Optional[list[str]] = None,
    verbose: bool = False,
) -> Tuple[float, float, float, Dict[str, Any]]:
    """
    Evaluate a multi-agent policy on a PettingZoo-style ParallelEnv.

    Parameters
    ----------
    policy_fn : Callable
        Maps Dict[agent_id, obs] -> Dict[agent_id, action]
    make_env_fn : Callable
        Factory to construct a fresh multi-agent environment per episode
    episodes : int
        Number of episodes to evaluate
    horizon : int
        Max steps per episode
    seeds : Optional[list[int]]
        If provided, use these seeds per episode (index-aligned). If fewer than
        `episodes`, remaining episodes fall back to 0..N-1. Defaults to None.
    track_metrics : list[str]
        Supported: ['fairness', 'coordination', 'per_agent_rewards', 'violations']
    verbose : bool
        Print progress
    """
    if track_metrics is None:
        track_metrics = []

    ep_returns: list[float] = []
    ep_costs: list[float] = []
    metrics: Dict[str, list[Any]] = defaultdict(list)

    for ep in range(int(episodes)):
        if verbose:
            log_info(f"Evaluating MA episode {ep+1}/{episodes}")

        env = make_env_fn()
        seed_to_use = seeds[ep] if (seeds is not None and ep < len(seeds)) else ep
        obs_dict, info_dict = env.reset(seed=seed_to_use)

        # Fixed agent list determined at reset for logging
        agent_ids = list(obs_dict.keys())

        # Per-episode trackers
        agent_rewards: Dict[str, float] = defaultdict(float)
        agent_costs: Dict[str, float] = defaultdict(float)
        last_cum_cost: Dict[str, float] = defaultdict(float)
        # Collect actions per agent for coordination
        actions_over_time: list[list[float]] = [[] for _ in agent_ids]  # index-aligned with agent_ids

        done = False
        t = 0
        try:
            while not done and t < horizon:
                # Policy actions
                actions = policy_fn(obs_dict)

                # Scalarize actions robustly and store for coordination
                for idx, aid in enumerate(agent_ids):
                    a = np.asarray(actions.get(aid, 0.0)).reshape(-1)
                    a_scalar = float(a.mean()) if a.size else 0.0
                    actions_over_time[idx].append(a_scalar)

                obs_next, rewards, terms, truncs, infos = env.step(actions)

                # Aggregate rewards and costs using returned dict keys (not env.agents)
                for aid in rewards.keys():
                    agent_rewards[aid] += float(rewards.get(aid, 0.0))
                    rb = infos.get(aid, {}).get('reward_breakdown', {})
                    cum = float(rb.get('excess_charge', 0.0))
                    inc = max(0.0, cum - last_cum_cost.get(aid, 0.0))
                    agent_costs[aid] += inc
                    last_cum_cost[aid] = cum

                # Termination when all agents done
                done = (len(terms) > 0 and all(bool(v) for v in terms.values())) or \
                       (len(truncs) > 0 and all(bool(v) for v in truncs.values()))
                obs_dict = obs_next
                t += 1
        finally:
            try:
                env.close()
            except Exception:
                pass

        # Episode aggregations
        mean_ep_return = float(np.mean(list(agent_rewards.values()))) if agent_rewards else 0.0
        mean_ep_cost = float(np.mean(list(agent_costs.values()))) if agent_costs else 0.0
        ep_returns.append(mean_ep_return)
        ep_costs.append(mean_ep_cost)

        # Additional metrics
        if 'fairness' in track_metrics:
            rvals = np.array(list(agent_rewards.values()), dtype=float)
            if rvals.size > 0:
                total = float(np.sum(rvals))
                if total <= 1e-12:
                    fairness = 1.0
                else:
                    sorted_rvals = np.sort(rvals)
                    n = len(sorted_rvals)
                    index = np.arange(1, n + 1, dtype=float)
                    gini = (2.0 * float(np.sum(index * sorted_rvals))) / (n * total) - (n + 1.0) / n
                    fairness = 1.0 - float(gini)
            else:
                fairness = 1.0
            metrics['fairness'].append(fairness)

        if 'coordination' in track_metrics:
            # Build [T, N] action matrix and compute mean |corr| off-diagonal
            T = min(len(seq) for seq in actions_over_time) if actions_over_time else 0
            if T > 1 and len(actions_over_time) > 1:
                A = np.stack([np.array(seq[:T], dtype=float) for seq in actions_over_time], axis=1)  # [T, N]
                try:
                    C = np.corrcoef(A, rowvar=False)
                    tri = np.triu_indices_from(C, k=1)
                    coord = float(np.mean(np.abs(C[tri]))) if len(tri[0]) > 0 else 0.0
                except Exception:
                    coord = 0.0
            else:
                coord = 0.0
            metrics['coordination'].append(coord)

        if 'action_fairness' in track_metrics:
            # Episode-level fairness based on per-agent total action
            L = min(len(seq) for seq in actions_over_time) if actions_over_time else 0
            if L > 0:
                totals = np.array([float(np.sum(seq[:L])) for seq in actions_over_time], dtype=float)
                std_totals = float(np.std(totals)) if totals.size else 0.0
                a_fair = 1.0 / (1.0 + std_totals)
            else:
                a_fair = 1.0
            metrics['action_fairness'].append(a_fair)

        if 'per_agent_rewards' in track_metrics:
            metrics['per_agent_rewards'].append(dict(agent_rewards))

        if 'violations' in track_metrics:
            metrics['violations'].append(mean_ep_cost)

    # Aggregate metrics across episodes
    aggregated: Dict[str, Any] = {}
    for k, values in metrics.items():
        if k in ['per_agent_rewards']:
            aggregated[k] = values
        else:
            try:
                aggregated[f"{k}_mean"] = float(np.mean(values))
                aggregated[f"{k}_std"] = float(np.std(values))
            except Exception:
                pass

    return float(np.mean(ep_returns)), float(np.std(ep_returns)), float(np.mean(ep_costs)), aggregated


def collect_ma_trajectory(
    make_env_fn: Callable[..., Any],
    policy_fn: Callable[[Dict[str, np.ndarray]], Dict[str, np.ndarray]],
    horizon: int = 288,
    seed: Optional[int] = None,
    track_full_info: bool = False,
) -> Dict[str, Any]:
    """
    Roll out a single multi-agent episode and collect per-agent trajectories.
    Returns a dict with keys: 'agents', 'global_rewards', 'episode_length', optional 'infos'.
    """
    env = make_env_fn()
    try:
        obs_dict, info_dict = env.reset(seed=seed)
        agent_ids = list(obs_dict.keys())
        traj: Dict[str, Any] = {
            'agents': defaultdict(list),
            'global_rewards': [],
            'episode_length': 0,
        }
        if track_full_info:
            traj['infos'] = []

        for t in range(int(horizon)):
            actions = policy_fn(obs_dict)
            obs_next, rewards, terms, truncs, infos = env.step(actions)

            # Per-agent records (use fixed agent_ids to keep alignment)
            for aid in agent_ids:
                a = np.asarray(actions.get(aid, 0.0)).reshape(-1)
                a_scalar = float(a.mean()) if a.size else 0.0
                traj['agents'][aid].append({
                    't': t,
                    'obs': obs_dict.get(aid, np.array([])).copy(),
                    'action': a_scalar,
                    'reward': float(rewards.get(aid, 0.0)),
                    'done': bool(terms.get(aid, False) or truncs.get(aid, False)),
                })

            # Global reward as mean per-agent reward
            if len(rewards) > 0:
                traj['global_rewards'].append(float(np.mean(list(rewards.values()))))
            else:
                traj['global_rewards'].append(0.0)

            if track_full_info:
                traj['infos'].append(dict(infos))

            traj['episode_length'] = t + 1
            done = (len(terms) > 0 and all(bool(v) for v in terms.values())) or \
                   (len(truncs) > 0 and all(bool(v) for v in truncs.values()))
            if done:
                break
            obs_dict = obs_next
        return traj
    finally:
        try:
            env.close()
        except Exception:
            pass


def run_multiagent_scenario_sweep(
    algorithms: Dict[str, Any],
    ma_env_factory: Callable[..., Any],
    scenarios: Dict[str, Dict[str, Any]],
    metrics_to_track: list[str] = ['fairness', 'coordination'],
    episodes_per_scenario: int = 1,
    horizon: int = 288,
    seeds: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Run multiple multi-agent policies across scenarios and return a tidy DataFrame.
    Columns include ['scenario','algorithm','mean_return','std_return','mean_cost', ... plus *_mean metrics].
    """

    # Use shared multi-agent policy wrapper from utils_common

    def _make_env_fn_from(env_kwargs: Dict[str, Any]) -> Callable[..., Any]:
        def _fn():
            return ma_env_factory(**env_kwargs)
        return _fn

    rows: list[Dict[str, Any]] = []
    for scen_name, scen_cfg in scenarios.items():
        env_kwargs = dict(scen_cfg.get('env_kwargs', {}))
        make_env_fn = _make_env_fn_from(env_kwargs)

        for algo_name, algo_impl in algorithms.items():
            policy_fn = wrap_ma_policy(algo_impl)

            mean_ret, std_ret, mean_cost, metrics = evaluate_multi_agent_policy(
                policy_fn=policy_fn,
                make_env_fn=make_env_fn,
                episodes=episodes_per_scenario,
                horizon=horizon,
                seeds=seeds,
                track_metrics=metrics_to_track,
            )

            row: Dict[str, Any] = {
                'scenario': scen_name,
                'algorithm': algo_name,
                'mean_return': mean_ret,
                'std_return': std_ret,
                'mean_cost': mean_cost,
            }
            for k, v in metrics.items():
                if isinstance(k, str) and k.endswith('_mean'):
                    row[k] = v
            rows.append(row)

    return pd.DataFrame(rows)


def save_ma_models(
    models: Dict[str, Any],
    tag: str,
    epoch: int,
    kind: str = "marl",
) -> str:
    """Save multi-agent model checkpoints to a pickle file."""
    import pickle

    cache_dir = get_cache_dir(kind)
    path = os.path.join(cache_dir, f"{tag}_ma_models_epoch_{epoch}.pkl")

    save_dict: Dict[str, Any] = {}
    for key, model in models.items():
        try:
            if hasattr(model, 'state_dict'):
                save_dict[key] = model.state_dict()
            else:
                save_dict[key] = model
        except Exception as e:
            raise RuntimeError(f"Failed to serialize model for key '{key}': {e}")

    with open(path, 'wb') as f:
        pickle.dump(save_dict, f)

    return path


def load_ma_models(
    model_class: Any,
    tag: str,
    epoch: int,
    kind: str = "marl",
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Load multi-agent model checkpoints saved by save_ma_models."""
    import pickle

    cache_dir = get_cache_dir(kind)
    path = os.path.join(cache_dir, f"{tag}_ma_models_epoch_{epoch}.pkl")

    with open(path, 'rb') as f:
        state_dicts = pickle.load(f)

    model_kwargs = model_kwargs or {}
    models: Dict[str, Any] = {}
    for key, state in state_dicts.items():
        model = model_class(**model_kwargs)
        if hasattr(model, 'load_state_dict'):
            model.load_state_dict(state)
            models[key] = model
        else:
            # If not a torch model, return the raw object
            models[key] = state
    return models


def plot_ma_training_comparison(
    training_dfs: Dict[str, pd.DataFrame],
    metrics: list[str] = ['reward', 'fairness', 'coordination'],
    figsize: Tuple[int, int] = (12, 4),
) -> None:
    """Compare training curves for multiple multi-agent algorithms."""
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]

    for i, metric in enumerate(metrics):
        ax = axes[i]
        for algo_name, df in training_dfs.items():
            if metric in df.columns:
                x_col = 'timestep' if 'timestep' in df.columns else ('episode' if 'episode' in df.columns else None)
                x = df[x_col] if x_col else np.arange(len(df))
                window = max(1, min(20, (len(df) // 10) or 1))
                try:
                    smoothed = df[metric].rolling(window=window, min_periods=1).mean()
                except Exception:
                    smoothed = df[metric]
                ax.plot(x, smoothed, label=algo_name, alpha=0.8)
        ax.set_xlabel('timestep' if any(('timestep' in d.columns) for d in training_dfs.values()) else 'episode')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
        ax.grid(True, alpha=0.3)
        ax.legend()
    plt.tight_layout()
    plt.show()


def plot_agent_coordination_matrix(
    trajectory: Dict[str, Any],
    figsize: Tuple[int, int] = (10, 8),
) -> None:
    """Visualize agent coordination via correlation matrix from a MA trajectory."""
    agents = list(trajectory['agents'].keys())
    n_agents = len(agents)
    if n_agents == 0:
        log_warning("Empty trajectory: no agents")
        return

    # Extract equal-length action sequences
    seqs = [np.array([step['action'] for step in trajectory['agents'][aid]], dtype=float) for aid in agents]
    L = min((len(s) for s in seqs), default=0)
    if L < 2:
        log_warning("Not enough steps to compute coordination")
        return
    seqs = [s[:L] for s in seqs]

    A = np.stack(seqs, axis=1)  # [T, N]
    C = np.corrcoef(A, rowvar=False)

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    im = axes[0].imshow(C, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
    axes[0].set_xticks(range(n_agents)); axes[0].set_yticks(range(n_agents))
    axes[0].set_xticklabels([f'A{i}' for i in range(n_agents)], rotation=45)
    axes[0].set_yticklabels([f'A{i}' for i in range(n_agents)])
    axes[0].set_title('Agent Action Correlation')
    plt.colorbar(im, ax=axes[0])

    for i, aid in enumerate(agents[:min(5, n_agents)]):
        axes[1].plot(A[:, i], label=f'Agent {i}', alpha=0.7)
    axes[1].set_xlabel('Time Step'); axes[1].set_ylabel('Action')
    axes[1].set_title('Agent Action Trajectories')
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout(); plt.show()


def create_ma_wrapper(
    ma_env: Any,
    wrapper_type: str = 'flatten',
    **kwargs,
) -> Any:
    """Create common multi-agent environment wrappers (lightweight)."""
    if wrapper_type == 'flatten':
        class MAFlattenWrapper:
            def __init__(self, env):
                self.env = env; self.agents = env.agents
            def reset(self, **kw):
                obs, info = self.env.reset(**kw)
                return obs, info
            def step(self, actions):
                return self.env.step(actions)
            def __getattr__(self, name):
                return getattr(self.env, name)
        return MAFlattenWrapper(ma_env)

        
    if wrapper_type == 'mask_obs':
        mask_keys = kwargs.get('mask_keys', [])
        class MAObsMaskWrapper:
            def __init__(self, env, mask_keys):
                self.env = env; self.mask_keys = mask_keys; self.agents = env.agents
            def reset(self, **kw):
                obs, info = self.env.reset(**kw)
                return self._mask(obs), info
            def step(self, actions):
                obs, r, te, tr, inf = self.env.step(actions)
                return self._mask(obs), r, te, tr, inf
            def _mask(self, obs_dict):
                # Placeholder: requires dict-structure knowledge to mask precisely
                return obs_dict
            def __getattr__(self, name):
                return getattr(self.env, name)
        return MAObsMaskWrapper(ma_env, mask_keys)

    if wrapper_type == 'add_channels':
        include = kwargs.get('include', ['neighbor_actions'])
        class MAChannelWrapper:
            def __init__(self, env, include):
                self.env = env; self.include = include; self.agents = env.agents
                self.last_actions = {a: 0.0 for a in getattr(env, 'agents', [])}
            def reset(self, **kw):
                obs, info = self.env.reset(**kw)
                self.last_actions = {a: 0.0 for a in getattr(self, 'agents', [])}
                return self._augment(obs), info
            def step(self, actions):
                self.last_actions = {k: float(np.asarray(v).reshape(-1).mean()) for k, v in actions.items()}
                obs, r, te, tr, inf = self.env.step(actions)
                return self._augment(obs), r, te, tr, inf
            def _augment(self, obs_dict):
                augmented = {}
                for aid, obs in obs_dict.items():
                    arr = np.asarray(obs, dtype=float).reshape(-1)
                    if 'neighbor_actions' in self.include:
                        other = [v for k, v in self.last_actions.items() if k != aid]
                        mean_a = float(np.mean(other)) if other else 0.0
                        arr = np.concatenate([arr, np.array([mean_a], dtype=float)], axis=0)
                    augmented[aid] = arr
                return augmented
            def __getattr__(self, name):
                return getattr(self.env, name)
        return MAChannelWrapper(ma_env, include)

    raise ValueError(f"Unknown wrapper type: {wrapper_type}")

# -----------------------------
# Plotting helpers
# -----------------------------


def plot_robustness_heatmap(
    df: pd.DataFrame,
    value: str = "return_mean",
    title: str = "Robustness heatmap",
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    pt = df.pivot(index="noise", columns="noise_action", values=value)
    plt.figure(figsize=(4, 3))
    im = plt.imshow(pt.values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, label=value)
    plt.xlabel("noise_action")
    plt.ylabel("noise")
    plt.title(title)
    plt.xticks(range(len(pt.columns)), pt.columns)
    plt.yticks(range(len(pt.index)), pt.index)
    plt.tight_layout()
    plt.show()


def plot_action_saturation(
    actions: np.ndarray,
    title: str = "Action distribution",
    thresholds: Tuple[float, float] = (0.001, 0.999),
) -> Tuple[float, float]:
    """
    Simple action saturation analysis. 
    For comprehensive analysis, use plot_action_distribution instead.
    """
    # Delegate to comprehensive function
    stats = plot_action_distribution(
        actions=actions,
        title=title,
        saturation_thresholds=thresholds,
        show_stats=False,
        show_saturation=True,
        return_stats=True,
        figsize=(8, 4)
    )
    
    # Return saturation values for backward compatibility
    return stats.get('saturation_high', 0.0), stats.get('saturation_low', 0.0)


def plot_action_distribution(
    actions: Union[np.ndarray, Dict[str, np.ndarray], pd.DataFrame],
    title: str = "Action Distribution Analysis",
    bins: int = 50,
    show_stats: bool = True,
    show_saturation: bool = True,
    saturation_thresholds: Tuple[float, float] = (0.001, 0.999),
    figsize: Tuple[int, int] = (12, 5),
    return_stats: bool = True,
) -> Optional[Dict[str, float]]:
    """
    Comprehensive action distribution visualization and analysis.

    Parameters
    ----------
    actions : Union[np.ndarray, Dict[str, np.ndarray], pd.DataFrame]
        - np.ndarray: Actions array of shape [T, A] or [T]
        - Dict: Multiple action arrays with labels as keys
        - pd.DataFrame: DataFrame with 'action' column and optional 'label' column
    title : str
        Main title for the plot
    bins : int
        Number of histogram bins
    show_stats : bool
        Display statistical summary panel
    show_saturation : bool
        Include saturation analysis panel
    saturation_thresholds : Tuple[float, float]
        Thresholds for low and high saturation
    figsize : Tuple[int, int]
        Figure size
    return_stats : bool
        Whether to return statistics dictionary

    Returns
    -------
    stats : Optional[Dict[str, float]]
        Dictionary of statistics if return_stats=True
    """
    # Normalize input to dictionary format
    if isinstance(actions, np.ndarray):
        action_dict = {"actions": actions.reshape(-1)}
    elif isinstance(actions, pd.DataFrame):
        if 'label' in actions.columns:
            action_dict = {label: group['action'].values for label, group in actions.groupby('label')}
        else:
            action_dict = {"actions": actions['action'].values}
    else:
        action_dict = {k: v.reshape(-1) for k, v in actions.items()}

    # Determine subplot layout: histogram + optional saturation panel
    # Stats are printed below using show()/show_metrics() instead of a center text panel.
    n_panels = 1 + int(show_saturation)
    fig, axes = plt.subplots(1, n_panels, figsize=figsize)
    if n_panels == 1:
        axes = [axes]

    # Panel 1: Histogram + optional KDE
    ax_hist = axes[0]
    colors = plt.cm.Set2(np.linspace(0, 1, len(action_dict)))

    for (label, acts), color in zip(action_dict.items(), colors):
        ax_hist.hist(acts, bins=bins, alpha=0.6, label=label, color=color, density=True)

        # Add KDE for smooth distribution (if SciPy available)
        if len(acts) > 10:
            try:
                from scipy.stats import gaussian_kde  # type: ignore
                kde = gaussian_kde(acts)
                x_range = np.linspace(0, 1, 200)
                ax_hist.plot(x_range, kde(x_range), '-', color=color, linewidth=2, alpha=0.8)
            except Exception:
                # SciPy not available or KDE failed; skip
                pass

    ax_hist.set_xlabel('Action Value')
    ax_hist.set_ylabel('Density')
    ax_hist.set_title('Action Distribution')
    ax_hist.grid(True, alpha=0.3)
    ax_hist.set_xlim([0, 1])
    if len(action_dict) > 1:
        ax_hist.legend()

    # Collect statistics
    all_stats: Dict[str, float] = {}
    panel_idx = 1

    # Optional: Print statistical summary below the plots via show()/show_metrics()
    if show_stats:
        summary: Dict[str, Any] = {}
        for label, acts in action_dict.items():
            mean_val = float(np.mean(acts))
            std_val = float(np.std(acts))
            median_val = float(np.median(acts))
            q25, q75 = np.percentile(acts, [25, 75]) if acts.size else (float('nan'), float('nan'))

            # Aggregate to all_stats for return
            prefix = f"{label}_" if len(action_dict) > 1 else ""
            all_stats[f"{prefix}mean"] = mean_val
            all_stats[f"{prefix}std"] = std_val
            all_stats[f"{prefix}median"] = median_val
            all_stats[f"{prefix}q25"] = float(q25)
            all_stats[f"{prefix}q75"] = float(q75)
            all_stats[f"{prefix}min"] = float(np.min(acts)) if acts.size else float('nan')
            all_stats[f"{prefix}max"] = float(np.max(acts)) if acts.size else float('nan')

            # Build nested summary per label
            label_key = str(label) if len(action_dict) > 1 else 'actions'
            summary[label_key] = {
                'Mean': f"{mean_val:.3f}",
                'Std Dev': f"{std_val:.3f}",
                'Median': f"{median_val:.3f}",
                'IQR': f"[{q25:.3f}, {q75:.3f}]",
                'Range': f"[{(np.min(acts) if acts.size else np.nan):.3f}, {(np.max(acts) if acts.size else np.nan):.3f}]",
            }

        # Render below figure using notebook-style helpers
        try:
            show("section: Action distribution summary")
            show_metrics(summary)
        except Exception:
            # If show() is unavailable, skip gracefully
            pass

    # Panel 2: Saturation analysis (if requested)
    if show_saturation:
        ax_sat = axes[panel_idx]

        sat_data = []
        labels = []

        for label, acts in action_dict.items():
            low_sat = np.mean(acts <= saturation_thresholds[0]) * 100
            mid_range = np.mean((acts > saturation_thresholds[0]) & (acts < saturation_thresholds[1])) * 100
            high_sat = np.mean(acts >= saturation_thresholds[1]) * 100

            sat_data.append([low_sat, mid_range, high_sat])
            labels.append(label)

            prefix = f"{label}_" if len(action_dict) > 1 else ""
            all_stats[f"{prefix}saturation_low"] = low_sat / 100
            all_stats[f"{prefix}saturation_mid"] = mid_range / 100
            all_stats[f"{prefix}saturation_high"] = high_sat / 100

        sat_data = np.array(sat_data)
        x = np.arange(len(labels))
        width = 0.25

        bars1 = ax_sat.bar(x - width, sat_data[:, 0], width, label=f'Low (≤{saturation_thresholds[0]})',
                           color='lightcoral', alpha=0.7)
        bars2 = ax_sat.bar(x, sat_data[:, 1], width, label='Mid',
                           color='lightblue', alpha=0.7)
        bars3 = ax_sat.bar(x + width, sat_data[:, 2], width, label=f'High (≥{saturation_thresholds[1]})',
                           color='lightgreen', alpha=0.7)

        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                if height > 1:
                    ax_sat.text(bar.get_x() + bar.get_width()/2., height,
                                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

        ax_sat.set_ylabel('Percentage (%)')
        ax_sat.set_title('Action Saturation Analysis')
        ax_sat.set_xticks(x)
        ax_sat.set_xticklabels(labels)
        ax_sat.legend()
        ax_sat.grid(True, alpha=0.3, axis='y')
        ax_sat.set_ylim([0, 105])

    plt.suptitle(title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()

    if return_stats:
        return all_stats


def plot_actions_heatmap(traj_actions: Dict[str, np.ndarray]) -> None:
    agents = sorted(traj_actions)
    A = np.stack([traj_actions[a] for a in agents], axis=1)  # T × N
    plt.figure(figsize=(10, 3))
    plt.imshow(A.T, aspect="auto", cmap="viridis")
    plt.colorbar(label="pilot")
    plt.xlabel("time")
    plt.ylabel("agent (station)")
    plt.title("Per-agent actions")
    plt.show()


def plot_fairness(traj_actions: Dict[str, np.ndarray]) -> None:
    agents = sorted(traj_actions)
    S = [np.sum(traj_actions[a]) for a in agents]
    plt.figure(figsize=(8, 3))
    plt.bar(np.arange(len(agents)), S)
    plt.title("Per-station served proxy")
    plt.show()


def plot_cost_return_pareto(pareto_df: pd.DataFrame, budget_col: str = "budget") -> None:
    plt.figure(figsize=(5, 4))
    sc = plt.scatter(pareto_df["cost"], pareto_df["return"], c=pareto_df[budget_col], cmap="viridis")
    plt.colorbar(sc, label=budget_col)
    plt.xlabel("mean episode cost")
    plt.ylabel("mean return")
    plt.title("Cost–Return Pareto")
    plt.show()


def plot_dual_dynamics(dual_df: pd.DataFrame, run_id: Optional[str] = None) -> None:
    groups = [(run_id, dual_df[dual_df["run_id"] == run_id])] if run_id is not None else dual_df.groupby("run_id")
    for rid, df in groups:
        fig, ax1 = plt.subplots(figsize=(6, 3))
        ax1.plot(df["update"], df["lambda"], label="lambda", color="tab:blue")
        ax1.set_xlabel("update"); ax1.set_ylabel("lambda", color="tab:blue")
        ax2 = ax1.twinx()
        ax2.plot(df["update"], df["mean_ep_cost"], label="mean_ep_cost", color="tab:red")
        if "budget" in df.columns:
            ax2.axhline(df["budget"].iloc[0], color="tab:gray", linestyle="--", label="budget")
        ax2.set_ylabel("mean_ep_cost", color="tab:red")
        fig.suptitle(f"Dual dynamics (run {rid})")
        fig.tight_layout(); plt.show()


def plot_violation_heatmap(viol_df: pd.DataFrame) -> None:
    """
    Expects columns: ['t','excess_charge','run_id'].
    Note: 'excess_charge' should be per-step (not cumulative). If you have cumulative
    series, convert to per-step (e.g., with np.diff) before plotting.
    """
    pv = viol_df.pivot_table(index="run_id", columns="t", values="excess_charge", aggfunc="mean")
    plt.figure(figsize=(8, 3))
    im = plt.imshow(pv.values, aspect="auto", cmap="Reds")
    plt.colorbar(im, label="Violation cost ($)")
    # Discrete y-axis: one row per algorithm name
    plt.yticks(range(len(pv.index)), list(pv.index))
    # Time ticks: show a manageable number (about 24 across the day)
    n_cols = len(pv.columns)
    if n_cols > 0:
        step = max(1, n_cols // 24)
        tick_positions = list(range(0, n_cols, step))
        tick_labels = [pv.columns[i] for i in tick_positions]
        plt.xticks(tick_positions, tick_labels)
    plt.xlabel("Time (5-min steps)")
    plt.ylabel("Algorithm")
    plt.title("Violation Cost Heatmap")
    plt.tight_layout()
    plt.show()


def plot_training_curves(
    data,
    x: str = "episode",
    y: str = "reward",
    label_col: Optional[str] = None,
    seed_col: Optional[str] = None,
    smoothing_window: int = 200,
    band: str = "std",  # one of {"none", "std", "ci", "quantile"}
    ci: float = 0.95,
    quantiles: Tuple[float, float] = (0.1, 0.9),
    normalize: bool = False,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    ax: Optional[plt.Axes] = None,
    fill_alpha: float = 0.3,
    return_ax: bool = False,
) -> Optional[plt.Axes]:
    """
    Plot one or more training curves with smoothing and variability bands using
    the unified timeseries DataFrames saved by tutorials (e.g., via save_timeseries).

    Parameters
    ----------
    data : Union[pd.DataFrame, Dict[str, pd.DataFrame], Iterable]
        - pd.DataFrame: expects columns [x, y] and optionally [label_col, seed_col].
        - Dict[label, DataFrame]: each DataFrame is one run for that label.
        - Iterable: of DataFrames or (label, DataFrame) tuples.
    x : str
        Column name to use for x-axis (default "episode").
    y : str
        Column name to use for y-axis (default "reward").
    label_col : Optional[str]
        Column name containing labels (algorithm/setting). If None, labels are
        inferred from dict keys or default to a single label.
    seed_col : Optional[str]
        Column identifying different runs/seeds. If provided, each seed is treated
        as an independent run under the same label for aggregation.
    smoothing_window : int
        Rolling window size used to smooth each run before aggregation.
    band : str
        Variability band type: "none", "std" (±1 std), "ci" (mean ± z*std/sqrt(n)),
        or "quantile" (between specified quantiles across runs per x).
    ci : float
        Confidence level for CI band (default 0.95). Used only if band == "ci".
    quantiles : Tuple[float, float]
        Low/high quantiles for the quantile band (default (0.1, 0.9)). Used only if band == "quantile".
    normalize : bool
        If True, normalize y per run to [0,1] before smoothing (min-max per run).
    title, xlabel, ylabel : Optional[str]
        Title and axis labels. If None, defaults to x/y names when appropriate.
    ax : Optional[plt.Axes]
        Existing axes to plot on. If None, creates a new figure/axes.
    fill_alpha : float
        Alpha for the shaded variability band.

    Returns
    -------
    Optional[plt.Axes]
        The axes the curves were plotted on if return_ax=True; otherwise None.
    """

    def _z_for_ci(alpha: float) -> float:
        # Simple mapping to avoid SciPy dependency
        mapping = {
            0.80: 1.2816,
            0.90: 1.6449,
            0.95: 1.96,
            0.98: 2.3263,
            0.99: 2.5758,
        }
        # find nearest key
        key = min(mapping.keys(), key=lambda k: abs(k - alpha))
        return mapping[key]

    # Normalize input into {label: [run_df, ...]}
    runs_by_label: Dict[str, list[pd.DataFrame]] = defaultdict(list)

    if isinstance(data, dict):
        for lbl, df in data.items():
            runs_by_label[str(lbl)].append(df.copy())
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
        if label_col is not None and label_col in df.columns:
            if seed_col is not None and seed_col in df.columns:
                for (lbl, _seed), g in df.groupby([label_col, seed_col]):
                    runs_by_label[str(lbl)].append(g.copy())
            else:
                for lbl, g in df.groupby(label_col):
                    runs_by_label[str(lbl)].append(g.copy())
        else:
            runs_by_label["run"].append(df)
    else:
        try:
            for item in data:  # type: ignore[assignment]
                if isinstance(item, tuple) and len(item) == 2:
                    lbl, df = item  # type: ignore[misc]
                    runs_by_label[str(lbl)].append(df.copy())
                elif isinstance(item, pd.DataFrame):
                    runs_by_label["run"].append(item.copy())
        except TypeError:
            raise ValueError("Unsupported 'data' type for plot_training_curves().")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))

    # Aggregate per label across runs
    for lbl, runs in runs_by_label.items():
        # Collect per-x lists of smoothed y across runs
        per_x_values: Dict[float, list[float]] = defaultdict(list)

        for run_df in runs:
            if x not in run_df.columns or y not in run_df.columns:
                raise KeyError(f"Required columns '{x}' and '{y}' not found in a run DataFrame.")

            run = run_df[[x, y]].dropna().sort_values(by=x).reset_index(drop=True)

            # Normalize per run if requested
            if normalize:
                y_min = float(run[y].min())
                y_max = float(run[y].max())
                denom = y_max - y_min
                if denom > 0:
                    run[y] = (run[y] - y_min) / denom

            y_smooth = run[y].rolling(window=max(1, int(smoothing_window)), min_periods=1).mean()

            xs = run[x].to_numpy()
            ys = y_smooth.to_numpy()
            for xv, yv in zip(xs, ys):
                per_x_values[float(xv)].append(float(yv))

        if not per_x_values:
            continue

        xs_sorted = np.array(sorted(per_x_values.keys()))
        means = np.array([np.mean(per_x_values[xv]) for xv in xs_sorted])

        lower = upper = None
        band = band.lower() if isinstance(band, str) else "none"
        if band == "std":
            stds = np.array([np.std(per_x_values[xv]) for xv in xs_sorted])
            lower = means - stds
            upper = means + stds
        elif band == "ci":
            z = _z_for_ci(ci)
            lowers, uppers = [], []
            for xv in xs_sorted:
                vals = per_x_values[xv]
                n = max(1, len(vals))
                s = np.std(vals)
                mu = np.mean(vals)
                delta = z * s / np.sqrt(n)
                lowers.append(mu - delta)
                uppers.append(mu + delta)
            lower = np.array(lowers)
            upper = np.array(uppers)
        elif band == "quantile":
            ql, qh = quantiles
            lower = np.array([np.quantile(per_x_values[xv], ql) for xv in xs_sorted])
            upper = np.array([np.quantile(per_x_values[xv], qh) for xv in xs_sorted])

        ax.plot(xs_sorted, means, label=str(lbl))
        if lower is not None and upper is not None:
            ax.fill_between(xs_sorted, lower, upper, alpha=fill_alpha)

    if title:
        ax.set_title(title)
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()
    return ax if return_ax else None


def plot_critical_timeline(
    trajectory: Dict[str, np.ndarray] | list[Dict[str, Any]],
     env,
    critical_windows: Optional[list] = None,
    title: str = "Critical Moment Timeline",
    topk: int = 3,
    window: int = 12,
) -> None:
    """
    Create a reusable timeline visualization of key signals with optional critical window highlights.

    Accepts either a dict of arrays (e.g., {'t':..., 'reward':..., 'action':...}) or a list of per-step dicts
    as produced by evaluate_policy when 'trajectory' is tracked.

    See also
    --------
    tutorials.utils_diagnostics.interactive_critical_timeline : Interactive Plotly version with widgets.
    """
    # Normalize trajectory structure
    if isinstance(trajectory, dict):
        T = len(next(iter(trajectory.values()))) if trajectory else 0
        t = np.arange(T) if 't' not in trajectory else np.asarray(trajectory['t']).reshape(-1)
        rewards = np.asarray(trajectory.get('reward', np.zeros(T))).reshape(-1)
        actions = trajectory.get('action')
    else:
        steps = trajectory  # list of dicts
        T = len(steps)
        t = np.arange(T)
        rewards = np.array([s.get('reward', 0.0) for s in steps], dtype=float)
        actions = np.array([np.asarray(s.get('action', 0.0)).reshape(-1) for s in steps], dtype=float)

    # Reduce actions to a single trace if multi-dimensional
    if actions is None:
        action_trace = np.zeros(T)
    else:
        A = np.asarray(actions)
        if A.ndim == 1:
            action_trace = A
        else:
            action_trace = np.mean(A, axis=-1)

    # Try to compute min slack over constraints if env exposes simulator internals
    min_slack_series: Optional[np.ndarray] = None
    try:
        sim = getattr(env, '_simulator', None)
        if sim is not None and hasattr(sim, 'charging_rates') and hasattr(sim, 'network'):
            rates = np.asarray(sim.charging_rates)  # N × T
            T_env = rates.shape[1]
            mins = []
            for ti in range(min(T, T_env)):
                amps = rates[:, ti]
                cur = sim.network.constraint_current(amps)
                mag = np.abs(cur)
                slack = sim.network.magnitudes - mag
                mins.append(np.min(slack))
            if mins:
                min_slack_series = np.array(mins)
    except Exception:
        min_slack_series = None

    # If critical windows not provided, attempt to infer from slack minima
    if critical_windows is None and min_slack_series is not None and len(min_slack_series) > 0:
        # Simple non-overlapping selection by lowest slack
        score = -min_slack_series
        idx = np.argsort(-score)
        used = np.zeros_like(score, dtype=bool)
        critical_windows = []
        for i in idx:
            L = max(0, int(i - window // 2))
            R = min(len(score), int(i + window // 2))
            if not used[L:R].any():
                critical_windows.append((L, R))
                used[L:R] = True
            if len(critical_windows) >= topk:
                break

    # Build the figure
    n_rows = 2 + (1 if min_slack_series is not None else 0)
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 6), sharex=True)
    if n_rows == 1:
        axes = [axes]

    ax = axes[0]
    ax.plot(t, rewards, label='reward', color='tab:blue')
    ax.set_ylabel('reward')
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    ax2.plot(t, action_trace, label='action (avg)', color='tab:orange')
    ax2.set_ylabel('action')
    ax2.grid(True, alpha=0.3)

    if min_slack_series is not None:
        ax3 = axes[2]
        x_sl = np.arange(len(min_slack_series))
        ax3.plot(x_sl, min_slack_series, label='min slack', color='tab:red')
        ax3.axhline(0.0, color='k', linestyle='--', alpha=0.5)
        ax3.set_ylabel('min slack (A)')
        ax3.grid(True, alpha=0.3)

    # Highlight critical windows
    if critical_windows is not None:
        for (L, R) in critical_windows:
            for axi in axes:
                axi.axvspan(L, R, color='gold', alpha=0.2)

    axes[-1].set_xlabel('time')
    fig.suptitle(title)
    fig.tight_layout()
    plt.show()


def plot_algorithm_comparison_matrix(
    results_df: pd.DataFrame,
    metrics: list[str] = ['reward', 'violations', 'satisfaction'],
    algorithms: Optional[list[str]] = None,
    scenarios: Optional[list[str]] = None,
    figsize: Tuple[int, int] = (15, 10),
) -> None:
    """
    Create a matrix of comparison plots for multiple metrics across algorithms and scenarios.

    The function maps common metric nicknames to the columns produced by run_scenario_sweep.
    """
    if results_df is None or len(results_df) == 0:
        log_warning("Empty results DataFrame; nothing to plot.")
        return

    # Determine algorithms and scenarios
    if algorithms is None:
        algorithms = sorted(results_df['algorithm'].unique().tolist())
    if scenarios is None:
        scenarios = sorted(results_df['scenario'].unique().tolist())

    # Map friendly metric names to DataFrame columns
    metric_map = {
        'reward': 'mean_return',
        'return': 'mean_return',
        'violations': 'mean_cost',
        'cost': 'mean_cost',
        'satisfaction': 'satisfaction_mean',
        'profit': 'profit_mean',
        'carbon': 'carbon_cost_mean',
        'fairness': 'fairness_mean',
        'coordination': 'coordination_mean',
        'action_fairness': 'action_fairness_mean',
    }
    resolved_metrics = [metric_map.get(m, m) for m in metrics]

    n_rows = len(resolved_metrics)
    n_cols = len(scenarios)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, squeeze=False)

    for r, metric_col in enumerate(resolved_metrics):
        for c, scen in enumerate(scenarios):
            ax = axes[r][c]
            df_sc = results_df[results_df['scenario'] == scen]
            vals = []
            errs = []
            labels = []
            for algo in algorithms:
                dfa = df_sc[df_sc['algorithm'] == algo]
                if metric_col in dfa.columns:
                    val = float(dfa[metric_col].mean())
                    # Provide std bars for return if available
                    if metric_col == 'mean_return' and 'std_return' in dfa.columns:
                        err = float(dfa['std_return'].mean())
                    else:
                        err = 0.0
                else:
                    val, err = np.nan, 0.0
                vals.append(val)
                errs.append(err)
                labels.append(algo)

            x = np.arange(len(labels))
            ax.bar(x, vals, yerr=errs, color='steelblue', alpha=0.8, capsize=3)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45, ha='right')
            ax.set_title(f"{metric_col} — {scen}")
            ax.grid(True, axis='y', alpha=0.3)

    fig.tight_layout()
    plt.show()

# -----------------------------
# Evaluation results persistence (single-agent)
# -----------------------------


def save_evaluation_results(
    tag: str,
    policy_name: str,
    mean_return: float,
    std_return: float,
    mean_cost: float,
    metrics: Dict[str, Any],
    env_config: Optional[Dict[str, Any]] = None,
    policy_config: Optional[Dict[str, Any]] = None,
    notes: Optional[str] = None,
    kind: str = "rl",
    save_full_trajectory: bool = False,
) -> str:
    """
    Save comprehensive evaluation results with metadata.

    Returns the path to the JSON evaluation file.
    """
    import json
    from datetime import datetime

    eval_data: Dict[str, Any] = {
        'tag': tag,
        'timestamp': datetime.now().isoformat(),
        'policy_name': policy_name,
        'results': {
            'mean_return': float(mean_return),
            'std_return': float(std_return),
            'mean_cost': float(mean_cost),
        }
    }

    serializable_metrics: Dict[str, Any] = {}
    trajectory_data: Dict[str, Any] = {}

    for key, value in metrics.items():
        if key in ['trajectories', 'actions_per_episode'] and not save_full_trajectory:
            serializable_metrics[f"{key}_length"] = len(value) if isinstance(value, list) else 1
        elif isinstance(value, np.ndarray):
            serializable_metrics[key] = value.tolist()
        elif isinstance(value, list):
            try:
                if len(value) > 0 and isinstance(value[0], (int, float, np.number)):
                    serializable_metrics[key] = value
                else:
                    if save_full_trajectory:
                        trajectory_data[key] = value
            except Exception:
                pass
        elif isinstance(value, (int, float, bool, str, type(None))):
            serializable_metrics[key] = value
        elif np.isscalar(value):
            serializable_metrics[key] = float(value)

    eval_data['metrics'] = serializable_metrics

    if env_config:
        eval_data['env_config'] = env_config
    if policy_config:
        eval_data['policy_config'] = policy_config
    if notes:
        eval_data['notes'] = notes

    cache_dir = get_cache_dir(kind)
    eval_path = os.path.join(cache_dir, f"{tag}_evaluation.json")

    with open(eval_path, 'w') as f:
        json.dump(eval_data, f, indent=2, default=str)

    if save_full_trajectory and trajectory_data:
        traj_path = os.path.join(cache_dir, f"{tag}_trajectory.npz")
        np_data: Dict[str, Any] = {}
        for k, v in trajectory_data.items():
            try:
                np_data[k] = np.array(v)
            except Exception:
                import pickle
                pickle_path = os.path.join(cache_dir, f"{tag}_{k}.pkl")
                with open(pickle_path, 'wb') as pf:
                    pickle.dump(v, pf)
        if np_data:
            np.savez_compressed(traj_path, **np_data)

    summary_df = pd.DataFrame([{
        'tag': tag,
        'policy': policy_name,
        'mean_return': mean_return,
        'std_return': std_return,
        'mean_cost': mean_cost,
        'satisfaction_mean': serializable_metrics.get('satisfaction_mean', np.nan),
        'profit_mean': serializable_metrics.get('profit_mean', np.nan),
        'carbon_cost_mean': serializable_metrics.get('carbon_cost_mean', np.nan),
        'action_saturation_high': serializable_metrics.get('action_saturation_high_mean', np.nan),
        'timestamp': eval_data['timestamp'],
    }])

    summary_path = os.path.join(cache_dir, f"{tag}_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    log_success(f"Evaluation saved to {eval_path}")
    if save_full_trajectory and trajectory_data:
        log_info("Trajectory data saved separately")
    log_success(f"Summary saved to {summary_path}")

    return eval_path


def load_evaluation_results(tag: str, kind: str = "rl", load_trajectory: bool = False) -> Dict[str, Any]:
    """
    Load saved evaluation results. Optionally load trajectory NPZ bundle.
    """
    import json

    cache_dir = get_cache_dir(kind)
    eval_path = os.path.join(cache_dir, f"{tag}_evaluation.json")

    with open(eval_path, 'r') as f:
        eval_data: Dict[str, Any] = json.load(f)

    if load_trajectory:
        traj_path = os.path.join(cache_dir, f"{tag}_trajectory.npz")
        if os.path.exists(traj_path):
            traj_npz = np.load(traj_path)
            eval_data['trajectory_data'] = {k: traj_npz[k] for k in traj_npz.files}

    return eval_data

# -----------------------------
# MARL helpers (persistence)
# -----------------------------


def save_marl_traj(tag: str, traj_actions: Dict[str, np.ndarray], ts: pd.DataFrame, cache_dir: Optional[str] = None) -> Tuple[str, str]:
    cache_dir = cache_dir or get_cache_dir("marl")
    ensure_dir(cache_dir)
    agents = sorted(traj_actions)
    T = len(next(iter(traj_actions.values())))
    act_df = pd.DataFrame({
        "t": np.repeat(np.arange(T), len(agents)),
        "agent": np.tile(agents, T),
        "action": np.concatenate([traj_actions[a] for a in agents])
    })
    actions_path = os.path.join(cache_dir, f"{tag}_actions.csv.gz")
    ts_path = os.path.join(cache_dir, f"{tag}_timeseries.csv.gz")
    save_df_gz(act_df, actions_path)
    save_df_gz(ts, ts_path)
    return actions_path, ts_path

# -----------------------------
# Critical windows (optional hooks)
# -----------------------------


def find_critical_windows_from_series(
    moer: np.ndarray,
    agg_demand: np.ndarray,
    slack: Optional[np.ndarray] = None,
    topk: int = 3,
    window: int = 12,
) -> Tuple[list[Tuple[int, int]], Optional[np.ndarray]]:
    """
    Lightweight scorer combining moer, aggregated demand, and (optional) slack.
    Returns non-overlapping [L,R) windows and the slack array as-is.
    """
    T = len(moer)
    if slack is None:
        slack = np.zeros(T)
    score = 0.5 * agg_demand + 0.5 * moer - 0.2 * (-slack)
    idx = np.argsort(-score)
    selected, used = [], np.zeros(T, dtype=bool)
    for i in idx:
        L = max(0, i - window // 2)
        R = min(T, i + window // 2)
        if not used[L:R].any():
            selected.append((L, R))
            used[L:R] = True
        if len(selected) >= topk:
            break
    return selected, slack

# -----------------------------
# Safe RL comparison utilities
# -----------------------------

def run_safe_rl_comparison(
    algorithms: Dict[str, Any],
    env_factory: Callable[..., Any],
    cost_limit: float,
    episodes_per_algorithm: int = 10,
    scenarios: Optional[Dict[str, Dict[str, Any]]] = None,
    track_metrics: List[str] = ['satisfaction', 'components', 'actions'],
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Compare Safe RL algorithms across one or more scenarios.

    algorithms may map names to:
      - callable policy_fn(obs)->action
      - dict with {'policy': callable, 'config': {...}}
      - objects exposing one of {get_action, predict, act}
    """
    if scenarios is None:
        scenarios = {'default': {'env_kwargs': {}}}

    def _wrap_policy(algo_impl: Any) -> Tuple[Callable[[Any], np.ndarray], Dict[str, Any]]:
        cfg: Dict[str, Any] = {}
        if callable(algo_impl):
            return (lambda obs: np.asarray(algo_impl(obs))), cfg
        if isinstance(algo_impl, dict) and 'policy' in algo_impl:
            pol = algo_impl['policy']
            cfg = dict(algo_impl.get('config', {}))
            return (lambda obs: np.asarray(pol(obs))), cfg
        if hasattr(algo_impl, 'predict'):
            return (lambda obs: np.asarray(algo_impl.predict(obs, deterministic=True)[0])), cfg
        if hasattr(algo_impl, 'get_action'):
            def _pf(obs):
                out = algo_impl.get_action(obs)
                if isinstance(out, tuple) and len(out) > 0:
                    out = out[0]
                return np.asarray(out)
            return _pf, cfg
        if hasattr(algo_impl, 'act'):
            def _pf(obs):
                out = algo_impl.act(obs)
                if isinstance(out, tuple) and len(out) > 0:
                    out = out[0]
                return np.asarray(out)
            return _pf, cfg
        raise ValueError("Unsupported algorithm interface: provide callable or object with get_action/predict/act")

    def _make_env_fn_from(env_kwargs: Dict[str, Any]) -> Callable[..., Any]:
        def _fn(noise: Optional[float] = None, noise_action: Optional[float] = None):
            try:
                return env_factory(**{**env_kwargs, 'noise': noise, 'noise_action': noise_action})
            except TypeError:
                return env_factory(**env_kwargs)
        return _fn

    rows: list[Dict[str, Any]] = []
    for algo_name, algo_impl in algorithms.items():
        if verbose:
            print(f"\nEvaluating {algo_name}...")
        policy_fn, cfg = _wrap_policy(algo_impl)

        for scen_name, scen_cfg in scenarios.items():
            env_kwargs = dict(scen_cfg.get('env_kwargs', {}))
            make_env_fn = _make_env_fn_from(env_kwargs)

            if verbose and len(scenarios) > 1:
                print(f"  Scenario: {scen_name}")

            mean_ret, std_ret, mean_cost, metrics = evaluate_policy(
                policy_fn=policy_fn,
                make_env_fn=make_env_fn,
                episodes=episodes_per_algorithm,
                track_metrics=track_metrics,
                verbose=False,
            )

            ep_costs = metrics.get('episode_costs', [mean_cost] * episodes_per_algorithm)
            ep_costs = np.asarray(ep_costs, dtype=float)
            safe_episodes = int(np.sum(ep_costs <= float(cost_limit)))
            safety_rate = float(safe_episodes) / max(1, len(ep_costs))
            sorted_costs = np.sort(ep_costs)
            tail_start = int(0.95 * len(sorted_costs)) if len(sorted_costs) > 0 else 0
            cvar_95 = float(np.mean(sorted_costs[tail_start:])) if tail_start < len(sorted_costs) else float(mean_cost)

            row: Dict[str, Any] = {
                'algorithm': algo_name,
                'scenario': scen_name,
                'mean_return': float(mean_ret),
                'std_return': float(std_ret),
                'mean_cost': float(mean_cost),
                'std_cost': float(np.std(ep_costs)) if ep_costs.size else 0.0,
                'safety_rate': safety_rate,
                'cvar_95_cost': cvar_95,
                'constraint_violation': float(mean_cost) - float(cost_limit),
                'episodes': int(episodes_per_algorithm),
            }

            # Merge *_mean metrics produced by evaluate_policy
            for k, v in metrics.items():
                if isinstance(k, str) and k.endswith('_mean'):
                    row[k] = v
            # Attach algo config if provided
            for k, v in cfg.items():
                row[k] = v
            rows.append(row)

    return pd.DataFrame(rows)


def plot_safety_metrics(
    results_df: pd.DataFrame,
    algorithms: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (15, 10),
    cost_limit: Optional[float] = None,
) -> None:
    """
    Multi-panel visualization of safety metrics for multiple algorithms.
    Panels: cost-return, safety rate, CVaR, violations, stability, composite score.
    """
    if results_df is None or len(results_df) == 0:
        print("Empty results DataFrame; nothing to plot.")
        return

    if algorithms is None:
        algorithms = results_df['algorithm'].unique().tolist()
    df = results_df[results_df['algorithm'].isin(algorithms)].copy()

    # Infer cost limit if possible
    if cost_limit is None and 'constraint_violation' in df.columns and 'mean_cost' in df.columns and not df.empty:
        sample = df.iloc[0]
        cost_limit = float(sample['mean_cost'] - sample['constraint_violation'])

    fig, axes = plt.subplots(2, 3, figsize=figsize)

    # 1) Cost-Return trade-off
    ax = axes[0, 0]
    for algo in algorithms:
        adf = df[df['algorithm'] == algo]
        if 'mean_cost' in adf.columns and 'mean_return' in adf.columns and len(adf) > 0:
            ax.scatter(adf['mean_cost'], adf['mean_return'], label=algo, s=100, alpha=0.7)
            if 'std_cost' in adf.columns and 'std_return' in adf.columns:
                try:
                    ax.errorbar(adf['mean_cost'], adf['mean_return'], xerr=adf.get('std_cost'), yerr=adf.get('std_return'), fmt='none', alpha=0.3)
                except Exception:
                    pass
    if cost_limit is not None:
        ax.axvline(x=cost_limit, color='red', linestyle='--', label='Cost Limit')
    ax.set_xlabel('Mean Cost'); ax.set_ylabel('Mean Return'); ax.set_title('Safety-Performance Trade-off')
    ax.legend(); ax.grid(True, alpha=0.3)

    # 2) Safety rate comparison
    ax = axes[0, 1]
    if 'safety_rate' in df.columns:
        safety_data = df.groupby('algorithm')['safety_rate'].mean().sort_values(ascending=False)
        bars = ax.bar(range(len(safety_data)), safety_data.values)
        ax.set_xticks(range(len(safety_data)))
        ax.set_xticklabels(safety_data.index, rotation=45, ha='right')
        ax.set_ylabel('Safety Rate'); ax.set_title('Constraint Satisfaction Rate'); ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5)
        ax.set_ylim([0, 1.05])
        for bar, rate in zip(bars, safety_data.values):
            if rate >= 0.95: bar.set_color('green')
            elif rate >= 0.8: bar.set_color('orange')
            else: bar.set_color('red')
    else:
        ax.text(0.5, 0.5, 'safety_rate not available', ha='center'); ax.axis('off')

    # 3) CVaR analysis
    ax = axes[0, 2]
    if 'cvar_95_cost' in df.columns:
        cvar_data = df.groupby('algorithm')['cvar_95_cost'].mean().sort_values()
        ax.barh(range(len(cvar_data)), cvar_data.values)
        ax.set_yticks(range(len(cvar_data))); ax.set_yticklabels(cvar_data.index)
        ax.set_xlabel('95% CVaR (Cost)'); ax.set_title('Tail Risk (Costs)')
        if cost_limit is not None:
            ax.axvline(x=cost_limit, color='red', linestyle='--')
    else:
        ax.text(0.5, 0.5, 'cvar_95_cost not available', ha='center'); ax.axis('off')

    # 4) Violation distribution across scenarios or average
    ax = axes[1, 0]
    if 'constraint_violation' in df.columns:
        if 'scenario' in df.columns and df['scenario'].nunique() > 1:
            try:
                viol = df.pivot_table(index='algorithm', columns='scenario', values='constraint_violation', aggfunc='mean')
                viol.plot(kind='bar', ax=ax)
                ax.set_ylabel('Constraint Violation'); ax.set_title('Violation Across Scenarios'); ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
                ax.legend(title='Scenario')
            except Exception:
                pass
        else:
            viol_flat = df.groupby('algorithm')['constraint_violation'].mean().sort_values()
            colors = ['green' if v <= 0 else 'red' for v in viol_flat.values]
            ax.bar(range(len(viol_flat)), viol_flat.values, color=colors)
            ax.set_xticks(range(len(viol_flat))); ax.set_xticklabels(viol_flat.index, rotation=45, ha='right')
            ax.set_ylabel('Constraint Violation'); ax.set_title('Average Constraint Violation'); ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    else:
        ax.text(0.5, 0.5, 'constraint_violation not available', ha='center'); ax.axis('off')

    # 5) Performance stability
    ax = axes[1, 1]
    if 'std_return' in df.columns:
        stability = df.groupby('algorithm').agg({'mean_return': 'mean', 'std_return': 'mean'})
        x = np.arange(len(stability)); width = 0.35
        ax.bar(x - width/2, stability['mean_return'], width, label='Mean Return', alpha=0.8)
        ax.bar(x + width/2, stability['std_return'], width, label='Std Return', alpha=0.8)
        ax.set_xticks(x); ax.set_xticklabels(stability.index, rotation=45, ha='right')
        ax.set_ylabel('Value'); ax.set_title('Performance Stability'); ax.legend()
    else:
        ax.text(0.5, 0.5, 'std_return not available', ha='center'); ax.axis('off')

    # 6) Composite score
    ax = axes[1, 2]
    required = {'mean_return', 'safety_rate', 'constraint_violation'}
    if required.issubset(df.columns):
        score_df = df.groupby('algorithm').agg({
            'mean_return': 'mean',
            'safety_rate': 'mean',
            'constraint_violation': 'mean'
        })
        # Normalize return
        ret_min, ret_max = score_df['mean_return'].min(), score_df['mean_return'].max()
        denom = (ret_max - ret_min) if (ret_max > ret_min) else 1.0
        score_df['return_score'] = (score_df['mean_return'] - ret_min) / denom
        score_df['safety_score'] = score_df['safety_rate']
        # Heuristic scaling for violation (smaller is better)
        score_df['violation_score'] = 1 - np.clip(score_df['constraint_violation'] / 10.0, 0, 1)
        score_df['composite_score'] = 0.4 * score_df['return_score'] + 0.4 * score_df['safety_score'] + 0.2 * score_df['violation_score']
        comp = score_df['composite_score'].sort_values(ascending=False)
        bars = ax.bar(range(len(comp)), comp.values)
        ax.set_xticks(range(len(comp))); ax.set_xticklabels(comp.index, rotation=45, ha='right')
        ax.set_ylabel('Composite Score'); ax.set_title('Overall Performance Score'); ax.set_ylim([0, 1])
        for bar, score in zip(bars, comp.values):
            if score >= 0.8: bar.set_color('green')
            elif score >= 0.6: bar.set_color('yellow')
            else: bar.set_color('red')
    else:
        ax.text(0.5, 0.5, 'insufficient columns for composite score', ha='center'); ax.axis('off')

    plt.tight_layout(); plt.show()


class SafeRLExperimentSuite:
    """
    Experiment runner for Safe RL algorithms across cost limits, noise levels, and seeds.
    """

    def __init__(
        self,
        env_factory: Callable[..., Any],
        cost_limits: List[float] = [15.0, 25.0, 35.0],
        noise_levels: List[float] = [0.0, 0.1, 0.2],
        seeds: List[int] = [0, 1, 2],
        base_config: Optional[Dict[str, Any]] = None,
    ):
        self.env_factory = env_factory
        self.cost_limits = cost_limits
        self.noise_levels = noise_levels
        self.seeds = seeds
        self.base_config = base_config or {}
        self.results: List[Dict[str, Any]] = []
        self.trained_models: Dict[str, Any] = {}

    def _make_env_for(self, cost_limit: float, noise: Optional[float] = None) -> Callable[..., Any]:
        def _fn():
            try:
                return self.env_factory(cost_limit=cost_limit, noise=noise)
            except TypeError:
                # Fallback if env doesn't accept noise
                return self.env_factory(cost_limit=cost_limit)
        return _fn

    def run_algorithm(
        self,
        algorithm_class: type,
        algorithm_name: str,
        training_steps: int = 100_000,
        specific_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        cfg = {**self.base_config, **(specific_config or {})}
        algo_pack: Dict[str, Any] = {'algorithm': algorithm_name, 'models': {}, 'training_curves': {}, 'evaluation': []}

        for cl in self.cost_limits:
            for seed in self.seeds:
                # Seeds
                try:
                    import torch
                    torch.manual_seed(seed)
                except Exception:
                    pass
                np.random.seed(seed)

                # Prepare default actor_critic if algorithm requires it (lazy import to avoid circulars)
                actor_critic = None
                try:
                    from .safe_rl import SafeActorCritic as _SafeActorCritic  # type: ignore
                except Exception:
                    _SafeActorCritic = None  # type: ignore
                if _SafeActorCritic is not None:
                    try:
                        probe_env = self._make_env_for(cl, None)()
                        try:
                            # Prefer declared shape if available
                            obs_shape = getattr(probe_env.observation_space, 'shape', None)
                            if obs_shape is None:
                                sample = probe_env.observation_space.sample()
                                obs_dim = int(np.array(sample, dtype=float).reshape(-1).shape[0])
                            else:
                                obs_dim = int(np.prod(obs_shape))
                            act_dim = int(np.prod(probe_env.action_space.shape))
                        finally:
                            try:
                                probe_env.close()
                            except Exception:
                                pass
                        actor_critic = _SafeActorCritic(obs_dim=obs_dim, act_dim=act_dim)
                    except Exception:
                        actor_critic = None

                # Instantiate algorithm (expects env_factory, cost_limit, experiment_tag, and possibly actor_critic)
                algo = algorithm_class(
                    env_factory=lambda: self.env_factory(cost_limit=cl),
                    actor_critic=actor_critic if actor_critic is not None else None,
                    cost_limit=cl,
                    experiment_tag=f"{algorithm_name}_cl{cl}_s{seed}",
                    **cfg,
                )

                # Train
                print(f"Training {algorithm_name} with cost_limit={cl}, seed={seed}")
                try:
                    results_df = algo.train(total_steps=training_steps)
                except TypeError:
                    # Some implementations may use 'steps' keyword
                    results_df = algo.train(steps=training_steps)

                key = f"cl{cl}_s{seed}"
                algo_pack['models'][key] = algo
                algo_pack['training_curves'][key] = results_df

                # Build a policy_fn for evaluation (lazy import to avoid circulars)
                policy_fn: Callable[[Any], np.ndarray]
                try:
                    from .safe_rl import create_safe_policy_wrapper as _create_safe_policy_wrapper  # type: ignore
                except Exception:
                    _create_safe_policy_wrapper = None  # type: ignore
                if _create_safe_policy_wrapper is not None:
                    try:
                        policy_fn = _create_safe_policy_wrapper(algo.actor_critic, deterministic=True)
                    except Exception:
                        policy_fn = None  # will fall through
                else:
                    policy_fn = None
                if policy_fn is None:
                    # Fallback using actor_critic.get_action if present
                    if hasattr(algo, 'actor_critic') and hasattr(algo.actor_critic, 'get_action'):
                        def _pf(obs):
                            out = algo.actor_critic.get_action(obs)
                            # unwrap (action, log_prob, value, cost_value)
                            if isinstance(out, tuple):
                                out = out[0]
                            try:
                                # torch.Tensor -> numpy
                                import torch
                                if isinstance(out, torch.Tensor):
                                    return out.detach().cpu().numpy().squeeze()
                            except Exception:
                                pass
                            # numpy or list -> numpy (1D)
                            return np.asarray(out).squeeze()
                        policy_fn = _pf
                    else:
                        # Last resort: assume algo itself exposes get_action
                        def _pf(obs):
                            out = algo.get_action(obs)
                            if isinstance(out, tuple):
                                out = out[0]
                            try:
                                import torch
                                if isinstance(out, torch.Tensor):
                                    return out.detach().cpu().numpy().squeeze()
                            except Exception:
                                pass
                            return np.asarray(out).squeeze()
                        policy_fn = _pf

                # Evaluate across noise levels
                for noise in self.noise_levels:
                    mr, sr, mc, metrics = evaluate_policy(
                        policy_fn=policy_fn,
                        make_env_fn=self._make_env_for(cl, noise),
                        episodes=10,
                        track_metrics=['actions'],
                    )

                    ep_costs = np.asarray(metrics.get('episode_costs', [mc] * 10), dtype=float)
                    safe_rate = float(np.mean(ep_costs <= cl)) if ep_costs.size else float(mc <= cl)
                    std_cost = float(np.std(ep_costs)) if ep_costs.size else 0.0
                    sorted_costs = np.sort(ep_costs)
                    tail_start = int(0.95 * len(sorted_costs)) if len(sorted_costs) > 0 else 0
                    cvar_95 = float(np.mean(sorted_costs[tail_start:])) if tail_start < len(sorted_costs) else float(mc)

                    algo_pack['evaluation'].append({
                        'algorithm': algorithm_name,
                        'cost_limit': cl,
                        'seed': seed,
                        'noise': noise,
                        'mean_return': float(mr),
                        'std_return': float(sr),
                        'mean_cost': float(mc),
                        'std_cost': std_cost,
                        'cvar_95_cost': cvar_95,
                        'safety_rate': safe_rate,
                        'constraint_violation': float(mc) - float(cl),
                    })

        self.trained_models[algorithm_name] = algo_pack['models']
        self.results.append(algo_pack)
        return algo_pack

    def run_all_algorithms(self, algorithms_config: Dict[str, Dict[str, Any]], training_steps: int = 100_000) -> pd.DataFrame:
        all_rows: list[Dict[str, Any]] = []
        for name, spec in algorithms_config.items():
            cls = spec['class']
            cfg = spec.get('config', {})
            res = self.run_algorithm(cls, name, training_steps=training_steps, specific_config=cfg)
            all_rows.extend(res['evaluation'])
        return pd.DataFrame(all_rows)

    def generate_report(self, save_path: Optional[str] = None, include_plots: bool = True) -> Dict[str, Any]:
        if not self.results:
            raise ValueError("No results to report. Run algorithms first.")

        eval_df = pd.DataFrame([row for pack in self.results for row in pack['evaluation']])

        report: Dict[str, Any] = {
            'summary_statistics': {},
            'rankings': {},
            'robustness_analysis': {},
            'recommendations': [],
        }

        # Summary per algorithm
        for algo in sorted(eval_df['algorithm'].unique().tolist()):
            adf = eval_df[eval_df['algorithm'] == algo]
            report['summary_statistics'][algo] = {
                'mean_return': float(adf['mean_return'].mean()),
                'std_return': float(adf['std_return'].mean()) if 'std_return' in adf.columns else np.nan,
                'mean_cost': float(adf['mean_cost'].mean()),
                'std_cost': float(adf['std_cost'].mean()) if 'std_cost' in adf.columns else np.nan,
                'safety_rate': float(adf['safety_rate'].mean()) if 'safety_rate' in adf.columns else np.nan,
                'return_range': (float(adf['mean_return'].min()), float(adf['mean_return'].max())),
                'cost_range': (float(adf['mean_cost'].min()), float(adf['mean_cost'].max())),
            }

        # Robustness score (lower variance across noise and cost limits => higher score)
        robustness_scores: Dict[str, float] = {}
        for algo in sorted(eval_df['algorithm'].unique().tolist()):
            adf = eval_df[eval_df['algorithm'] == algo]
            ret_by_cond = adf.groupby(['cost_limit', 'noise'])['mean_return'].mean()
            cost_by_cond = adf.groupby(['cost_limit', 'noise'])['mean_cost'].mean()
            r_var = float(ret_by_cond.std()) if len(ret_by_cond) > 1 else 0.0
            c_var = float(cost_by_cond.std()) if len(cost_by_cond) > 1 else 0.0
            robustness_scores[algo] = float(1 / (1 + r_var + c_var))

        # Rankings
        algo_names = list(report['summary_statistics'].keys())
        report['rankings']['return'] = sorted([(a, report['summary_statistics'][a]['mean_return']) for a in algo_names], key=lambda x: x[1], reverse=True)
        report['rankings']['safety'] = sorted([(a, report['summary_statistics'][a]['safety_rate']) for a in algo_names], key=lambda x: x[1] if not np.isnan(x[1]) else -np.inf, reverse=True)
        report['rankings']['robustness'] = sorted([(a, robustness_scores[a]) for a in algo_names], key=lambda x: x[1], reverse=True)

        overall_scores: list[Tuple[str, float]] = []
        ret_vals = [report['summary_statistics'][a]['mean_return'] for a in algo_names]
        ret_norm = {a: (report['summary_statistics'][a]['mean_return'] / (max(ret_vals) or 1.0)) for a in algo_names}
        for a in algo_names:
            s = 0.4 * ret_norm[a] + 0.4 * (report['summary_statistics'][a]['safety_rate'] if not np.isnan(report['summary_statistics'][a]['safety_rate']) else 0.0) + 0.2 * robustness_scores[a]
            overall_scores.append((a, float(s)))
        report['rankings']['overall'] = sorted(overall_scores, key=lambda x: x[1], reverse=True)

        # Robustness analysis details
        for a in algo_names:
            adf = eval_df[eval_df['algorithm'] == a]
            noise_impact = adf.groupby('noise').agg({'mean_return': 'mean', 'mean_cost': 'mean', 'safety_rate': 'mean'}).to_dict()
            def _get(d: Dict[Any, Any], key: Any, default: float = 0.0) -> float:
                try:
                    return float(d.get(key, default))
                except Exception:
                    return default
            try:
                rn0 = _get(noise_impact['mean_return'], 0.0, default=np.nan)
                rnM = _get(noise_impact['mean_return'], max(self.noise_levels), default=np.nan)
                sn0 = _get(noise_impact.get('safety_rate', {}), 0.0, default=np.nan)
                snM = _get(noise_impact.get('safety_rate', {}), max(self.noise_levels), default=np.nan)
            except Exception:
                rn0 = rnM = sn0 = snM = np.nan
            report['robustness_analysis'][a] = {
                'noise_impact': noise_impact,
                'return_degradation': float(rn0 - rnM) if (not np.isnan(rn0) and not np.isnan(rnM)) else np.nan,
                'safety_degradation': float(sn0 - snM) if (not np.isnan(sn0) and not np.isnan(snM)) else np.nan,
            }

        # Recommendations
        best_overall = report['rankings']['overall'][0][0]
        best_safety = report['rankings']['safety'][0][0]
        best_return = report['rankings']['return'][0][0]
        best_robust = report['rankings']['robustness'][0][0]
        report['recommendations'] = [
            f"Best Overall: {best_overall}",
            f"Best Safety: {best_safety}",
            f"Best Return: {best_return}",
            f"Most Robust: {best_robust}",
        ]

        if include_plots:
            try:
                plot_safety_metrics(eval_df.rename(columns={'cost_limit': 'scenario'}), algorithms=None)
            except Exception:
                pass
            try:
                self._plot_learning_curves()
            except Exception:
                pass
            try:
                self._plot_robustness_heatmap(eval_df)
            except Exception:
                pass

        if save_path:
            import json
            with open(save_path, 'w') as f:
                json.dump(self._make_json_serializable(report), f, indent=2)
            print(f"Report saved to {save_path}")

        return report

    def _plot_learning_curves(self) -> None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        titles = ['Learning Curves - Reward', 'Learning Curves - Cost', 'Dual Variable (lambda)']

        for pack in self.results:
            name = pack['algorithm']
            all_rewards, all_costs, all_lambdas = [], [], []
            for _, df in pack['training_curves'].items():
                if not isinstance(df, pd.DataFrame):
                    continue
                if 'reward' in df.columns:
                    all_rewards.append(df['reward'].to_numpy())
                if 'mean_cost' in df.columns:
                    all_costs.append(df['mean_cost'].to_numpy())
                if 'lambda' in df.columns:
                    all_lambdas.append(df['lambda'].to_numpy())

            def _plot_series(ax, series_list, label):
                if series_list:
                    arr = np.array(series_list, dtype=object)
                    # Pad to equal length by truncating to min length
                    L = min(map(len, series_list))
                    if L > 0:
                        arr2 = np.stack([s[:L] for s in series_list])
                        m = arr2.mean(axis=0); s = arr2.std(axis=0)
                        xs = np.arange(L)
                        ax.plot(xs, m, label=label)
                        ax.fill_between(xs, m - s, m + s, alpha=0.3)

            _plot_series(axes[0], all_rewards, name)
            _plot_series(axes[1], all_costs, name)
            if all_lambdas:
                L = min(map(len, all_lambdas))
                xs = np.arange(L)
                axes[2].plot(xs, np.mean(np.stack([s[:L] for s in all_lambdas]), axis=0), label=name)

        for ax, title in zip(axes, titles):
            ax.set_title(title); ax.set_xlabel('Episode'); ax.grid(True, alpha=0.3); ax.legend()
        if len(self.cost_limits) > 0:
            for limit in self.cost_limits:
                axes[1].axhline(y=limit, color='red', linestyle='--', alpha=0.4)
        plt.tight_layout(); plt.show()

    def _plot_robustness_heatmap(self, eval_df: pd.DataFrame) -> None:
        if eval_df is None or len(eval_df) == 0:
            return
        # Pivot tables
        try:
            pivot_return = eval_df.pivot_table(index='algorithm', columns=['cost_limit', 'noise'], values='mean_return', aggfunc='mean')
            pivot_safety = eval_df.pivot_table(index='algorithm', columns=['cost_limit', 'noise'], values='safety_rate', aggfunc='mean')
        except Exception:
            return
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        if sns is not None:
            sns.heatmap(pivot_return, annot=True, fmt='.1f', cmap='viridis', ax=axes[0], cbar_kws={'label': 'Mean Return'})
            axes[0].set_title('Performance Across Conditions'); axes[0].set_xlabel('(Cost Limit, Noise)')
            sns.heatmap(pivot_safety, annot=True, fmt='.2f', cmap='RdYlGn', ax=axes[1], cbar_kws={'label': 'Safety Rate'}, vmin=0, vmax=1)
            axes[1].set_title('Safety Rate Across Conditions'); axes[1].set_xlabel('(Cost Limit, Noise)')
        else:
            im0 = axes[0].imshow(pivot_return.values, aspect='auto', cmap='viridis')
            plt.colorbar(im0, ax=axes[0], label='Mean Return'); axes[0].set_title('Performance Across Conditions')
            im1 = axes[1].imshow(pivot_safety.values, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
            plt.colorbar(im1, ax=axes[1], label='Safety Rate'); axes[1].set_title('Safety Rate Across Conditions')
        plt.tight_layout(); plt.show()

    def _make_json_serializable(self, obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._make_json_serializable(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(self._make_json_serializable(v) for v in obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        return obj
