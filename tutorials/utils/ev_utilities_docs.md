# EV Charging Tutorials - Comprehensive Utilities Documentation

## Overview

The `tutorials/utils/` package provides a unified API for EV charging experiments, supporting single-agent RL, multi-agent RL (MARL), safe RL, and comprehensive diagnostics. This documentation covers all utility modules and their usage patterns.

## Module Structure

```
tutorials/utils/
├── __init__.py         # Unified API exports
├── common.py          # Core wrappers and environment utilities
├── ev.py              # EV-specific evaluation, persistence, and plotting
├── rl.py              # RL infrastructure (buffers, networks, vectorization)
├── marl.py            # Multi-agent RL utilities
├── safe_rl.py         # Constrained MDP and safe RL utilities
├── morl.py            # Multi-objective RL utilities
├── diagnostics.py     # Advanced analysis and visualization
├── external.py        # Integrations with external libraries
└── notebook.py        # Jupyter notebook helpers
```

## 1. Core Utilities (`common.py`)

### Environment Creation and Wrapping

#### `create_ev_env(site, date_range, seed, safe_rl, flatten, noise, noise_action, **env_kwargs)`
Standard factory for creating EV charging environments.
- **Parameters**:
  - `site`: 'caltech' or 'jpl'
  - `date_range`: Tuple of date strings ('YYYY-MM-DD', 'YYYY-MM-DD')
  - `seed`: Random seed for reproducibility
  - `safe_rl`: If True, flattens observation space to Box (for OmniSafe compatibility)
  - `flatten`: If True, applies DictFlatteningWrapper
  - `noise`: Observation noise level [0, 1]
  - `noise_action`: Action noise level [0, 1]
- **Returns**: EVChargingEnv or wrapped version

#### `wrap_policy(policy_or_model)`
Universal policy wrapper supporting multiple interfaces (callable, get_action, predict, act).
```python
policy_fn = wrap_policy(model)  # Works with any interface
action = policy_fn(obs)
```

#### `wrap_ma_policy(policy_or_model)`
Multi-agent version returning Dict[agent_id, action].

### Utility Functions

#### `extract_from_obs(obs, key, env_space=None)`
Extract component from observation regardless of format (Dict or flattened).

#### `solve_cvx_with_fallback(prob, solvers=None, verbose=False)`
Solve CVXPY problem with automatic solver fallback.

## 2. EV-Specific Utilities (`ev.py`)

### Evaluation Functions

#### `evaluate_policy(policy_fn, make_env_fn, episodes=2, noise=None, noise_action=None, horizon=288, track_metrics=None, verbose=False)`
Comprehensive policy evaluation with optional metric tracking.
- **track_metrics** options:
  - `'satisfaction'`: Track demand satisfaction rate
  - `'components'`: Track profit and carbon cost breakdown
  - `'actions'`: Collect action statistics
  - `'trajectory'`: Full trajectory recording
  - `'demands'`: Track demand reduction
- **Returns**: `(mean_return, std_return, mean_cost, metrics_dict)`

#### `evaluate_policy_batch(policy_fn, make_env_fn, episodes=10, num_envs=4, horizon=288, **env_kwargs)`
Vectorized evaluation using Gymnasium's SyncVectorEnv.

#### `sweep_noise(policy_fn, make_env_fn, Ns=(0.0, 0.05, 0.1, 0.2), N_as=(0.0, 0.1), episodes=2)`
Grid sweep over observation and action noise levels.
- **Returns**: DataFrame with columns ['noise', 'noise_action', 'return_mean', 'return_std', 'safety_cost']

### Persistence and Caching

#### Cache Management
```python
get_cache_dir(kind)  # kind in ['baselines', 'rl', 'safe_rl', 'marl']
# Default: ./cache/<kind>/, override with TUTORIALS_CACHE_DIR env var
```

#### Timeseries Data
```python
save_timeseries(tag, ts_df, kind='rl')  # Save training curves
load_timeseries(tag, kind='rl')         # Load training curves
```

#### Model Checkpoints
```python
save_model_checkpoint(model, tag, epoch, kind='rl')
load_model_checkpoint(model_class, tag, epoch, kind='rl', **kwargs)
```

#### Evaluation Results
```python
save_evaluation_results(
    tag, policy_name, mean_return, std_return, mean_cost,
    metrics, env_config=None, policy_config=None, 
    notes=None, kind='rl', save_full_trajectory=False
)
results = load_evaluation_results(tag, kind='rl', load_trajectory=False)
```

### Plotting Functions

#### `plot_robustness_heatmap(df, value='return_mean', title='Robustness heatmap')`
Visualize robustness across noise grid from sweep_noise results.

#### `plot_action_distribution(actions, title, bins=50, show_stats=True, show_saturation=True, saturation_thresholds=(0.001, 0.999), return_stats=True)`
Comprehensive action distribution analysis with:
- Histogram with optional KDE overlay
- Statistical summary panel
- Saturation analysis (low/mid/high percentages)
- **Standardized thresholds**: Low ≤ 0.001, High ≥ 0.999

#### `plot_action_saturation(actions, title, thresholds=(0.001, 0.999))`
Simple saturation analysis (delegates to plot_action_distribution).
- **Returns**: `(saturation_high, saturation_low)` tuple

#### `plot_training_curves(data, x='episode', y='reward', smoothing_window=200, band='std')`
Flexible training curve visualization with:
- **band** options: 'none', 'std', 'ci' (confidence interval), 'quantile'
- Supports DataFrame, Dict[label, DataFrame], or list of DataFrames
- Automatic smoothing and aggregation across seeds

#### `plot_critical_timeline(trajectory, env, critical_windows=None, title, topk=3, window=12)`
Static timeline visualization (Matplotlib) showing:
- Reward, actions, and constraint slack over time
- Critical window highlights
- Can auto-detect critical periods from slack data

#### `plot_algorithm_comparison_matrix(results_df, metrics=['reward', 'violations', 'satisfaction'], algorithms=None, scenarios=None)`
Matrix comparison across algorithms and scenarios.

### Experiment Management

#### `ExperimentTracker`
```python
tracker = ExperimentTracker(base_dir='./experiments')
tracker.add_experiment(tag, config, results_df)
tracker.compare_experiments(['exp1', 'exp2'], metric='reward')
```

#### `run_scenario_sweep(algorithms, env_factory, scenarios, metrics_to_track=['satisfaction'], episodes_per_scenario=1)`
Compare multiple algorithms across scenarios.
- **Returns**: Tidy DataFrame with columns ['scenario', 'algorithm', 'mean_return', ...]

## 3. RL Infrastructure (`rl.py`)

### Environment Wrappers

#### `DictFlatteningWrapper(env)`
Flattens Dict observation spaces to Box for compatibility.

#### `make_vec_envs(env_id, num_envs=1, seed=0, device='cpu', wrapper_fn=None)`
Create vectorized environments for parallel rollouts.

### Network Initialization

#### `layer_init(layer, std=np.sqrt(2), bias_const=0.0)`
Orthogonal initialization for neural network layers.

### Data Structures

#### `RolloutBuffer(obs_shape, act_shape, capacity, device='cpu')`
Storage for on-policy algorithms.
```python
buffer = RolloutBuffer((obs_dim,), (act_dim,), capacity=2048)
buffer.add(obs, action, logprob, reward, done, value)
data = buffer.get()  # Returns dict with all data
```

#### `RunningMeanStd(shape=(), epsilon=1e-4)`
Online statistics for observation normalization.

### Utilities

- `get_obs_shape(observation_space)`: Extract shape from any space type
- `get_action_dim(action_space)`: Get action dimension
- `set_random_seed(seed)`: Set seeds for all RNGs
- `explained_variance(y_pred, y_true)`: Value function quality metric
- `LinearSchedule(initial, final, total_steps)`: Linear parameter scheduling

## 4. Multi-Agent Utilities (`marl.py`)

### State Construction

#### `build_central_state(obs_dict, agent_order=None)`
Build centralized observation by concatenating per-agent observations.

### Value Normalization

#### `ValueNormalizer`
```python
normalizer = ValueNormalizer()
normalizer.update(returns_array)
normalized = normalizer.normalize_torch(tensor)
```

### Multi-Agent Components

#### `MultiAgentRolloutBuffer(agent_ids, obs_shapes, act_shapes, capacity, device)`
Per-agent trajectory storage with synchronized stepping.

#### `CentralizedCritic(agent_obs_dims, hidden_dim=256, use_layer_norm=False)`
Centralized value function observing all agents.

#### `DecentralizedActorCritic(agent_configs, share_parameters=False)`
Per-agent or shared actor-critic networks.

#### `CommNetwork(agent_ids, message_dim=32, comm_type='broadcast')`
Agent communication with broadcast, targeted, or graph-based protocols.

### Multi-Agent Evaluation

#### `evaluate_multi_agent_policy(policy_fn, make_env_fn, episodes=2, horizon=288, track_metrics=None)`
Evaluate PettingZoo-style multi-agent environments.
- **track_metrics** options: 'fairness', 'coordination', 'per_agent_rewards', 'violations'

#### `run_multiagent_scenario_sweep(algorithms, ma_env_factory, scenarios, metrics_to_track, episodes_per_scenario=1)`
Multi-agent version of scenario sweep.

## 5. Safe RL Utilities (`safe_rl.py`)

### Core Components

#### `SafeRolloutBuffer(obs_shape, act_shape, capacity, device)`
Extended buffer tracking costs and cost values.
```python
buffer.add_safe(obs, action, reward, cost, value, cost_value, log_prob, done)
buffer.compute_returns_and_advantages(gamma, gae_lambda)
```

#### `LagrangianMultiplier(cost_limit, variant='standard', init_value=0.0, lr=0.035)`
Adaptive Lagrangian multiplier with variants:
- `'standard'`: Gradient-based update
- `'pid'`: PID controller
- `'adaptive'`: Dynamic learning rate

#### `BaseSafeRLTrainer`
Abstract base class providing:
- Standardized rollout collection
- Logging and checkpointing
- Constraint tracking
- Override `update()` for algorithm-specific logic

### Safe Policy Wrapper

#### `create_safe_policy_wrapper(actor_critic, deterministic=True, device='cpu', normalize_obs=False)`
Create evaluation-ready policy from safe RL model.

## 6. Multi-Objective RL (`morl.py`)

### Gradient Conflict Resolution

#### `GradientConflictResolver`
Static methods for gradient combination:
- `cagrad(gradients, c=0.5)`: Conflict-Averse Gradient
- `pcgrad(gradients)`: Projecting Conflicting Gradients
- `mgda(gradients, normalize=True)`: Multiple Gradient Descent Algorithm
- `imtl_g(gradients)`: Impartial Multi-Task Learning

#### `MultiObjectiveWrapper(objectives, weights=None, combination_method='weighted_sum', conflict_resolver=None)`
Manages multiple objectives with:
- Weight adaptation methods
- Loss combination strategies
- Performance tracking

## 7. Diagnostics (`diagnostics.py`)

### Constraint Analysis

#### `pilot_signals_df(env)`
Extract pilot signals as DataFrame with columns ['t', 'station', 'amps'].

#### `constraint_slack_t(env, t_index)`
Compute constraint magnitudes and slack at specific timestep.

#### `constraint_slack_series(env)`
Full episode constraint analysis.
- **Returns**: `(magnitudes[T, C], slacks[T, C])`

#### `get_constraint_binding_stats(env, threshold=5.0)`
Comprehensive binding statistics including:
- Per-constraint frequency
- Temporal patterns
- Violation counts

### Policy Analysis

#### `get_policy_diagnostics(policy_fn, env, num_samples=100, thresholds=(0.001, 0.999))`
Lightweight diagnostics from sampled observations.
- **Returns**: Dict with action statistics, saturation rates, entropy

#### `compare_policies(policies, env_factory, diagnostic_fn=get_policy_diagnostics)`
Compare multiple policies using diagnostics.
- **Returns**: DataFrame with one row per policy

#### `policy_behavior_fingerprint(policy_fn, env_fn, n_samples=100, seed=0)`
Generate behavioral signature including:
- Action statistics and distribution
- Station correlation patterns

### Interactive Visualizations

#### `interactive_critical_timeline(trajectory, env, policy_fn, window_size=12, top_k=5)`
Interactive Plotly timeline with:
- Click-to-jump navigation
- Critical window detection
- State inspection panels
- Safety overlays (cost, lambda, MOER)
- **Requires**: plotly, ipywidgets

#### `algorithm_arena(algorithms, env_fn, seed=42, max_steps=288, metrics_to_track=['reward', 'excess_charge'])`
Side-by-side algorithm comparison with:
- Synchronized playback
- Action heatmaps per algorithm
- Cumulative return curves
- Safety metrics scoreboard

#### `animate_policy_evolution(checkpoints, env_fn, episodes_per_checkpoint=1, interval=500)`
Animate learning progress across training checkpoints.

### Safety Analysis

#### `analyze_violation_patterns(env, trajectory)`
Correlate violations with demand, time-of-day, MOER, action saturation.

#### `test_safety_robustness(policy_fn, env_fn, cost_limit, perturbations)`
Test under various perturbations:
```python
perturbations = {
    'obs_noise': [0.0, 0.1, 0.2],
    'action_noise': [0.0, 0.05],
    'demand_scale': [1.0, 1.5, 2.0]
}
```

#### `compute_safety_pareto_frontier(policy_fn, env_fn, lambda_values)`
Estimate cost-reward Pareto frontier by varying λ.

## 8. External Integrations (`external.py`)

### Stable-Baselines3
```python
vec_env = sb3_make_vec_env(env_factory, n_envs=4)
model = sb3_make_model('PPO', 'MlpPolicy', vec_env, verbose=1)
policy_fn = sb3_policy_fn(model)  # Convert to standard interface
```

### RLlib + PettingZoo
```python
rllib_register_parallel_env('ev_ma', pz_env_factory)
trajectory = rllib_collect_one_episode(algo, env, policy_mapping_fn)
```

### OmniSafe
```python
env = make_omnisafe_ready_env(env_factory)  # Ensures info['safety_cost']
```

### SafePO
```python
ppo_lag = safepo_import_single_agent('ppo_lag', base_dir='Safe-Policy-Optimization-main')
mappo = safepo_import_multi_agent('mappo', base_dir='Safe-Policy-Optimization-main')
share_obs = build_share_obs_dict(obs_dict, agent_order)
```

## 9. Notebook Helpers (`notebook.py`)

### Formatted Output
```python
show("header: Results")
show("metric: Reward = {r:.2f}", r=100.5)
show("table", df=results_df)
show("progress: Training", step=50, total=100)
show_metrics({'reward': 100, 'cost': 5}, title="Performance")
```

### Quick Plotting
```python
quick_plot({
    'type': 'line',
    'x': timesteps,
    'y': {'PPO': ppo_rewards, 'SAC': sac_rewards},
    'title': 'Algorithm Comparison',
    'legend': True
})
```

## Usage Patterns

### Basic Evaluation
```python
from tutorials.utils import evaluate_policy, save_evaluation_results

mean_ret, std_ret, mean_cost, metrics = evaluate_policy(
    policy_fn, make_env_fn,
    episodes=10,
    track_metrics=['satisfaction', 'components', 'actions']
)

save_evaluation_results(
    tag="experiment_001",
    policy_name="PPO-Robust",
    mean_return=mean_ret,
    std_return=std_ret,
    mean_cost=mean_cost,
    metrics=metrics
)
```

### Robustness Analysis
```python
from tutorials.utils import sweep_noise, plot_robustness_heatmap

df = sweep_noise(policy_fn, make_env_fn, episodes=5)
plot_robustness_heatmap(df, value='return_mean')
```

### Training with Monitoring
```python
from tutorials.utils import RolloutBuffer, save_model_checkpoint

buffer = RolloutBuffer((obs_dim,), (act_dim,), capacity=2048)
# ... collect rollouts ...
save_model_checkpoint(model, "ppo_robust", epoch=100)
```

### Safe RL Training
```python
from tutorials.utils import SafeRolloutBuffer, LagrangianMultiplier

buffer = SafeRolloutBuffer((obs_dim,), (act_dim,), capacity=2048)
lagrange = LagrangianMultiplier(cost_limit=25.0, variant='pid')
# ... training loop ...
stats = lagrange.update(mean_cost)
```

## Configuration

### Environment Variables
- `TUTORIALS_CACHE_DIR`: Override default cache location
- Example: `export TUTORIALS_CACHE_DIR=/data/experiments/cache`

### Noise Configuration
The noise implementation in `env.py`:
- **Observation noise**: Applied to prev_moer, est_departures, demands, forecasted_moer
- **Action noise**: Applied to pilot signals
- Range: [0, 1] representing percentage of signal magnitude

### Standardized Metrics
- **Action saturation thresholds**: Low ≤ 0.001, High ≥ 0.999
- **Episode horizon**: 288 steps (24 hours at 5-minute intervals)
- **Default cache structure**: `<CACHE_BASE>/<kind>/<tag>_<suffix>.csv.gz`

## Dependencies

### Required
- numpy, pandas, matplotlib
- gymnasium
- torch (for RL utilities)
- cvxpy (for action projection)

### Optional
- plotly, ipywidgets (interactive visualizations)
- scipy (KDE in plot_action_distribution)
- stable-baselines3 (external baselines)
- ray[rllib] (multi-agent baselines)

## Common Issues and Solutions

### Issue: Memory leaks during training
**Solution**: Ensure `vec_env.close()` is called after training

### Issue: LSTM hidden state persists across episodes
**Solution**: Use StatefulPolicy class with reset() method

### Issue: Noise not applied in safe_rl mode
**Solution**: Safe RL mode flattens observations; noise implementation assumes Dict format

### Issue: Low episode count for statistics
**Solution**: Use at least 10-20 episodes for reliable metrics

### Issue: Action saturation metrics inconsistent
**Solution**: Use standardized thresholds (0.001, 0.999) across all utilities

## See Also

- Tutorial 01: Introduction and environment basics
- Tutorial 02: Baseline algorithms (greedy, MPC, optimal)
- Tutorial 03: Standard RL (PPO, SAC, LSTM-PPO, GRPO)
- Tutorial 04: Safe RL approaches
- Tutorial 05: Multi-agent coordination
- Tutorial 06: Advanced diagnostics