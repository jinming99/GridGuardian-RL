"""
Utilities for deeper debugging and diagnostics (Tutorial 06).
These are not minimal standard utils; they introspect ACN-Sim internals
(e.g., constraint currents, pilot signals) and create analysis plots.
"""
from __future__ import annotations

from typing import Dict, Any, Callable, Optional, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
try:
    import ipywidgets as widgets  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    widgets = None  # type: ignore
try:
    import plotly.graph_objects as go  # type: ignore
    from plotly.subplots import make_subplots  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    go = None  # type: ignore
    def make_subplots(*args, **kwargs):  # type: ignore
        raise ImportError("plotly is required for interactive_critical_timeline and algorithm_arena. pip install plotly")
from collections import defaultdict
from .notebook import log_info, log_warning, log_error

def safe_get_simulator_data(env, attr: str, default=None):
    """Safely access simulator attributes with fallback."""
    try:
        if hasattr(env, '_simulator') and env._simulator is not None:
            return getattr(env._simulator, attr, default)
    except Exception:
        pass
    return default


def pilot_signals_df(env) -> pd.DataFrame:
    """
    Build a long-form DataFrame of pilot signals (Amps) from the simulator.
    Safely handles missing simulator data by returning an empty DataFrame with expected columns.
    Columns: ['t', 'station', 'amps']
    """
    rates = safe_get_simulator_data(env, "charging_rates")
    if rates is None:
        # Return empty DataFrame with expected structure
        return pd.DataFrame(columns=['t', 'station', 'amps'])
    rates = np.asarray(rates)
    T = rates.shape[1]
    N = rates.shape[0]
    return pd.DataFrame({
        "t": np.repeat(np.arange(T), N),
        "station": np.tile(np.arange(N), T),
        "amps": rates.reshape(-1),
    })


def constraint_slack_t(env, t_index: int):
    """
    Compute constraint magnitudes and slack at a specific timestep.
    Returns (magnitudes, slack) arrays.
    """
    amps = env._simulator.charging_rates[:, t_index]
    cur = env._simulator.network.constraint_current(amps)
    mag = np.abs(cur)
    slack = env._simulator.network.magnitudes - mag
    return mag, slack


def constraint_slack_series(env):
    """
    Compute per-timestep arrays of (magnitudes, slack) across the episode.
    Returns: mags[T, C], slacks[T, C]
    """
    rates = env._simulator.charging_rates  # N × T
    T = rates.shape[1]
    mags, slacks = [], []
    for ti in range(T):
        mag, sl = constraint_slack_t(env, ti)
        mags.append(mag)
        slacks.append(sl)
    return np.stack(mags), np.stack(slacks)


def reconstruct_state(env, policy_fn: Callable, t: int, seed: int = 42):
    """
    Deterministically reconstruct the environment observation at timestep t
    by rolling out from reset using the provided policy function.

    Parameters
    ----------
    env : Any
        Environment instance supporting reset(seed) and step(action)
    policy_fn : Callable
        Function mapping observation -> action
    t : int
        Target timestep index (number of steps to advance from reset)
    seed : int
        Reset seed to ensure determinism

    Returns
    -------
    obs : Any
        Observation at timestep t (or terminal state if episode ended earlier)
    """
    obs, _ = env.reset(seed=seed)
    for _ in range(int(t)):
        action = policy_fn(obs)
        obs, _, done, _, _ = env.step(action)
        if done:
            break
    return obs


def get_constraint_binding_stats(env, threshold: float = 5.0) -> Dict[str, Any]:
    """
    Compute comprehensive constraint binding statistics.
    
    Returns dict with:
    - per_constraint_freq: binding frequency for each constraint
    - temporal_pattern: binding over time
    - min_slack_series: minimum slack across constraints per timestep
    - peak_binding_periods: top periods with most binding
    - total_violations: count of slack < 0 entries
    """
    mags, slacks = constraint_slack_series(env)
    binding = slacks < threshold
    per_constraint_freq = np.mean(binding, axis=0)
    temporal_pattern = np.mean(binding, axis=1)
    min_slack_series = np.min(slacks, axis=1)
    peak_binding_periods = np.argsort(min_slack_series)[:10]
    total_violations = int(np.sum(slacks < 0))

    return {
        'per_constraint_freq': per_constraint_freq,
        'temporal_pattern': temporal_pattern,
        'min_slack_series': min_slack_series,
        'peak_binding_periods': peak_binding_periods,
        'total_violations': total_violations,
    }


def plot_slack_from_env(env) -> None:
    """
    Convenience plot for constraint slack heatmap over time (Tutorial 06 Section 2).
    """
    _, slacks = constraint_slack_series(env)  # T × C
    plt.figure(figsize=(10, 3))
    # Use reversed colormap so low slack (tight/alert) is red and high slack is blue
    plt.imshow(slacks.T, aspect="auto", cmap="coolwarm_r")
    plt.colorbar(label="slack (A)")
    plt.xlabel("time")
    plt.ylabel("constraint index")
    plt.title("Constraint slack over time")
    plt.show()


def get_policy_diagnostics(
    policy_fn: Callable,
    env,
    num_samples: int = 100,
    thresholds: Tuple[float, float] = (0.001, 0.999),
) -> Dict[str, float]:
    """
    Compute lightweight policy diagnostics from randomly sampled observations.

    Attempts to call policy_fn(obs, return_diagnostics=True) to obtain (action, log_prob, value).
    If unsupported, falls back to policy_fn(obs) and reports only action statistics.

    Parameters
    ----------
    thresholds : Tuple[float, float]
        (low_threshold, high_threshold) for saturation; counts action ≤ low as low-saturation,
        and action ≥ high as high-saturation. Defaults to (0.001, 0.999) to match
        tutorials.utils_ev.plot_action_saturation.
    """
    actions_list = []
    log_probs_list = []
    values_list = []

    for _ in range(int(num_samples)):
        obs = env.observation_space.sample()
        # Try to fetch richer diagnostics if policy supports it
        try:
            out = policy_fn(obs, return_diagnostics=True)
            if isinstance(out, tuple) and len(out) >= 3:
                action, log_prob, value = out[:3]
            else:
                action = out
                log_prob = None
                value = None
        except TypeError:
            action = policy_fn(obs)
            log_prob = None
            value = None

        actions_list.append(np.asarray(action))
        if log_prob is not None:
            try:
                import torch  # type: ignore
                log_probs_list.append(float(torch.as_tensor(log_prob).mean().item()))
            except Exception:
                log_probs_list.append(float(np.mean(np.asarray(log_prob))))
        if value is not None:
            try:
                import torch  # type: ignore
                values_list.append(float(torch.as_tensor(value).mean().item()))
            except Exception:
                values_list.append(float(np.mean(np.asarray(value))))

    # Aggregate
    A = np.concatenate([a.reshape(-1) for a in actions_list]) if actions_list else np.array([])
    action_mean = float(np.mean(A)) if A.size else float("nan")
    action_std = float(np.std(A)) if A.size else float("nan")
    lo_thr, hi_thr = thresholds
    sat_hi = float(np.mean(A >= hi_thr)) if A.size else float("nan")
    sat_lo = float(np.mean(A <= lo_thr)) if A.size else float("nan")

    entropy = float(-np.mean(log_probs_list)) if len(log_probs_list) > 0 else float("nan")
    value_mean = float(np.mean(values_list)) if len(values_list) > 0 else float("nan")
    value_std = float(np.std(values_list)) if len(values_list) > 0 else float("nan")

    return {
        'action_mean': action_mean,
        'action_std': action_std,
        'action_saturation_high': sat_hi,
        'action_saturation_low': sat_lo,
        'entropy': entropy,
        'value_mean': value_mean,
        'value_std': value_std,
    }


def compare_policies(
    policies: Dict[str, Callable],
    env_factory: Callable,
    diagnostic_fn: Callable = get_policy_diagnostics,
) -> pd.DataFrame:
    """
    Compare multiple policies using the provided diagnostic function.
    Returns a DataFrame where each row corresponds to one policy's diagnostics.
    """
    results = []
    for name, pol in policies.items():
        env = env_factory()
        try:
            diags = diagnostic_fn(pol, env)
        finally:
            try:
                env.close()
            except Exception:
                pass
        diags['policy'] = name
        results.append(diags)
    return pd.DataFrame(results)


def interactive_critical_timeline(
    trajectory: Dict[str, np.ndarray],
    env,
    policy_fn: Callable,
    window_size: int = 12,
    top_k: int = 5,
) -> widgets.VBox:
    """
    Interactive timeline showing critical moments with a time slider and click-to-jump.

    Click on traces to jump to a timestep and inspect state and simple counterfactuals.

    Parameters
    ----------
    trajectory : Dict[str, np.ndarray]
        Trajectory data with required keys: 'reward', 'action'.
        Optional keys:
          - 'obs': observation sequence for inspection
          - 'excess_charge' or 'cost': either per-step safety cost or a cumulative cost series
          - 'lambda': per-step lambda (dual variable) series if available
          - 'moer': marginal operating emissions rate series
        If a cumulative cost series is provided, per-step cost will be computed via difference.
    env : EVChargingEnv
        Environment instance (must have run at least one episode so _simulator has data) for slack overlay.
    policy_fn : Callable
        Policy function for comparison (not currently used in plotting; reserved for future comparisons).
    window_size : int
        Size of zoom window in timesteps (used for highlighting critical windows).
    top_k : int
        Number of critical moments to highlight.

    Returns
    -------
    widgets.VBox
        Interactive widget with timeline and detail panels.

    See also
    --------
    tutorials.utils.plot_critical_timeline : Static Matplotlib version suitable for headless runs.
    """
    # Require ipywidgets
    if widgets is None:
        raise ImportError("ipywidgets is required for interactive_critical_timeline. pip install ipywidgets")

    # Coerce arrays
    reward = np.asarray(trajectory.get('reward', np.array([]))).astype(float)
    action = np.asarray(trajectory.get('action', np.array([])))
    obs_seq = trajectory.get('obs', None)
    moer = trajectory.get('moer', None)
    excess_charge = trajectory.get('excess_charge', None)

    T = int(len(reward))
    if T == 0:
        return widgets.VBox([widgets.HTML("<p>No trajectory data provided.</p>")])
    t_range = np.arange(T)

    # Identify critical moments (low reward, high constraint violation)
    criticality_score = np.zeros(T, dtype=float)
    criticality_score -= reward
    if excess_charge is not None:
        ex = np.asarray(excess_charge).reshape(-1)
        ex = ex[:T]
        criticality_score += 10.0 * ex

    # Find top-k disjoint windows around peaks
    critical_windows: List[Tuple[int, int, int]] = []
    used = np.zeros(T, dtype=bool)
    for _ in range(top_k):
        if np.all(used):
            break
        scores_masked = np.where(used, -np.inf, criticality_score)
        peak = int(np.argmax(scores_masked))
        start = max(0, peak - window_size // 2)
        end = min(T, peak + window_size // 2)
        critical_windows.append((start, end, peak))
        used[start:end] = True

    # Optional channels
    cost_key = 'cost' if 'cost' in trajectory else ('excess_charge' if 'excess_charge' in trajectory else None)
    lambda_series = trajectory.get('lambda', None)
    moer = trajectory.get('moer', None)

    cost_present = cost_key is not None and len(np.asarray(trajectory.get(cost_key, []))) > 0
    lambda_present = lambda_series is not None and len(np.asarray(lambda_series).reshape(-1)) > 0
    moer_present = moer is not None and len(np.asarray(moer).reshape(-1)) > 0

    # Compute cost step and cumulative if present
    cost_step = None
    cost_cum = None
    if cost_present:
        c = np.asarray(trajectory[cost_key]).reshape(-1)[:T]
        diffs = np.diff(c)
        is_cumulative = np.all(diffs >= -1e-9) and (np.max(c) >= np.min(c))
        if is_cumulative:
            cost_cum = c
            cost_step = np.diff(np.concatenate([[0.0], c]))
        else:
            cost_step = c
            cost_cum = np.cumsum(c)

    # Recompute criticality to use safety cost per-step if available
    criticality_score = -reward.copy()
    if cost_present and cost_step is not None:
        criticality_score += 10.0 * cost_step[:T]

    # Recompute top-k critical windows using updated criticality_score
    critical_windows = []
    used = np.zeros(T, dtype=bool)
    for _ in range(top_k):
        if np.all(used):
            break
        scores_masked = np.where(used, -np.inf, criticality_score)
        peak = int(np.argmax(scores_masked))
        start = max(0, peak - window_size // 2)
        end = min(T, peak + window_size // 2)
        critical_windows.append((start, end, peak))
        used[start:end] = True

    # Determine dynamic rows
    titles: List[str] = []
    specs: List[Dict[str, Any]] = []
    heights: List[float] = []

    # Row 1: Reward (+ Cum Cost on secondary y if available)
    titles.append('Reward' + (' + Cumulative Cost' if cost_present else ''))
    specs.append({'secondary_y': bool(cost_present)})
    heights.append(0.28)
    # Row 2: Actions(mean)
    titles.append('Actions (mean)')
    specs.append({})
    heights.append(0.20)
    # Row 3: Constraint Slack (min)
    titles.append('Constraint Slack (min)')
    specs.append({})
    heights.append(0.20)
    # Row 4: Safety Cost (per-step) if present
    if cost_present:
        titles.append('Safety Cost (per-step)')
        specs.append({})
        heights.append(0.16)
    # Row 5: Lambda if present
    if lambda_present:
        titles.append('Lambda (dual)')
        specs.append({})
        heights.append(0.16)
    # Last: MOER if present
    if moer_present:
        titles.append('MOER')
        specs.append({})
        heights.append(0.20)

    n_rows = len(titles)
    base_fig = make_subplots(
        rows=n_rows, cols=1,
        subplot_titles=tuple(titles),
        shared_xaxes=True,
        vertical_spacing=0.1,
        specs=[[specs[i]] for i in range(n_rows)],
        row_heights=heights,
    )

    # Reward
    base_fig.add_trace(
        go.Scatter(x=t_range, y=reward, name='Reward', line=dict(color='blue')),
        row=1, col=1
    )

    # Action mean per step
    act = action
    if act.ndim == 1:
        actions_mean = act
    else:
        try:
            actions_mean = np.mean(act, axis=-1)
        except Exception:
            actions_mean = act.reshape(T, -1).mean(axis=1)
    base_fig.add_trace(
        go.Scatter(x=t_range, y=actions_mean, name='Action(mean)', line=dict(color='orange')),
        row=2, col=1
    )

    # Constraint slack (min across constraints) if simulator available
    try:
        if hasattr(env, '_simulator') and env._simulator is not None:
            _, slacks = constraint_slack_series(env)  # shape [T_env, C]
            min_slack = np.min(slacks, axis=1) if slacks.ndim > 1 else slacks
            t_slack = np.arange(len(min_slack))
            # Align lengths
            U = min(T, len(min_slack))
            base_fig.add_trace(
                go.Scatter(x=t_slack[:U], y=min_slack[:U], name='Min Slack', line=dict(color='red')),
                row=3, col=1
            )
            try:
                base_fig.add_hline(y=0, row=3, col=1, line_dash="dash", line_color="black")
            except Exception:
                pass
        else:
            # Graceful message when simulator hasn't been initialized
            log_warning("Constraint slack data not available (simulator not initialized)")
    except Exception as e:
        log_warning(f"Could not extract constraint data: {e}")

    # Cost overlays
    row_idx = 1
    if cost_present and cost_cum is not None:
        try:
            base_fig.add_trace(
                go.Scatter(x=t_range, y=cost_cum, name='Cumulative Cost', line=dict(color='purple', dash='dash')),
                row=row_idx, col=1, secondary_y=True
            )
        except Exception:
            # If secondary axis is not supported (older plotly), fall back to primary
            base_fig.add_trace(
                go.Scatter(x=t_range, y=cost_cum, name='Cumulative Cost (approx)', line=dict(color='purple', dash='dash')),
                row=row_idx, col=1
            )

    # Safety cost per-step row (if present)
    current_row = 4  # After Reward, Actions, Slack
    if cost_present and cost_step is not None:
        base_fig.add_trace(
            go.Scatter(x=t_range, y=cost_step, name='Cost (step)', line=dict(color='crimson')),
            row=current_row, col=1
        )
        current_row += 1

    # Lambda row (if present)
    if lambda_present:
        lam_arr = np.asarray(lambda_series).reshape(-1)[:T]
        base_fig.add_trace(
            go.Scatter(x=t_range, y=lam_arr, name='Lambda', line=dict(color='purple')),
            row=current_row, col=1
        )
        current_row += 1

    # MOER channel (optional)
    if moer_present:
        moer_arr = np.asarray(moer).reshape(-1)[:T]
        base_fig.add_trace(
            go.Scatter(x=t_range, y=moer_arr, name='MOER', line=dict(color='green')),
            row=current_row, col=1
        )

    # Highlight critical windows
    for start, end, peak in critical_windows:
        for row in range(1, n_rows + 1):
            try:
                base_fig.add_vrect(x0=start, x1=end, fillcolor="red", opacity=0.2,
                                   layer="below", line_width=0, row=row, col=1)
            except Exception:
                pass

    base_fig.update_layout(
        title="Critical Moments Timeline (click points to jump)",
        height=int(220 * n_rows),
        showlegend=True,
        hovermode='x unified'
    )
    base_fig.update_xaxes(title_text="Timestep", row=n_rows, col=1)

    # Convert to FigureWidget to enable click callbacks
    figw = go.FigureWidget(base_fig)

    # Detail panel (single)
    detail_output = widgets.Output()

    def show_detail_at_time(t: int):
        t = int(max(0, min(T - 1, t)))
        with detail_output:
            detail_output.clear_output(wait=True)
            print(f"\n=== State at t={t} ===")
            print(f"Reward: {float(reward[t]):.3f}")
            if obs_seq is not None and t < len(obs_seq):
                obs = obs_seq[t]
                if isinstance(obs, dict):
                    print("\nObservation:")
                    for k, v in obs.items():
                        if isinstance(v, np.ndarray):
                            print(f"  {k}: shape {v.shape}, mean={np.mean(v):.3f}")
                        else:
                            print(f"  {k}: {v}")
            if excess_charge is not None:
                ex_t = float(np.asarray(excess_charge)[t])
                print(f"\nExcess charge (step): {ex_t:.3f}")
            print(f"\nAction taken: {np.asarray(action[t])}")

    # Time slider
    time_slider = widgets.IntSlider(value=0, min=0, max=T - 1, step=1,
                                    description='Time:', continuous_update=False)

    def on_time_change(change):
        if change.get('name') == 'value':
            show_detail_at_time(change['new'])

    time_slider.observe(on_time_change, names='value')

    # Click callback on reward trace to set slider
    if len(figw.data) > 0:
        def _on_click(trace, points, state):
            if points.point_inds:
                idx = int(points.point_inds[0])
                time_slider.value = idx
        try:
            figw.data[0].on_click(_on_click)  # first trace = Reward
        except Exception:
            pass

    # Assemble widget
    widget = widgets.VBox([
        widgets.HTML("<h3>Interactive Critical Timeline</h3>"),
        figw,
        time_slider,
        widgets.VBox([widgets.HTML("<b>State Details</b>"), detail_output])
    ])

    # Initialize view at first critical moment if available
    if critical_windows:
        time_slider.value = int(critical_windows[0][2])
    else:
        show_detail_at_time(0)

    return widget


def analyze_dual_dynamics(training_stats: pd.DataFrame, cost_limit: float):
    """
    Analyze Lagrangian multiplier dynamics and constraint satisfaction.

    Shows:
    - Lambda evolution over time (with mean episode cost on secondary axis)
    - Lambda vs. constraint violation scatter
    - Cost-reward trade-off colored by lambda
    - Violation histogram

    Returns a Plotly FigureWidget, wrapped in a VBox if ipywidgets is available.
    """
    if go is None:
        raise ImportError("plotly is required for analyze_dual_dynamics. pip install plotly")

    # Flexible column names
    lam_col = 'lambda' if 'lambda' in training_stats.columns else ('lam' if 'lam' in training_stats.columns else None)
    cost_col = 'mean_cost' if 'mean_cost' in training_stats.columns else ('mean_ep_cost' if 'mean_ep_cost' in training_stats.columns else None)
    rew_col = 'mean_reward' if 'mean_reward' in training_stats.columns else ('reward' if 'reward' in training_stats.columns else None)

    if lam_col is None or cost_col is None:
        raise ValueError("training_stats must contain lambda and mean_cost (or mean_ep_cost) columns")

    x_index = training_stats.index if training_stats.index.name is not None else np.arange(len(training_stats))

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Lambda Evolution', 'Lambda Response', 'Cost-Reward Trade-off', 'Violation Distribution'),
        specs=[[{'secondary_y': True}, {}], [{}, {}]]
    )

    # Lambda and cost evolution
    fig.add_trace(
        go.Scatter(x=x_index, y=training_stats[lam_col], name='λ', line=dict(color='blue')),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=x_index, y=training_stats[cost_col], name='Mean Cost', line=dict(color='red')),
        row=1, col=1, secondary_y=True
    )
    try:
        fig.add_hline(y=cost_limit, row=1, col=1, secondary_y=True, line_dash='dash', annotation_text='Cost Limit')
    except Exception:
        pass

    # Lambda vs violation scatter
    violations = training_stats[cost_col] - float(cost_limit)
    try:
        fig.add_trace(
            go.Scatter(x=violations, y=training_stats[lam_col], mode='markers',
                       marker=dict(color=np.arange(len(training_stats)), colorscale='Viridis'), name='λ Response'),
            row=1, col=2
        )
        fig.add_vline(x=0, row=1, col=2, line_dash='dash')
    except Exception:
        pass

    # Cost-reward trade-off
    if rew_col is not None:
        fig.add_trace(
            go.Scatter(x=training_stats[cost_col], y=training_stats[rew_col], mode='markers',
                       marker=dict(color=training_stats[lam_col], colorscale='Hot', showscale=True), name='Trade-off'),
            row=2, col=1
        )

    # Violation histogram
    fig.add_trace(
        go.Histogram(x=violations, name='Violations', marker_color='orange'),
        row=2, col=2
    )

    fig.update_layout(height=700, showlegend=True, title_text='Dual Dynamics Analysis')
    figw = go.FigureWidget(fig)
    return widgets.VBox([figw]) if widgets is not None else figw


def explain_safety_decision(
    actor_critic: Any,
    obs: np.ndarray,
    cost_limit: float,
    lambda_value: float
) -> Dict[str, Any]:
    """
    Explain why the safe policy takes a specific action.

    Returns dict with:
    - predicted_reward: Expected future reward
    - predicted_cost: Expected future cost
    - safety_margin: Cost limit - predicted cost
    - action_conservativeness: How conservative the action is (1 - |action| mean)
    - decision_rationale: Text explanation
    - action: The chosen action (deterministic if available)
    """
    try:
        import torch  # type: ignore
    except Exception as e:
        raise ImportError(f"explain_safety_decision requires PyTorch: {e}")

    obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        out = actor_critic(obs_tensor)
        dist = value = cost_value = None
        if isinstance(out, tuple):
            if len(out) >= 3:
                dist, value, cost_value = out[:3]
            elif len(out) == 2:
                dist, value = out
        # Fallbacks for common actor-critic APIs
        if value is None and hasattr(actor_critic, 'critic'):
            value = actor_critic.critic(obs_tensor)
        if cost_value is None and hasattr(actor_critic, 'cost_critic'):
            cost_value = actor_critic.cost_critic(obs_tensor)

        # Deterministic action from distribution if possible
        action_tensor = None
        if dist is not None:
            if hasattr(dist, 'mean'):
                action_tensor = dist.mean
            elif hasattr(dist, 'mode'):
                action_tensor = dist.mode
            else:
                action_tensor = dist.sample()
        else:
            # Try a generic get_action
            if hasattr(actor_critic, 'get_action'):
                action_tensor = torch.as_tensor(actor_critic.get_action(obs), dtype=torch.float32).unsqueeze(0)
            else:
                # Last resort: zero action
                action_tensor = torch.zeros_like(obs_tensor)

    pred_reward = float(torch.as_tensor(value).view(-1)[0].item()) if value is not None else float('nan')
    pred_cost = float(torch.as_tensor(cost_value).view(-1)[0].item()) if cost_value is not None else float('nan')
    safety_margin = float(cost_limit - pred_cost) if not np.isnan(pred_cost) else float('nan')

    action_mag = float(torch.mean(torch.abs(action_tensor)).item())

    if not np.isnan(safety_margin) and safety_margin < 5:
        rationale = f"High violation risk (margin={safety_margin:.1f}). Taking conservative action."
    elif lambda_value > 1.0:
        rationale = f"High λ={lambda_value:.2f} indicates past violations. Prioritizing safety."
    elif not np.isnan(pred_reward) and not np.isnan(safety_margin) and pred_reward > 10 and safety_margin > 10:
        rationale = f"Safe margin ({safety_margin:.1f}) allows reward optimization."
    else:
        rationale = f"Balancing reward ({pred_reward:.1f}) and safety (margin={safety_margin:.1f})."

    return {
        'predicted_reward': pred_reward,
        'predicted_cost': pred_cost,
        'safety_margin': safety_margin,
        'action_conservativeness': float(1.0 - action_mag),
        'lambda_value': float(lambda_value),
        'decision_rationale': rationale,
        'action': action_tensor.cpu().numpy(),
    }


def analyze_violation_patterns(env, trajectory: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyze when and why constraint violations occur, correlating with
    demand levels, time-of-day, MOER values, and action saturation.
    """
    stats = get_constraint_binding_stats(env)
    min_slack_series = np.asarray(stats.get('min_slack_series', []))
    if min_slack_series.size == 0:
        return {'violation_times': [], 'total_violations': 0, 'violation_rate': 0.0}

    violation_mask = min_slack_series < 0
    violation_times = np.where(violation_mask)[0]

    patterns: Dict[str, Any] = {
        'violation_times': violation_times.tolist(),
        'total_violations': int(len(violation_times)),
        'violation_rate': float(len(violation_times) / len(min_slack_series)),
    }

    obs_seq = trajectory.get('obs') if isinstance(trajectory, dict) else None
    actions_seq = trajectory.get('action') if isinstance(trajectory, dict) else None
    moer_seq = trajectory.get('moer') if isinstance(trajectory, dict) else None

    # Demand analysis during violations
    if violation_times.size > 0 and obs_seq is not None:
        demands_at_violations: list[float] = []
        overall_demands: list[float] = []
        for t in range(min(len(obs_seq), len(min_slack_series))):
            obs_t = obs_seq[t]
            if isinstance(obs_t, dict) and 'demands' in obs_t:
                val = float(np.sum(obs_t['demands']))
                overall_demands.append(val)
                if t in violation_times:
                    demands_at_violations.append(val)
        if demands_at_violations and overall_demands:
            patterns['avg_demand_at_violation'] = float(np.mean(demands_at_violations))
            patterns['avg_demand_overall'] = float(np.mean(overall_demands))
            patterns['demand_correlation'] = float(patterns['avg_demand_at_violation'] / max(patterns['avg_demand_overall'], 1e-6))

    # Time-of-day pattern (5-min steps)
    if violation_times.size > 0:
        hour_of_violations = ((violation_times * 5) / 60.0).astype(int)
        counts = np.bincount(hour_of_violations, minlength=24)
        patterns['peak_violation_hours'] = counts.tolist()

    # Action saturation during violations
    if actions_seq is not None and violation_times.size > 0:
        act_arr = np.asarray(actions_seq, dtype=object)
        sat_flags: list[float] = []
        for t in violation_times:
            if t < len(act_arr):
                a = np.asarray(act_arr[t]).astype(float)
                sat_flags.append(float(np.mean(a > 0.9)))
        if sat_flags:
            patterns['action_saturation_at_violations'] = float(np.mean(sat_flags))

    # MOER correlation
    if moer_seq is not None and violation_times.size > 0:
        moer_arr = np.asarray(moer_seq).reshape(-1)
        moer_arr = moer_arr[:len(min_slack_series)]
        moer_at_viol = moer_arr[violation_mask[:len(moer_arr)]]
        if moer_at_viol.size > 0:
            patterns['avg_moer_at_violation'] = float(np.mean(moer_at_viol))
            patterns['avg_moer_overall'] = float(np.mean(moer_arr))
            patterns['moer_correlation'] = float(patterns['avg_moer_at_violation'] / max(patterns['avg_moer_overall'], 1e-6))

    return patterns


def test_safety_robustness(
    policy_fn: Callable,
    env_fn: Callable,
    cost_limit: float,
    perturbations: Dict[str, List[float]]
) -> pd.DataFrame:
    """
    Test policy safety under various perturbations.

    perturbations: dict with keys like 'obs_noise', 'action_noise', 'demand_scale'
    """
    results: List[Dict[str, Any]] = []

    def _perturb_obs(obs: Any, noise_std: float | None, scale_demands: float | None) -> Any:
        ob = obs
        if isinstance(ob, dict):
            ob = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in ob.items()}
            if scale_demands is not None and 'demands' in ob and isinstance(ob['demands'], np.ndarray):
                ob['demands'] = ob['demands'] * float(scale_demands)
            if noise_std is not None:
                for k, v in ob.items():
                    if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.number):
                        ob[k] = v + np.random.normal(0, float(noise_std), size=v.shape)
        elif isinstance(ob, np.ndarray) and noise_std is not None:
            ob = ob + np.random.normal(0, float(noise_std), size=ob.shape)
        return ob

    for pert_type, pert_values in perturbations.items():
        for pert_level in pert_values:
            env = env_fn()

            episode_costs: List[float] = []
            episode_rewards: List[float] = []

            for ep in range(5):
                obs, _ = env.reset(seed=ep)
                ep_cost = 0.0
                ep_reward = 0.0
                done = False
                while not done:
                    # Apply observation perturbation or demand scaling
                    if pert_type == 'obs_noise':
                        obs_pert = _perturb_obs(obs, noise_std=float(pert_level), scale_demands=None)
                    elif pert_type == 'demand_scale':
                        obs_pert = _perturb_obs(obs, noise_std=None, scale_demands=float(pert_level))
                    else:
                        obs_pert = obs

                    action = np.asarray(policy_fn(obs_pert))

                    # Action noise
                    if pert_type == 'action_noise':
                        action = action + np.random.normal(0, float(pert_level), size=action.shape)
                        action = np.clip(action, -1, 1)

                    obs, reward, term, trunc, info = env.step(action)
                    rb = info.get('reward_breakdown', {})
                    ep_cost += float(rb.get('excess_charge', 0.0))
                    ep_reward += float(reward)
                    done = bool(term) or bool(trunc)

            episode_costs.append(ep_cost)
            episode_rewards.append(ep_reward)

            results.append({
                'perturbation_type': pert_type,
                'perturbation_level': float(pert_level),
                'mean_cost': float(np.mean(episode_costs)) if episode_costs else float('nan'),
                'std_cost': float(np.std(episode_costs)) if episode_costs else float('nan'),
                'mean_reward': float(np.mean(episode_rewards)) if episode_rewards else float('nan'),
                'violation_rate': float(np.mean([c > cost_limit for c in episode_costs])) if episode_costs else float('nan'),
                'max_violation': float(max(0.0, (max(episode_costs) - cost_limit))) if episode_costs else float('nan'),
            })

            try:
                env.close()
            except Exception:
                pass

    return pd.DataFrame(results)


def compute_safety_pareto_frontier(
    policy_fn: Callable,
    env_fn: Callable,
    lambda_values: List[float]
) -> pd.DataFrame:
    """
    Compute Pareto frontier by varying lambda values.
    Attempts to call policy_fn(obs, lambda_weight=lam); if unsupported, tries policy_fn.set_lambda(lam).
    """
    rows: List[Dict[str, Any]] = []

    for lam in lambda_values:
        # Prepare policy wrapper respecting lambda if possible
        def _policy(obs, _lam=lam):
            try:
                return np.asarray(policy_fn(obs, lambda_weight=_lam))
            except TypeError:
                if hasattr(policy_fn, 'set_lambda'):
                    try:
                        policy_fn.set_lambda(_lam)
                    except Exception:
                        pass
                return np.asarray(policy_fn(obs))

        rewards: List[float] = []
        costs: List[float] = []
        for ep in range(10):
            env = env_fn()
            obs, _ = env.reset(seed=ep)
            ep_reward = 0.0
            ep_cost = 0.0
            done = False
            while not done:
                action = _policy(obs)
                obs, reward, term, trunc, info = env.step(action)
                ep_reward += float(reward)
                ep_cost += float(info.get('reward_breakdown', {}).get('excess_charge', 0.0))
                done = bool(term) or bool(trunc)
            rewards.append(ep_reward)
            costs.append(ep_cost)
            try:
                env.close()
            except Exception:
                pass

        rows.append({
            'lambda': float(lam),
            'mean_reward': float(np.mean(rewards)) if rewards else float('nan'),
            'std_reward': float(np.std(rewards)) if rewards else float('nan'),
            'mean_cost': float(np.mean(costs)) if costs else float('nan'),
            'std_cost': float(np.std(costs)) if costs else float('nan'),
        })

    return pd.DataFrame(rows)


def animate_policy_evolution(
    checkpoints: List[Tuple[int, Any]],
    env_fn: Callable,
    episodes_per_checkpoint: int = 1,
    interval: int = 500,
) -> HTML:
    """
    Animate policy behavior evolution across training checkpoints.

    Creates an animated heatmap showing how actions evolve during training.
    """
    heatmaps: List[np.ndarray] = []
    timesteps: List[int] = []

    for ts, model_or_fn in checkpoints:
        if callable(model_or_fn):
            policy_fn = model_or_fn
        else:
            def policy_fn(obs, _m=model_or_fn):
                return _m.predict(obs, deterministic=True)[0]

        env = env_fn()
        actions_collected: List[np.ndarray] = []

        for ep in range(int(episodes_per_checkpoint)):
            obs, _ = env.reset(seed=ep)
            episode_actions: List[np.ndarray] = []
            done = False
            t = 0
            while not done and t < 288:
                a = policy_fn(obs)
                episode_actions.append(np.asarray(a))
                obs, _, term, trunc, _ = env.step(a)
                done = bool(term) or bool(trunc)
                t += 1
            if episode_actions:
                actions_collected.append(np.asarray(episode_actions))

        try:
            env.close()
        except Exception:
            pass

        if actions_collected:
            avg_actions = np.mean(actions_collected, axis=0)  # [T, A]
            heatmaps.append(avg_actions.T)  # [A, T]
            timesteps.append(int(ts))

    if not heatmaps:
        return HTML("<p>No valid checkpoints found</p>")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})

    im = ax1.imshow(heatmaps[0], aspect='auto', cmap='viridis', vmin=0, vmax=1)
    ax1.set_ylabel('Station')
    ax1.set_xlabel('Time (5-min steps)')
    title = ax1.set_title(f'Policy Actions at Training Step {timesteps[0]}')
    plt.colorbar(im, ax=ax1, label='Action (normalized)')

    ax2.set_xlim(0, max(timesteps))
    ax2.set_ylim(0, 1)
    ax2.set_xlabel('Training Steps')
    ax2.set_yticks([])
    progress_line = ax2.axvline(timesteps[0], color='red', linewidth=2)
    ax2.plot(timesteps, [0.5] * len(timesteps), 'o', markersize=8, color='blue')

    def _animate(frame):
        im.set_array(heatmaps[frame])
        title.set_text(f'Policy Actions at Training Step {timesteps[frame]}')
        progress_line.set_xdata([timesteps[frame]])
        return [im, title, progress_line]

    anim = FuncAnimation(fig, _animate, frames=len(heatmaps), interval=interval, blit=True)
    plt.tight_layout()
    return HTML(anim.to_jshtml())


def algorithm_arena(
    algorithms: Dict[str, Callable],
    env_fn: Callable,
    seed: int = 42,
    max_steps: int = 288,
    metrics_to_track: List[str] = ['reward', 'excess_charge', 'profit', 'carbon_cost'],
) -> widgets.VBox:
    """
    Side-by-side algorithm comparison with synchronized playback.

    Note: For scoreboard metrics coming from info['reward_breakdown'], the totals are
    taken as the final cumulative values (not sum-over-time of cumulative series).
    """
    # Require ipywidgets
    if widgets is None:
        raise ImportError("ipywidgets is required for algorithm_arena. pip install ipywidgets")

    # Collect trajectories
    trajectories: Dict[str, Dict[str, np.ndarray]] = {}
    for name, policy_fn in algorithms.items():
        env = env_fn()
        obs, info = env.reset(seed=seed)
        traj = defaultdict(list)
        for t in range(int(max_steps)):
            a = policy_fn(obs)
            obs_next, r, term, trunc, info = env.step(a)
            traj['action'].append(np.asarray(a))
            traj['reward'].append(float(r))
            traj['obs'].append(obs)
            rb = info.get('reward_breakdown', {})
            for k, v in rb.items():
                traj[k].append(float(v))  # cumulative
            obs = obs_next
            if term or trunc:
                break
        try:
            env.close()
        except Exception:
            pass
        # Convert to arrays
        for k in traj:
            try:
                trajectories[name] = trajectories.get(name, {})
                trajectories[name][k] = np.array(traj[k])
            except Exception:
                trajectories[name][k] = np.asarray(traj[k], dtype=object)

    n_algos = len(trajectories)
    if n_algos == 0:
        return widgets.VBox([widgets.HTML("<p>No trajectories to compare.</p>")])

    # Build figure (one column). Each algo row shows actions heatmap; last row shows cumulative objective curves.
    titles = [f"Actions — {name}" for name in trajectories.keys()] + ["Cumulative Objective (Return − Cost)"]
    fig = make_subplots(rows=n_algos + 1, cols=1, subplot_titles=tuple(titles), shared_xaxes=True, vertical_spacing=0.05)

    # Add action heatmaps per algorithm
    first = True
    for i, (name, traj) in enumerate(trajectories.items(), start=1):
        actions = traj.get('action', np.zeros((0, 1)))
        if actions.ndim == 1:
            actions = actions.reshape(-1, 1)
        elif actions.ndim > 2:
            actions = actions.reshape(actions.shape[0], -1)
        # Build heatmap trace and attach a compact colorbar only for the first one
        hm = go.Heatmap(
            z=actions.T,
            colorscale='Viridis',
            showscale=first,
            name=name,
            showlegend=False,  # keep legend clean for line plot
        )
        if first:
            hm.update(colorbar=dict(len=0.45, thickness=10, outlinewidth=0, x=1.02,
                                    tickfont=dict(size=10), title='Action', titlefont=dict(size=11)))
        fig.add_trace(hm, row=i, col=1)
        fig.update_yaxes(title_text=name, row=i, col=1)
        first = False

    # Add cumulative objective (return − cost) in last row
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'teal']
    for i, (name, traj) in enumerate(trajectories.items()):
        rew = traj.get('reward', np.array([]))
        if rew.size:
            cumsum_reward = np.cumsum(rew)
            # Derive cumulative cost from 'excess_charge' if present
            ec = traj.get('excess_charge', np.array([]))
            if isinstance(ec, np.ndarray) and ec.size > 0:
                ec_float = ec.astype(float)
                diffs = np.diff(ec_float)
                is_cumulative = np.all(diffs >= -1e-9) and (np.max(ec_float) >= np.min(ec_float))
                cost_cum = ec_float if is_cumulative else np.cumsum(ec_float)
            else:
                cost_cum = np.zeros_like(cumsum_reward)
            U = min(len(cumsum_reward), len(cost_cum))
            objective_cum = cumsum_reward[:U] - cost_cum[:U]
            fig.add_trace(
                go.Scatter(x=np.arange(U), y=objective_cum,
                           name=f"{name} Objective", line=dict(color=colors[i % len(colors)])),
                row=n_algos + 1, col=1
            )

    # Overall layout and legend positioning (legend anchored to top-left of last subplot)
    fig.update_layout(title="Algorithm Arena — Synchronized Comparison", height=220 * (n_algos + 1), showlegend=True)
    # Axis labels for cumulative objective plot
    fig.update_xaxes(title_text='Time (5-min steps)', row=n_algos + 1, col=1)
    fig.update_yaxes(title_text='Cumulative Objective', row=n_algos + 1, col=1)
    # Compute y-domain for the last row to anchor legend near its top-left corner
    yaxis_key = 'yaxis' if (n_algos + 1) == 1 else f'yaxis{n_algos + 1}'
    try:
        last_domain = getattr(fig.layout, yaxis_key).domain
        y_top = float(last_domain[1])
    except Exception:
        y_top = 0.30  # reasonable fallback near bottom portion of the figure
    fig.update_layout(
        legend=dict(
            x=0.01,
            y=max(0.0, y_top - 0.02),
            xanchor='left',
            yanchor='top',
            orientation='h',
            bgcolor='rgba(255,255,255,0.7)',
            bordercolor='rgba(0,0,0,0)',
            font=dict(size=11),
            itemsizing='constant',
            itemwidth=30,
        )
    )

    # Scoreboard
    scoreboard_data = []
    for name, traj in trajectories.items():
        row = {
            'Algorithm': name,
            'Total Return': float(np.sum(traj.get('reward', np.array([0.0])))),
            'Avg Reward': float(np.mean(traj.get('reward', np.array([0.0])))),
        }
        # Safety metrics derived from 'excess_charge' if present (robust to cumulative vs per-step)
        ec = traj.get('excess_charge', np.array([]))
        if isinstance(ec, np.ndarray) and ec.size > 0:
            ec_float = ec.astype(float)
            diffs = np.diff(ec_float)
            is_cumulative = np.all(diffs >= -1e-9) and (np.max(ec_float) >= np.min(ec_float))
            step_cost = np.diff(np.concatenate(([0.0], ec_float))) if is_cumulative else ec_float
            step_pos = step_cost[step_cost > 0]
            row['Total Cost'] = float(np.sum(step_cost))
            row['Violation Steps'] = int(np.sum(step_cost > 0))
            row['Mean Violation'] = float(np.mean(step_pos)) if step_pos.size else 0.0
            row['Max Violation'] = float(np.max(step_pos)) if step_pos.size else 0.0
        for metric in metrics_to_track:
            if metric == 'reward':
                # Avoid duplicating reward; already summarized above
                continue
            if metric in traj and len(traj[metric]) > 0:
                # Use final cumulative value (env exposes cumulative per-step)
                row[f'Total {metric}'] = float(traj[metric][-1])
        scoreboard_data.append(row)
    scoreboard_df = pd.DataFrame(scoreboard_data)
    # Modern, minimal scoreboard styling (compact, readable)
    display_df = scoreboard_df.copy()
    # Format numbers to 3 decimals in HTML generation
    num_cols = [c for c in display_df.columns if pd.api.types.is_numeric_dtype(display_df[c])]
    formatters = {c: (lambda x: f"{x:.3f}") for c in num_cols}
    scoreboard_css = """
    <style>
    .gg-scoreboard { font-family: -apple-system,BlinkMacSystemFont,Segoe UI,Roboto,Helvetica,Arial,sans-serif; font-size: 13px; color: #111827; border-collapse: separate; border-spacing: 0; width: 100%; border: 1px solid #E5E7EB; border-radius: 12px; overflow: hidden; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }
    .gg-scoreboard thead th { background: #F9FAFB; font-weight: 600; text-align: center; padding: 8px 10px; border-bottom: 1px solid #E5E7EB; }
    .gg-scoreboard tbody tr:nth-child(even) { background: #FCFCFD; }
    .gg-scoreboard tbody tr:hover { background: #F3F4F6; }
    .gg-scoreboard td { padding: 8px 10px; border-bottom: 1px solid #F3F4F6; text-align: right; }
    .gg-scoreboard td:first-child { text-align: left; font-weight: 600; color: #111827; }
    </style>
    """
    scoreboard_table = display_df.to_html(index=False, classes='gg-scoreboard', border=0, justify='center', escape=False, formatters=formatters)
    scoreboard_html = scoreboard_css + scoreboard_table

    widget = widgets.VBox([
        widgets.HTML("<h3>Algorithm Face-Off</h3>"),
        go.FigureWidget(fig),
        widgets.HTML('<div style="font-weight:700; font-size:16px; margin:8px 0 6px 0;">Scoreboard</div>'),
        widgets.HTML(scoreboard_html)
    ])
    return widget


def policy_behavior_fingerprint(
    policy_fn: Callable,
    env_fn: Callable,
    n_samples: int = 100,
    seed: int = 0,
) -> Dict[str, Any]:
    """
    Generate a behavioral fingerprint for a policy.

    Samples diverse states and analyzes policy responses to create a
    characteristic signature of the policy's behavior.
    """
    np.random.seed(int(seed))
    env = env_fn()

    states: List[Any] = []
    actions: List[np.ndarray] = []
    for i in range(int(n_samples)):
        obs, _ = env.reset(seed=i)
        # Vary time within episode with random rollout length
        steps = int(np.random.randint(0, 100))
        term = trunc = False
        for _ in range(steps):
            if term or trunc:
                break
            a_rand = env.action_space.sample()
            obs, _, term, trunc, _ = env.step(a_rand)
        states.append(obs)
        a = policy_fn(obs)
        actions.append(np.asarray(a))
    try:
        env.close()
    except Exception:
        pass

    A = np.asarray(actions)
    A_flat = A.reshape(-1) if A.size else np.array([])

    fingerprint: Dict[str, Any] = {}
    fingerprint['action_stats'] = {
        'mean': float(np.mean(A_flat)) if A_flat.size else float('nan'),
        'std': float(np.std(A_flat)) if A_flat.size else float('nan'),
        'min': float(np.min(A_flat)) if A_flat.size else float('nan'),
        'max': float(np.max(A_flat)) if A_flat.size else float('nan'),
        'saturation_low': float(np.mean(A_flat < 0.01)) if A_flat.size else float('nan'),
        'saturation_high': float(np.mean(A_flat > 0.99)) if A_flat.size else float('nan'),
    }

    hist, bins = np.histogram(A_flat, bins=20) if A_flat.size else (np.array([]), np.array([]))
    fingerprint['action_distribution'] = {
        'histogram': hist.tolist() if hist.size else [],
        'bins': bins.tolist() if bins.size else [],
    }

    if A.ndim > 1 and A.shape[1] > 1:
        try:
            action_corr = np.corrcoef(A.reshape(len(A), -1).T)
            tri = np.triu_indices_from(action_corr, k=1)
            vals = np.abs(action_corr[tri])
            fingerprint['station_correlation'] = {
                'mean_correlation': float(np.mean(vals)),
                'max_correlation': float(np.max(vals)),
            }
        except Exception:
            pass

    return fingerprint
