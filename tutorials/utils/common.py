"""Common utilities shared across modules."""
import numpy as np
from collections import deque
from typing import Any, Callable, List, Dict, Tuple, Optional
from dataclasses import dataclass


def wrap_policy(policy_or_model: Any) -> Callable[[Any], np.ndarray]:
    """Universal policy wrapper supporting multiple interfaces."""
    if callable(policy_or_model):
        return lambda obs: np.asarray(policy_or_model(obs))
    
    if hasattr(policy_or_model, 'get_action'):
        def _wrapper(obs):
            out = policy_or_model.get_action(obs)
            if isinstance(out, tuple):
                out = out[0]
            return np.asarray(out)
        return _wrapper
    
    if hasattr(policy_or_model, 'predict'):
        return lambda obs: np.asarray(policy_or_model.predict(obs, deterministic=True)[0])
    
    if hasattr(policy_or_model, 'act'):
        return lambda obs: np.asarray(policy_or_model.act(obs))
    
    raise ValueError("Unsupported policy interface")


def check_env_interface(env: Any, required_attrs: List[str]) -> bool:
    """Safely check if environment has required attributes."""
    for attr_path in required_attrs:
        obj = env
        for attr in attr_path.split('.'):
            if not hasattr(obj, attr):
                return False
            obj = getattr(obj, attr)
    return True


def wrap_ma_policy(policy_or_model: Any) -> Callable[[Dict[str, np.ndarray]], Dict[str, np.ndarray]]:
    """Wrap a multi-agent policy interface into a standard callable.

    Expects and returns Dict[agent_id, np.ndarray].
    Supports callables and objects exposing get_actions/act/predict that return a dict.
    """
    if callable(policy_or_model):
        def _fn(obs_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
            out = policy_or_model(obs_dict)
            # Ensure numpy arrays
            return {aid: np.asarray(act) for aid, act in out.items()}
        return _fn

    if hasattr(policy_or_model, 'get_actions'):
        def _fn(obs_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
            out = policy_or_model.get_actions(obs_dict)
            return {aid: np.asarray(act) for aid, act in out.items()}
        return _fn

    if hasattr(policy_or_model, 'act'):
        def _fn(obs_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
            out = policy_or_model.act(obs_dict)
            return {aid: np.asarray(act) for aid, act in out.items()}
        return _fn

    if hasattr(policy_or_model, 'predict'):
        def _fn(obs_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
            out = policy_or_model.predict(obs_dict)
            if isinstance(out, tuple):
                out = out[0]
            return {aid: np.asarray(act) for aid, act in out.items()}
        return _fn

    raise ValueError("Unsupported multi-agent policy interface")


# -----------------------------
# New common helpers
# -----------------------------

def extract_from_obs(obs, key: str, env_space=None):
    """Extract component from observation regardless of format.

    Parameters
    ----------
    obs : Union[dict, np.ndarray]
        Observation (dict or flattened)
    key : str
        Key to extract ('demands', 'est_departures', etc.)
    env_space : Optional[gym.Space]
        Original observation space for unflattening
    """
    if isinstance(obs, dict):
        return np.asarray(obs.get(key, np.array([])))
    elif env_space is not None:
        try:
            from gymnasium import spaces  # type: ignore
            obs_dict = spaces.unflatten(env_space, obs)
            return np.asarray(obs_dict.get(key, np.array([])))
        except Exception:
            return np.array([])
    else:
        return np.array([])


from .notebook import log_info, log_warning


def solve_cvx_with_fallback(prob, solvers=None, verbose: bool = False):
    """Solve CVXPY problem with fallback to multiple solvers.

    Returns the status string if solved; raises RuntimeError otherwise.
    """
    try:
        import cvxpy as cp  # type: ignore
    except Exception as e:
        raise RuntimeError(f"CVXPY not available: {e}")

    if solvers is None:
        solvers = ['MOSEK', 'GUROBI', 'ECOS', 'OSQP', 'SCS']

    for solver_name in solvers:
        try:
            solver_const = getattr(cp, solver_name, None)
            if solver_const is None:
                continue
            if solver_name not in cp.installed_solvers():
                continue
            prob.solve(solver=solver_const, verbose=False, warm_start=True)
            if prob.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                if verbose:
                    log_info(f"Solved with {solver_name}")
                return prob.status
        except Exception as e:
            if verbose:
                log_warning(f"Failed with {solver_name}: {e}")
            continue

    raise RuntimeError(f"No solver succeeded. Installed: {getattr(cp, 'installed_solvers', lambda: [])()}")


def create_ev_env(
    site: str = 'caltech',
    date_range: Tuple[str, str] = ('2019-05-01', '2019-05-31'),
    seed: int = 42,
    safe_rl: bool = False,
    flatten: bool = True,
    noise: Optional[float] = None,
    noise_action: Optional[float] = None,
    dense_mode: bool = False,
    density_multiplier: float = 3.0,
    violation_weight: Optional[float] = None,
    data_generator: Any = None,
    **env_kwargs,
):
    """Standard environment factory for tutorials.

    Returns a single-agent EVChargingEnv; optionally flattens observations.
    """
    from envs.evcharging import EVChargingEnv, GMMsTraceGenerator  # local import to avoid heavy deps on module import
    try:
        from .rl import DictFlatteningWrapper  # type: ignore
    except Exception:
        DictFlatteningWrapper = None  # type: ignore

    # Lightweight, local subclass to increase daily session counts deterministically
    class DenseGMMsTraceGenerator(GMMsTraceGenerator):  # type: ignore
        def __init__(self, *args, density_mult: float = 3.0, **kwargs):
            super().__init__(*args, **kwargs)
            self._density_mult = max(float(density_mult), 0.0)

        def _create_events(self):  # override to scale the daily session count
            import pandas as pd  # local to avoid global dependency at import time
            import numpy as _np
            base_n = int(self.rng.choice(self.cnt))
            n = int(_np.ceil(base_n * self._density_mult)) if self._density_mult > 0 else base_n
            samples = self._sample(n)

            events = pd.DataFrame({
                'arrival': samples[:, self.ARRCOL].astype(int),
                'departure': samples[:, self.DEPCOL].astype(int),
                'estimated_departure': samples[:, self.ESTCOL].astype(int),
                'requested_energy (kWh)': _np.clip(samples[:, self.EREQCOL], 0, self.requested_energy_cap),
                'session_id': [str(__import__('uuid').uuid4()) for _ in range(n)]
            })
            events.sort_values('arrival', inplace=True)
            station_cnts = self.station_usage / self.station_usage.sum()
            station_dep = _np.full(len(self.station_ids), -1, dtype=_np.int32)
            station_ids = []
            for i in range(n):
                avail = _np.where(station_dep < events['arrival'].iloc[i])[0]
                if len(avail) == 0:
                    station_ids.append('NOT_AVAIL')
                else:
                    station_cnts_sum = station_cnts[avail].sum()
                    if station_cnts_sum <= 1e-5:
                        idx = self.rng.choice(avail)
                    else:
                        idx = self.rng.choice(avail, p=station_cnts[avail] / station_cnts_sum)
                    station_dep[idx] = max(events['departure'].iloc[i], station_dep[idx])
                    station_ids.append(self.station_ids[idx])
            events['station_id'] = station_ids
            events = events[events['station_id'] != 'NOT_AVAIL']
            return events.reset_index()

    # Choose generator based on override or dense_mode
    if data_generator is not None:
        gen = data_generator
    else:
        if dense_mode:
            gen = DenseGMMsTraceGenerator(
                site=site,
                date_period=date_range,
                seed=seed,
                density_mult=density_multiplier,
            )
        else:
            gen = GMMsTraceGenerator(
                site=site,
                date_period=date_range,
                seed=seed,
            )

    # Allow overrides via env_kwargs without passing duplicate keywords
    env_kwargs = dict(env_kwargs)  # shallow copy to avoid side effects
    env_kwargs.setdefault('moer_forecast_steps', 36)
    env_kwargs.setdefault('project_action_in_env', True)

    env = EVChargingEnv(
        data_generator=gen,
        safe_rl=safe_rl,
        noise=noise,
        noise_action=noise_action,
        **env_kwargs,
    )

    # Apply violation_weight convenience knob (update factor accordingly)
    if violation_weight is not None:
        try:
            vw = float(violation_weight)
            setattr(env, 'VIOLATION_WEIGHT', vw)
            # Ensure dependent factor stays consistent
            a2k = float(getattr(env, 'A_PERS_TO_KWH', 0.0))
            setattr(env, 'VIOLATION_FACTOR', a2k * vw)
        except Exception:
            # Non-fatal: keep env usable even if attributes are missing
            pass

    if flatten and not safe_rl and DictFlatteningWrapper is not None:
        return DictFlatteningWrapper(env)
    return env


def create_ma_ev_env(
    site: str = 'caltech',
    date_range: Tuple[str, str] = ('2019-05-01', '2019-05-31'),
    seed: int = 42,
    moer_forecast_steps: int = 36,
    project_action_in_env: bool = True,
    periods_delay: int = 0,
    dense_mode: bool = False,
    density_multiplier: float = 3.0,
    violation_weight: Optional[float] = None,
    data_generator: Any = None,
    **env_kwargs,
):
    """Factory for multi-agent EV charging env with optional dense tutorial mode."""
    from envs.evcharging import MultiAgentEVChargingEnv, GMMsTraceGenerator  # type: ignore

    # Local dense generator subclass
    class DenseGMMsTraceGenerator(GMMsTraceGenerator):  # type: ignore
        def __init__(self, *args, density_mult: float = 3.0, **kwargs):
            super().__init__(*args, **kwargs)
            self._density_mult = max(float(density_mult), 0.0)

        def _create_events(self):
            import pandas as pd
            import numpy as _np
            base_n = int(self.rng.choice(self.cnt))
            n = int(_np.ceil(base_n * self._density_mult)) if self._density_mult > 0 else base_n
            samples = self._sample(n)
            events = pd.DataFrame({
                'arrival': samples[:, self.ARRCOL].astype(int),
                'departure': samples[:, self.DEPCOL].astype(int),
                'estimated_departure': samples[:, self.ESTCOL].astype(int),
                'requested_energy (kWh)': _np.clip(samples[:, self.EREQCOL], 0, self.requested_energy_cap),
                'session_id': [str(__import__('uuid').uuid4()) for _ in range(n)]
            })
            events.sort_values('arrival', inplace=True)
            station_cnts = self.station_usage / self.station_usage.sum()
            station_dep = _np.full(len(self.station_ids), -1, dtype=_np.int32)
            station_ids = []
            for i in range(n):
                avail = _np.where(station_dep < events['arrival'].iloc[i])[0]
                if len(avail) == 0:
                    station_ids.append('NOT_AVAIL')
                else:
                    ssum = station_cnts[avail].sum()
                    if ssum <= 1e-5:
                        idx = self.rng.choice(avail)
                    else:
                        idx = self.rng.choice(avail, p=station_cnts[avail] / ssum)
                    station_dep[idx] = max(events['departure'].iloc[i], station_dep[idx])
                    station_ids.append(self.station_ids[idx])
            events['station_id'] = station_ids
            events = events[events['station_id'] != 'NOT_AVAIL']
            return events.reset_index()

    if data_generator is not None:
        gen = data_generator
    else:
        gen = (
            DenseGMMsTraceGenerator(site=site, date_period=date_range, seed=seed, density_mult=density_multiplier)
            if dense_mode else
            GMMsTraceGenerator(site=site, date_period=date_range, seed=seed)
        )

    env_kwargs = dict(env_kwargs)
    env_kwargs.setdefault('moer_forecast_steps', moer_forecast_steps)
    env_kwargs.setdefault('project_action_in_env', project_action_in_env)
    env_kwargs.setdefault('periods_delay', periods_delay)

    env = MultiAgentEVChargingEnv(
        data_generator=gen,
        **env_kwargs,
    )
    # Apply violation_weight if provided
    if violation_weight is not None:
        try:
            vw = float(violation_weight)
            setattr(env, 'VIOLATION_WEIGHT', vw)
            a2k = float(getattr(env, 'A_PERS_TO_KWH', 0.0))
            setattr(env, 'VIOLATION_FACTOR', a2k * vw)
        except Exception:
            pass
    return env


def apply_wrappers(env: Any, wrappers: Optional[list] = None) -> Any:
    """Apply a sequence of wrappers to an environment.

    Each element in `wrappers` can be:
    - a callable `fn(env) -> env`, or
    - a tuple `(WrapperClass, kwargs_dict)` to be instantiated as `WrapperClass(env, **kwargs_dict)`.

    Returns the final wrapped environment. If `wrappers` is None or empty, returns `env` unchanged.
    """
    if not wrappers:
        return env
    wrapped = env
    for w in wrappers:
        try:
            if callable(w) and not isinstance(w, tuple):
                wrapped = w(wrapped)
            elif isinstance(w, tuple) and len(w) in (1, 2):
                W = w[0]
                kw = (w[1] if len(w) == 2 and isinstance(w[1], dict) else {})
                wrapped = W(wrapped, **kw)
            else:
                # Unsupported spec; skip gracefully
                continue
        except Exception:
            # Do not fail composition on individual wrapper errors
            continue
    return wrapped


# -----------------------------
# Scenario wrappers (consolidated here for easier reuse)
# -----------------------------

class DepartureUncertaintyWrapper:
    """Add realistic uncertainty to departure estimates in observations.

    Modes
    -----
    - 'gaussian': zero-mean noise with std that grows with time-until-departure.
    - 'optimistic_bias': systematic underestimation of stay duration.

    Parameters
    ----------
    env : Any
        Base environment exposing reset()/step() with dict observations that include
        'est_departures' and 'demands'.
    uncertainty_mode : str
        One of {'gaussian','optimistic_bias'}.
    params : Optional[dict]
        Optional parameters per mode.
    rng_seed : Optional[int]
        Seed for the internal RNG used by the wrapper.
    verbose : bool
        If True, print a brief diagnostic of how many entries were perturbed.
    """

    def __init__(self, env: Any, uncertainty_mode: str = 'gaussian', params: Optional[dict] = None, rng_seed: Optional[int] = 42, verbose: bool = False):
        self.env = env
        self.uncertainty_mode = str(uncertainty_mode or 'gaussian')
        self.params = dict(params or {})
        self.true_departures: dict[Any, int] = {}
        self.rng = np.random.RandomState(None if rng_seed is None else int(rng_seed))
        self.verbose = bool(verbose)
        # Cache station IDs if available for stable keys across observation reshuffles
        try:
            self._station_ids = list(getattr(getattr(env, 'cn', object()), 'station_ids', []))
        except Exception:
            self._station_ids = []

    def _has_keys(self, obs: Any, keys: Tuple[str, ...]) -> bool:
        return isinstance(obs, dict) and all(k in obs for k in keys)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.true_departures = {}
        obs = self._add_departure_uncertainty(obs, is_reset=True)
        return obs, info

    def step(self, action):
        obs, r, d, t, info = self.env.step(action)
        obs = self._add_departure_uncertainty(obs, is_reset=False)
        return obs, r, d, t, info

    def _add_departure_uncertainty(self, obs: Any, is_reset: bool = False):
        if not self._has_keys(obs, ('demands', 'est_departures')):
            return obs

        demands = np.asarray(obs['demands']).reshape(-1)
        est = np.asarray(obs['est_departures']).astype(int).copy()
        n_modified = 0

        if self.uncertainty_mode == 'gaussian':
            # Std grows with time until departure; near departures are more certain.
            base_max_frac = float(self.params.get('max_std_frac', 0.2))  # cap at 20%
            for i, (demand, est_dep) in enumerate(zip(demands, est)):
                if demand <= 0:
                    continue
                key = self._station_ids[i] if (i < len(self._station_ids)) else i
                if is_reset and (key not in self.true_departures):
                    self.true_departures[key] = int(est_dep)
                true_dep = int(self.true_departures.get(key, est_dep))
                time_until = max(0, true_dep)
                # 5% at ~1h (12 steps), up to 20% at 4h+ (48+ steps)
                std_fraction = min(base_max_frac, 0.05 * np.sqrt(time_until / 12.0))
                std_steps = max(0.0, std_fraction * time_until)
                noise = self.rng.normal(0.0, std_steps)
                obs['est_departures'][i] = int(np.clip(true_dep + noise, 0, min(288, int(true_dep * 1.5) if true_dep > 0 else 288)))
                n_modified += 1

        elif self.uncertainty_mode == 'optimistic_bias':
            # People systematically underestimate how long they'll stay (workplace charging)
            avg_ext = float(self.params.get('avg_extension_steps', 12))  # ~1 hour
            for i, (demand, est_dep) in enumerate(zip(demands, est)):
                if demand <= 0:
                    continue
                key = self._station_ids[i] if (i < len(self._station_ids)) else i
                if is_reset and (key not in self.true_departures):
                    extension = float(self.rng.exponential(scale=avg_ext))
                    self.true_departures[key] = int(est_dep + extension)
                # MPC sees the optimistic estimate (original)
                obs['est_departures'][i] = int(est_dep)
                n_modified += 1

        else:
            # Unknown mode: no-op
            return obs

        if self.verbose:
            try:
                n_active = int(np.sum(demands > 0))
            except Exception:
                n_active = 0
            print(f"DepartureUncertainty: Perturbed {n_modified}/{n_active} entries (mode={self.uncertainty_mode})")
        return obs

    def __getattr__(self, name):
        return getattr(self.env, name)


# -----------------------------
# Shared planning evaluation settings
# -----------------------------

@dataclass
class PlanningEvalConfig:
    """Standardized evaluation-time configuration to make planning salient.

    Attributes
    ----------
    density_multiplier : float
        Scales daily EV session count. 3–5 restores headroom; 10+ oversaturates.
    violation_weight : float
        Scales safety penalty. Higher discourages aggressive violations.
    carbon_multiplier : float
        Emphasizes MOER timing.
    departure_extension_steps : int
        Average extension for optimistic-bias departure uncertainty (~12 steps = 1 hour).
    enable_forecast_error : bool
        Add MOER forecast errors that grow with horizon.
    forecast_error_model : str
        Model for forecast error ('linear_growth', 'sqrt_growth', 'bias_and_variance').
    forecast_linear_scale : float
        Per-step scale for linear growth model (default ~0.5% of true MOER per step).
    enable_demand_charge : bool
        Optional demand charge penalty toggle.
    demand_charge_per_kw : float
        $/kW for 15-min rolling average window when demand charge is enabled.
    seed : int
        Base seed for reproducibility.
    """

    density_multiplier: float = 3.0
    violation_weight: float = 5.0
    carbon_multiplier: float = 3.0
    departure_extension_steps: int = 24
    enable_forecast_error: bool = True
    forecast_error_model: str = 'linear_growth'
    forecast_linear_scale: float = 0.0075
    enable_demand_charge: bool = False
    demand_charge_per_kw: float = 200.0
    seed: int = 42


def get_planning_eval_env_fn(
    cfg: PlanningEvalConfig,
    *,
    site: str = 'caltech',
    date_range: Tuple[str, str] = ('2019-05-01', '2019-08-31'),
    flatten: bool = False,
    moer_forecast_steps: int = 36,
    project_action_in_env: bool = False,
    dense_mode: bool = True,
) -> Callable[[], Any]:
    """Return an env_fn() that constructs a planning-friendly evaluation env.

    The returned env applies:
    - Reduced density
    - Increased violation penalty
    - Carbon emphasis
    - Departure uncertainty (optimistic bias)
    - Optional forecast error and demand charge
    """

    def _env_fn():
        base = create_ev_env(
            site=site,
            date_range=date_range,
            seed=cfg.seed,
            flatten=flatten,
            moer_forecast_steps=moer_forecast_steps,
            project_action_in_env=project_action_in_env,
            dense_mode=dense_mode,
            density_multiplier=cfg.density_multiplier,
            violation_weight=cfg.violation_weight,
        )
        wrappers = [
            (CarbonEmphasisWrapper, {'multiplier': cfg.carbon_multiplier}),
            (DepartureUncertaintyWrapper, {
                'uncertainty_mode': 'optimistic_bias',
                'params': {'avg_extension_steps': int(cfg.departure_extension_steps)},
                'rng_seed': int(cfg.seed) + 1,
            }),
        ]
        if cfg.enable_forecast_error:
            wrappers.append((GrowingForecastErrorWrapper, {
                'error_model': cfg.forecast_error_model,
                'params': {'linear_scale': float(cfg.forecast_linear_scale)},
                'rng_seed': int(cfg.seed) + 2,
            }))
        if cfg.enable_demand_charge:
            wrappers.append((DemandChargeWrapper, {
                'charge_per_kw': float(cfg.demand_charge_per_kw),
                'window_minutes': 15,
                'expose_peak_in_obs': False,
            }))
        return apply_wrappers(base, wrappers)

    return _env_fn


class DemandChargeWrapper:
    """Add demand charge penalties based on rolling 15-minute average peak kW.

    This wrapper infers executed aggregate current from the reward breakdown's
    profit term and converts it to kW using env constants and step duration.
    A penalty is applied only when the rolling window average exceeds the
    historical peak (incremental billing), keeping rewards additive.
    """

    def __init__(
        self,
        env: Any,
        charge_per_kw: float = 200.0,
        window_minutes: int = 15,
        expose_peak_in_obs: bool = False,
        amps_source: str = 'reward',
        verbose: bool = False,
    ):
        self.env = env
        self.charge_per_kw = float(charge_per_kw)
        self.window_minutes = int(window_minutes)
        self.expose_peak_in_obs = bool(expose_peak_in_obs)
        self.amps_source = str(amps_source)
        self.verbose = bool(verbose)

        # Infer step duration (minutes)
        self.step_minutes = self._infer_step_minutes()
        self.window_steps = int(np.ceil(self.window_minutes / max(self.step_minutes, 1e-9)))

        # State
        self.kw_window = deque(maxlen=self.window_steps)
        self.peak_kw_so_far = 0.0
        self.cumulative_demand_charge = 0.0
        self.peak_tolerance = 1e-6
        # Track previous cumulative profit to derive per-step profit
        self._prev_cum_profit = 0.0

    def _infer_step_minutes(self) -> float:
        if hasattr(self.env, 'step_minutes'):
            try:
                return float(self.env.step_minutes)
            except Exception:
                pass
        if hasattr(self.env, 'max_timestep'):
            try:
                return 1440.0 / float(getattr(self.env, 'max_timestep'))
            except Exception:
                pass
        return 5.0

    def reset(self, **kwargs):
        self.kw_window.clear()
        self.peak_kw_so_far = 0.0
        self.cumulative_demand_charge = 0.0
        self._prev_cum_profit = 0.0
        obs, info = self.env.reset(**kwargs)
        if self.expose_peak_in_obs and isinstance(obs, dict):
            obs['peak_kw_so_far'] = np.array([self.peak_kw_so_far], dtype=np.float32)
        return obs, info

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        rb = info.get('reward_breakdown', {})

        # Prefer base profit if present (in case dynamic pricing is layered on)
        cum_profit = float(rb.get('profit_base', rb.get('profit', 0.0)))
        profit_factor = float(getattr(self.env, 'PROFIT_FACTOR', 0.0))

        # Convert cumulative to per-step profit via differencing
        profit_step = max(0.0, cum_profit - self._prev_cum_profit)
        self._prev_cum_profit = cum_profit

        if self.amps_source == 'reward' and profit_factor > 0:
            agg_amps = profit_step / profit_factor
        else:
            agg_amps = 0.0
            if self.verbose and profit_factor <= 0 and self.amps_source == 'reward':
                print(f"[DemandCharge] PROFIT_FACTOR={profit_factor} → cannot infer executed amps; skipping demand charge this step.")

        # Convert to kW using step duration
        a2k = float(getattr(self.env, 'A_PERS_TO_KWH', 0.0))
        kw_instant = agg_amps * a2k * (60.0 / max(self.step_minutes, 1e-9))

        # Update rolling average over window (15-min avg)
        self.kw_window.append(kw_instant)
        window_avg_kw = float(np.mean(self.kw_window)) if len(self.kw_window) > 0 else 0.0

        # Incremental penalty only when new peak is set
        penalty = 0.0
        if window_avg_kw > self.peak_kw_so_far + self.peak_tolerance:
            delta = window_avg_kw - self.peak_kw_so_far
            penalty = delta * self.charge_per_kw
            self.peak_kw_so_far = window_avg_kw
            self.cumulative_demand_charge += penalty
            if self.verbose:
                print(f"[DemandCharge] New 15-min avg peak: {window_avg_kw:.2f} kW → penalty ${penalty:.2f}")

        if penalty > 0:
            r -= penalty
            # Step-specific penalty for transparency (not cumulative)
            rb['demand_charge_step'] = penalty

        # Update breakdown and optionally obs
        rb['demand_charge'] = self.cumulative_demand_charge
        rb['instant_kw'] = kw_instant
        rb['window_avg_kw'] = window_avg_kw
        # Backward compatibility key (peak within window historically mislabeled)
        rb['window_peak_kw'] = window_avg_kw
        rb['peak_kw_so_far'] = self.peak_kw_so_far
        rb['profit_step'] = profit_step
        info['reward_breakdown'] = rb

        if self.expose_peak_in_obs and isinstance(obs, dict):
            obs['peak_kw_so_far'] = np.array([self.peak_kw_so_far], dtype=np.float32)

        return obs, r, term, trunc, info

    def close(self):
        try:
            return self.env.close()
        except Exception:
            return None

    def __getattr__(self, name):
        return getattr(self.env, name)


class GrowingForecastErrorWrapper:
    """Add MOER forecast errors that grow with horizon.

    Modes: 'linear_growth', 'sqrt_growth', 'bias_and_variance'
    """

    def __init__(self, env: Any, error_model: str = 'linear_growth', params: Optional[dict] = None, rng_seed: Optional[int] = 42, verbose: bool = False):
        self.env = env
        self.error_model = str(error_model or 'linear_growth')
        self.params = dict(params or {})
        self.rng = np.random.RandomState(None if rng_seed is None else int(rng_seed))
        self.t = 0
        self.true_moer_trajectory: Optional[np.ndarray] = None  # type: ignore
        self.verbose = bool(verbose)

    def reset(self, **kwargs):
        self.t = 0
        obs, info = self.env.reset(**kwargs)
        # Try to build a true MOER trajectory from env.moer if available
        self._generate_true_moer(getattr(self.env, 'moer', None))
        obs = self._add_forecast_errors(obs)
        return obs, info

    def step(self, action):
        self.t += 1
        obs, r, d, t, info = self.env.step(action)
        obs = self._add_forecast_errors(obs)
        return obs, r, d, t, info

    def _generate_true_moer(self, moer_matrix: Optional[np.ndarray]):  # type: ignore
        # Use environment's episode length when available
        T = int(getattr(self.env, 'max_timestep', 288))
        if moer_matrix is not None and isinstance(moer_matrix, np.ndarray) and moer_matrix.ndim >= 2 and moer_matrix.shape[0] >= T:
            base = np.asarray(moer_matrix[:T, 0], dtype=float).copy()
        else:
            # Fallback synthetic base: daily cycle
            t = np.arange(T)
            base = 0.4 + 0.2 * np.sin(2 * np.pi * t / 144)
        # Add smoothed noise
        variations = 0.05 * self.rng.randn(T)
        kernel = np.ones(6) / 6.0
        variations = np.convolve(variations, kernel, mode='same')
        # Add occasional spike events
        for _ in range(int(self.params.get('n_events', 5))):
            event_t = int(self.rng.randint(0, T))
            magnitude = float(self.rng.uniform(0.1, 0.3))
            duration = int(self.rng.randint(3, 12))
            start = max(0, event_t - duration // 2)
            end = min(T, event_t + duration // 2)
            window = np.hanning(max(1, end - start))
            base[start:end] += magnitude * window
        self.true_moer_trajectory = np.clip(base, 0.0, 1.0)

    def _add_forecast_errors(self, obs: Any):
        if not isinstance(obs, dict) or 'forecasted_moer' not in obs:
            return obs
        if self.true_moer_trajectory is None:
            return obs

        forecast = np.asarray(obs['forecasted_moer'], dtype=float)
        k = len(forecast)

        model = self.error_model
        n_modified = 0
        for i in range(k):
            idx = min(self.t + i + 1, len(self.true_moer_trajectory) - 1)
            true_m = float(self.true_moer_trajectory[idx])
            if model == 'linear_growth':
                # 0% at t+0, ~20% at t+36 as default
                scale = float(self.params.get('linear_scale', 0.005))  # per step
                error_std = max(0.0, scale * i) * true_m
                err = self.rng.normal(0.0, error_std)
                forecast[i] = np.clip(true_m + err, 0.0, 1.0)
                n_modified += 1
            elif model == 'sqrt_growth':
                # ~15% at 36 steps
                scale = float(self.params.get('sqrt_scale', 0.025))
                error_std = max(0.0, scale * np.sqrt(max(i, 0))) * true_m
                err = self.rng.normal(0.0, error_std)
                forecast[i] = np.clip(true_m + err, 0.0, 1.0)
            elif model == 'bias_and_variance':
                bias_per_step = float(self.params.get('bias_per_step', -0.002))
                var_scale = float(self.params.get('var_scale', 0.02))
                bias = bias_per_step * i
                error_std = max(0.0, var_scale * np.sqrt(max(i, 0))) * true_m
                err = self.rng.normal(0.0, error_std)
                forecast[i] = np.clip(true_m + bias + err, 0.0, 1.0)
                n_modified += 1
            else:
                # Unknown model -> leave unchanged
                pass
        obs['forecasted_moer'] = forecast.astype(np.float32)

        # Optionally synchronize prev_moer to true value if present
        if 'prev_moer' in obs and isinstance(obs['prev_moer'], (list, np.ndarray)) and len(obs['prev_moer']) > 0:
            obs['prev_moer'][0] = float(self.true_moer_trajectory[min(self.t, len(self.true_moer_trajectory)-1)])

        if self.verbose:
            print(f"ForecastError: Perturbed {n_modified}/{k} entries (model={self.error_model}) at t={self.t}")

        return obs

    def __getattr__(self, name):
        return getattr(self.env, name)


class ConstraintTighteningWrapper:
    """Scale env.cn.magnitudes by a factor and restore on close if requested."""
    def __init__(self, env: Any, scale_factor: float = 0.8, restore_on_close: bool = True):
        self.env = env
        self.scale_factor = float(scale_factor)
        self.restore_on_close = bool(restore_on_close)
        self._original_magnitudes = None
        try:
            if hasattr(env, 'cn') and hasattr(env.cn, 'magnitudes') and getattr(env.cn, 'magnitudes') is not None:
                self._original_magnitudes = np.array(env.cn.magnitudes, copy=True)
                env.cn.magnitudes = env.cn.magnitudes * self.scale_factor
        except Exception:
            # Graceful no-op if attributes are missing
            pass

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)

    def close(self):
        try:
            if self.restore_on_close and (self._original_magnitudes is not None):
                self.env.cn.magnitudes = self._original_magnitudes
        except Exception:
            pass
        try:
            return self.env.close()
        except Exception:
            return None

    def __getattr__(self, name):
        return getattr(self.env, name)


class CarbonEmphasisWrapper:
    """Increase carbon weight coherently.

    If only multiplier is provided, scales CARBON_COST_FACTOR by that multiplier.
    If absolute_price is provided (e.g., $/metric-ton), sets both attributes coherently.
    Restores original values on close.
    """
    def __init__(self, env: Any, multiplier: Optional[float] = None, absolute_price: Optional[float] = None):
        self.env = env
        self._original = {}
        try:
            if absolute_price is not None:
                self._original['CO2_COST_PER_METRIC_TON'] = getattr(env, 'CO2_COST_PER_METRIC_TON', None)
                self._original['CARBON_COST_FACTOR'] = getattr(env, 'CARBON_COST_FACTOR', None)
                setattr(env, 'CO2_COST_PER_METRIC_TON', float(absolute_price))
                a_pers_to_kwh = float(getattr(env, 'A_PERS_TO_KWH', 0.0))
                setattr(env, 'CARBON_COST_FACTOR', a_pers_to_kwh * (float(absolute_price) / 1000.0))
            elif multiplier is not None:
                self._original['CARBON_COST_FACTOR'] = getattr(env, 'CARBON_COST_FACTOR', None)
                setattr(env, 'CARBON_COST_FACTOR', float(getattr(env, 'CARBON_COST_FACTOR', 0.0)) * float(multiplier))
        except Exception:
            pass

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)

    def close(self):
        try:
            if 'CO2_COST_PER_METRIC_TON' in self._original and self._original['CO2_COST_PER_METRIC_TON'] is not None:
                setattr(self.env, 'CO2_COST_PER_METRIC_TON', self._original['CO2_COST_PER_METRIC_TON'])
            if 'CARBON_COST_FACTOR' in self._original and self._original['CARBON_COST_FACTOR'] is not None:
                setattr(self.env, 'CARBON_COST_FACTOR', self._original['CARBON_COST_FACTOR'])
        except Exception:
            pass
        try:
            return self.env.close()
        except Exception:
            return None

    def __getattr__(self, name):
        return getattr(self.env, name)
