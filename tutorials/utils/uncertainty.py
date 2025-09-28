"""
Uncertainty wrappers for EV Charging notebooks.

These wrappers modify observation fields to inject realistic uncertainty patterns
without changing the base environment's internal dynamics.

- DepartureUncertaintyWrapper: perturbs 'est_departures' to simulate estimation error.
- GrowingForecastErrorWrapper: perturbs 'forecasted_moer' with horizon-dependent error
  and can also synchronize 'prev_moer' to a generated true trajectory.

Wrappers are light-weight (no Gym dependency) and simply forward reset/step.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple
import numpy as np


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

        return obs

    def __getattr__(self, name):
        return getattr(self.env, name)


class GrowingForecastErrorWrapper:
    """Add MOER forecast errors that grow with horizon.

    Modes
    -----
    - 'linear_growth': error std grows linearly with horizon.
    - 'sqrt_growth': error std grows ~ sqrt(horizon).
    - 'bias_and_variance': adds systematic bias plus horizon-dependent variance.

    Parameters
    ----------
    env : Any
        Base environment exposing reset()/step() with dict observations that include
        'forecasted_moer' and 'prev_moer'. The base env should expose attribute
        `moer` after reset for ground-truth guidance.
    error_model : str
        One of {'linear_growth','sqrt_growth','bias_and_variance'}.
    params : Optional[dict]
        Optional parameters per model.
    rng_seed : Optional[int]
        Seed for the internal RNG used by the wrapper.
    """

    def __init__(self, env: Any, error_model: str = 'linear_growth', params: Optional[dict] = None, rng_seed: Optional[int] = 42, verbose: bool = False):
        self.env = env
        self.error_model = str(error_model or 'linear_growth')
        self.params = dict(params or {})
        self.rng = np.random.RandomState(None if rng_seed is None else int(rng_seed))
        self.t = 0
        self.true_moer_trajectory: Optional[np.ndarray] = None
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

    def _generate_true_moer(self, moer_matrix: Optional[np.ndarray]):
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
