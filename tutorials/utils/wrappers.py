"""
Robust attribute wrappers for EVChargingEnv to support scenario tweaks safely.

- ConstraintTighteningWrapper: scales network constraint magnitudes with optional restore on close.
- CarbonEmphasisWrapper: scales carbon cost parameters coherently.

These wrappers avoid fragile inline lambdas and encapsulate state changes.
"""
from __future__ import annotations

from typing import Any, Optional
import numpy as np


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
    """Increase carbon weight coherently by setting CO2_COST_PER_METRIC_TON and CARBON_COST_FACTOR.

    If only multiplier is provided, scales CARBON_COST_FACTOR by that multiplier.
    If absolute_price is provided (e.g., $/metric-ton), sets both attributes coherently.
    """
    def __init__(self, env: Any, multiplier: Optional[float] = None, absolute_price: Optional[float] = None):
        self.env = env
        self._original = {}
        try:
            if absolute_price is not None:
                # Save originals
                self._original['CO2_COST_PER_METRIC_TON'] = getattr(env, 'CO2_COST_PER_METRIC_TON', None)
                self._original['CARBON_COST_FACTOR'] = getattr(env, 'CARBON_COST_FACTOR', None)
                setattr(env, 'CO2_COST_PER_METRIC_TON', float(absolute_price))
                # Recompute CARBON_COST_FACTOR from constants in env when possible
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
            # Restore originals if recorded
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
