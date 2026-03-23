# ============================================================
#  op_threshold.py  –  Phase 7: Operational-Aware Dynamic Threshold
#  RLS: E(t) = α·O(t) + β   →   θ(t) = predicted + ANOMALY_SIGMA·σ_res
# ============================================================
import numpy as np
from collections import deque
from config import RLS_LAMBDA, REG_MIN, OP_HIGH_FRAC, ANOMALY_SIGMA, DELTA


class _RLS:
    def __init__(self):
        self.theta = np.zeros(2)
        self.P     = np.eye(2) * 1e4
        self.n     = 0

    def update(self, O: float, E: float):
        phi   = np.array([O, 1.0])
        err   = E - phi @ self.theta
        denom = RLS_LAMBDA + phi @ self.P @ phi
        K     = (self.P @ phi) / denom
        self.theta += K * err
        self.P      = (self.P - np.outer(K, phi @ self.P)) / RLS_LAMBDA
        self.n     += 1

    def predict(self, O: float) -> float:
        return float(self.theta[0]*O + self.theta[1])

    @property
    def ready(self): return self.n >= REG_MIN


class OperationalThreshold:
    def __init__(self):
        self.rls    = _RLS()
        self._E_buf = deque(maxlen=600)
        self._O_buf = deque(maxlen=600)

    def op_energy(self, op_vals: np.ndarray) -> float:
        return float(np.nanmean(np.abs(op_vals))) if len(op_vals) else 0.0

    def update(self, E: float, op_vals: np.ndarray):
        O = self.op_energy(op_vals)
        self._E_buf.append(E)
        self._O_buf.append(O)
        if np.isfinite(E) and E > 0:
            self.rls.update(O, E)
        return O, self._theta(O)

    def is_high_ops(self, op_vals: np.ndarray) -> bool:
        if not len(op_vals): return False
        return float(np.nanmean(np.abs(op_vals) > 0.5)) > OP_HIGH_FRAC

    def _theta(self, O: float) -> float:
        if not self.rls.ready:
            if len(self._E_buf) > 5:
                return float(np.percentile(list(self._E_buf), 95)) * 1.5
            return float("inf")
        resid   = np.array(self._E_buf) - np.array(
                      [self.rls.predict(o) for o in self._O_buf])
        sigma   = float(np.std(resid)) + DELTA
        base    = self.rls.predict(O)
        return float(max(base + ANOMALY_SIGMA*sigma, base*1.1, 0.0))

    @property
    def alpha(self): return float(self.rls.theta[0])
    @property
    def beta(self):  return float(self.rls.theta[1])
