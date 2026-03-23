# ============================================================
#  peer_deviation.py  –  Phase 3: MAD Peer Deviation + Hysteresis
# ============================================================
import numpy as np
from collections import defaultdict
from config import PEER_Z, PEER_HYS, DELTA

_counters: dict = defaultdict(int)


def peer_deviation_check(sensor_vals: dict, healthy: list) -> tuple:
    if len(healthy) < 3:
        return [], {}
    vals  = np.array([sensor_vals.get(s, np.nan) for s in healthy], dtype=float)
    ok    = np.isfinite(vals)
    if ok.sum() < 3:
        return [], {}
    v_ok   = vals[ok]
    n_ok   = [s for s, f in zip(healthy, ok) if f]
    median = np.median(v_ok)
    sigma  = 1.4826 * np.median(np.abs(v_ok - median)) + DELTA
    z_scores  = {}
    deviating = []
    for name, val in zip(n_ok, v_ok):
        z = float(abs(val - median) / sigma)
        z_scores[name] = z
        _counters[name] = _counters[name] + 1 if z > PEER_Z else 0
        if _counters[name] >= PEER_HYS:
            deviating.append(name)
    return deviating, z_scores


def reset_all():
    _counters.clear()
