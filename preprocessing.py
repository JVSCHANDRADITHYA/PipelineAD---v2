# ============================================================
#  preprocessing.py  –  Phase 2: Cold-Start & Frozen Detection
# ============================================================
import numpy as np
from collections import Counter
from config import COLD_START, DELTA


class SensorStateManager:
    EXCLUDED        = "EXCLUDED"
    NON_OPERATIONAL = "NON-OPERATIONAL"
    DEVIATING       = "DEVIATING"
    HEALTHY         = "HEALTHY"

    def __init__(self, sensors: list):
        self.states  = {s: self.HEALTHY for s in sensors}
        self._n      = {s: 0            for s in sensors}
        self._mean   = {s: 0.0          for s in sensors}
        self._M2     = {s: 0.0          for s in sensors}
        self.t       = 0

    def update(self, vals: dict) -> None:
        self.t += 1
        for s, v in vals.items():
            if self.states.get(s) in (self.EXCLUDED, self.NON_OPERATIONAL):
                continue
            if not np.isfinite(v):
                self.states[s] = self.EXCLUDED
                continue
            # Welford
            n = self._n[s] + 1
            self._n[s]  = n
            d           = v - self._mean[s]
            self._mean[s] += d / n
            self._M2[s]   += d * (v - self._mean[s])

        if self.t == COLD_START:
            frozen = 0
            for s in list(self.states):
                if self.states[s] not in (self.EXCLUDED, self.NON_OPERATIONAL):
                    std = (self._M2[s] / max(self._n[s]-1, 1)) ** 0.5
                    if std < DELTA:
                        self.states[s] = self.NON_OPERATIONAL
                        frozen += 1
            if frozen:
                print(f"[PreProc] Cold-start: {frozen} frozen → NON-OPERATIONAL")

    def set_deviating(self, sensors: list) -> None:
        for s in sensors:
            if self.states.get(s) == self.HEALTHY:
                self.states[s] = self.DEVIATING

    def clear_deviating(self, sensors: list) -> None:
        for s in sensors:
            if self.states.get(s) == self.DEVIATING:
                self.states[s] = self.HEALTHY

    def get_healthy(self, group_sensors: list) -> list:
        return [s for s in group_sensors if self.states.get(s) == self.HEALTHY]

    def get_pca_eligible(self, group_sensors: list) -> list:
        bad = {self.EXCLUDED, self.NON_OPERATIONAL}
        return [s for s in group_sensors if self.states.get(s) not in bad]

    def is_warm(self) -> bool:
        return self.t >= COLD_START

    def progress(self) -> float:
        return min(self.t / COLD_START, 1.0)

    def summary(self) -> dict:
        return dict(Counter(self.states.values()))
