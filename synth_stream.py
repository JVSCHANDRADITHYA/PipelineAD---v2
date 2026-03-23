# ============================================================
#  synth_stream.py  –  Synthetic Pipeline Stream
#  Drop-in replacement for CSVStreamer during testing.
# ============================================================
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from config import COLD_START

P_COLS  = [f"P-SIM-{i:04d}" for i in range(1, 25)]
T_COLS  = [f"T-SIM-{i:04d}" for i in range(1, 21)]
F_COLS  = [f"F-SIM-{i:04d}" for i in range(1, 13)]
OP_COLS = [f"MOV-SIM-{i:02d}" for i in range(1, 16)] + \
          [f"PUMP-SIM-{i:02d}" for i in range(1, 9)]
ALL_DATA = P_COLS + T_COLS + F_COLS + OP_COLS


class _OU:
    def __init__(self, mu, theta, sigma, dt=5.0, seed=None):
        r = np.random.RandomState(seed)
        self.mu=mu; self.theta=theta; self.sigma=sigma; self.dt=dt
        self.x = mu + r.randn()*sigma
        self._r = r
    def step(self):
        self.x += (-self.theta*(self.x-self.mu)*self.dt
                   + self.sigma*np.sqrt(self.dt)*self._r.randn())
        return float(self.x)


class SyntheticStreamer:
    def __init__(self, n_rows=2000, dt=5.0, seed=42, leak_magnitude=20.0):
        rng = np.random.default_rng(seed)
        self.n_rows = n_rows
        self.columns = ["Seconds", "Timestamp_IST"] + ALL_DATA
        self.groups  = {"P": P_COLS, "T": T_COLS, "F": F_COLS,
                        "OP": OP_COLS, "OTHER": []}

        # OU processes
        self._p  = [_OU(60+rng.uniform(-3,3), 0.03, 2.0, dt, seed+i) for i in range(len(P_COLS))]
        self._t  = [_OU(38+rng.uniform(-2,2), 0.02, 1.2, dt, seed+100+i) for i in range(len(T_COLS))]
        self._f  = [_OU(800+rng.uniform(-20,20), 0.04, 25, dt, seed+200+i) for i in range(len(F_COLS))]
        self._op = {c: float(rng.random()<0.6) for c in OP_COLS}
        self._rng = rng
        self._dt  = dt
        self._t0  = datetime(2025, 5, 16, 15, 24, 45)

        # Fault schedule
        warm = COLD_START
        self._faults = [
            {"type":"sensor_drift",  "start":warm+60,  "end":warm+160,
             "sensor":P_COLS[3],     "mag":60.0},   # 3× larger → clear MAD outlier
            {"type":"op_burst",      "start":warm+300,  "end":warm+340},
            {"type":"sensor_spike",  "start":warm+450,  "end":warm+510,
             "sensor":T_COLS[5],     "mag":40.0},   # large spike > PEER_Z
            {"type":"system_leak",   "start":warm+600,  "end":warm+780,
             "mag": leak_magnitude},
        ]

        self._df  = self._generate()
        self._idx = 0
        print("="*56)
        print(f"  SyntheticStreamer : {n_rows} rows, dt={dt}s")
        print(f"  Faults : {[f['type'] for f in self._faults]}")
        print(f"  Leak magnitude : {leak_magnitude} bar / flow_units")
        print("="*56)

    def _generate(self):
        fault_map = {}
        for f in self._faults:
            for i in range(f["start"], min(f["end"], self.n_rows)):
                fault_map.setdefault(i, []).append(f)

        rows = []
        t    = 0.0
        for idx in range(self.n_rows):
            ts = (self._t0 + timedelta(seconds=t)).strftime("%Y-%m-%d %H:%M:%S")
            p  = {c: ou.step() for c, ou in zip(P_COLS, self._p)}
            te = {c: ou.step() for c, ou in zip(T_COLS, self._t)}
            f  = {c: ou.step() for c, ou in zip(F_COLS, self._f)}
            # Occasional valve flips
            for c in OP_COLS:
                if self._rng.random() < 0.003:
                    self._op[c] = 1.0 - self._op[c]
            op = dict(self._op)

            for fault in fault_map.get(idx, []):
                prog = (idx - fault["start"]) / max(fault["end"]-fault["start"], 1)
                ft   = fault["type"]
                if ft == "sensor_drift":
                    s = fault["sensor"]
                    if s in p: p[s] += prog * fault["mag"]
                elif ft == "sensor_spike":
                    s = fault["sensor"]
                    spike = np.sin(prog * np.pi)
                    if s in te: te[s] += spike * fault["mag"]
                elif ft == "system_leak":
                    drop = prog * fault["mag"]
                    for c in p:  p[c]  -= drop * (1 + self._rng.normal(0, 0.01))
                    for c in f:  f[c]  -= drop * 50 * (1 + self._rng.normal(0, 0.01))
                elif ft == "op_burst":
                    for c in OP_COLS:
                        if self._rng.random() < 0.08:
                            self._op[c] = 1.0 - self._op[c]
                    op = dict(self._op)

            # Frozen sensor
            p[P_COLS[0]] = 0.0

            row = {"Seconds": t, "Timestamp_IST": ts}
            row.update(p); row.update(te); row.update(f); row.update(op)
            rows.append(row)
            t += self._dt

        return pd.DataFrame(rows, columns=self.columns)

    def __len__(self):  return self.n_rows
    def __iter__(self): return self
    def __next__(self):
        if self._idx >= self.n_rows: raise StopIteration
        r = self._df.iloc[self._idx].to_dict()
        self._idx += 1
        return r

    @property
    def idx(self): return self._idx
    def reset(self): self._idx = 0

    def get_group_values(self, row, group):
        return {c: float(row.get(c, np.nan)) for c in self.groups.get(group, [])}

    def save_csv(self, path):
        self._df.to_csv(path, index=False)
        return path
