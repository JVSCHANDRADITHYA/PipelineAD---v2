# ============================================================
#  ingestion.py  –  Phase 1: Data Ingestion & Sensor Classification
# ============================================================
import pandas as pd
import numpy as np
from config import MAIN_SENSORS, OP_KEYS


def classify_column(col: str) -> str:
    tokens = col.replace("_", "-").split("-")
    for tok in tokens:
        t = tok.upper()
        for grp, prefixes in MAIN_SENSORS.items():
            for p in prefixes:
                if t == p.upper() or t.startswith(p.upper()):
                    return grp
        for k in OP_KEYS:
            if t == k.upper() or t.startswith(k.upper()):
                return "OP"
    return "OTHER"


def classify_sensors(columns: list) -> dict:
    skip   = {"Seconds", "Timestamp_IST"}
    groups = {k: [] for k in list(MAIN_SENSORS.keys()) + ["OP", "OTHER"]}
    for col in columns:
        if col not in skip:
            groups[classify_column(col)].append(col)
    return groups


class CSVStreamer:
    def __init__(self, path: str):
        self.df      = pd.read_csv(path)
        self.idx     = 0
        self.columns = list(self.df.columns)
        self.groups  = classify_sensors(self.columns)
        print("=" * 56)
        print(f"  Loaded  : {len(self.df):,} rows × {len(self.columns)} cols")
        for g, c in self.groups.items():
            if c: print(f"  Group {g:5s}: {len(c):3d} sensors")
        print("=" * 56)

    def __len__(self):   return len(self.df)
    def __iter__(self):  return self

    def __next__(self) -> dict:
        if self.idx >= len(self.df): raise StopIteration
        row = self.df.iloc[self.idx].to_dict()
        self.idx += 1
        return row

    def reset(self): self.idx = 0

    def get_group_values(self, row: dict, group: str) -> dict:
        out = {}
        for col in self.groups.get(group, []):
            try:    out[col] = float(row[col])
            except: out[col] = np.nan
        return out
