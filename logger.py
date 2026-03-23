# ============================================================
#  logger.py  –  Structured Event Logger (SCADA-ready)
# ============================================================
import csv, json, os
from datetime import datetime, timezone
from pathlib import Path
from config import STATES

_ALARM_STATES = {"ANOMALY", "SENSOR_FAULT"}

_FIELDS = ["wall_utc","t_s","timestamp","state","E","theta","O",
           "n_faults","fault_sensors",
           "pca_P0","pca_P1","pca_T0","pca_T1","pca_F0","pca_F1",
           "mean_P","mean_T","mean_F","mean_F_cusum",
           "healthy_P","healthy_T","healthy_F","non_op","deviating"]


class EventLogger:
    def __init__(self, out_dir=".", run_tag=None):
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        tag  = run_tag or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        self._csv_path   = Path(out_dir) / f"events_{tag}.csv"
        self._alarm_path = Path(out_dir) / f"alarms_{tag}.jsonl"
        self._cf  = open(self._csv_path,   "w", newline="", buffering=1)
        self._af  = open(self._alarm_path, "w",              buffering=1)
        self._wr  = csv.DictWriter(self._cf, fieldnames=_FIELDS, extrasaction="ignore")
        self._wr.writeheader()
        self._n_alarm = 0; self._n_fault = 0
        print(f"[Logger] events → {self._csv_path}")
        print(f"[Logger] alarms → {self._alarm_path}")

    def log(self, result, pipeline=None):
        wall   = datetime.now(timezone.utc).isoformat()
        states = result.get("sensor_states", {})
        summ   = result.get("state_summary", {})
        pca    = result.get("pca_vecs", {})
        gm     = result.get("group_means", [0,0])
        faults = result.get("faults", [])
        E      = result["E"]; theta = result["theta"]; O = result["O"]
        from config import MAIN_SENSORS, CUSUM_GROUPS
        from ingestion import classify_column

        def cnt(grp):
            return sum(1 for k,v in states.items()
                       if v=="HEALTHY" and classify_column(k)==grp)

        def pv(g, i):
            v = pca.get(g)
            return round(float(v[i]),6) if v is not None and len(v)>i else 0.0

        row = {
            "wall_utc": wall, "t_s": result["t"],
            "timestamp": result.get("timestamp",""),
            "state": result["state"],
            "E": round(E,8), "theta": round(theta,8) if theta<1e15 else "inf",
            "O": round(O,6), "n_faults": len(faults),
            "fault_sensors": "|".join(faults),
            "pca_P0":pv("P",0),"pca_P1":pv("P",1),
            "pca_T0":pv("T",0),"pca_T1":pv("T",1),
            "pca_F0":pv("F",0),"pca_F1":pv("F",1),
            "mean_P": round(float(gm[0]),4) if len(gm)>0 else 0,
            "mean_T": 0,
            "mean_F": round(float(gm[1]),4) if len(gm)>1 else 0,
            "mean_F_cusum": 0,
            "healthy_P":cnt("P"),"healthy_T":cnt("T"),"healthy_F":cnt("F"),
            "non_op": summ.get("NON-OPERATIONAL",0),
            "deviating": summ.get("DEVIATING",0),
        }
        self._wr.writerow(row)

        if result["state"] in _ALARM_STATES:
            alarm = {"event": result["state"], "wall": wall,
                     "t": result["t"], "ts": result.get("timestamp",""),
                     "E": round(E,8), "theta": round(theta,8) if theta<1e15 else None,
                     "O": round(O,6), "faults": faults}
            self._af.write(json.dumps(alarm)+"\n")
            if result["state"]=="ANOMALY": self._n_alarm+=1
            else:                          self._n_fault+=1

    def close(self):
        self._cf.close(); self._af.close()
        print(f"[Logger] Closed. Alarms={self._n_alarm}  Faults={self._n_fault}")

    def __enter__(self): return self
    def __exit__(self,*_): self.close()
