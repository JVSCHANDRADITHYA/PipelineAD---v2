# ============================================================
#  pipeline.py  –  PipelineAD Orchestrator
# ============================================================
import numpy as np
from typing import Any
from preprocessing   import SensorStateManager
from peer_deviation  import peer_deviation_check
from pca_module      import OnlineGroupPCA
from temporal_model  import make_temporal_model
from op_threshold    import OperationalThreshold
from decision        import decide
from config          import MAIN_SENSORS, PCA_COMPONENTS, CUSUM_GROUPS


class PipelineAD:
    def __init__(self, groups: dict):
        self.groups = groups

        # Phase 2 – analog sensors only (OP sensors excluded: binary, legit frozen → flse NON-OPEATIONAL otherwise
        analog = sum([groups.get(k, []) for k in MAIN_SENSORS], [])
        self.state_mgr = SensorStateManager(analog)

        # Phase 4
        self.pcas = {g: OnlineGroupPCA(g) for g in MAIN_SENSORS}

        # Phase 6 – CUSUM groups index into CUSUM_GROUPS list
        cusum_n = len(CUSUM_GROUPS)
        pca_dim = len(MAIN_SENSORS) * PCA_COMPONENTS
        self.ae = make_temporal_model(pca_dim, cusum_n)

        # Phase 7
        self.op_thresh   = OperationalThreshold()
        self._prev_devs  = {g: [] for g in MAIN_SENSORS}
        self.last_group_vals: dict = {}

    # ── main entry point ─────────────────────────────────────────────────────
    def process(self, row: dict) -> dict[str, Any]:
        result: dict[str, Any] = {
            "t": float(row.get("Seconds", 0)),
            "timestamp": row.get("Timestamp_IST", ""),
            "sensor_states": {}, "z_scores": {},
            "pca_vecs": {}, "fused_z": np.zeros(len(MAIN_SENSORS)*PCA_COMPONENTS),
            "group_means": np.zeros(len(CUSUM_GROUPS)),
            "E": 0.0, "theta": float("inf"), "O": 0.0,
            "state": "COLD_START", "faults": [], "state_summary": {},
        }

        # ── Phase 1: extract group values ────────────────────────────
        gv = {}
        for g in list(MAIN_SENSORS.keys()) + ["OP"]:
            d = {}
            for col in self.groups.get(g, []):
                try:    d[col] = float(row[col])
                except: d[col] = np.nan
            gv[g] = d
        self.last_group_vals = gv

        # ── Phase 2: cold-start / frozen detection ───────────────────
        analog_vals = {}
        for g in MAIN_SENSORS:
            analog_vals.update(gv[g])
        self.state_mgr.update(analog_vals)
        result["sensor_states"] = dict(self.state_mgr.states)
        result["state_summary"] = self.state_mgr.summary()

        # ── Phase 3: peer deviation ──────────────────────────────────
        all_faults, all_z = [], {}
        for g in MAIN_SENSORS:
            cols    = self.groups.get(g, [])
            healthy = self.state_mgr.get_healthy(cols)
            self.state_mgr.clear_deviating(self._prev_devs[g])
            faults, zscores = peer_deviation_check(gv[g], healthy)
            self.state_mgr.set_deviating(faults)
            self._prev_devs[g] = faults
            all_faults.extend(faults)
            all_z.update(zscores)
        result["faults"]   = all_faults
        result["z_scores"] = all_z

        # ── Phase 4: incremental PCA (stable dim via median imputation)
        pca_vecs = {}
        for g in MAIN_SENSORS:
            cols     = self.groups.get(g, [])
            eligible = self.state_mgr.get_pca_eligible(cols)
            healthy  = self.state_mgr.get_healthy(cols)
            if eligible:
                healthy_vals = {s: gv[g][s] for s in healthy if s in gv[g]}
                median_g = float(np.nanmedian(list(healthy_vals.values()))) if healthy_vals else 0.0
                imputed = {
                    s: median_g if self.state_mgr.states.get(s) == "DEVIATING"
                    else float(gv[g].get(s, median_g))
                    for s in eligible
                }
                pca_vecs[g] = self.pcas[g].update(imputed)
            else:
                pca_vecs[g] = self.pcas[g].update({})
        result["pca_vecs"] = pca_vecs

        # ── Phase 5: latent fusion ───────────────────────────────────
        z_fused = np.concatenate([pca_vecs[g] for g in MAIN_SENSORS])
        result["fused_z"] = z_fused

        # Group means for CUSUM (healthy sensors only → noise cancels)
        group_means = np.array([
            float(np.nanmean([gv[g].get(s, np.nan)
                               for s in self.state_mgr.get_healthy(self.groups.get(g,[]))])
                  ) if self.state_mgr.get_healthy(self.groups.get(g,[]))
                  else 0.0
            for g in CUSUM_GROUPS
        ], dtype=np.float64)
        result["group_means"] = group_means

        # ── Phase 6: temporal model ──────────────────────────────────
        warm      = self.state_mgr.is_warm()
        pca_ready = all(self.pcas[g].is_ready for g in MAIN_SENSORS)
        raw_E     = self.ae.push(z_fused, group_means,
                                 warm=warm, pca_ready=pca_ready)
        result["E"] = raw_E if warm else 0.0

        # ── Phase 7: operational threshold ──────────────────────────
        op_arr = np.array(list(gv["OP"].values()), dtype=np.float64)
        O, theta     = self.op_thresh.update(result["E"], op_arr)
        high_ops     = self.op_thresh.is_high_ops(op_arr)
        result["O"]  = O;  result["theta"] = theta

        # ── Phase 8: decision ────────────────────────────────────────
        result["state"] = decide(
            warm=warm, sensor_faults=all_faults,
            E=result["E"], theta=theta, high_ops=high_ops,
        )
        return result
