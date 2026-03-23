# ============================================================
#  decision.py  –  Phase 8: Decision Engine
# ============================================================
def decide(*, warm, sensor_faults, E, theta, high_ops) -> str:
    if not warm:          return "COLD_START"
    if sensor_faults:     return "SENSOR_FAULT"
    if E <= theta:        return "NORMAL"
    if high_ops:          return "OPERATIONAL"
    return "ANOMALY"
