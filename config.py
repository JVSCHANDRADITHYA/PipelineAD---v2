# ============================================================
#  config.py  –  All tunable constants
# ============================================================

DATA_PATH = r"F:\MultiDimensionalAD\data\202505161524GMT+5x30.h24.csv"

# ── Streaming simulation ──────────────────────────────────────
STREAM_INTERVAL_MS = 80          # ms between animation frames

# ── Pipeline hyper-parameters ─────────────────────────────────
COLD_START     = 100             # rows before anomaly logic activates
WINDOW         = 60              # raw-panel rolling display length
HISTORY_LEN    = 500             # dashboard history deque length
PCA_COMPONENTS = 2               # PCs per sensor group
AE_SEQ_LEN     = 20             # (LSTM only) sliding window length
PEER_Z         = 4.5             # MAD Z-score for peer deviation
PEER_HYS       = 3               # consecutive steps to confirm fault
REG_MIN        = 30              # RLS warm-up steps
DELTA          = 1e-8

# ── CUSUM (slow-drift / leak detector) ───────────────────────
CUSUM_K        = 0.75            # slack (multiples of σ)
CUSUM_H        = 6.0             # decision threshold (multiples of σ)
CUSUM_GROUPS   = ["P", "F"]      # groups whose means feed CUSUM
                                 # (pressure + flow drop together in a leak)

# ── RLS threshold ─────────────────────────────────────────────
RLS_LAMBDA     = 0.98
ANOMALY_SIGMA  = 2.5

# ── Operational threshold ─────────────────────────────────────
OP_HIGH_FRAC   = 0.40

# ── LSTM AE (only used when PyTorch is available) ─────────────
AE_HIDDEN          = 32
AE_LATENT          = 8
AE_LR              = 3e-4
AE_WEIGHT_DECAY    = 1e-5
AE_TRAIN_EVERY     = 5
AE_BUFFER_SIZE     = 500
AE_BATCH_SIZE      = 16
AE_EPOCHS          = 3
AE_GRAD_CLIP       = 1.0

# ── Sensor taxonomy ───────────────────────────────────────────
MAIN_SENSORS = {
    "P": ["PI", "PIC"],
    "T": ["TI", "TIC"],
    "F": ["FI", "FIC"],
}
OP_KEYS = ["MOV","PUMP","SCR","XXI","DRA","SBV","TYPE","MP","PMS","PHSD","CBN"]

MAX_RAW_LINES = 18

STATES = {
    "NORMAL"       : "NORMAL",
    "SENSOR_FAULT" : "SENSOR FAULT",
    "OPERATIONAL"  : "OPERATIONAL DRIVEN",
    "ANOMALY"      : "LEAK-LIKE ANOMALY",
    "COLD_START"   : "COLD START",
}
