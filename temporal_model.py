# ============================================================
#  temporal_model.py  –  Phase 6: Temporal Anomaly Detection
#
#  Two channels, each calibrated on cold-start data:
#
#  FAST – Diagonal Kalman on Welford-normalised PCA scores.
#         Calibrated NIS score, normalised to cold-start units.
#         Catches: spikes, step changes, variance bursts.
#
#  SLOW – Page's CUSUM on group means (P + F).
#         S(t) = max(0, S(t-1) + (z(t) - k))
#         z(t) = (mean(t) - μ_ref) / σ_ref    (frozen reference)
#         k    = CUSUM_K  (slack, typically 0.5–1.0)
#         Alarm when S(t) > CUSUM_H
#         Catches: sustained slow pressure / flow drops (leaks).
#
#  Physics:
#   A leak causes a slow monotone decline in P and F group means.
#   Individual sensor noise averages out (σ_mean = σ_sensor/√n).
#   CUSUM accumulates each small deviation → S rises steadily.
#   No adaptive baseline → leak stays visible throughout.
# ============================================================

from __future__ import annotations
import numpy as np
from collections import deque
from config import CUSUM_K, CUSUM_H, COLD_START

_KAL_F    = 0.95
_KAL_Q0   = 1e-4
_KAL_QMAX = 0.5
_KAL_R0   = 5e-3
_FAST_W   = 0.35
_SLOW_W   = 0.65


# ── Welford normalizer (frozen after cold-start) ──────────────────────────────
class _WelfordNorm:
    def __init__(self, dim: int):
        self.n    = 0
        self.mean = np.zeros(dim)
        self._M2  = np.zeros(dim)
        self._frz = False

    def update(self, x: np.ndarray):
        if self._frz: return
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self._M2  += d * (x - self.mean)

    @property
    def std(self):
        return np.sqrt(self._M2 / max(self.n - 1, 1)) + 1e-8

    def transform(self, x):
        return (x - self.mean) / self.std

    def freeze(self): self._frz = True

    @property
    def ready(self): return self.n >= 5


# ── Fast: diagonal Kalman on normalised PCA scores ───────────────────────────
class _FastKalman:
    def __init__(self, dim: int):
        self.dim    = dim
        self._norm  = _WelfordNorm(dim)
        self.x      = np.zeros(dim)
        self.P      = np.ones(dim)
        self.Q      = np.ones(dim) * _KAL_Q0
        self.R      = np.ones(dim) * _KAL_R0
        self._ebuf  : deque = deque(maxlen=20)
        self._csbuf : list  = []   # cold-start NIS values
        self._mu    = 1.0
        self._sig   = 1.0
        self._frozen= False

    def _step(self, z_n: np.ndarray) -> float:
        x_p = _KAL_F * self.x
        P_p = _KAL_F**2 * self.P + self.Q
        e   = z_n - x_p
        S   = P_p + self.R + 1e-12
        nis = float(np.clip(np.mean(e**2 / S), 0, 200))
        K      = P_p / S
        self.x = x_p + K * e
        self.P = np.maximum((1 - K) * P_p, 1e-12)
        self._ebuf.append(e.copy())
        if len(self._ebuf) >= 8:
            q = np.mean(np.array(self._ebuf)**2, axis=0) - P_p
            self.Q = np.clip(.97*self.Q + .03*q, 1e-10, _KAL_QMAX)
        return nis

    def collect(self, z_raw: np.ndarray):
        self._norm.update(z_raw)
        if not self._norm.ready: return
        self._csbuf.append(self._step(self._norm.transform(z_raw)))

    def freeze(self):
        if self._frozen or len(self._csbuf) < 5: return
        a = np.array(self._csbuf)
        self._mu  = float(np.median(a))
        self._sig = float(max(a.std(), 0.1))
        self._norm.freeze()
        self._frozen = True

    def score(self, z_raw: np.ndarray) -> float:
        if not self._frozen: return 0.0
        z_n = self._norm.transform(z_raw)
        nis = self._step(z_n)
        return float(np.clip((nis - self._mu) / self._sig, -3, 50))

    @property
    def ready(self): return self._frozen


# ── Slow: CUSUM on group means ────────────────────────────────────────────────
class _CUSUMDetector:
    """
    Page's CUSUM applied to the mean of selected sensor groups.
    Reference μ and σ frozen from cold-start data.
    Detects downward drift (pressure / flow drop during a leak).
    """
    def __init__(self, n_groups: int):
        self.n       = n_groups
        self._csbuf  : list[np.ndarray] = []
        self._frozen = False
        self.ref_mean: np.ndarray | None = None
        self.ref_std : np.ndarray | None = None
        # Two-sided CUSUM accumulators (one per group)
        self._S_pos  = np.zeros(n_groups)   # upward cusum
        self._S_neg  = np.zeros(n_groups)   # downward cusum
        # Calibration
        self._cal_mu  = 0.0
        self._cal_sig = 1.0

    def collect(self, means: np.ndarray):
        if not self._frozen and np.all(np.isfinite(means)):
            self._csbuf.append(means.copy())

    def freeze(self) -> bool:
        if self._frozen or len(self._csbuf) < 15: return False
        arr = np.array(self._csbuf)
        self.ref_mean = arr.mean(0)
        self.ref_std  = arr.std(0) + 1e-6
        # Self-calibrate: run CUSUM on cold-start data to get typical max-S
        dummy_pos = np.zeros(self.n)
        dummy_neg = np.zeros(self.n)
        maxs = []
        for v in arr:
            z = (v - self.ref_mean) / self.ref_std
            dummy_pos = np.maximum(0, dummy_pos + z  - CUSUM_K)
            dummy_neg = np.maximum(0, dummy_neg - z  - CUSUM_K)
            maxs.append(float(np.max(np.maximum(dummy_pos, dummy_neg))))
        m = np.array(maxs)
        self._cal_mu  = float(np.median(m))
        self._cal_sig = float(max(m.std(), 0.1))
        self._frozen = True
        return True

    def score(self, means: np.ndarray) -> float:
        if not self._frozen or not np.all(np.isfinite(means)): return 0.0
        z = (means - self.ref_mean) / self.ref_std
        self._S_pos = np.maximum(0, self._S_pos + z  - CUSUM_K)
        self._S_neg = np.maximum(0, self._S_neg - z  - CUSUM_K)
        raw = float(np.max(np.maximum(self._S_pos, self._S_neg)))
        return float(np.clip((raw - self._cal_mu) / self._cal_sig, -3, 100))

    def reset(self):
        self._S_pos[:] = 0
        self._S_neg[:] = 0

    @property
    def frozen(self): return self._frozen


# ── Combined model ────────────────────────────────────────────────────────────
class TwoScaleModel:
    """
    push(z_pca, group_means, warm, pca_ready) → float  E(t)
    """
    def __init__(self, pca_dim: int, n_cusum_groups: int):
        self._fast   = _FastKalman(pca_dim)
        self._slow   = _CUSUMDetector(n_cusum_groups)
        self._step   = 0
        self._warmed = False

    def push(self, z_pca: np.ndarray, group_means: np.ndarray,
             warm: bool = True, pca_ready: bool = True) -> float:
        if not warm:
            if pca_ready:
                self._fast.collect(z_pca)
                self._slow.collect(group_means)
            return 0.0
        if not self._warmed:
            self._fast.freeze()
            self._slow.freeze()
            self._warmed = True
        self._step += 1
        f = self._fast.score(z_pca)        if self._fast.ready  else 0.0
        s = self._slow.score(group_means)  if self._slow.frozen else 0.0
        return float(np.clip(np.sqrt((_FAST_W*f)**2 + (_SLOW_W*s)**2), 0, 100))

    @property
    def is_trained(self): return self._warmed and self._step >= 5

    def channel_scores(self, z_pca, group_means):
        f = self._fast.score(z_pca)       if self._fast.ready  else 0.0
        s = self._slow.score(group_means) if self._slow.frozen else 0.0
        return {"fast": f, "slow": s, "cusum_S": self._slow._S_pos.copy()}


# ── LSTM AE (PyTorch, preferred) ─────────────────────────────────────────────
class _LSTMEncoder(object): pass   # imported lazily below


def make_temporal_model(pca_dim: int, n_cusum_groups: int):
    try:
        import torch, torch.nn as nn

        class Enc(nn.Module):
            def __init__(self):
                super().__init__()
                from config import AE_HIDDEN, AE_LATENT
                self.lstm = nn.LSTM(pca_dim, AE_HIDDEN, 2, batch_first=True, dropout=0.1)
                self.proj = nn.Linear(AE_HIDDEN, AE_LATENT)
            def forward(self, x):
                _, (h, _) = self.lstm(x)
                return torch.tanh(self.proj(h[-1]))

        class Dec(nn.Module):
            def __init__(self, seq):
                super().__init__()
                from config import AE_HIDDEN, AE_LATENT
                self.seq  = seq
                self.exp  = nn.Linear(AE_LATENT, AE_HIDDEN)
                self.lstm = nn.LSTM(AE_HIDDEN, AE_HIDDEN, 2, batch_first=True, dropout=0.1)
                self.out  = nn.Linear(AE_HIDDEN, pca_dim)
            def forward(self, z):
                h = self.exp(z).unsqueeze(1).expand(-1, self.seq, -1)
                o, _ = self.lstm(h)
                return self.out(o)

        from config import AE_SEQ_LEN, AE_LR, AE_WEIGHT_DECAY, AE_TRAIN_EVERY
        from config import AE_BUFFER_SIZE, AE_BATCH_SIZE, AE_EPOCHS, AE_GRAD_CLIP
        import random
        from collections import deque as dq

        class LSTMAEModel:
            def __init__(self):
                self.enc  = Enc();  self.dec = Dec(AE_SEQ_LEN)
                self.opt  = torch.optim.Adam(
                    list(self.enc.parameters())+list(self.dec.parameters()),
                    lr=AE_LR, weight_decay=AE_WEIGHT_DECAY)
                self.loss = nn.MSELoss()
                self._win = dq(maxlen=AE_SEQ_LEN)
                self._buf = dq(maxlen=AE_BUFFER_SIZE)
                self._norm= None
                self._step= 0
                self.last_E = 0.0
                self._warm  = False

            def push(self, z_pca, group_means=None, warm=True, pca_ready=True):
                z = np.asarray(z_pca, dtype=np.float32)
                if self._norm is None:
                    self._norm_n = 0; self._norm_m = np.zeros_like(z); self._norm_v = np.ones_like(z)
                self._norm_n += 1
                d = z - self._norm_m; self._norm_m += d/self._norm_n
                self._norm_v += d*(z-self._norm_m)
                z_n = (z - self._norm_m) / (np.sqrt(self._norm_v/max(self._norm_n-1,1))+1e-8)
                self._win.append(z_n)
                if len(self._win)==AE_SEQ_LEN: self._buf.append(np.array(self._win))
                self._step += 1
                if len(self._buf)>=max(AE_BATCH_SIZE,4) and self._step%AE_TRAIN_EVERY==0:
                    self.enc.train(); self.dec.train()
                    k = min(AE_BATCH_SIZE, len(self._buf))
                    x = torch.from_numpy(np.stack(random.sample(list(self._buf),k)))
                    for _ in range(AE_EPOCHS):
                        self.opt.zero_grad()
                        xh = self.dec(self.enc(x))
                        l  = self.loss(xh, x); l.backward()
                        nn.utils.clip_grad_norm_(
                            list(self.enc.parameters())+list(self.dec.parameters()),AE_GRAD_CLIP)
                        self.opt.step()
                    self._warm = True
                if self._warm and len(self._win)==AE_SEQ_LEN:
                    self.enc.eval(); self.dec.eval()
                    with torch.no_grad():
                        x = torch.from_numpy(np.array(self._win)).unsqueeze(0)
                        self.last_E = float(self.loss(self.dec(self.enc(x)), x))
                return self.last_E if warm else 0.0

            @property
            def is_trained(self): return self._warm
            def channel_scores(self, *a, **kw): return {"fast":0,"slow":0}

        print("[TemporalModel] LSTM Autoencoder  (PyTorch)")
        return LSTMAEModel()

    except ImportError:
        print("[TemporalModel] TwoScale CUSUM+Kalman  (numpy)")
        return TwoScaleModel(pca_dim, n_cusum_groups)
