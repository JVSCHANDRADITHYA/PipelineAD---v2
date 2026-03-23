# ============================================================
#  pca_module.py  –  Phase 4: Incremental PCA per Group
# ============================================================
import numpy as np
from sklearn.decomposition import IncrementalPCA
from config import PCA_COMPONENTS

_MIN_FIT = max(PCA_COMPONENTS + 2, 12)


class OnlineGroupPCA:
    def __init__(self, group: str):
        self.name      = group
        self._pca      = None
        self._n_feat   = None
        self._buf      = []
        self._fitted   = False
        self._last_pc  = np.zeros(PCA_COMPONENTS, dtype=np.float32)

    def update(self, vals: dict) -> np.ndarray:
        if len(vals) < PCA_COMPONENTS:
            return self._last_pc.copy()
        x = np.array(list(vals.values()), dtype=np.float64)
        x = np.where(np.isfinite(x), x, np.nanmean(x) if np.isfinite(x).any() else 0.0)
        n = len(x)
        if self._n_feat is not None and n != self._n_feat:
            self._reset()
        self._n_feat = n
        n_comp = min(PCA_COMPONENTS, n - 1)
        if n_comp < 1:
            return self._last_pc.copy()
        self._buf.append(x)
        if not self._fitted and len(self._buf) >= _MIN_FIT:
            X = np.array(self._buf)
            ac = min(n_comp, X.shape[0] - 1)
            self._pca = IncrementalPCA(n_components=ac)
            self._pca.partial_fit(X)
            self._fitted = True
        if self._fitted and self._pca.n_features_in_ == n:
            try:
                proj = self._pca.transform(x.reshape(1, -1))[0]
                if len(proj) < PCA_COMPONENTS:
                    proj = np.pad(proj, (0, PCA_COMPONENTS - len(proj)))
                self._last_pc = proj.astype(np.float32)
            except Exception:
                pass
        return self._last_pc.copy()

    @property
    def is_ready(self):
        return self._fitted

    @property
    def explained_variance_ratio(self):
        return self._pca.explained_variance_ratio_ if self._fitted else None

    def _reset(self):
        self._pca    = None
        self._fitted = False
        self._buf.clear()
        self._last_pc = np.zeros(PCA_COMPONENTS, dtype=np.float32)
