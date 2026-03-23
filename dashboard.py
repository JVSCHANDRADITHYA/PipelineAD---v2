# ============================================================
#  dashboard.py  –  Real-Time Pipeline Dashboard
#
#  Layout (4-row mosaic):
#   ┌──────────┬──────────┬──────────┐
#   │  P raw   │  T raw   │  F raw   │
#   ├──────────┼──────────┴──────────┤
#   │  Avg     │  PCA latent space   │
#   ├──────────┴──────────┬──────────┤
#   │  Error E vs θ(t)    │  O(t)    │
#   ├─────────────────────┴──────────┤
#   │        SYSTEM STATE            │
#   └────────────────────────────────┘
# ============================================================
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyBboxPatch
from collections import deque
from config import (WINDOW, HISTORY_LEN, MAIN_SENSORS, STATES,
                    PCA_COMPONENTS, MAX_RAW_LINES, COLD_START)

BG = "#0d1117"; PAN = "#161b22"; BDR = "#30363d"
DIM = "#8b949e"; BRIG = "#e6edf3"
GC  = {"P": "#5dade2", "T": "#ec7063", "F": "#58d68d"}
STC = {"NORMAL":"#27ae60","SENSOR_FAULT":"#f39c12",
       "OPERATIONAL":"#2980b9","ANOMALY":"#e74c3c","COLD_START":"#6c7a8d"}
STI = {"NORMAL":"●  NORMAL","SENSOR_FAULT":"▲  SENSOR FAULT",
       "OPERATIONAL":"◆  OPERATIONAL DRIVEN",
       "ANOMALY":"✖  LEAK-LIKE ANOMALY","COLD_START":"○  COLD START"}


class PipelineDashboard:
    def __init__(self, streamer, pipeline):
        self.streamer = streamer
        self.pipeline = pipeline
        self._done    = False

        H = HISTORY_LEN
        self.t_h    = deque(maxlen=H)
        self.E_h    = deque(maxlen=H)
        self.thr_h  = deque(maxlen=H)
        self.O_h    = deque(maxlen=H)
        self.pca_h  = {g: [deque(maxlen=H) for _ in range(PCA_COMPONENTS)] for g in MAIN_SENSORS}
        self.avg_h  = {g: deque(maxlen=H) for g in MAIN_SENSORS}
        self.raw_buf= {g: {c: deque(maxlen=WINDOW)
                           for c in streamer.groups.get(g,[])} for g in MAIN_SENSORS}
        self.disp   = {g: streamer.groups.get(g,[])[:MAX_RAW_LINES] for g in MAIN_SENSORS}
        self._sc    = {k: 0 for k in STATES}
        self._tick  = 0
        self._build()

    def _build(self):
        plt.style.use("dark_background")
        self.fig, self.ax = plt.subplot_mosaic(
            [["P","T","F"],["AVG","PCA","PCA"],
             ["ERR","ERR","OP"],["ST","ST","ST"]],
            figsize=(22,14), constrained_layout=True)
        self.fig.patch.set_facecolor(BG)
        for a in self.ax.values():
            a.set_facecolor(PAN)
            a.tick_params(colors=DIM, labelsize=7)
            for sp in a.spines.values(): sp.set_edgecolor(BDR)
        self.fig.suptitle("Pipeline Anomaly Detection  —  Real-Time Stream",
                          color=BRIG, fontsize=13, fontweight="bold")
        titles = {"P":"Pressure (raw)","T":"Temperature (raw)","F":"Flow (raw)",
                  "AVG":"Group Averages","PCA":"PCA Latent Space (PC1 & PC2 per group)",
                  "ERR":"Reconstruction Error  E(t)  vs  Dynamic Threshold  θ(t)",
                  "OP":"Operational Proxy  O(t)","ST":""}
        for k,v in titles.items():
            self.ax[k].set_title(v, color=DIM, fontsize=7.5, loc="left", pad=4)

        self._avg_ln = {}
        for g,c in GC.items():
            ln, = self.ax["AVG"].plot([],[],color=c,lw=1.5,label=f"Avg-{g}")
            self._avg_ln[g] = ln
        self.ax["AVG"].legend(fontsize=7, facecolor=PAN, labelcolor=DIM, loc="upper left")

        self._pca_ln = {}
        for g,c in GC.items():
            for i,ls in enumerate(["-","--"]):
                ln, = self.ax["PCA"].plot([],[],color=c,ls=ls,lw=1.1,alpha=0.85,label=f"{g}-PC{i+1}")
                self._pca_ln[(g,i)] = ln
        self.ax["PCA"].legend(fontsize=6,ncol=3,facecolor=PAN,labelcolor=DIM)

        self._err_ln, = self.ax["ERR"].plot([],[],color="#ff6b6b",lw=0.9,label="E(t)")
        self._thr_ln, = self.ax["ERR"].plot([],[],color="#ffd700",lw=1.3,ls="--",label="θ(t)")
        self.ax["ERR"].legend(fontsize=7,facecolor=PAN,labelcolor=DIM)

        self._op_ln, = self.ax["OP"].plot([],[],color="#bb8fce",lw=1.2)
        self.ax["OP"].set_ylim(0,1.1)
        self.ax["OP"].axhline(0.4,color="#555",lw=0.8,ls=":")

        self.ax["ST"].axis("off")
        self._sbg = FancyBboxPatch((0.02,0.10),0.96,0.82,
            boxstyle="round,pad=0.015", transform=self.ax["ST"].transAxes,
            facecolor="#1e2028", edgecolor="#444", lw=1.5, zorder=1)
        self.ax["ST"].add_patch(self._sbg)
        self._stxt = self.ax["ST"].text(0.5,0.60,STI["COLD_START"],
            transform=self.ax["ST"].transAxes, fontsize=26, fontweight="bold",
            ha="center", va="center", color=STC["COLD_START"], zorder=2)
        self._mtxt = self.ax["ST"].text(0.5,0.32,"",
            transform=self.ax["ST"].transAxes, fontsize=9,
            ha="center", va="center", color=DIM, family="monospace", zorder=2)
        self._ttxt = self.ax["ST"].text(0.98,0.92,"",
            transform=self.ax["ST"].transAxes, fontsize=7.5,
            ha="right", va="top", color=DIM, zorder=2)
        self._wtxt = self.ax["ST"].text(0.5,0.88,"",
            transform=self.ax["ST"].transAxes, fontsize=8,
            ha="center", va="top", color="#7f8c8d", zorder=2)

    def step(self, _frame):
        if self._done: return []
        try:
            row    = next(self.streamer)
            result = self.pipeline.process(row)
            self._ingest(result)
            self._redraw(result)
        except StopIteration:
            self._done = True
            self._stxt.set_text("▶  STREAM ENDED"); self._stxt.set_color(DIM)
        return []

    def _ingest(self, r):
        self.t_h.append(r["t"]); self.E_h.append(r["E"])
        th = r["theta"]
        self.thr_h.append(th if np.isfinite(th) else np.nan)
        self.O_h.append(r["O"])
        self._tick += 1
        self._sc[r["state"]] = self._sc.get(r["state"],0)+1
        gv = self.pipeline.last_group_vals
        for g in MAIN_SENSORS:
            pca = r["pca_vecs"].get(g, np.zeros(PCA_COMPONENTS))
            for i in range(PCA_COMPONENTS):
                self.pca_h[g][i].append(float(pca[i]) if i<len(pca) else 0.0)
            hs = [c for c in self.streamer.groups.get(g,[])
                  if r["sensor_states"].get(c)=="HEALTHY"]
            avg = float(np.nanmean([gv[g].get(c,np.nan) for c in hs])) if hs else np.nan
            self.avg_h[g].append(avg)
            for col in self.disp[g]:
                self.raw_buf[g][col].append(float(gv[g].get(col,np.nan)))

    def _redraw(self, r):
        t  = np.array(self.t_h)
        xl = (float(t[0]), float(t[-1])+1e-6) if len(t) else (0,1)

        self._draw_raw(r)

        for g in MAIN_SENSORS:
            self._avg_ln[g].set_data(t, np.array(self.avg_h[g]))
        self.ax["AVG"].set_xlim(xl)
        _ay(self.ax["AVG"], np.concatenate([np.array(self.avg_h[g]) for g in MAIN_SENSORS]))

        all_y = []
        for g in MAIN_SENSORS:
            for i in range(PCA_COMPONENTS):
                y = np.array(self.pca_h[g][i])
                self._pca_ln[(g,i)].set_data(t, y); all_y.append(y)
        self.ax["PCA"].set_xlim(xl)
        _ay(self.ax["PCA"], np.concatenate(all_y))

        E   = np.array(self.E_h); thr = np.array(self.thr_h)
        self._err_ln.set_data(t, E); self._thr_ln.set_data(t, thr)
        self.ax["ERR"].set_xlim(xl); _ay(self.ax["ERR"], np.concatenate([E, thr[np.isfinite(thr)]]))
        try:
            for c in list(self.ax["ERR"].collections): c.remove()
            valid = np.isfinite(thr)
            if valid.any():
                self.ax["ERR"].fill_between(t[valid], E[valid], thr[valid],
                    where=(E[valid]>thr[valid]), alpha=0.18, color="#e74c3c")
        except Exception: pass

        O = np.array(self.O_h)
        self._op_ln.set_data(t, O); self.ax["OP"].set_xlim(xl)
        self.ax["OP"].set_ylim(0, max(float(np.nanmax(O))*1.15, 1.05) if len(O) else 1.05)

        self._draw_state(r)
        self.fig.canvas.draw_idle()

    def _draw_raw(self, r):
        # Shared time axis for all raw panels — last WINDOW timestamps
        t_raw = np.array(self.t_h)[-WINDOW:]

        for g in MAIN_SENSORS:
            ax = self.ax[g]; ax.cla(); ax.set_facecolor(PAN)
            ax.set_title({"P":"Pressure (raw)","T":"Temperature (raw)","F":"Flow (raw)"}[g],
                         color=DIM, fontsize=7.5, loc="left", pad=4)
            ax.tick_params(colors=DIM, labelsize=6)
            for sp in ax.spines.values(): sp.set_edgecolor(BDR)
            stacks=[]; t_stacks=[]; fault_g=[f for f in r["faults"] if f in self.disp[g]]
            for col in self.disp[g]:
                vals = np.array(self.raw_buf[g][col])
                if not len(vals): continue
                # Align time axis to vals length (buffer may not be full yet)
                t_col = t_raw[-len(vals):]
                st = r["sensor_states"].get(col,"HEALTHY")
                if st=="HEALTHY":
                    ax.plot(t_col,vals,color=GC[g],alpha=0.18,lw=0.7)
                    stacks.append(vals); t_stacks.append(t_col)
                elif st=="DEVIATING":
                    ax.plot(t_col,vals,color="#e74c3c",alpha=0.8,lw=1.2,ls="--",zorder=3)
                else:
                    ax.plot(t_col,vals,color="#444",alpha=0.35,lw=0.5)
            if stacks:
                ml = min(len(v) for v in stacks)
                t_med = t_stacks[0][-ml:]
                ax.plot(t_med,
                        np.nanmedian(np.stack([v[-ml:] for v in stacks]),axis=0),
                        color=GC[g],lw=2.0,alpha=0.95,label="Median",zorder=4)
                ax.legend(fontsize=6,facecolor=PAN,labelcolor=DIM,loc="upper right")
            # Set x-limits to match the rolling time window
            if len(t_raw):
                ax.set_xlim(float(t_raw[0]), float(t_raw[-1]) + 1e-6)
            if fault_g:
                ax.set_title(f"⚠ FAULT: {', '.join(fault_g[:3])}",
                             color="#e74c3c",fontsize=7,loc="right",pad=4)
            if not self.pipeline.state_mgr.is_warm():
                p = self.pipeline.state_mgr.progress()
                ax.axhline(ax.get_ylim()[0],xmin=0,xmax=p,color="#ffd700",lw=3,alpha=0.4)

    def _draw_state(self, r):
        st  = r["state"]; clr = STC.get(st,DIM); lbl = STI.get(st,st)
        self._sbg.set_facecolor(clr+"18"); self._sbg.set_edgecolor(clr+"99")
        self._stxt.set_text(lbl); self._stxt.set_color(clr)
        E = r["E"]; th = r["theta"]; O = r["O"]
        ths = f"{th:.5f}" if np.isfinite(th) else "∞"
        self._mtxt.set_text(
            f"E(t) = {E:.5f}    θ(t) = {ths}    O(t) = {O:.3f}"
            f"    Faults = {len(r['faults'])}    t = {r['t']:.0f} s")
        self._ttxt.set_text(r.get("timestamp",""))
        if not self.pipeline.state_mgr.is_warm():
            self._wtxt.set_text(f"Cold-start  {self.pipeline.state_mgr.progress()*100:.0f}% complete")
        else:
            total = max(self._tick,1)
            self._wtxt.set_text("   ".join(
                f"{k}: {v/total*100:.0f}%" for k,v in self._sc.items() if v>0))


def _ay(ax, v, pad=0.10):
    v = v[np.isfinite(v)] if len(v) else np.array([])
    if not len(v): return
    lo,hi = v.min(),v.max(); r = hi-lo+1e-8
    ax.set_ylim(lo-pad*r, hi+pad*r)