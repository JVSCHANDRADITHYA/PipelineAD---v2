# ============================================================
#  analyzer.py  –  Post-Run Static Report + Figure
# ============================================================
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

BG="#0d1117"; PAN="#161b22"; DIM="#8b949e"; BRIG="#e6edf3"
STC={"NORMAL":"#27ae60","SENSOR_FAULT":"#f39c12",
     "OPERATIONAL":"#2980b9","ANOMALY":"#e74c3c","COLD_START":"#4a4e54"}


def analyze_results(results: list, out_dir=".") -> str:
    df = pd.DataFrame([{
        "t": r["t"], "state": r["state"],
        "E": r["E"], "theta": r["theta"] if r["theta"]<1e15 else np.nan,
        "O": r["O"], "n_faults": len(r["faults"]),
        "pca_P0": r["pca_vecs"].get("P",[0,0])[0] if "pca_vecs" in r else 0,
        "pca_T0": r["pca_vecs"].get("T",[0,0])[0] if "pca_vecs" in r else 0,
        "pca_F0": r["pca_vecs"].get("F",[0,0])[0] if "pca_vecs" in r else 0,
    } for r in results])
    return _run(df, out_dir, "batch_run")


def analyze_csv(csv_path: str, out_dir=".") -> str:
    return _run(pd.read_csv(csv_path), out_dir, Path(csv_path).stem)


def _run(df, out_dir, title):
    _report(df, title)
    return _figure(df, out_dir, title)


def _report(df, title):
    n = len(df)
    print(f"\n{'='*60}\n  Post-Run  ·  {title}  ·  {n:,} steps  ·  {df['t'].max():.0f}s\n{'='*60}")
    for st,cnt in df["state"].value_counts().items():
        bar="█"*int(cnt/n*40)
        print(f"  {st:<22s}  {cnt:5d}  ({cnt/n*100:4.1f}%)  {bar}")
    warm = df[df["state"]!="COLD_START"]
    if len(warm):
        print(f"\n  E(t) post cold-start: mean={warm['E'].mean():.5f} "
              f"std={warm['E'].std():.5f} max={warm['E'].max():.5f}")
    anoms = df[df["state"]=="ANOMALY"]
    if len(anoms):
        print(f"\n  ⚠  {len(anoms)} ANOMALY rows  "
              f"(t={anoms['t'].min():.0f}s → {anoms['t'].max():.0f}s)")
    print("="*60+"\n")


def _figure(df, out_dir, title):
    t  = df["t"].values
    fig,axes = plt.subplots(5,1,figsize=(18,16),sharex=True,
        gridspec_kw={"height_ratios":[1,1.5,1.5,1,0.8]})
    fig.patch.set_facecolor(BG)
    for ax in axes:
        ax.set_facecolor(PAN); ax.tick_params(colors=DIM,labelsize=7)
        for sp in ax.spines.values(): sp.set_edgecolor("#333")
    fig.suptitle(f"Pipeline AD — Post-Run · {title}", color=BRIG, fontsize=12, fontweight="bold")

    # State bar
    ax=axes[0]; ax.set_title("System State", color=DIM, fontsize=8, loc="left")
    clrs=[STC.get(s,"#555") for s in df["state"]]
    ax.scatter(t,np.zeros_like(t),c=clrs,s=8,marker="|",linewidths=2)
    ax.set_yticks([]); ax.set_ylim(-1,1)
    ax.legend(handles=[mpatches.Patch(color=v,label=k) for k,v in STC.items()],
              fontsize=6.5,ncol=5,facecolor=PAN,labelcolor=DIM,loc="upper right")

    # E vs θ
    ax=axes[1]; ax.set_title("E(t) vs θ(t)", color=DIM, fontsize=8, loc="left")
    E=df["E"].values; thr=pd.to_numeric(df["theta"],errors="coerce").values
    ax.plot(t,E,color="#ff6b6b",lw=0.8,label="E(t)")
    ax.plot(t,thr,color="#ffd700",lw=1.0,ls="--",label="θ(t)")
    ax.fill_between(t,E,thr,where=np.nan_to_num(E)>np.nan_to_num(thr),
                    alpha=0.22,color="#e74c3c",label="Anomalous")
    ax.legend(fontsize=7,facecolor=PAN,labelcolor=DIM)

    # PCA PC1 per group
    ax=axes[2]; ax.set_title("PCA PC1 per Group", color=DIM, fontsize=8, loc="left")
    for col,clr,lbl in [("pca_P0","#5dade2","P-PC1"),("pca_T0","#ec7063","T-PC1"),("pca_F0","#58d68d","F-PC1")]:
        if col in df.columns: ax.plot(t,df[col].values,color=clr,lw=0.8,label=lbl,alpha=0.85)
    ax.legend(fontsize=7,facecolor=PAN,labelcolor=DIM)

    # O(t)
    ax=axes[3]; ax.set_title("Operational Proxy O(t)", color=DIM, fontsize=8, loc="left")
    ax.plot(t,df["O"].values,color="#bb8fce",lw=0.8)
    ax.axhline(0.4,color="#555",lw=0.8,ls=":"); ax.set_ylim(0,1.1)

    # Fault count
    ax=axes[4]; ax.set_title("Sensor Fault Count", color=DIM, fontsize=8, loc="left")
    if "n_faults" in df.columns:
        ax.fill_between(t,0,df["n_faults"].values,color="#f39c12",alpha=0.5,step="mid")
    ax.set_xlabel("Pipeline Time (s)",color=DIM,fontsize=8)

    plt.tight_layout(rect=[0,0,1,0.97])
    out=Path(out_dir)/f"summary_{title}.png"
    fig.savefig(out,dpi=150,bbox_inches="tight",facecolor=BG,edgecolor="none")
    plt.close(fig)
    print(f"[Analyzer] Saved → {out}")
    return str(out)
