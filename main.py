#!/usr/bin/env python3
# ============================================================
#  main.py  –  Entry Point
#
#  python main.py                    streaming dashboard (real CSV)
#  python main.py --batch            headless batch (real CSV)
#  python main.py --synth            streaming dashboard (synthetic)
#  python main.py --synth --batch    headless batch (synthetic)
#  python main.py --data path.csv    override data path
# ============================================================
import argparse, time, collections
import numpy as np


def _args():
    p = argparse.ArgumentParser()
    p.add_argument("--batch",    action="store_true")
    p.add_argument("--synth",    action="store_true")
    p.add_argument("--data",     default=None)
    p.add_argument("--interval", type=int, default=None)
    p.add_argument("--rows",     type=int, default=2000)
    return p.parse_args()


def _batch(streamer, pipeline, label=""):
    from logger   import EventLogger
    from analyzer import analyze_results
    import tempfile, pathlib
    out = pathlib.Path(tempfile.mkdtemp())
    log = EventLogger(str(out))
    results=[]; t0=time.perf_counter()
    for row in streamer:
        r = pipeline.process(row)
        log.log(r, pipeline); results.append(r)
    elapsed = time.perf_counter()-t0
    log.close()
    print(f"\n[Batch] {len(results):,} rows in {elapsed:.2f}s "
          f"({len(results)/elapsed:.0f} rows/s)")
    counts = collections.Counter(r["state"] for r in results)
    print("\n  ┌─ State distribution ──────────────────────────────┐")
    for st,n in counts.most_common():
        bar="█"*min(int(n/max(len(results),1)*40),40)
        print(f"  │  {st:<22s} {n:5d}  {bar}")
    print("  └──────────────────────────────────────────────────┘")
    analyze_results(results, str(out))
    return results


def _stream(streamer, pipeline, interval_ms):
    import matplotlib
    try:    matplotlib.use("TkAgg")
    except: matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from dashboard import PipelineDashboard
    db   = PipelineDashboard(streamer, pipeline)
    anim = FuncAnimation(db.fig, db.step, interval=interval_ms,
                         blit=False, cache_frame_data=False)
    try:    plt.show()
    except: pass
    return streamer.idx


def main():
    args = _args()
    from pipeline import PipelineAD

    if args.synth:
        from synth_stream import SyntheticStreamer
        streamer = SyntheticStreamer(n_rows=args.rows, leak_magnitude=20.0)
    else:
        from ingestion import CSVStreamer
        from config    import DATA_PATH
        streamer = CSVStreamer(args.data or DATA_PATH)

    pipeline = PipelineAD(streamer.groups)

    if args.batch:
        _batch(streamer, pipeline)
    else:
        from config import STREAM_INTERVAL_MS
        ms = args.interval or STREAM_INTERVAL_MS
        consumed = _stream(streamer, pipeline, ms)
        if streamer.idx < len(streamer):
            print(f"\n[Main] Dashboard closed at row {consumed}."
                  f" Processing remaining {len(streamer)-streamer.idx} rows …")
            _batch(streamer, pipeline)


if __name__ == "__main__":
    main()
