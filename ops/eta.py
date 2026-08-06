"""Recent per-lane commit throughput for a phase, and an ETA for the remaining owned cells."""
import sqlite3, sys, datetime as dt

PHASE = sys.argv[1] if len(sys.argv) > 1 else "campaign"
now = None
for L in (0, 1):
    root = f"/storage/nvme/shapeflow-data/runner-lane{L}"
    c = sqlite3.connect(f"file:{root}/runs/ledger.sqlite?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    q = ("select updated_at as t from work_items where phase_id=? "
         "and state in ('COMMITTED','FAILED_FINAL') order by updated_at")
    ts = [r["t"] for r in c.execute(q, (PHASE,))]
    if not ts:
        print(f"lane{L}: nothing terminal"); continue
    last = ts[-1]
    n60 = sum(1 for t in ts if t >= last - 3600)
    n180 = sum(1 for t in ts if t >= last - 10800)
    span = ts[-1] - ts[0]
    if now is None:
        now = dt.datetime.now(dt.timezone.utc).timestamp()
    print(f"lane{L}: terminal={len(ts)}  span={span/3600:.1f}h  "
          f"last-60min={n60}  last-3h={n180}  idle={int(now-last)}s")
    rate = n180 / 3.0 if n180 else 0.0
    q2 = "select count(*) as n from work_items where phase_id=? and state='PENDING'"
    print(f"        recent rate {rate:.1f} cells/h")
