"""Did the C-boundary publication cause the failure? Compare what the close did in cells that
committed against cells that failed, within each C-treating arm."""
import collections, json, sqlite3, sys

sys.path.insert(0, "/storage/nvme/shapeflow-p1-study/src")
from shapeflow.object_store import ObjectStore  # noqa: E402

agg = collections.defaultdict(lambda: collections.defaultdict(list))
for L in (0, 1):
    root = f"/storage/nvme/shapeflow-data/runner-lane{L}"
    store = ObjectStore(root + "/object_store")
    c = sqlite3.connect(f"file:{root}/runs/ledger.sqlite?mode=ro", uri=True)
    c.row_factory = sqlite3.Row
    q = ("select w.arm_id a, w.state s, x.result_object_ref r from work_items w "
         "join attempts x on x.work_key=w.work_key where w.phase_id='campaign' "
         "and w.state in ('COMMITTED','FAILED_FINAL')")
    for row in c.execute(q):
        if not row["r"]:
            continue
        b = json.loads(store.get_bytes(row["r"]).decode())
        counts = b.get("counts") or {}
        agg[row["a"]][row["s"]].append((
            int(counts.get("close_reduced", 0) or 0),
            int(counts.get("close_failed", 0) or 0),
            int(counts.get("page_batches_reduced", 0) or 0),
            len((b.get("work_summary") or {}).get("unavailable_attempt_ids") or []),
        ))

print(f"{'arm':<18} {'state':<14} {'n':>4} {'close_pub':>10} {'close_fail':>11} "
      f"{'page_batches':>13} {'unavail':>8}")
for arm in sorted(agg):
    for state in ("COMMITTED", "FAILED_FINAL"):
        rows = agg[arm][state]
        if not rows:
            continue
        n = len(rows)
        m = [sum(col) / n for col in zip(*rows)]
        print(f"{arm:<18} {state:<14} {n:>4} {m[0]:>10.2f} {m[1]:>11.2f} "
              f"{m[2]:>13.2f} {m[3]:>8.2f}")
